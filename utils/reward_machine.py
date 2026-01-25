import numpy as np
import random
import torch
import torch.optim as optim
from collections import deque

from .reward_model import Critic, Reward


class RewardFunction():
    def __init__(self, env, args, device):
        """
        Initialize the reward function used for upper-level optimization

        Args:
            env: Gym-like environment
            args: 
            device: "cpu" or "cuda"
        """
        activation_function_list = {
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "None": None
        }
        self.hidden_dim = args.hidden_dim
        self.encode_dim = args.encode_dim
        self.gamma = args.gamma
        self.lr = args.reward_lr
        self.activate_function = activation_function_list[args.activate_function]
        self.last_activate_function = activation_function_list[args.last_activate_function]
        self.device = device

        obs_space = getattr(env, "single_observation_space", None) or getattr(env, "observation_space", None)
        act_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
        self.obs_space = obs_space
        self.act_space = act_space

        if obs_space is not None:
            self.state_dim = int(np.prod(obs_space.shape))
        else:
            # Fallback for non-Gym custom envs (e.g., casestudy3 data center)
            if not hasattr(args, "state_dim"):
                raise ValueError("env has no observation_space; please provide args.state_dim")
            self.state_dim = int(args.state_dim)

        # Action encoding
        if act_space is not None:
            if hasattr(act_space, "n"):
                # Discrete
                self.is_discrete = True
                self.n_actions = int(act_space.n)
                self.action_dim = self.n_actions
            else:
                # Continuous Box
                self.is_discrete = False
                self.n_actions = None
                self.action_dim = int(np.prod(act_space.shape))
        else:
            # Fallback: assume discrete actions with size from args
            n_actions = None
            if hasattr(args, "action_dim"):
                n_actions = int(args.action_dim)
            elif hasattr(args, "number_actions"):
                n_actions = int(args.number_actions)
            if not n_actions or n_actions <= 0:
                raise ValueError("env has no action_space; please provide args.action_dim (discrete) or use a Gym env")
            self.is_discrete = True
            self.n_actions = n_actions
            self.action_dim = n_actions

        # Value function V(s) used to approximate standard state values
        self.value_function = Critic(layer_num=3, input_dim=self.state_dim, output_dim=1,\
                                    hidden_dim=self.hidden_dim,
                                    activation_function=self.activate_function, last_activation = None).to(device=self.device)
        self.value_function_optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)
        # Reward function R(s, a)
        self.reward_function = Reward(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, \
                                     encode_dim=self.encode_dim,\
                                     output_dim=1,\
                                     activation_function=self.activate_function,\
                                     last_activation=self.last_activate_function)\
                                    .to(self.device)
        self.reward_function_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.lr)
        self.D_xi = deque(maxlen=args.reward_buffer_size) # Trajectory buffer

        # Number of sampled actions for estimating action set (continuous only)
        self.n_samples = int(getattr(args, "n_samples", 10))

    def _to_state_tensor(self, state):
        if isinstance(state, torch.Tensor):
            state_t = state
        else:
            state_t = torch.as_tensor(state)
        state_t = state_t.to(self.device, dtype=torch.float32)
        state_t = state_t.view(-1, self.state_dim)
        return state_t

    def _encode_action(self, action, batch_size: int | None = None):
        if isinstance(action, torch.Tensor):
            action_t = action
        else:
            action_t = torch.as_tensor(action)

        if self.is_discrete:
            action_t = action_t.to(self.device)
            action_t = action_t.view(-1).long()
            return torch.nn.functional.one_hot(action_t, num_classes=self.n_actions).to(dtype=torch.float32)

        action_t = action_t.to(self.device, dtype=torch.float32)
        action_t = action_t.view(-1, self.action_dim)
        return action_t

    def _get_action_probs(self, mu):
        # mu can be a Categorical distribution or raw logits/q-values
        if hasattr(mu, "probs"):
            probs = mu.probs
            return probs.to(self.device, dtype=torch.float32).view(-1)

        vec = mu
        if not isinstance(vec, torch.Tensor):
            vec = torch.as_tensor(vec, device=self.device, dtype=torch.float32)
        else:
            vec = vec.to(self.device, dtype=torch.float32)
        vec = vec.view(-1)

        # Heuristic: if it already looks like probabilities, normalize gently.
        if torch.all(vec >= 0) and torch.all(vec <= 1.0 + 1e-3) and torch.isfinite(vec).all():
            s = vec.sum()
            if torch.isfinite(s) and (s > 0.99) and (s < 1.01):
                return vec / s

        return torch.softmax(vec, dim=0)

    def observe_reward(self, state, action, next_state=None):
        """
        Give the current reward function on a given state-action pair
        Args:
            state: Current state
            action: Taken action
            next_state (optional): Next state (unused)

        Returns:
            Reward signal for the state-action pair
        """
        state_t = self._to_state_tensor(state)
        action_t = self._encode_action(action)
        reward = self.reward_function.forward(state_t, action_t).detach().cpu().numpy().squeeze(-1)
        return reward

    # Backward-compatible typo
    def ovserve_reward(self, state, action, next_state=None):
        return self.observe_reward(state, action, next_state=next_state)

    def optimize_reward(self, agent):
        """
        Perform the upper-level optimization step to update the reward function
        
        """
        if len(self.D_xi) == 0:
            return

        # Decompose trajectories into transitions
        transitions = [step for traj in self.D_xi for step in traj]
        if len(transitions) == 0:
            return

        # Keep optimization cost bounded.
        max_transitions = 2048
        if len(transitions) > max_transitions:
            transitions = random.sample(transitions, k=max_transitions)
        else:
            np.random.shuffle(transitions)

        states_batch, overline_V_batch = [], []
        losses = []

        for step in transitions:
            s, a, reward_hat, _log_probs, mu, overline_V = step

            state_t = self._to_state_tensor(s)  # [1, state_dim]
            V_s = self.value_function(state_t).detach().view(())
            overline_V_t = torch.as_tensor(overline_V, device=self.device, dtype=torch.float32).view(())

            if self.is_discrete:
                probs = self._get_action_probs(mu)  # [n_actions]
                all_actions = torch.eye(self.n_actions, device=self.device, dtype=torch.float32)
                state_rep = state_t.repeat(self.n_actions, 1)
                r_all = self.reward_function(state_rep, all_actions).squeeze(-1)  # [n_actions]
                reward_center = (probs * r_all).sum()
                a_idx = int(a)
                pi_a = probs[a_idx].detach()
            else:
                # Continuous: approximate with samples from the current policy (requires agent helper)
                action_bs, log_probs_action_bs = agent.get_action_prob_from_mu(mu, self.n_samples)
                action_enc = self._encode_action(action_bs)
                state_rep = state_t.repeat(action_enc.shape[0], 1)
                reward_bs = self.reward_function(state_rep, action_enc).squeeze(-1)
                probs_bs = torch.exp(log_probs_action_bs.to(self.device, dtype=torch.float32)).view(-1)
                reward_center = (probs_bs * reward_bs).sum()
                pi_a = torch.tensor(1.0, device=self.device)

            reward_hat_t = torch.as_tensor(reward_hat, device=self.device, dtype=torch.float32).view(())
            term1 = reward_hat_t - reward_center
            term2 = pi_a * (overline_V_t - V_s)
            losses.append(term2 * term1)

            states_batch.append(np.asarray(s))
            overline_V_batch.append(float(overline_V))

        # Update reward model first.
        # Some training loops call this alongside other model updates; doing the reward step
        # before value-function regression avoids accidental inplace-version conflicts.
        loss = torch.mean(torch.stack(losses))
        self.reward_function_optimizer.zero_grad()
        loss.backward()
        self.reward_function_optimizer.step()

        # Then regress value function V(s) toward overline_V.
        self.optimize_value_function(np.asarray(states_batch), np.asarray(overline_V_batch))

    def optimize_value_function(self, states_batch, overline_V_batch):
        """
        Optimize the value function V(s) to regress toward the standardized return overline_V
        
        """
        # array --> tensor
        states_batch = torch.as_tensor(states_batch, device=self.device, dtype=torch.float32)
        overline_V_batch = torch.as_tensor(overline_V_batch, device=self.device, dtype=torch.float32)
        pred_batch = self.value_function.forward(states_batch)
        loss = torch.nn.functional.smooth_l1_loss(pred_batch.view(-1), overline_V_batch.view(-1))
        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()

    def store_V(self, epidata):
        """
        Calculate and store standardized return overline_V for each step in a trajectory

        Args:
            epidata: A list of trajectory steps.

        Returns:
            A new list of steps with computed overline_V added to each step.
        """
        new_epidata = []
        overline_V = 0.0
        for step in reversed(epidata):
            overline_V = step.reward + self.gamma * overline_V
            updated_step = step._replace(overline_V=overline_V)
            new_epidata.insert(0, updated_step)
        return new_epidata
