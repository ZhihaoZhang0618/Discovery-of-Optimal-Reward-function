# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time
from dataclasses import dataclass
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

#from utils.reward_machine import RewardFunction

from utils.reward_machine_re1 import RewardFunction



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    # PPO
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # runtime filled
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Reward learning (upper-level)
    reward_mode: str = "learned"  # 'learned' or 'env'
    reward_frequency: int = 4096  # ✅ 更慢：默认从 1024 提高到 4096（以 global_step 为单位）
    reward_lr: float = 1e-5
    reward_buffer_size: int = 100

    hidden_dim: int = 256
    encode_dim: int = 64
    activate_function: str = "tanh"
    last_activate_function: str = "None"

    # ✅ 新增：reward 更新步数 & 冻结 reward（用于排查漂移）
    reward_updates_per_call: int = 1
    """how many gradient steps to run inside each optimize_reward() call"""

    freeze_reward_after: int = 100000
    """stop updating reward model after this global_step (set <=0 to disable)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), probs

    def get_action_prob_from_mu(self, mu, n):
        actions = mu.sample((n,))
        log_probs = mu.log_prob(actions)
        return actions.transpose(0, 1), log_probs.transpose(0, 1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.reward_mode}__seed{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()

    # Reward model
    reward_function = RewardFunction(env=envs, args=args, device=device) if args.reward_mode == "learned" else None

    # Upper-level buffer: only env0 (kept as your simplification)
    Transition = namedtuple("Transition", ["state", "action", "env_reward", "log_probs", "mu", "overline_V"])
    epidata = []

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # LR anneal
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            now_ob = next_obs

            obs[step] = next_obs
            dones[step] = next_done

            # Action
            with torch.no_grad():
                action, logprob, _, value, mu = agent.get_action_and_value(now_ob)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Env step
            next_obs_np, env_rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            # PPO reward: learned or env
            with torch.no_grad():
                if reward_function is not None:
                    learned_reward = reward_function.observe_reward(now_ob, action, next_obs_np)
                    ppo_reward = np.asarray(learned_reward)
                else:
                    ppo_reward = np.asarray(env_rewards)

            rewards[step] = torch.as_tensor(ppo_reward, device=device, dtype=torch.float32).view(-1)

            # Advance
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)

            # =========================
            # Upper-level data: env0 only
            # =========================
            if reward_function is not None:
                env_r0 = float(np.asarray(env_rewards).reshape(-1)[0])
                transition = Transition(
                    state=now_ob[0].detach().cpu().numpy(),
                    action=int(action[0].detach().cpu().item()),
                    env_reward=env_r0,
                    log_probs=float(logprob[0].detach().cpu().item()),
                    mu=mu.probs[0].detach().cpu().numpy(),
                    overline_V=0.0,
                )
                epidata.append(transition)

            # Logging: env0 final_info only
            if "final_info" in infos and infos["final_info"] is not None:
                finfo0 = infos["final_info"][0]
                if finfo0 is not None and "episode" in finfo0:
                    writer.add_scalar("charts/episodic_return", finfo0["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", finfo0["episode"]["l"], global_step)

            # ✅ 必须修：只在 env0 结束时 store_V
            if reward_function is not None and bool(next_done_np[0]):
                reward_function.D_xi.append(reward_function.store_V(epidata))
                epidata = []

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize PPO
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Explained variance
        y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # =========================
        # Reward learning update (slower + optional freeze)
        # =========================
        if reward_function is not None and (global_step % args.reward_frequency == 0):
            do_freeze = (args.freeze_reward_after > 0) and (global_step >= args.freeze_reward_after)
            writer.add_scalar("reward/frozen", float(do_freeze), global_step)

            if not do_freeze:
                # multiple gradient steps (usually 1 is enough; keep small for stability)
                for _ in range(int(max(1, args.reward_updates_per_call))):
                    reward_function.optimize_reward(agent)

        # Log stats
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if len(clipfracs) else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"iter={iteration}/{args.num_iterations}  step={global_step}  SPS={sps}")

    envs.close()
    writer.close()

