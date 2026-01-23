import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import namedtuple
from utils.data_center import Environment
from utils import reward_machine
from utils.agent import DQN, Brain
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--train", default=True)
    parser.add_argument("--number_epochs", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.0)
    
    parser.add_argument("--number_actions", type=int, default=5)
    parser.add_argument("--temperature_step", type=float, default=1.5)
    parser.add_argument("--initial_month", type=int, default=0)
    parser.add_argument("--initial_users", type=int, default=20)
    parser.add_argument("--initial_rate_data", type=int, default=30)
    
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_memory", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)

    
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encode_dim", type=int, default=16)
    parser.add_argument("--reward_lr", type=float, default=1e-4)
    parser.add_argument("--activate_function", type=str, default="relu")
    parser.add_argument("--last_activate_function", type=str, default="None")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--reward_buffer_size", type=int, default=100)

    parser.add_argument("--max_steps", type=int, default=5 * 30 * 24 * 60)
    parser.add_argument(
        "--steps_per_month",
        type=int,
        default=30 * 24 * 60,
        help="How many environment steps correspond to one month when computing the month index.",
    )

    parser.add_argument("--seed", type=int, default=1037)
    parser.add_argument("--reward_mode", type=str, default="learned", choices=["env", "learned"])

    return parser.parse_args()


def month_from_timestep(env: Environment, timestep: int, steps_per_month: int) -> int:
    if steps_per_month <= 0:
        return int(env.initial_month) % 12
    return (int(env.initial_month) + (int(timestep) // int(steps_per_month))) % 12


if __name__ == "__main__":
    args = get_args()
    
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direction_boundary = (args.number_actions - 1) / 2
    
    env = Environment(
        optimal_temperature=(18.0, 24.0),
        initial_month=args.initial_month,
        initial_number_users=args.initial_users,
        initial_rate_data=args.initial_rate_data
    )
    env.train = args.train
    
    brain = Brain(args).to(device)
    dqn = DQN(args)

    run_name = f"casestudy3__dqn__{args.reward_mode}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Reward Setup
    reward_function = reward_machine.RewardFunction(env=env,args = args,device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V'])
    
    model_path = "./model/DQN+.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if env.train:
        for epoch in range(1, args.number_epochs + 1):
            env_ep_return = 0.0
            train_ep_return = 0.0
            loss = 0.0
            new_month = np.random.randint(0, 12)
            env.reset(new_month=new_month)
            current_state, _, _ = env.observe()
            game_over = False
            timestep = 0
            epidata = []

            while not game_over and timestep <= args.max_steps:
                
                if np.random.rand() <= args.epsilon:
                    action = np.random.randint(0, args.number_actions)
                    log_prob = torch.zeros((1, args.number_actions), dtype=torch.float32)
                else:
                    with torch.no_grad():
                        current_state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device)
                        q_values = brain(current_state_tensor.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                        log_prob = F.log_softmax(q_values, dim=1)

                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * args.temperature_step

                next_state, env_reward, game_over = env.update_env(
                    direction, energy_ai, month_from_timestep(env, timestep, args.steps_per_month)
                )

                learned_reward = reward_function.observe_reward(
                    np.asarray(current_state), action, np.asarray(next_state)
                )
                learned_reward = float(np.asarray(learned_reward).reshape(-1)[0])
                reward_train = float(env_reward) if args.reward_mode == "env" else learned_reward

                with torch.no_grad():
                    q_values = brain(torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0))
                    transition = Transition(
                        state=np.asarray(current_state),
                        action=action,
                        reward=learned_reward,
                        log_probs=log_prob,
                        mu=q_values.detach().cpu().numpy().reshape(-1),
                        overline_V=0.0,
                    )
                    epidata.append(transition)

                env_ep_return += float(env_reward)
                train_ep_return += reward_train

                
                dqn.remember([current_state, action, reward_train, next_state], game_over)

                
                if len(dqn.memory) >= args.batch_size:
                    inputs, targets = dqn.get_batch(brain, batch_size=args.batch_size)

                    inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=device)
                    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

                    brain.optimizer.zero_grad()
                    predictions = brain(inputs_tensor)
                    batch_loss = brain.loss_fn(predictions, targets_tensor)
                    loss += batch_loss.item()
                    batch_loss.backward()
                    brain.optimizer.step()

                timestep += 1
                current_state = next_state

            
            epidata = reward_function.store_V(epidata)
            
            reward_function.D_xi.append(epidata)
            
            reward_function.optimize_reward(agent=dqn)

            writer.add_scalar("charts/episodic_return", env_ep_return, epoch)
            writer.add_scalar("charts/train_reward", train_ep_return, epoch)
            writer.add_scalar("losses/epoch_loss", loss, epoch)

            # Align with the paper's reporting style: energy reduction rate vs. no-RL baseline.
            denom = float(env.total_energy_noai)
            raw_energy_reduction_rate = (float(env.total_energy_noai) - float(env.total_energy_ai)) / (denom + 1e-8)
            clipped_energy_reduction_rate = max(min(raw_energy_reduction_rate, 1.0), -1.0)
            energy_reduction_valid = 1.0 if denom > 1e-6 else 0.0
            writer.add_scalar("charts/energy_reduction_rate", float(raw_energy_reduction_rate), epoch)
            writer.add_scalar("charts/energy_reduction_rate_clipped", float(clipped_energy_reduction_rate), epoch)
            writer.add_scalar("charts/energy_reduction_rate_valid", float(energy_reduction_valid), epoch)
            writer.add_scalar("charts/total_energy_ai", float(env.total_energy_ai), epoch)
            writer.add_scalar("charts/total_energy_noai", float(env.total_energy_noai), epoch)

            
            torch.save(brain.state_dict(), model_path)

        writer.close()