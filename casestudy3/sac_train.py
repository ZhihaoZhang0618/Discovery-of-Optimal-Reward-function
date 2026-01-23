
import os
import sys
import torch
import numpy as np
import random
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import namedtuple
from utils import data_center
from utils import reward_machine
from utils.agent import SACD_agent
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", type=int, default=1, help="1=train, 0=eval")
    parser.add_argument("--number_epochs", type=int, default=200)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--temperature_step", type=float, default=1.5)
    parser.add_argument("--initial_month", type=int, default=0)
    parser.add_argument("--initial_users", type=int, default=20)
    parser.add_argument("--initial_rate_data", type=int, default=30)
    parser.add_argument("--hid_shape", type=int, nargs="+", default=[64, 64, 16])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--adaptive_alpha", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_memory", type=int, default=100000)
    
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encode_dim", type=int, default=16)
    parser.add_argument("--reward_lr", type=float, default=1e-4)
    parser.add_argument("--activate_function", type=str, default="relu")
    parser.add_argument("--last_activate_function", type=str, default="None")
    parser.add_argument("--reward_buffer_size", type=int, default=100)
    parser.add_argument("--reward_n_samples", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=10)

    parser.add_argument("--max_steps", type=int, default=5 * 30 * 24 * 60)
    parser.add_argument(
        "--steps_per_month",
        type=int,
        default=30 * 24 * 60,
        help="How many environment steps correspond to one month when computing the month index.",
    )

    parser.add_argument("--seed", type=int, default=1037)
    parser.add_argument("--reward_mode", type=str, default="learned", choices=["env", "learned"])
    
    args = parser.parse_args()
    return args


def month_from_timestep(env: data_center.Environment, timestep: int, steps_per_month: int) -> int:
    if steps_per_month <= 0:
        return int(env.initial_month) % 12
    return (int(env.initial_month) + (int(timestep) // int(steps_per_month))) % 12


if __name__ == "__main__":
    args = get_args()
    
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    direction_boundary = (args.action_dim - 1) / 2
    env = data_center.Environment(
            optimal_temperature=(18.0, 24.0),
            initial_month=args.initial_month,
            initial_number_users=args.initial_users,
            initial_rate_data=args.initial_rate_data
        )
    env.train = args.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACD_agent(args, device)

    run_name = f"casestudy3__sac__{args.reward_mode}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    reward_function = reward_machine.RewardFunction(env=env, args=args, device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V'])
    if env.train:
        for epoch in range(1, args.number_epochs + 1):
            env_ep_return = 0.0
            train_ep_return = 0.0
            new_month = np.random.randint(0, 12)
            env.reset(new_month=new_month)
            game_over = False
            current_state, _, _ = env.observe()
            timestep = 0
            epidata = []

            while not game_over and timestep <= args.max_steps:
                action, logprob, pi_s = agent.select_action(current_state, deterministic=False)

                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * args.temperature_step

                next_state, env_reward, game_over = env.update_env(
                    direction,
                    energy_ai,
                    month_from_timestep(env, timestep, args.steps_per_month),
                )

                learned_reward = reward_function.observe_reward(np.asarray(current_state), action, np.asarray(next_state))
                learned_reward = float(np.asarray(learned_reward).reshape(-1)[0])
                reward_train = float(env_reward) if args.reward_mode == "env" else learned_reward

                agent.replay_buffer.add(np.asarray(current_state), action, reward_train, np.asarray(next_state), game_over)

                env_ep_return += float(env_reward)
                train_ep_return += reward_train
                timestep += 1

                transition = Transition(
                    state=np.asarray(current_state),
                    action=action,
                    reward=learned_reward,
                    log_probs=logprob,
                    mu=pi_s,
                    overline_V=0.0,
                )
                epidata.append(transition)

                current_state = next_state

                if agent.replay_buffer.size % args.batch_size == 0:
                    for _ in range(args.batch_size):
                        agent.train()

            epidata = reward_function.store_V(epidata)
            
            reward_function.D_xi.append(epidata)
            
            reward_function.optimize_reward(agent)

            agent.save()

            writer.add_scalar("charts/episodic_return", env_ep_return, epoch)
            writer.add_scalar("charts/train_reward", train_ep_return, epoch)

            denom = float(env.total_energy_noai)
            raw_energy_reduction_rate = (float(env.total_energy_noai) - float(env.total_energy_ai)) / (denom + 1e-8)
            clipped_energy_reduction_rate = max(min(raw_energy_reduction_rate, 1.0), -1.0)
            energy_reduction_valid = 1.0 if denom > 1e-6 else 0.0
            writer.add_scalar("charts/energy_reduction_rate", float(raw_energy_reduction_rate), epoch)
            writer.add_scalar("charts/energy_reduction_rate_clipped", float(clipped_energy_reduction_rate), epoch)
            writer.add_scalar("charts/energy_reduction_rate_valid", float(energy_reduction_valid), epoch)
            writer.add_scalar("charts/total_energy_ai", float(env.total_energy_ai), epoch)
            writer.add_scalar("charts/total_energy_noai", float(env.total_energy_noai), epoch)

        writer.close()
