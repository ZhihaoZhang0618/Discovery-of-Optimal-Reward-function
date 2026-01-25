#!/usr/bin/env bash
set -euo pipefail

# One-shot smoke run at a fixed seed, comparing env vs learned.
# This wraps tools/run_longer_multiseed.sh with 100k-ish budgets.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
SEED="${SEED:-114514}"
LOG_DIR="${LOG_DIR:-logs/smoke_${DATE_TAG}_seed${SEED}_100k}"

mkdir -p "$LOG_DIR"

# Common knobs
export NO_CUDA="${NO_CUDA:-1}"
export PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

# Case1 (CartPole)
export CASE1_SEEDS="$SEED"
export CASE1_MODES="env learned"
export CASE1_ALGOS="ppo dqn"
export CASE1_TIMESTEPS="${CASE1_TIMESTEPS:-100000}"
export CASE1_REWARD_FREQUENCY="${CASE1_REWARD_FREQUENCY:-5000}"

# Case2 (Reacher)
export CASE2_SEEDS="$SEED"
export CASE2_MODES="env learned"
export CASE2_ALGOS="ppo sac"
export CASE2_PPO_TIMESTEPS="${CASE2_PPO_TIMESTEPS:-100000}"
export CASE2_SAC_TIMESTEPS="${CASE2_SAC_TIMESTEPS:-100000}"
export CASE2_REWARD_FREQUENCY="${CASE2_REWARD_FREQUENCY:-5000}"

# Case3 (data center): approximate ~100k steps via 9 * 12000 = 108k
export CASE3_SEEDS="$SEED"
export CASE3_MODES="env learned"
export CASE3_ALGOS="dqn ppo sac"
export CASE3_EPOCHS="${CASE3_EPOCHS:-9}"
export CASE3_MAX_STEPS="${CASE3_MAX_STEPS:-12000}"
export CASE3_STEPS_PER_MONTH="${CASE3_STEPS_PER_MONTH:-1000}"

# Case4 (QuadX)
export CASE4_SEEDS="$SEED"
export CASE4_MODES="env learned"
export CASE4_ALGOS="ppo sac td3"
export CASE4_PPO_TIMESTEPS="${CASE4_PPO_TIMESTEPS:-100000}"
export CASE4_SAC_TIMESTEPS="${CASE4_SAC_TIMESTEPS:-100000}"
export CASE4_TD3_TIMESTEPS="${CASE4_TD3_TIMESTEPS:-100000}"
export CASE4_REWARD_FREQUENCY="${CASE4_REWARD_FREQUENCY:-5000}"

export LOG_DIR

echo "[smoke] logs -> $LOG_DIR"

echo "[smoke] case1"
bash tools/run_longer_multiseed.sh case1

echo "[smoke] case2"
bash tools/run_longer_multiseed.sh case2

echo "[smoke] case3"
bash tools/run_longer_multiseed.sh case3

echo "[smoke] case4"
bash tools/run_longer_multiseed.sh case4

echo "[smoke] done"
