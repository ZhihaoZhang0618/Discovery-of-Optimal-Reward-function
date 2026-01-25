#!/usr/bin/env bash
set -euo pipefail

# Sequential (non-parallel) demos for casestudy3 + casestudy4.
# Designed to surface bugs early: runs one job at a time and stops on the first failure.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

SEED="${SEED:-114514}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
LOG_DIR="${LOG_DIR:-logs/smoke_${DATE_TAG}_seed${SEED}_100k}"
mkdir -p "$LOG_DIR"

NO_CUDA="${NO_CUDA:-1}"
NO_CUDA_FLAG=""
if [[ "$NO_CUDA" == "1" ]]; then
  NO_CUDA_FLAG="--no-cuda"
fi

# -------- casestudy3 (data center) --------
# Approximate ~100k env steps by epochs*max_steps.
CASE3_EPOCHS="${CASE3_EPOCHS:-9}"
CASE3_MAX_STEPS="${CASE3_MAX_STEPS:-12000}"
CASE3_STEPS_PER_MONTH="${CASE3_STEPS_PER_MONTH:-1000}"

# casestudy3 scripts don't accept --no-cuda; force CPU by hiding GPUs.
CASE3_PREFIX=""
if [[ "$NO_CUDA" == "1" ]]; then
  CASE3_PREFIX="CUDA_VISIBLE_DEVICES=''"
fi

run_case3_one() {
  local algo="$1"  # dqn|ppo|sac
  local mode="$2"  # env|learned
  local out="$LOG_DIR/case3_${algo}_${mode}_seed${SEED}.log"

  echo "[case3] $algo $mode -> $out"
  case "$algo" in
    dqn)  eval "$CASE3_PREFIX python casestudy3/dqn_train.py --reward_mode '$mode' --number_epochs '$CASE3_EPOCHS' --max_steps '$CASE3_MAX_STEPS' --steps_per_month '$CASE3_STEPS_PER_MONTH' --seed '$SEED' >'$out' 2>&1" ;;
    ppo)  eval "$CASE3_PREFIX python casestudy3/ppo_train.py --reward_mode '$mode' --number_epochs '$CASE3_EPOCHS' --max_steps '$CASE3_MAX_STEPS' --steps_per_month '$CASE3_STEPS_PER_MONTH' --seed '$SEED' >'$out' 2>&1" ;;
    sac)  eval "$CASE3_PREFIX python casestudy3/sac_train.py --reward_mode '$mode' --number_epochs '$CASE3_EPOCHS' --max_steps '$CASE3_MAX_STEPS' --steps_per_month '$CASE3_STEPS_PER_MONTH' --seed '$SEED' >'$out' 2>&1" ;;
    *) echo "unknown case3 algo: $algo" >&2; exit 2 ;;
  esac
}

# -------- casestudy4 (QuadX) --------
CASE4_TIMESTEPS="${CASE4_TIMESTEPS:-100000}"
CASE4_REWARD_FREQUENCY="${CASE4_REWARD_FREQUENCY:-5000}"

run_case4_one() {
  local algo="$1"  # ppo|sac|td3
  local mode="$2"  # env|learned
  local out="$LOG_DIR/case4_${algo}_${mode}_seed${SEED}.log"

  echo "[case4] $algo $mode -> $out"
  case "$algo" in
    ppo)
      python casestudy4/ppo_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps "$CASE4_TIMESTEPS" $NO_CUDA_FLAG --seed "$SEED" --num-envs 1 --num-steps 128 --reward-mode "$mode" --reward-frequency "$CASE4_REWARD_FREQUENCY" >"$out" 2>&1
      ;;
    sac)
      python casestudy4/sac_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps "$CASE4_TIMESTEPS" $NO_CUDA_FLAG --seed "$SEED" --learning-starts 1000 --reward-mode "$mode" --reward-frequency "$CASE4_REWARD_FREQUENCY" >"$out" 2>&1
      ;;
    td3)
      python casestudy4/td3_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps "$CASE4_TIMESTEPS" $NO_CUDA_FLAG --seed "$SEED" --learning-starts 1000 --reward-mode "$mode" --reward-frequency "$CASE4_REWARD_FREQUENCY" >"$out" 2>&1
      ;;
    *) echo "unknown case4 algo: $algo" >&2; exit 2 ;;
  esac
}

main() {
  # Case3: env vs learned per algo
  run_case3_one dqn env
  run_case3_one dqn learned
  run_case3_one ppo env
  run_case3_one ppo learned
  run_case3_one sac env
  run_case3_one sac learned

  # Case4: env vs learned per algo
  run_case4_one ppo env
  run_case4_one ppo learned
  run_case4_one sac env
  run_case4_one sac learned
  run_case4_one td3 env
  run_case4_one td3 learned

  echo "done"
}

main "$@"
