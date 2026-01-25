#!/usr/bin/env bash
set -euo pipefail

# Runs "longer" multi-seed experiments for casestudy1/2/4 (env vs learned) and
# a scaled-year setup for casestudy3.
#
# Usage examples:
#   ./tools/run_longer_multiseed.sh case3
#   ./tools/run_longer_multiseed.sh case1
#   NO_CUDA=1 ./tools/run_longer_multiseed.sh case1
#   ./tools/run_longer_multiseed.sh all

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

NO_CUDA_FLAG=""
if [[ "${NO_CUDA:-0}" == "1" ]]; then
  NO_CUDA_FLAG="--no-cuda"
fi

CUDA_PREFIX=""
if [[ "${NO_CUDA:-0}" == "1" ]]; then
  # casestudy3 scripts don't accept --no-cuda; force CPU by hiding GPUs.
  CUDA_PREFIX="CUDA_VISIBLE_DEVICES=''"
fi

PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

run_commands() {
  # Usage: run_commands cmd1 cmd2 ...
  # If PARALLEL_JOBS>1, execute commands in parallel.
  local -a cmds=("$@")
  if [[ "${PARALLEL_JOBS}" -le 1 ]]; then
    local cmd
    for cmd in "${cmds[@]}"; do
      echo "$cmd"
      eval "$cmd"
    done
    return
  fi

  if [[ "${NO_CUDA:-0}" != "1" ]]; then
    echo "[warn] PARALLEL_JOBS=${PARALLEL_JOBS} with CUDA enabled may OOM or slow down due to GPU contention." >&2
    echo "[warn] Consider setting NO_CUDA=1 or PARALLEL_JOBS=1." >&2
  fi

  printf '%s\0' "${cmds[@]}" | xargs -0 -n1 -P "${PARALLEL_JOBS}" bash -c
}

run_case3() {
  # Scaled-year configuration: 12 "months" per episode via steps_per_month.
  # This makes energy-related metrics meaningful without 500k+ steps per episode.
  local epochs="${CASE3_EPOCHS:-10}"
  local max_steps="${CASE3_MAX_STEPS:-12000}"
  local steps_per_month="${CASE3_STEPS_PER_MONTH:-1000}"

  local seeds_str="${CASE3_SEEDS:-1 2 3 4 5}"
  local modes_str="${CASE3_MODES:-env learned}"
  local algos_str="${CASE3_ALGOS:-dqn ppo sac}"
  seeds_str="${seeds_str//,/ }"
  modes_str="${modes_str//,/ }"
  algos_str="${algos_str//,/ }"

  local -a cmds=()
  for seed in $seeds_str; do
    for mode in $modes_str; do
      for algo in $algos_str; do
        case "$algo" in
          dqn)
            cmds+=("$CUDA_PREFIX python casestudy3/dqn_train.py --reward_mode '$mode' --number_epochs '$epochs' --max_steps '$max_steps' --steps_per_month '$steps_per_month' --seed '$seed' >'$LOG_DIR/case3_dqn_${mode}_seed${seed}.log' 2>&1")
            ;;
          ppo)
            cmds+=("$CUDA_PREFIX python casestudy3/ppo_train.py --reward_mode '$mode' --number_epochs '$epochs' --max_steps '$max_steps' --steps_per_month '$steps_per_month' --seed '$seed' >'$LOG_DIR/case3_ppo_${mode}_seed${seed}.log' 2>&1")
            ;;
          sac)
            cmds+=("$CUDA_PREFIX python casestudy3/sac_train.py --reward_mode '$mode' --number_epochs '$epochs' --max_steps '$max_steps' --steps_per_month '$steps_per_month' --seed '$seed' >'$LOG_DIR/case3_sac_${mode}_seed${seed}.log' 2>&1")
            ;;
          *)
            echo "Unknown CASE3_ALGOS entry: $algo" >&2
            exit 2
            ;;
        esac
      done
    done
  done
  run_commands "${cmds[@]}"
}

run_case1() {
  local timesteps="${CASE1_TIMESTEPS:-200000}"
  local reward_frequency="${CASE1_REWARD_FREQUENCY:-}"

  local reward_frequency_flag=""
  if [[ -n "${reward_frequency}" ]]; then
    reward_frequency_flag="--reward-frequency '${reward_frequency}'"
  fi

  local seeds_str="${CASE1_SEEDS:-1 2 3 4 5}"
  local modes_str="${CASE1_MODES:-env learned}"
  local algos_str="${CASE1_ALGOS:-dqn ppo}"
  seeds_str="${seeds_str//,/ }"
  modes_str="${modes_str//,/ }"
  algos_str="${algos_str//,/ }"

  local -a cmds=()
  for seed in $seeds_str; do
    for mode in $modes_str; do
      for algo in $algos_str; do
        case "$algo" in
          dqn)
            cmds+=("python casestudy1/dqn.py --env-id CartPole-v1 --total-timesteps '$timesteps' $NO_CUDA_FLAG --seed '$seed' --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case1_dqn_${mode}_seed${seed}.log' 2>&1")
            ;;
          ppo)
            cmds+=("python casestudy1/ppo.py --env-id CartPole-v1 --total-timesteps '$timesteps' $NO_CUDA_FLAG --seed '$seed' --num-envs 1 --num-steps 128 --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case1_ppo_${mode}_seed${seed}.log' 2>&1")
            ;;
          *)
            echo "Unknown CASE1_ALGOS entry: $algo" >&2
            exit 2
            ;;
        esac
      done
    done
  done
  run_commands "${cmds[@]}"
}

run_case2() {
  # Reacher-v4 is much slower than CartPole; adjust timesteps as needed.
  local ppo_timesteps="${CASE2_PPO_TIMESTEPS:-200000}"
  local sac_timesteps="${CASE2_SAC_TIMESTEPS:-200000}"
  local reward_frequency="${CASE2_REWARD_FREQUENCY:-}"

  local reward_frequency_flag=""
  if [[ -n "${reward_frequency}" ]]; then
    reward_frequency_flag="--reward-frequency '${reward_frequency}'"
  fi

  local seeds_str="${CASE2_SEEDS:-1 2 3 4 5}"
  local modes_str="${CASE2_MODES:-env learned}"
  local algos_str="${CASE2_ALGOS:-ppo sac}"
  seeds_str="${seeds_str//,/ }"
  modes_str="${modes_str//,/ }"
  algos_str="${algos_str//,/ }"

  local -a cmds=()
  for seed in $seeds_str; do
    for mode in $modes_str; do
      for algo in $algos_str; do
        case "$algo" in
          ppo)
            cmds+=("python casestudy2/ppo_continuous_action.py --env-id Reacher-v4 --total-timesteps '$ppo_timesteps' $NO_CUDA_FLAG --seed '$seed' --num-envs 1 --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case2_ppo_${mode}_seed${seed}.log' 2>&1")
            ;;
          sac)
            cmds+=("python casestudy2/sac_continuous_action.py --env-id Reacher-v4 --total-timesteps '$sac_timesteps' $NO_CUDA_FLAG --seed '$seed' --learning-starts 1000 --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case2_sac_${mode}_seed${seed}.log' 2>&1")
            ;;
          *)
            echo "Unknown CASE2_ALGOS entry: $algo" >&2
            exit 2
            ;;
        esac
      done
    done
  done
  run_commands "${cmds[@]}"
}

run_case4() {
  # PyFlyt/QuadX-Waypoints-v4 can be slow and sometimes CPU-bound; tune timesteps.
  local ppo_timesteps="${CASE4_PPO_TIMESTEPS:-200000}"
  local sac_timesteps="${CASE4_SAC_TIMESTEPS:-200000}"
  local td3_timesteps="${CASE4_TD3_TIMESTEPS:-200000}"
  local reward_frequency="${CASE4_REWARD_FREQUENCY:-}"

  local reward_frequency_flag=""
  if [[ -n "${reward_frequency}" ]]; then
    reward_frequency_flag="--reward-frequency '${reward_frequency}'"
  fi

  local seeds_str="${CASE4_SEEDS:-1 2 3 4 5}"
  local modes_str="${CASE4_MODES:-env learned}"
  local algos_str="${CASE4_ALGOS:-ppo sac td3}"
  seeds_str="${seeds_str//,/ }"
  modes_str="${modes_str//,/ }"
  algos_str="${algos_str//,/ }"

  local -a cmds=()
  for seed in $seeds_str; do
    for mode in $modes_str; do
      for algo in $algos_str; do
        case "$algo" in
          ppo)
            cmds+=("python casestudy4/ppo_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps '$ppo_timesteps' $NO_CUDA_FLAG --seed '$seed' --num-envs 1 --num-steps 128 --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case4_ppo_${mode}_seed${seed}.log' 2>&1")
            ;;
          sac)
            cmds+=("python casestudy4/sac_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps '$sac_timesteps' $NO_CUDA_FLAG --seed '$seed' --learning-starts 1000 --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case4_sac_${mode}_seed${seed}.log' 2>&1")
            ;;
          td3)
            cmds+=("python casestudy4/td3_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps '$td3_timesteps' $NO_CUDA_FLAG --seed '$seed' --learning-starts 1000 --reward-mode '$mode' ${reward_frequency_flag} >'$LOG_DIR/case4_td3_${mode}_seed${seed}.log' 2>&1")
            ;;
          *)
            echo "Unknown CASE4_ALGOS entry: $algo" >&2
            exit 2
            ;;
        esac
      done
    done
  done
  run_commands "${cmds[@]}"
}

case "${1:-}" in
  case1) run_case1 ;;
  case2) run_case2 ;;
  case3) run_case3 ;;
  case4) run_case4 ;;
  all) run_case3; run_case1; run_case2; run_case4 ;;
  *)
    echo "Usage: $0 {case1|case2|case3|case4|all}" >&2
    exit 2
    ;;
esac

echo "Done. TensorBoard runs are under ./runs/"
