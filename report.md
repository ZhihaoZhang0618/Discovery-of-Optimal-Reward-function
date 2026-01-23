# 复现报告（持续更新，中文版）

仓库：https://github.com/zhshao17/Discovery-of-Optimal-Reward-function

本文件是当前唯一“主报告”，用于持续记录复现过程与对比结果。

---

## 1. 总览

目标：按照仓库 README/脚本跑通各 casestudy，并对比两种训练奖励设置：

- `reward_mode=env`：用**环境奖励**训练（baseline）
- `reward_mode=learned`：用**学习到的奖励模型**训练（论文核心）

对比时坚持同一口径：无论训练奖励来自哪里，最终指标统一使用环境真实回报 `charts/episodic_return`（由 `gym.wrappers.RecordEpisodeStatistics` 记录），保证可比性。

说明：当前记录均为 smoke-run（短跑）用于验证脚本可运行、指标可抽取；要得出稳定结论，需要拉长步数并做多 seed 重复。

为了方便批量运行，我新增了脚本 [tools/run_longer_multiseed.sh](tools/run_longer_multiseed.sh)（默认 seeds=1–5，env vs learned），可按需调大 timesteps/epochs（见脚本内环境变量）。

## 2. 环境与限制

- OS：Linux
- Python：3.10（通过 `uv` 创建虚拟环境）
- 当前默认以 CPU 为主（`--no-cuda`），但已完成 GPU 启用。
- GPU：NVIDIA GeForce RTX 5070 Ti（sm_120），驱动显示 CUDA 12.8。
  - 仓库默认依赖 pin 的 `torch==1.12.1+cu102` / 以及 `torch==2.6.0+cu124` 均不包含 sm_120 的 kernel（会报 `no kernel image is available`）。
  - 为启用 GPU，本环境改用 PyTorch nightly（CUDA 12.8）并验证可运行 CUDA kernel：
    - `python -m pip install -U --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio`
    - 当前版本：`torch==2.11.0.dev20260116+cu128`（会偏离仓库 pinned 依赖）。

## 3. 口径与公平性（防“trick”原则）

- **统一评估指标**：比较使用环境 episodic return（`charts/episodic_return`），而不是 learned reward 自己的数。
- **同一组超参**：同一算法下，env vs learned 仅切换 `--reward-mode`，其它训练超参保持一致。
- **随机种子**：脚本会设置 `random/numpy/torch` seed，并调用 `envs.reset(seed=seed)`。

重要说明（逻辑一致性）：2026-01-18 我修复了一个会影响 `reward_mode=learned` 结论的实现问题：用于上层 reward-learning 的轨迹缓冲 `epidata`/`store_V` 在部分脚本中曾误存了环境奖励而非 learned reward（导致 reward-learning 目标与 RL 更新不一致）。因此，修复前跑出的 case1/case2 “longer”结果只能作为参考；若要严格对照论文，需要在修复后重新跑同配置。

## 4. 复现进度与结果（Smoke runs）

### 4.1 Casestudy1：CartPole-v1（离散控制）

运行命令（CPU、seed=1、50k steps）：

- DQN env：`python casestudy1/dqn.py --env-id CartPole-v1 --total-timesteps 50000 --no-cuda --seed 1 --reward-mode env`
- DQN learned：`python casestudy1/dqn.py --env-id CartPole-v1 --total-timesteps 50000 --no-cuda --seed 1 --reward-mode learned`
- PPO env：`python casestudy1/ppo.py --env-id CartPole-v1 --total-timesteps 50000 --no-cuda --seed 1 --num-envs 1 --num-steps 128 --reward-mode env`
- PPO learned：`python casestudy1/ppo.py --env-id CartPole-v1 --total-timesteps 50000 --no-cuda --seed 1 --num-envs 1 --num-steps 128 --reward-mode learned`

对比指标：TensorBoard scalar `charts/episodic_return`。

结果汇总（CartPole-v1, 50k steps, seed=1）：

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| DQN | env | 217.70 | 500 |
| DQN | learned | 184.45 | 500 |
| PPO | env | 209.15 | 500 |
| PPO | learned | 265.30 | 500 |

运行目录：

- `runs/CartPole-v1__dqn__env__1__1768633906`
- `runs/CartPole-v1__dqn__learned__1__1768633933`
- `runs/CartPole-v1__ppo__env__1__1768633999`
- `runs/CartPole-v1__ppo__learned__1__1768634035`

简单解读（单 seed 短跑，仅供参考）：PPO 的 learned reward 在 mean(last20) 上更高；DQN 的 learned reward 更低。

多 seed 统计（CartPole-v1, 50k steps, seeds=1–5；统计对象为每个 run 的 mean(last 20 episodes)）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| DQN | env | 220.650 | 53.231 | 500 |
| DQN | learned | 239.530 | 84.352 | 500 |
| PPO | env | 257.590 | 57.161 | 500 |
| PPO | learned | 211.680 | 38.562 | 500 |

进一步的 longer multi-seed（CartPole-v1, 200k steps, seeds=1–5；统计对象为每个 run 的 mean(last 100 episodes)，默认启用 CUDA）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| DQN | env | 467.470 | 35.441 | 500 |
| DQN | learned | 398.902 | 118.953 | 500 |
| PPO | env | 496.468 | 6.826 | 500 |
| PPO | learned | 261.034 | 48.546 | 500 |

简单解读（longer 版本更可信）：在本仓库实现与默认超参下，CartPole 上 `reward_mode=learned` 并没有稳定优于 `reward_mode=env`；尤其 PPO 的 learned reward 表现显著更差且方差更大。

### 4.2 Casestudy2：Reacher-v4（MuJoCo 连续控制）

额外依赖：gymnasium 的 MuJoCo v4 环境需要 `mujoco` Python 包。

- 安装：
  - `python -m ensurepip --upgrade`
  - `python -m pip install mujoco==2.3.3`
- 快速检查：
  - `python -c "import gymnasium as gym; env=gym.make('Reacher-v4'); env.reset(seed=0); env.step(env.action_space.sample()); print('ok'); env.close()"`

PPO（8192 steps，CPU）：

- env：`python casestudy2/ppo_continuous_action.py --env-id Reacher-v4 --total-timesteps 8192 --num-envs 1 --no-cuda --reward-mode env`
- learned：`python casestudy2/ppo_continuous_action.py --env-id Reacher-v4 --total-timesteps 8192 --num-envs 1 --no-cuda --reward-mode learned`

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| PPO | env | -59.00 | -47.09 |
| PPO | learned | -62.19 | -46.64 |

运行目录：

- `runs/Reacher-v4__ppo_continuous_action__env__1__1768646623`
- `runs/Reacher-v4__ppo_continuous_action__learned__1__1768646658`

SAC（3000 steps，learning_starts=1000，CPU）：

- env：`python casestudy2/sac_continuous_action.py --env-id Reacher-v4 --total-timesteps 3000 --learning-starts 1000 --no-cuda --reward-mode env`
- learned：`python casestudy2/sac_continuous_action.py --env-id Reacher-v4 --total-timesteps 3000 --learning-starts 1000 --no-cuda --reward-mode learned`

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| SAC | env | -23.33 | -14.89 |
| SAC | learned | -44.86 | -34.46 |

运行目录：

- `runs/Reacher-v4__sac_continuous_action__env__1__1768646764`
- `runs/Reacher-v4__sac_continuous_action__learned__1__1768646822`

备注：gymnasium 可能会提示 `Reacher-v4` 的 observation dtype/range 警告（float64/float32），但不影响 smoke-run 跑通与记录指标。

多 seed 统计（Reacher-v4, PPO=8192 steps / SAC=3000 steps, seeds=1–5；统计对象为每个 run 的 mean(last 20 episodes)）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| PPO | env | -58.942 | 0.847 | -45.733 |
| PPO | learned | -61.587 | 0.590 | -46.637 |
| SAC | env | -23.464 | 0.793 | -14.885 |
| SAC | learned | -43.885 | 1.005 | -33.952 |

进一步的 longer multi-seed（Reacher-v4, PPO=20k / SAC=20k timesteps, seeds=1–5；统计对象为每个 run 的 mean(last 20 episodes)，默认启用 CUDA）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| PPO | env | -56.569 | 1.631 | -43.084 |
| PPO | learned | -64.385 | 3.064 | -46.237 |
| SAC | env | -5.691 | 0.490 | -1.759 |
| SAC | learned | -50.336 | 9.984 | -20.000 |

简单解读（Reacher 20k）：在本仓库实现与默认超参下，`reward_mode=learned` 明显更差，且在 SAC 上差距非常大；这与论文中“learned reward 提升表现”的直觉不一致，需要后续通过更长 timesteps、严格对齐 reward 模型训练设置/归一化方式来进一步核对。

### 4.3 Casestudy3：Data center（离散控制）

说明：该 casestudy 的脚本与其它 case 风格不同（自带 `utils/data_center.py` 环境与 `utils/reward_machine.py` 训练 learned reward）。对比口径仍统一使用 `charts/episodic_return`（环境真实回报）。

Smoke-run（seed=1，epochs=5，max_steps=500）：

- DQN env：`python casestudy3/dqn_train.py --reward_mode env --number_epochs 5 --max_steps 500 --seed 1`
- DQN learned：`python casestudy3/dqn_train.py --reward_mode learned --number_epochs 5 --max_steps 500 --seed 1`
- PPO env：`python casestudy3/ppo_train.py --reward_mode env --number_epochs 5 --max_steps 500 --seed 1`
- PPO learned：`python casestudy3/ppo_train.py --reward_mode learned --number_epochs 5 --max_steps 500 --seed 1`
- SAC env：`python casestudy3/sac_train.py --reward_mode env --number_epochs 5 --max_steps 500 --seed 1`
- SAC learned：`python casestudy3/sac_train.py --reward_mode learned --number_epochs 5 --max_steps 500 --seed 1`

结果汇总（Data center, smoke-run, seed=1）：

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| DQN | env | 0.216 | 0.393 |
| DQN | learned | 0.222 | 0.393 |
| PPO | env | 0.326 | 0.757 |
| PPO | learned | 0.326 | 0.757 |
| SAC | env | 0.419 | 0.975 |
| SAC | learned | 0.522 | 0.930 |

运行目录：

- `runs/casestudy3__dqn__env__1__1768651181`
- `runs/casestudy3__dqn__learned__1__1768651189`
- `runs/casestudy3__ppo__env__1__1768651210`
- `runs/casestudy3__ppo__learned__1__1768651216`
- `runs/casestudy3__sac__env__1__1768651272`
- `runs/casestudy3__sac__learned__1__1768651279`

备注（为跑通所做的必要修复）：

- `utils/agent.py` 中 DQN 读取了不存在的 `args.ax_memory/args.discount`（已兼容到 `max_memory/gamma`）。
- `utils/agent.py` 的 `Categorical` 导入方式错误导致 PPO 报错（已修复）。
- `utils/agent.py` 的 `SACD_agent` 构造函数与 `casestudy3/sac_train.py` 调用不匹配（已做兼容）。
- `./model/` 目录缺失导致保存模型失败（已在保存前 `os.makedirs(..., exist_ok=True)` 处理）。

#### 4.3.1 Casestudy3：longer multi-seed（更贴近论文的“跨月份/全年”设定）

论文（PDF）强调评估口径是 **1-year energy consumption reduction rate**（相对 no-AI baseline 的能耗下降比例），而原始脚本默认每步按“分钟”推进：`month = timestep / (30*24*60)`。

问题：如果 `max_steps` 远小于 `30*24*60`，month 不会变化，且 baseline `total_energy_noai` 有时会变成 0（温度恰好一直在舒适区内），会导致 reduction rate 变成极端大负数（分母接近 0）。

因此我做了两点改动来支持“压缩时间”的复现：

- 在 [casestudy3/dqn_train.py](casestudy3/dqn_train.py)、[casestudy3/ppo_train.py](casestudy3/ppo_train.py)、[casestudy3/sac_train.py](casestudy3/sac_train.py) 增加 `--steps_per_month`，并将 month 计算修正为 `(env.initial_month + timestep // steps_per_month) % 12`。
- 能耗指标：同时记录 `charts/energy_reduction_rate`（raw）与 `charts/energy_reduction_rate_clipped`（裁剪到 [-1, 1]，用于聚合/画图更稳定），以及 `charts/energy_reduction_rate_valid`（baseline 分母是否 > 0）。

scaled-year 运行配置（seeds=1–5，env vs learned，DQN/PPO/SAC）：

- `--number_epochs 10 --max_steps 12000 --steps_per_month 1000`
  - 解释：把 1000 steps 视为 1 month，因此 12000 steps 覆盖 12 months（“一年”）

结果汇总（统计对象为每个 run 的 mean(last 5 epochs)）：

环境回报（`charts/episodic_return`）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| DQN | env | 0.353 | 0.385 | 3.055 |
| DQN | learned | 0.311 | 0.255 | 3.055 |
| PPO | env | 0.106 | 0.021 | 3.402 |
| PPO | learned | 0.106 | 0.021 | 3.402 |
| SAC | env | 0.250 | 0.229 | 2.665 |
| SAC | learned | 0.268 | 0.270 | 2.505 |

能耗下降比例（`charts/energy_reduction_rate_clipped`，裁剪版，用于稳定对比；值越大越省电）：

| 算法 | reward_mode | mean over seeds | std over seeds | max (any seed) |
|---|---|---:|---:|---:|
| DQN | env | 0.392 | 0.315 | 1.000 |
| DQN | learned | 0.261 | 0.244 | 1.000 |
| PPO | env | -0.090 | 0.148 | 1.000 |
| PPO | learned | -0.090 | 0.148 | 1.000 |
| SAC | env | 0.010 | 0.183 | 0.767 |
| SAC | learned | 0.096 | 0.204 | 0.767 |

与论文对照（当前可得结论）：

- 本仓库 casestudy3 的环境/基线设定会出现 `total_energy_noai=0` 的 epoch（舒适区内无需能耗），导致 raw reduction rate 不稳定；这与论文“全年能耗对比”的设定不完全一致。
- 在“压缩时间 + 裁剪指标”的设置下，learned reward 并没有稳定地在所有算法上带来更高的能耗下降；例如 DQN 下 env 反而更高，SAC 下 learned 略高但方差不小。
- 若要严格对齐论文的 200 rounds、每分钟决策、全年统计，需要把每个 epoch 的步数拉到 50 万级（12*30*24*60），这个计算量在本机上会非常大；当前结论应视为“按原环境动力学压缩时间尺度”的近似复现。

### 4.4 Casestudy4：PyFlyt/QuadX-Waypoints-v4（UAV 连续控制）

依赖与环境包装：

- 安装了 `pyflyt` / `pybullet`。
- PyFlyt 会带入 `pettingzoo`，而 pettingzoo 期望更高版本 `gymnasium`（>=1.0.0）。为兼容 `stable-baselines3==2.0.0`（要求 `gymnasium==0.28.1`），本环境保持 gymnasium 0.28.1，因此 pip 会提示依赖冲突；但实际 `PyFlyt/QuadX-Waypoints-v4` 仍可运行。
- Waypoints 原始 observation 结构较复杂，脚本使用 `FlattenWaypointEnv(context_length=2)` 进行扁平化以适配 MLP。

SAC（2000 steps，learning_starts=1000，CPU）：

- env：`python casestudy4/sac_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps 2000 --learning-starts 1000 --no-cuda --seed 1 --reward-mode env`
- learned：`python casestudy4/sac_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps 2000 --learning-starts 1000 --no-cuda --seed 1 --reward-mode learned`

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| SAC | env | -100.97 | 9.73 |
| SAC | learned | -103.95 | 9.73 |

运行目录：

- `runs/PyFlyt/QuadX-Waypoints-v4__sac_continuous_action__env__1__1768648320`
- `runs/PyFlyt/QuadX-Waypoints-v4__sac_continuous_action__learned__1__1768648343`

PPO（4096 steps，CPU）：

- env：`python casestudy4/ppo_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps 4096 --num-envs 1 --num-steps 128 --no-cuda --seed 1 --reward-mode env`
- learned：`python casestudy4/ppo_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps 4096 --num-envs 1 --num-steps 128 --no-cuda --seed 1 --reward-mode learned`

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| PPO | env | -96.11 | 12.61 |
| PPO | learned | -94.64 | 11.32 |

运行目录：

- `runs/PyFlyt/QuadX-Waypoints-v4__ppo_continuous_action__env__1__1768648446`
- `runs/PyFlyt/QuadX-Waypoints-v4__ppo_continuous_action__learned__1__1768648564`

TD3（2000 steps，learning_starts=1000，CPU）：

- env：`python casestudy4/td3_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps 2000 --learning-starts 1000 --no-cuda --seed 1 --reward-mode env`
- learned：`python casestudy4/td3_continuous_action.py --env-id PyFlyt/QuadX-Waypoints-v4 --total-timesteps 2000 --learning-starts 1000 --no-cuda --seed 1 --reward-mode learned`

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| TD3 | env | -102.09 | 9.73 |
| TD3 | learned | -98.04 | 9.73 |

运行目录：

- `runs/PyFlyt/QuadX-Waypoints-v4__td3_continuous_action__env__1__1768648737`
- `runs/PyFlyt/QuadX-Waypoints-v4__td3_continuous_action__learned__1__1768648834`

多 seed 统计（PyFlyt/QuadX-Waypoints-v4, SAC/TD3=2000 steps / PPO=4096 steps, seeds=1–5；统计对象为每个 run 的 mean(last 20 episodes)）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| SAC | env | -100.930 | 3.129 | 12.947 |
| SAC | learned | -102.058 | 1.500 | 12.947 |
| PPO | env | -97.016 | 3.662 | 18.722 |
| PPO | learned | -97.380 | 2.065 | 14.330 |
| TD3 | env | -101.536 | 1.551 | 12.947 |
| TD3 | learned | -100.065 | 1.644 | 12.947 |

进一步的 longer multi-seed（PyFlyt/QuadX-Waypoints-v4, PPO/SAC/TD3=2000 timesteps, seeds=1–5；统计对象为每个 run 的 mean(last 20 episodes)，默认启用 CUDA）：

| 算法 | reward_mode | mean over seeds | std over seeds | max episodic return (any seed) |
|---|---|---:|---:|---:|
| PPO | env | -95.002 | 2.629 | 15.916 |
| PPO | learned | -87.039 | 26.914 | 10.734 |
| SAC | env | -100.352 | 2.700 | 12.947 |
| SAC | learned | -88.737 | 22.011 | 15.014 |
| TD3 | env | -102.243 | 2.305 | 12.947 |
| TD3 | learned | -101.732 | 3.386 | 12.947 |

简单解读（PyFlyt 2k）：从均值看 PPO/SAC 的 learned reward 更高（更接近 0），但方差明显更大（不稳定）；TD3 基本持平。这个结论仍属于“短预算验证”，需要更长 timesteps 才能对齐论文的稳定提升叙事。

## 5. “是否有 trick/造假”检查结论（当前能覆盖范围）

- 从对比口径看：env vs learned 的比较使用同一个环境真实回报指标（`charts/episodic_return`），没有出现“用 learned reward 指标冒充环境成绩”的情况。
- 从实验设置看：seed 与超参在同一算法内保持一致，主要差异仅是 reward_mode。

## 6. 下一步

1. 对 Casestudy1/2/4 做多 seed（例如 1–5）重复，得到更可靠的均值/方差。
2. 若要更严谨：拉长 timesteps/epochs（当前均为 smoke-run），并固定/记录所有依赖版本（尤其是 torch nightly）。
