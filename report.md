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

### 2.1 GPU 是否“值得用”（casestudy3/4 的经验结论）

对 casestudy3/4 做了小规模 wall-time 对比（同一机器，同一脚本，仅切换 CUDA 可见性/`--no-cuda`）：

- casestudy3（SAC，`number_epochs=2,max_steps=2000`）：
  - CUDA 可用：约 3.77s
  - 强制 CPU（`CUDA_VISIBLE_DEVICES=''`）：约 1.90s
- casestudy4（PPO，`total_timesteps=5000`）：
  - CUDA 可用：约 58.17s
  - `--no-cuda`：约 51.38s

结论：这两类任务更偏 **环境/仿真 CPU-bound + 网络较小**，GPU 往往收益不明显甚至更慢。更实用的加速方式是 **禁用 CUDA + 提高并行进程数（PARALLEL_JOBS）**。

## 3. 口径与公平性（防“trick”原则）

- **统一评估指标**：比较使用环境 episodic return（`charts/episodic_return`），而不是 learned reward 自己的数。
- **同一组超参**：同一算法下，env vs learned 仅切换 `--reward-mode`，其它训练超参保持一致。
- **随机种子**：脚本会设置 `random/numpy/torch` seed，并调用 `envs.reset(seed=seed)`。

重要说明（逻辑一致性）：2026-01-18 我修复了一个会影响 `reward_mode=learned` 结论的实现问题：用于上层 reward-learning 的轨迹缓冲 `epidata`/`store_V` 在部分脚本中曾误存了环境奖励而非 learned reward（导致 reward-learning 目标与 RL 更新不一致）。因此，修复前跑出的 case1/case2 “longer”结果只能作为参考；若要严格对照论文，需要在修复后重新跑同配置。

另外：2026-01-25 又修复了 casestudy1 PPO learned 训练会“越跑越慢/被手动中断”的问题（reward-learning 更新频率过高导致 CPU-bound），并在修复后重新跑了 case1/2 的 100k 单 seed 对比；本报告的 case1/2 结果以该批次为准。

## 4. 复现进度与结果（Smoke runs）

### 4.1 Casestudy1：CartPole-v1（离散控制）

本节结果已用 **修复后的实现重新覆盖**（旧的 50k/200k multi-seed 结果受 bug 影响，不再引用）。

运行配置（2026-01-25，CPU，seed=114514，100k steps）：

- DQN env：`python casestudy1/dqn.py --env-id CartPole-v1 --total-timesteps 100000 --no-cuda --seed 114514 --reward-mode env --reward-frequency 5000`
- DQN learned：`python casestudy1/dqn.py --env-id CartPole-v1 --total-timesteps 100000 --no-cuda --seed 114514 --reward-mode learned --reward-frequency 5000`
- PPO env：`python casestudy1/ppo.py --env-id CartPole-v1 --total-timesteps 100000 --no-cuda --seed 114514 --num-envs 1 --num-steps 128 --reward-mode env --reward-frequency 5000`
- PPO learned：`python casestudy1/ppo.py --env-id CartPole-v1 --total-timesteps 100000 --no-cuda --seed 114514 --num-envs 1 --num-steps 128 --reward-mode learned --reward-frequency 5000`

对比指标：TensorBoard scalar `charts/episodic_return`（环境真实回报）。

结果汇总（CartPole-v1, 100k, seed=114514；统计 `mean(last 100 episodes)`）：

| 算法 | reward_mode | mean(last 100 episodes) | max episodic return |
|---|---|---:|---:|
| DQN | env | 292.55 | 500 |
| DQN | learned | 287.42 | 500 |
| PPO | env | 420.26 | 500 |
| PPO | learned | 500.00 | 500 |

运行目录（TensorBoard）：

- `runs/CartPole-v1__dqn__env__114514__1769277753`
- `runs/CartPole-v1__dqn__learned__114514__1769277789`
- `runs/CartPole-v1__ppo__env__114514__1769277484`
- `runs/CartPole-v1__ppo__learned__114514__1769277547`

日志目录（stdout/trace）：

- `logs/smoke_20260125_seed114514_100k/`

简要解读（单 seed）：

- PPO：learned reward 达到稳定解（500），明显优于 env（mean_last100≈420）。
- DQN：env 与 learned 很接近，env 略高。

### 4.2 Casestudy2：Reacher-v4（MuJoCo 连续控制）

额外依赖：gymnasium 的 MuJoCo v4 环境需要 `mujoco` Python 包。

- 安装：
  - `python -m ensurepip --upgrade`
  - `python -m pip install mujoco==2.3.3`
- 快速检查：
  - `python -c "import gymnasium as gym; env=gym.make('Reacher-v4'); env.reset(seed=0); env.step(env.action_space.sample()); print('ok'); env.close()"`


本节结果同样以 **修复后的实现重新覆盖**（旧的短跑/多 seed/longer 结果不再引用）。

运行配置（2026-01-25，CPU，seed=114514，PPO=100k, SAC=100k）：

- PPO env：`python casestudy2/ppo_continuous_action.py --env-id Reacher-v4 --total-timesteps 100000 --num-envs 1 --no-cuda --seed 114514 --reward-mode env --reward-frequency 5000`
- PPO learned：`python casestudy2/ppo_continuous_action.py --env-id Reacher-v4 --total-timesteps 100000 --num-envs 1 --no-cuda --seed 114514 --reward-mode learned --reward-frequency 5000`
- SAC env：`python casestudy2/sac_continuous_action.py --env-id Reacher-v4 --total-timesteps 100000 --learning-starts 1000 --no-cuda --seed 114514 --reward-mode env --reward-frequency 5000`
- SAC learned：`python casestudy2/sac_continuous_action.py --env-id Reacher-v4 --total-timesteps 100000 --learning-starts 1000 --no-cuda --seed 114514 --reward-mode learned --reward-frequency 5000`

结果汇总（Reacher-v4, 100k, seed=114514；统计 `mean(last 20 episodes)`）：

| 算法 | reward_mode | mean(last 20 episodes) | max episodic return |
|---|---|---:|---:|
| PPO | env | -37.80 | -25.75 |
| PPO | learned | -64.34 | -45.34 |
| SAC | env | -4.79 | -1.50 |
| SAC | learned | -52.96 | -17.78 |

运行目录（TensorBoard）：

- `runs/Reacher-v4__ppo_continuous_action__env__114514__1769277743`
- `runs/Reacher-v4__ppo_continuous_action__learned__114514__1769278535`
- `runs/Reacher-v4__sac_continuous_action__env__114514__1769277847`
- `runs/Reacher-v4__sac_continuous_action__learned__114514__1769278995`

日志目录（stdout/trace）：

- `logs/smoke_20260125_seed114514_100k/`

简要解读（单 seed）：

- PPO：env 明显好于 learned。
- SAC：env 显著好于 learned（差距很大）。

### 4.3 Casestudy3：Data center（离散控制）

本节已用 **修复后的实现** 重新跑并覆盖（采用“逐个 demo 顺序跑”的方式，避免并行批跑引入的新 bug）。

运行配置（2026-01-25，seed=114514，近似 100k env steps）：

- `--number_epochs 9 --max_steps 12000 --steps_per_month 1000`
  - 解释：每个 epoch 最多 12000 steps，总步数约 $9\times 12000=108000$。

运行命令（每个算法均分别跑 env vs learned；casestudy3 脚本不提供 `--no-cuda`，本次通过 `CUDA_VISIBLE_DEVICES=''` 强制 CPU）：

- DQN env/learned：`python casestudy3/dqn_train.py ... --reward_mode {env,learned}`
- PPO env/learned：`python casestudy3/ppo_train.py ... --reward_mode {env,learned}`
- SAC env/learned：`python casestudy3/sac_train.py ... --reward_mode {env,learned}`

分析口径（强调“整体趋势”，不只看最后几步）：

- `charts/episodic_return`：环境真实回报（每个 epoch 一条点）。
- `charts/energy_reduction_rate_clipped`：能耗下降比例（裁剪到 [-1, 1]；每个 epoch 一条点）。
- 统计方式：用全部曲线点计算 **全程均值（mean_all）**，并对比 **前 20% vs 后 20%** 的均值（粗略反映是否在训练中改善）。

结果（casestudy3, seed=114514；前/后 20% 是按 epoch 点数切分）：

环境回报（`charts/episodic_return`）：

| 算法 | reward_mode | mean_all | first20% mean | last20% mean | max |
|---|---|---:|---:|---:|---:|
| DQN | env | 0.0247 | 0.1413 | 0.0113 | 0.2225 |
| DQN | learned | 0.0247 | 0.1413 | 0.0113 | 0.2225 |
| PPO | env | 0.0941 | 0.0175 | -0.0225 | 0.4650 |
| PPO | learned | 0.0941 | 0.0175 | -0.0225 | 0.4650 |
| SAC | env | 0.4789 | 0.0725 | 2.0388 | 3.7350 |
| SAC | learned | 0.6422 | 0.0725 | 2.7738 | 5.5325 |

能耗下降比例（`charts/energy_reduction_rate_clipped`，越大越省电）：

| 算法 | reward_mode | mean_all | first20% mean | last20% mean | max |
|---|---|---:|---:|---:|---:|
| DQN | env | -0.3053 | 0.4582 | 0.1294 | 0.5528 |
| DQN | learned | -0.3053 | 0.4582 | 0.1294 | 0.5528 |
| PPO | env | -0.2917 | -0.3934 | -1.0000 | 0.6458 |
| PPO | learned | -0.2917 | -0.3934 | -1.0000 | 0.6458 |
| SAC | env | 0.0131 | -0.2213 | 0.5514 | 0.5592 |
| SAC | learned | -0.0218 | -0.2213 | 0.3947 | 0.5574 |

简要趋势解读（单 seed，epoch 数较少，结论偏“demo 排雷”性质）：

- DQN/PPO：env 与 learned 的两条曲线在本次配置下几乎一致（回报与能耗指标都一致），说明 reward_mode 切换在该短预算下没有带来可观察的行为差异（需要再拉长 epochs 或进一步检查训练代码是否真正用到了 reward_train）。
- SAC：learned 的环境回报整体更高（mean_all 与 max 更高，后 20% 明显上升），但能耗下降指标反而略差（env 的 mean_all 略正，learned 略负）。这提示 learned reward 在此 case 中可能更偏向“任务回报”，未必直接优化能耗指标。

本次趋势分析原始输出（可复核）：

- `logs/smoke_20260125_seed114514_100k/trend_case3_episodic_return.tsv`
- `logs/smoke_20260125_seed114514_100k/trend_case3_energy_reduction_rate_clipped.tsv`

### 4.4 Casestudy4：PyFlyt/QuadX-Waypoints-v4（UAV 连续控制）

本节同样以 **修复后** 的 100k 单 seed（顺序跑）结果为准。

运行配置（2026-01-25，CPU，seed=114514，100k steps；逐个 demo 顺序跑）：

- PPO：`python casestudy4/ppo_continuous_action.py --total-timesteps 100000 --reward-mode {env,learned} --reward-frequency 5000 --no-cuda --seed 114514`
- SAC：`python casestudy4/sac_continuous_action.py --total-timesteps 100000 --learning-starts 1000 --reward-mode {env,learned} --reward-frequency 5000 --no-cuda --seed 114514`
- TD3：`python casestudy4/td3_continuous_action.py --total-timesteps 100000 --learning-starts 1000 --reward-mode {env,learned} --reward-frequency 5000 --no-cuda --seed 114514`

分析口径：使用 `charts/episodic_return`（环境回报），并用 **全程均值 + 前/后 20% 均值** 来看整体趋势（而不是只看最后 20 个 episode）。

整体趋势结果（PyFlyt/QuadX-Waypoints-v4, 100k, seed=114514）：

| 算法 | reward_mode | mean_all | first20% mean | last20% mean | max |
|---|---|---:|---:|---:|---:|
| PPO | env | -35.49 | -91.74 | 16.66 | 177.63 |
| PPO | learned | -93.82 | -93.53 | -95.03 | 112.54 |
| SAC | env | -34.23 | -85.55 | -12.30 | 294.38 |
| SAC | learned | -93.44 | -78.78 | -102.17 | 92.48 |
| TD3 | env | -17.50 | -98.85 | 27.49 | 270.83 |
| TD3 | learned | -93.26 | -103.31 | -54.74 | 111.16 |

简要趋势解读（单 seed，但 episode 数量很大，趋势更可信）：

- env：三种算法都呈现明显“从负到接近 0 甚至转正”的改善趋势（first20% 很差，last20% 明显更高）。
- learned：PPO/SAC 的整体曲线基本卡在 -90 左右（first20% 与 last20% 接近甚至变差），表现显著弱于 env。
- TD3 learned 虽然 last20% 比 first20% 好，但 mean_all 仍非常低（整体大部分时间在低回报区间）。

本次趋势分析原始输出（可复核）：

- `logs/smoke_20260125_seed114514_100k/trend_case4_episodic_return.tsv`
- 顺序跑 driver log：`logs/smoke_20260125_seed114514_100k/driver_case3_case4_sequential.log`

## 5. “是否有 trick”检查结论（当前能覆盖范围）

- 从对比口径看：env vs learned 的比较使用同一个环境真实回报指标（`charts/episodic_return`），没有出现“用 learned reward 指标冒充环境成绩”的情况。
- 从实验设置看：seed 与超参在同一算法内保持一致，主要差异仅是 reward_mode。

## 6. 下一步

1. 对 Casestudy1/2/4 做多 seed（例如 1–5）重复，得到更可靠的均值/方差。
2. 若要更严谨：拉长 timesteps/epochs（当前均为 smoke-run），并固定/记录所有依赖版本（尤其是 torch nightly）。
