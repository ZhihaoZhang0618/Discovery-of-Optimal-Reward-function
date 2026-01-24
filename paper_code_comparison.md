# 论文与代码逻辑对照分析

**论文**: *Discovery of the Reward Function for Embodied Reinforcement Learning Agents*

**分析日期**: 2026-01-24

---

## 1. 核心算法框架

### 1.1 双层优化目标

**论文公式：**

- **Upper-level（奖励函数优化）**:
  $$\omega^* = \arg\max_{\omega \in \Omega} J^{upper}(\theta, \omega) \triangleq \mathbb{E}_{\pi^*_\theta \sim M[R_\omega]} \left[ \bar{G}^{\pi^*_\theta}_{\bar{R}} \right]$$

- **Lower-level（策略优化）**:
  $$\theta^* = \arg\max_{\theta \in \Theta} J^{lower}(\theta, \omega) \triangleq \mathbb{E}_{s \sim \rho^{\pi_\theta}, a \sim \pi_\theta(\cdot|s)} \left[ \sum_{t=0}^{\infty} \gamma^t \cdot R_\omega(s^{(t)}, a^{(t)}) \right]$$

**代码实现：** ✅ **一致**

| 层级 | 代码位置 | 说明 |
|------|---------|------|
| Lower-level | 各 casestudy 的 RL 训练循环 | 使用 `reward_function.observe_reward()` 生成的奖励训练策略 |
| Upper-level | `reward_function.optimize_reward()` | 优化 reward 函数 R(s,a) |

---

## 2. 奖励函数梯度更新

### 2.1 论文核心公式 (Theorem 2, Equation 21)

$$\omega' \approx \omega + \beta \cdot \mathbb{E}_{(s,a) \sim \mu}\left[\nabla_\omega \left[ A^{\pi_{\theta'}}_{\bar{R}}(s,a) - A^{\pi_\theta}_{R_\omega}(s,a) \right]\right]$$

其中 Advantage 函数:
$$A^{\pi_\theta}_R(s, a) = Q^{\pi_\theta}_R(s, a) - V^{\pi_\theta}_R(s)$$

### 2.2 代码实现 (`utils/reward_machine.py` L178-206)

```python
for step in transitions:
    s, a, reward_hat, _log_probs, mu, overline_V = step

    V_s = self.value_function(state_t).detach()           # V(s)
    overline_V_t = torch.as_tensor(overline_V, ...)       # 折扣回报 Ḡ

    # 离散动作: reward_center = Σ π(a|s) * R(s,a)
    r_all = self.reward_function(state_rep, all_actions)
    reward_center = (probs * r_all).sum()

    # term1 ≈ A^{R_ω}(s,a) = R(s,a) - E_π[R(s,·)]
    term1 = reward_hat_t - reward_center
    
    # term2 = π(a|s) * (Ḡ - V(s))
    term2 = pi_a * (overline_V_t - V_s)
    
    losses.append(term2 * term1)
```

### 2.3 对照分析

| 论文中的项 | 代码对应 | 一致性 | 说明 |
|-----------|---------|--------|------|
| $A^{\pi_{\theta'}}_{\bar{R}}(s,a)$ | `overline_V_t - V_s` | ⚠️ 近似 | 用蒙特卡洛回报 $\bar{V}$ 替代 Q(s,a)，单样本估计 |
| $A^{\pi_\theta}_{R_\omega}(s,a)$ | `reward_hat_t - reward_center` | ✅ 正确 | R(s,a) - E_π[R(s,·)] |
| $\nabla_\omega [A_{\bar{R}} - A_{R_\omega}]$ | `term2 * term1` | ⚠️ 乘积形式 | 见下方说明 |

**说明**：
- 代码用 `term2 * term1` 乘积形式，而论文是差的梯度
- `pi_a` 因子可能用于 off-policy 修正
- 这种实现方式在策略梯度方法中常见，本质上是用 advantage 加权

---

## 3. 折扣回报 $\bar{V}$ 的计算

**论文**: $\bar{V}_t = r_t + \gamma \cdot \bar{V}_{t+1}$

**代码** (`utils/reward_machine.py` L232-249):

```python
def store_V(self, epidata):
    overline_V = 0.0
    for step in reversed(epidata):
        overline_V = step.reward + self.gamma * overline_V
        updated_step = step._replace(overline_V=overline_V)
        new_epidata.insert(0, updated_step)
    return new_epidata
```

**结论**: ✅ **完全一致**

---

## 4. Reward 模型结构

**论文描述**: State encoder + Action encoder + Forward model

**代码** (`utils/reward_model.py`):

```
Reward Function R(s, a):
├── StateEncoder: s → FC → ReLU → FC → ReLU → FC → ReLU → state_embedding
├── ActionEncoder: a → FC → ReLU → FC → ReLU → FC → ReLU → action_embedding
└── ForwardModel: concat(state_emb, action_emb) → FC → reward
```

**结论**: ✅ **完全一致**

---

## 5. 超参数对比

### 5.1 Case Study 1 (CartPole 等)

| 参数 | 论文值 | 代码默认值 | 一致性 |
|------|--------|-----------|--------|
| `total_timesteps` | 500,000 | 500,000 | ✅ |
| `reward_frequency` (DQN) | 1,000 | 1,000 | ✅ |
| `reward_frequency` (PPO) | 1,024 | 1,024 | ✅ |
| `hidden_dim` | 256 | 256 | ✅ |
| `encode_dim` | 64 | 64 | ✅ |
| `reward_buffer_size` | 100 | 100 | ✅ |
| `reward_lr` | 1e-4 | 1e-4 | ✅ |
| `gamma` | 0.99 | 0.99 | ✅ |

### 5.2 Case Study 2 (Reacher 等)

| 参数 | 论文值 | 代码默认值 | 一致性 |
|------|--------|-----------|--------|
| `total_timesteps` | 1,000,000 | 1,000,000 | ✅ |
| `reward_frequency` | 2,048 | 2,048 | ✅ |
| `n_samples` (连续动作) | - | 1,000 | ✅ |

### 5.3 Case Study 3 (Data Center)

| 参数 | 论文值 | 代码默认值 | 一致性 |
|------|--------|-----------|--------|
| `number_epochs` | 200 | 200 | ✅ |
| `hidden_dim` | 64 | 64 | ✅ |
| `encode_dim` | 16 | 16 | ✅ |
| 训练时长/epoch | 30 天 | 可配置 | ⚠️ |

### 5.4 Case Study 4 (UAV)

| 参数 | 论文值 | 代码默认值 | 一致性 |
|------|--------|-----------|--------|
| `total_timesteps` | 1,000,000 | 1,000,000 | ✅ |
| `reward_frequency` | 5,000 | 5,000 | ✅ |

---

## 6. 评估口径

**论文**: 使用环境真实回报 $\bar{G}_{\bar{R}}$ 评估

**代码**: 所有脚本使用 `gym.wrappers.RecordEpisodeStatistics` 记录 `charts/episodic_return`

**结论**: ✅ **一致** — 无论训练用什么奖励，评估统一用环境真实回报

---

## 7. 潜在问题与差异

### 7.1 梯度公式的实现差异

论文公式是对 **差** 求梯度:
$$\nabla_\omega [A_{\bar{R}} - A_{R_\omega}]$$

代码实现成 **乘积** 形式:
```python
loss = term2 * term1 = π(a|s) * (Ḡ - V(s)) * (R(s,a) - E[R])
```

这种差异可能源于：
1. 策略梯度的 importance sampling 修正
2. 数值稳定性考虑
3. 论文简化描述 vs 实际工程实现

### 7.2 连续动作空间的近似

对于连续动作，代码通过采样估计 $\mathbb{E}_\pi[R(s, \cdot)]$：
```python
action_bs, log_probs_action_bs = agent.get_action_prob_from_mu(mu, self.n_samples)
reward_center = (probs_bs * reward_bs).sum()
```

默认 `n_samples=10` 可能导致高方差。

### 7.3 曾存在的 Bug（已修复）

`report.md` 记录：曾误存环境奖励到 `epidata` 而非 learned reward，影响 reward-learning 目标一致性。

**当前状态**: 已修复，`epidata` 现在正确存储 `learned_reward`。

---

## 8. 复现结果与论文叙事的差距

根据 `report.md` 的复现结果：

| Case Study | 论文叙事 | 复现结论 |
|------------|---------|---------|
| CartPole (200k) | learned > env | ❌ PPO: 261 vs 496 (learned 更差) |
| Reacher (20k) | learned > env | ❌ SAC: -50 vs -5.7 (learned 远差) |
| Data Center | learned > env | ⚠️ 不一致，结果混合 |
| PyFlyt (2k) | learned > env | ⚠️ 略好但方差大 |

**可能原因**:
1. 训练步数不足（smoke-run 50k-200k vs 论文 500k-1M）
2. seed 数量不足（1-3 vs 论文 5+）
3. Case3 时间压缩影响

---

## 9. 结论

### ✅ 代码与论文一致的部分
- 双层优化框架
- Reward 模型结构
- 折扣回报计算
- 超参数设置
- 评估口径

### ⚠️ 需要注意的部分
- 梯度更新的乘积形式 vs 差的梯度
- 连续动作采样数量（方差问题）
- 复现结果与论文声称存在差距

### 建议后续
1. 使用论文完整配置（500k-1M steps, 5 seeds）重新复现
2. 检查梯度公式是否需要调整
3. 增大连续动作的 `n_samples`
