# TensorBoard 指标快速参考

## 📋 所有可用指标列表

### 🎯 Episode 奖励指标

| 指标名称 | 类型 | 说明 | 期望趋势 |
|---------|------|------|---------|
| `global_agent/episode/reward` | 标量 | Global Agent 单个 episode 的总奖励 | ↗ 上升 |
| `global_agent/episode/mean_reward_100` | 标量 | Global Agent 最近100个 episode 的平均奖励 | ↗ 上升并稳定 |
| `global_agent/episode/length` | 标量 | Global Agent episode 长度（步数） | → 稳定 |
| `local_agent/episode/reward` | 标量 | Local Agent 单个 episode 的总奖励 | ↗ 上升 |
| `local_agent/episode/mean_reward_100` | 标量 | Local Agent 最近100个 episode 的平均奖励 | ↗ 上升并稳定 |
| `local_agent/episode/length` | 标量 | Local Agent episode 长度（步数） | → 稳定 |
| `episode/reward` | 标量 | 总 episode 奖励 (global + local) | ↗ 上升 |
| `episode/global_reward` | 标量 | Global Agent 平均单步奖励 | ↗ 上升 |
| `episode/local_reward` | 标量 | Local Agent 平均单步奖励 | ↗ 上升 |
| `episode/mean_reward` | 标量 | 滚动平均总奖励 (100 episodes) | ↗ 上升并稳定 |
| `episode/mean_global_reward` | 标量 | Global 滚动平均奖励 | ↗ 上升并稳定 |
| `episode/mean_local_reward` | 标量 | Local 滚动平均奖励 | ↗ 上升并稳定 |

---

### 🧠 PPO 神经网络损失

| 指标名称 | Agent | 说明 | 期望趋势 | 健康范围 |
|---------|-------|------|---------|---------|
| `global/train/policy_loss` | Global | 策略网络损失 | ↘ 下降并稳定 | 小幅波动 |
| `global/train/value_loss` | Global | 价值网络损失 | ↘ 下降并稳定 | 接近 0 |
| `global/train/entropy_loss` | Global | 熵损失 | ↘ 缓慢下降 | > 0 (保持探索) |
| `global/train/loss` | Global | 总损失 | ↘ 下降 | - |
| `local/train/policy_loss` | Local | 策略网络损失 | ↘ 下降并稳定 | 小幅波动 |
| `local/train/value_loss` | Local | 价值网络损失 | ↘ 下降并稳定 | 接近 0 |
| `local/train/entropy_loss` | Local | 熵损失 | ↘ 缓慢下降 | > 0 (保持探索) |
| `local/train/loss` | Local | 总损失 | ↘ 下降 | - |

---

### 📊 PPO 训练质量指标

| 指标名称 | Agent | 说明 | 健康范围 | 警告信号 |
|---------|-------|------|---------|---------|
| `global/train/approx_kl` | Global | KL 散度（策略变化幅度） | < 0.1 | > 0.2 (变化太快) |
| `global/train/clip_fraction` | Global | 被裁剪的样本比例 | 0.1 - 0.3 | > 0.5 或 < 0.05 |
| `global/train/explained_variance` | Global | 价值函数解释方差 | > 0.5 | < 0 (估计很差) |
| `global/train/learning_rate` | Global | 当前学习率 | 递减 (如果使用调度) | - |
| `global/train/n_updates` | Global | 累计更新次数 | 递增 | - |
| `local/train/approx_kl` | Local | KL 散度 | < 0.1 | > 0.2 |
| `local/train/clip_fraction` | Local | 裁剪比例 | 0.1 - 0.3 | > 0.5 或 < 0.05 |
| `local/train/explained_variance` | Local | 解释方差 | > 0.5 | < 0 |
| `local/train/learning_rate` | Local | 学习率 | 递减 | - |
| `local/train/n_updates` | Local | 更新次数 | 递增 | - |

---

### 🎲 Rollout 统计

| 指标名称 | Agent | 说明 | 用途 |
|---------|-------|------|-----|
| `global/rollout/mean_ep_reward` | Global | Rollout 期间平均 episode 奖励 | 监控采样质量 |
| `global/rollout/mean_ep_length` | Global | Rollout 期间平均 episode 长度 | 监控 episode 稳定性 |
| `local/rollout/mean_ep_reward` | Local | Rollout 期间平均 episode 奖励 | 监控采样质量 |
| `local/rollout/mean_ep_length` | Local | Rollout 期间平均 episode 长度 | 监控 episode 稳定性 |

---

## 🚨 异常模式识别

### ❌ **问题 1: Reward 不增长**

**症状：**
- `episode/reward` 曲线平坦或下降
- `episode/mean_reward` 没有改善

**诊断指标：**
```
global/train/approx_kl > 0.2        # KL 散度过大
global/train/explained_variance < 0  # 价值估计很差
global/train/policy_loss 剧烈震荡    # 策略不稳定
```

**可能原因：**
- 学习率太高
- 奖励函数设计问题
- 环境随机性太大

**解决方案：**
1. 降低学习率 (`learning_rate: 0.0001`)
2. 增大 `n_steps` (更多经验)
3. 增大 `batch_size` (更稳定更新)

---

### ❌ **问题 2: Loss 震荡剧烈**

**症状：**
- `train/policy_loss` 上下剧烈波动
- `train/value_loss` 不稳定

**诊断指标：**
```
train/approx_kl 波动很大            # 策略变化不稳定
train/clip_fraction > 0.5          # 过多样本被裁剪
```

**可能原因：**
- 学习率太高
- Batch size 太小
- Clip range 不合适

**解决方案：**
1. 降低 `learning_rate`
2. 增大 `batch_size`
3. 调整 `clip_range` (默认 0.2)
4. 减少 `n_epochs` (每次更新的训练轮数)

---

### ❌ **问题 3: 过早收敛（Premature Convergence）**

**症状：**
- `train/entropy_loss` 快速降至接近 0
- Reward 在次优水平停滞

**诊断指标：**
```
train/entropy_loss ≈ 0              # 没有探索
train/clip_fraction < 0.05          # 策略几乎不更新
episode/reward 停滞在次优值         # 陷入局部最优
```

**可能原因：**
- 熵系数太小
- 探索不足

**解决方案：**
1. 增大 `ent_coef` (熵系数，默认 0.01 → 0.05)
2. 使用 exploration noise
3. 增加训练时长

---

### ❌ **问题 4: 价值函数估计差**

**症状：**
- `train/explained_variance` 持续为负或接近 0
- Reward 提升缓慢

**诊断指标：**
```
train/explained_variance < 0        # 价值估计比平均值还差
train/value_loss 不下降            # 价值网络学不到东西
```

**可能原因：**
- 价值网络容量不足
- Reward 信号太稀疏
- Gamma 设置不当

**解决方案：**
1. 增大价值网络规模
2. 调整 `gamma` (折扣因子)
3. 改善 reward shaping
4. 增大 `n_steps` (更长的轨迹)

---

### ❌ **问题 5: Global 和 Local Agents 不协调**

**症状：**
- `global_agent/episode/reward` 上升
- `local_agent/episode/reward` 下降（或反之）
- 总 `episode/reward` 改善缓慢

**诊断指标：**
```
global_agent/episode/reward ↗      # Global 改进
local_agent/episode/reward ↘       # Local 退化
episode/reward 震荡                # 总体不稳定
```

**可能原因：**
- 交替训练周期不平衡
- 奖励函数冲突
- 学习率不匹配

**解决方案：**
1. 调整 `global_steps_per_cycle` 和 `local_steps_per_cycle` 比例
2. 重新设计奖励权重，确保协作
3. 使用相似的学习率
4. 考虑切换到 simultaneous training

---

## 📈 健康训练的指标特征

### ✅ **理想曲线模式**

#### **Episode Reward**
```
↑
│         ╱‾‾‾‾‾‾
│       ╱
│     ╱
│   ╱
│ ╱
└────────────→ timesteps
```
- 初期快速上升
- 中期稳步改进
- 后期趋于稳定（可能有小幅波动）

#### **Policy Loss**
```
↑
│╲
│ ╲___
│     ‾‾‾‾‾‾
└────────────→ timesteps
```
- 快速下降
- 稳定在一个低水平
- 允许小幅波动

#### **Value Loss**
```
↑
│╲
│ ╲__
│    ‾‾‾‾‾
└────────────→ timesteps
```
- 持续下降
- 趋近于 0
- 比 policy loss 更平滑

#### **Entropy**
```
↑
│╲
│ ╲
│  ╲__
│     ‾‾‾‾
└────────────→ timesteps
```
- 缓慢下降
- 保持在 > 0 的水平（保持探索）

#### **Explained Variance**
```
↑ 1.0 ┼        ╱‾‾
│       ╱
│     ╱
│ 0.5 ┤  ╱
│   ╱
│ 0.0 ┼
└────────────→ timesteps
```
- 快速上升
- 稳定在 > 0.5 的高水平

---

## 🎯 关键指标检查清单

训练完成后，按以下清单检查：

### Global Agent
- [ ] `episode/reward` 持续上升 ✓
- [ ] `train/policy_loss` 下降并稳定 ✓
- [ ] `train/value_loss` < 10 ✓
- [ ] `train/explained_variance` > 0.5 ✓
- [ ] `train/approx_kl` < 0.1 ✓
- [ ] `train/clip_fraction` 在 0.1-0.3 ✓
- [ ] `train/entropy_loss` > 0 (有探索) ✓

### Local Agent
- [ ] `episode/reward` 持续上升 ✓
- [ ] `train/policy_loss` 下降并稳定 ✓
- [ ] `train/value_loss` < 10 ✓
- [ ] `train/explained_variance` > 0.5 ✓
- [ ] `train/approx_kl` < 0.1 ✓
- [ ] `train/clip_fraction` 在 0.1-0.3 ✓
- [ ] `train/entropy_loss` > 0 ✓

### 协作性
- [ ] `episode/reward` = global + local 协同改进 ✓
- [ ] 两个 agent 的 reward 都在增长 ✓
- [ ] 没有一个 agent 严重拖后腿 ✓

---

## 💡 查看技巧

### 1. **使用正则表达式过滤**
在 TensorBoard 搜索框中：
- `.*policy_loss` → 显示所有 policy loss
- `global/.*` → 显示所有 global 指标
- `.*episode/reward` → 显示所有 episode reward

### 2. **调整时间轴**
- X 轴可选择：`Step`, `Relative`, `Wall Time`
- 对比实验时使用 `Step`

### 3. **使用 Tag 过滤器**
- 左侧可以按 tag 分组
- 折叠不需要的分组

### 4. **导出图片**
- 点击图表右上角的三个点
- 选择 "Download as PNG" 或 "Download as SVG"

---

## 📝 CSV 日志字段说明

`training_progress.csv` 包含以下字段：

| 字段 | 说明 |
|-----|-----|
| `timestep` | 当前训练步数 |
| `episode` | Episode 编号 |
| `episode_reward` | 当前 episode 的总奖励 |
| `episode_global_reward` | Global Agent 的平均单步奖励 |
| `episode_local_reward` | Local Agent 的平均单步奖励 |
| `mean_reward` | 最近100个 episode 的平均总奖励 |
| `mean_global_reward` | 最近100个 episode 的平均 global 奖励 |
| `mean_local_reward` | 最近100个 episode 的平均 local 奖励 |
| `best_mean_reward` | 迄今为止的最佳平均奖励 |

---

**祝训练顺利！有任何问题随时查阅此文档。** 🚀

