# 🌱 Carbon Emission Penalty 系数调整指南

## 📊 问题分析

### **当前观察到的数据：**

```python
Episode 1:
- Local agents reward 总和: -7,781
- Carbon emission: 1.01 kg CO2
- Carbon penalty (factor=1.0): 1.01
- Carbon penalty 占比: 1.01 / 7781 = 0.013%
```

**问题：** Carbon penalty 相对于 local rewards 量级太小，对 global agent 决策几乎没有影响。

---

## ✅ 已修改

**config.yml 第 1075 行：**

```yaml
# 之前
carbon_emission_penalty_coef: 1.0

# 现在
carbon_emission_penalty_coef: 100.0
```

**预期影响：**
- Carbon penalty = 100.0 × 1.01 = **101**
- 占总 reward 比例: 101 / 7781 ≈ **1.3%**

---

## 🎯 如何选择合适的系数

### **计算公式：**

```python
期望影响比例 = carbon_penalty / |local_rewards_总和|
carbon_penalty = carbon_emission_penalty_coef × carbon_kg
```

**推导：**
```python
carbon_emission_penalty_coef = (期望影响比例 × |local_rewards_总和|) / carbon_kg
```

### **示例计算（基于你的数据）：**

| 期望影响比例 | 计算 | 推荐系数 |
|------------|------|---------|
| **1%** | `0.01 × 7781 / 1.01` | **77** |
| **5%** | `0.05 × 7781 / 1.01` | **385** |
| **10%** | `0.10 × 7781 / 1.01` | **770** |
| **15%** | `0.15 × 7781 / 1.01` | **1156** |

---

## 🔄 调整策略

### **第一次训练（当前设置）：**

```yaml
carbon_emission_penalty_coef: 100.0  # 约 1.3% 影响
```

**训练后观察：**
1. TensorBoard 查看 `custom_metrics/total_carbon_kg_mean`
2. monitor.csv 查看 `total_carbon_kg` 列

**期望结果：**
- ✅ Carbon emission 有小幅下降（5-10%）
- ✅ Episode reward 略有变化但不会太剧烈
- ✅ Green energy ratio 略有上升

---

### **如果 carbon 没有明显下降：**

**诊断：** Penalty 太弱，agent 还是优先考虑 local rewards

**解决方案：** 增大系数到 **300-500**

```yaml
carbon_emission_penalty_coef: 300.0  # 约 3.9% 影响
```

或者更激进：

```yaml
carbon_emission_penalty_coef: 500.0  # 约 6.4% 影响
```

---

### **如果 episode reward 下降太多（> 20%）：**

**诊断：** Penalty 太强，agent 过度优化 carbon 而牺牲其他指标

**解决方案：** 降低系数到 **50-80**

```yaml
carbon_emission_penalty_coef: 50.0  # 约 0.64% 影响
```

---

### **如果 training 不收敛（reward 震荡）：**

**诊断：** Penalty 系数变化太快，破坏了原有的 reward balance

**解决方案：** 回退到更小的系数，渐进式增加

```yaml
carbon_emission_penalty_coef: 30.0  # 约 0.39% 影响
# 训练一段时间稳定后，再增加到 50 → 80 → 100
```

---

## 📈 理想的训练曲线

### **Carbon Emission（应该下降）：**

```
Before tuning (coef=1.0):
  ↑
碳|━━━━━━━━━  ← 几乎不变
排|
放└──────────→ Episodes

After tuning (coef=100):
  ↑
碳|╲
排|  ╲____  ← 明显下降趋势
放|       ‾‾‾
  └──────────→ Episodes
```

### **Episode Reward（可能略微下降）：**

```
Before tuning:
  ↑
奖|     ╱‾‾‾
励|   ╱
  | ╱
  └──────────→ Episodes

After tuning (合理的 penalty):
  ↑
奖|     ╱‾‾  ← 略低但还在上升
励|   ╱
  | ╱
  └──────────→ Episodes
```

**注意：** Reward 会略微下降是正常的（因为增加了 carbon penalty），但应该还是有上升趋势。

---

## 🎓 高级调整技巧

### **1. 动态调整（实验性）**

在不同训练阶段使用不同的系数：

**Phase 1 (0-20k steps):** 低 penalty，让 agent 先学会基本调度
```yaml
carbon_emission_penalty_coef: 30.0
```

**Phase 2 (20k-50k steps):** 中等 penalty，引入 carbon awareness
```yaml
carbon_emission_penalty_coef: 100.0
```

**Phase 3 (50k+ steps):** 高 penalty，强化 carbon 优化
```yaml
carbon_emission_penalty_coef: 300.0
```

**实现方式：** 需要修改代码，在不同的 checkpoint 时更新配置。

---

### **2. 相对 Penalty（自适应）**

根据当前 episode 的 local rewards 动态计算 penalty：

```python
# 伪代码
penalty = carbon_kg × (α × |episode_local_rewards|)
# α 是相对系数，例如 0.01 表示 1% 影响
```

**优点：** 自动适应 reward scale 的变化

**缺点：** 需要修改 reward 计算逻辑（在 Java 侧）

---

### **3. Multi-Objective Reward Shaping**

分别优化多个目标，使用加权和：

```python
global_reward = w1 × local_rewards - w2 × carbon_penalty - w3 × latency_penalty
```

**示例权重：**
```yaml
local_reward_weight: 1.0         # 基准
carbon_penalty_weight: 100.0     # 当前设置
latency_penalty_weight: 50.0     # 如果关心延迟
```

---

## 🔬 实验记录表

建议记录每次实验的结果，方便对比：

| Exp | carbon_coef | Avg Carbon (kg) | Avg Reward | Green Ratio | 备注 |
|-----|------------|----------------|-----------|------------|-----|
| 1 | 1.0 | 1.010 | -15,300 | 43.6% | Baseline (太弱) |
| 2 | 100.0 | ? | ? | ? | 当前设置 |
| 3 | 300.0 | ? | ? | ? | 如果效果不够 |
| 4 | 500.0 | ? | ? | ? | 更强的 penalty |

---

## 📊 监控指标

训练时重点关注这些指标：

### **在 TensorBoard：**

1. **`custom_metrics/total_carbon_kg_mean`** - Carbon emission 趋势
   - 期望：下降
   
2. **`episode_reward_mean`** - 总奖励趋势
   - 期望：上升，但可能比之前略低

3. **`custom_metrics/green_ratio_mean`** - 绿色能源比例
   - 期望：上升

4. **`agent_rewards/global_agent`** - Global agent 奖励
   - 期望：上升

### **在 monitor.csv：**

```python
import pandas as pd
df = pd.read_csv('monitor.csv')

# 对比前后变化
print(f"Average carbon: {df['total_carbon_kg'].mean():.3f} kg")
print(f"Average green ratio: {df['green_ratio'].mean():.2%}")
print(f"Average brown energy: {df['brown_used_wh'].mean():.2f} Wh")
```

---

## ⚠️ 注意事项

### **1. 训练时间可能更长**

增大 carbon penalty 后，agent 需要学习新的平衡策略，可能需要更多 timesteps 才能收敛。

**建议：** 至少运行 100k-200k timesteps。

---

### **2. 可能需要调整其他超参数**

如果发现训练不稳定：

```yaml
# 可以适当降低学习率
learning_rate: 1e-4  # 原来是 3e-4
```

---

### **3. Checkpoint 对比**

保存不同 carbon_coef 下的 checkpoint，方便对比：

```bash
logs/
├── experiment_carbon_coef_1/     # coef=1.0
├── experiment_carbon_coef_100/   # coef=100.0
├── experiment_carbon_coef_300/   # coef=300.0
```

---

## 🎯 推荐的调优流程

1. ✅ **第一次实验（已设置）：** `carbon_emission_penalty_coef: 100.0`
   - 训练 50k-100k steps
   - 观察 carbon 和 reward 趋势

2. **根据结果调整：**
   - 如果 carbon 下降 < 5%：增大到 **300**
   - 如果 carbon 下降 > 20%：保持 **100** 或降到 **50**
   - 如果 reward 下降 > 20%：降低到 **50**

3. **精细调优：**
   - 找到合适的范围后，在 ±30% 范围内微调
   - 例如：100 有效 → 尝试 80, 120, 150

4. **最终验证：**
   - 用最佳系数训练完整的 200k+ steps
   - 对比 baseline（coef=1.0）的改进

---

## 📝 快速命令

### **运行新实验：**

```bash
cd drl-manager
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5 --total-timesteps 100000
```

### **启动 TensorBoard：**

```bash
cd ..\logs\experiment_multi_dc_5
tensorboard --logdir=. --port=6006
```

### **对比不同实验：**

```python
import pandas as pd

# 读取多个实验的 monitor.csv
df1 = pd.read_csv('logs/experiment_multi_dc_5/20251122_203819/monitor.csv')  # coef=1.0
df2 = pd.read_csv('logs/experiment_multi_dc_5/<new_timestamp>/monitor.csv')  # coef=100.0

print(f"Baseline (coef=1.0): Carbon={df1['total_carbon_kg'].mean():.3f} kg")
print(f"New (coef=100.0): Carbon={df2['total_carbon_kg'].mean():.3f} kg")
print(f"Improvement: {(1 - df2['total_carbon_kg'].mean()/df1['total_carbon_kg'].mean())*100:.1f}%")
```

---

## 💡 总结

- ✅ **当前设置：** `carbon_emission_penalty_coef: 100.0` (约 1.3% 影响)
- 🎯 **期望效果：** Carbon 下降 5-15%，Reward 略微降低但还在改进
- 🔄 **如果效果不够：** 增大到 300-500
- ⚠️ **如果效果太强：** 降低到 50-80

**记住：** 强化学习是实验性的，需要多次尝试才能找到最佳平衡点！

---

**Good luck with your carbon-aware training! 🌱🚀**

