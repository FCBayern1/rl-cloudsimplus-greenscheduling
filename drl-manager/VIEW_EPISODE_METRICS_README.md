# 📊 查看 Episode-Level 指标说明

## 🎯 问题：TensorBoard 显示的不是 Episode-Level 数据

你在 TensorBoard 看到的问题：

1. ❌ **横轴是 Training Steps**，不是 Episode Number
2. ❌ **值是聚合的 mean/max/min**，不是每个 episode 的真实值
3. ❌ **前几千步显示为 0**，因为是聚合统计异常

## ✅ 解决方案：使用 monitor.csv

你的 `monitor.csv` 文件记录了**每个 episode 的真实数据**！

---

## 方法 1: 用 Python 脚本自动生成图表（推荐）

### **步骤 1: 安装依赖**

```bash
cd drl-manager
pip install pandas matplotlib
```

### **步骤 2: 运行脚本**

```bash
python view_core_metrics.py 20251122_203819
```

会自动生成图表，包含：
1. **Episode Reward** - 横轴 = Episode Number ✅
2. **Carbon Emission** - 每个 episode 的碳排放 ✅
3. **Brown Energy Used** - 每个 episode 的棕色能源使用 ✅
4. **Green Energy Ratio** - 绿色能源比例变化
5. **Agent Rewards** - Global vs Local Agent 奖励对比
6. **Energy Breakdown** - 绿色/棕色/浪费能源对比

图片会保存到：`logs/experiment_multi_dc_5/<timestamp>/episode_metrics.png`

---

## 方法 2: 用 Excel 查看 monitor.csv

### **步骤 1: 打开文件**

```
logs/experiment_multi_dc_5/20251122_203819/monitor.csv
```

### **步骤 2: 关键列**

| 列名 | 含义 |
|------|------|
| `episode` | Episode 编号（1, 2, 3...）|
| `episode_reward` | **总奖励** ← 你要的指标 3 |
| `total_carbon_kg` | **碳排放（kg）** ← 你要的指标 2 |
| `brown_used_wh` | **棕色能源使用（Wh）** |
| `green_ratio` | 绿色能源比例 |
| `global_agent_reward` | Global Agent 奖励 |
| `local_agents_avg_reward` | Local Agents 平均奖励 |

### **步骤 3: 绘图**

在 Excel 中：
1. 选择 `episode` 列（横轴）
2. 选择 `brown_used_wh` 列（纵轴）
3. 插入 → 折线图

就能看到**每个 episode 的棕色能源使用变化**！

---

## 方法 3: 用 Python 手动绘图

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('logs/experiment_multi_dc_5/20251122_203819/monitor.csv')

# 绘制 Brown Energy
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['brown_used_wh'], marker='o', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Brown Energy Used (Wh)')
plt.title('Brown Energy Usage per Episode')
plt.grid(True, alpha=0.3)
plt.show()

# 绘制 Carbon Emission
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['total_carbon_kg'], marker='o', linewidth=2, color='red')
plt.xlabel('Episode')
plt.ylabel('Carbon Emission (kg CO2)')
plt.title('Carbon Emission per Episode')
plt.grid(True, alpha=0.3)
plt.show()

# 绘制 Episode Reward
plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['episode_reward'], marker='o', linewidth=2, color='blue')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Episode Reward over Training')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 📊 你的实验数据分析

根据 `monitor.csv`，前 8 个 episodes 的数据：

| Episode | Brown Energy (Wh) | Carbon (kg) | Reward |
|---------|-------------------|-------------|---------|
| 1 | 1989.03 | 1.010 | -15562.96 |
| 2 | 1989.01 | 1.010 | -15684.36 |
| 3 | 1989.06 | 1.010 | -15256.12 |
| 4 | 1989.18 | 1.010 | -15405.34 |
| 5 | 1989.00 | 1.010 | -14860.45 |
| 6 | 1989.07 | 1.010 | -15007.33 |
| 7 | 1989.12 | 1.010 | -14312.65 |
| 8 | 1988.99 | 1.010 | -14438.00 |

**观察：**
- ✅ **每个 episode 从一开始就有 brown energy 数据**（~1989 Wh）
- ✅ 碳排放稳定在 ~1.01 kg CO2
- ✅ Reward 有小幅改善趋势（-15562 → -14438）

所以 TensorBoard 显示的 "前几千步为 0" 是**聚合统计的问题**，真实数据是正常的！

---

## 🆚 TensorBoard vs monitor.csv 对比

| 特性 | TensorBoard | monitor.csv |
|------|-------------|-------------|
| **横轴** | Training Steps | Episode Number ✅ |
| **数据类型** | 聚合统计（mean/max/min） | 每个 episode 真实值 ✅ |
| **初始数据** | 可能显示为 0（统计异常） | 从 episode 1 就有数据 ✅ |
| **Policy Loss** | 有（custom_metrics） | ❌ 没有 |
| **Value Loss** | 有（custom_metrics） | ❌ 没有 |
| **适用场景** | 查看训练过程和损失曲线 | 查看每个 episode 的性能指标 ✅ |

**结论：**
- **看 Policy/Value Loss** → 用 TensorBoard
- **看 Episode-Level 性能指标** → 用 monitor.csv（推荐）

---

## 💡 下次实验的建议

### **训练时同时监控两者：**

**终端 1: 运行训练**
```bash
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5 --total-timesteps 100000
```

**终端 2: 实时查看 monitor.csv**
```bash
# Windows PowerShell
Get-Content logs\experiment_multi_dc_5\<timestamp>\monitor.csv -Wait

# 或者用 Python 实时绘图
python -c "
import pandas as pd
import time
while True:
    df = pd.read_csv('logs/experiment_multi_dc_5/<timestamp>/monitor.csv')
    print(f'Episodes: {len(df)}, Latest Reward: {df[\"episode_reward\"].iloc[-1]:.2f}')
    time.sleep(10)
"
```

**终端 3: TensorBoard（查看损失）**
```bash
tensorboard --logdir=logs/experiment_multi_dc_5 --port=6006
```

这样你就能：
- ✅ 实时看到 Policy/Value Loss（TensorBoard）
- ✅ 实时看到每个 episode 的性能（monitor.csv）
- ✅ 同时监控训练进度

---

## 🎯 总结

- **TensorBoard 的 brown_used_wh 图表不合理** → 因为是聚合统计，不是真实 episode 数据
- **真实的 episode 数据在 monitor.csv** → 从第一个 episode 就有完整数据
- **推荐做法** → 用 `view_core_metrics.py` 脚本自动生成 episode-level 图表
- **Policy Loss 还是要看 TensorBoard** → monitor.csv 没有这些数据

下次训练完成后，直接运行：
```bash
python view_core_metrics.py <timestamp>
```

就能看到所有 episode-level 的核心指标了！

