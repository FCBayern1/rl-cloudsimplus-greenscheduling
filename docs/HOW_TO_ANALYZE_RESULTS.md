# 如何分析RL训练结果
# How to Analyze RL Training Results

## 目录 Table of Contents
1. [生成的文件说明](#1-生成的文件说明)
2. [快速分析方法](#2-快速分析方法)
3. [手动分析CSV](#3-手动分析csv)
4. [使用Python自定义分析](#4-使用python自定义分析)
5. [如何判断实验成功](#5-如何判断实验成功)

---

## 1. 生成的文件说明

### 训练过程中自动生成的文件

```
logs/SPEC_Authentic/exp10_spec_real/
├── monitor.csv                  ⭐ 最重要：每个episode的统计
├── progress.csv                 ⭐ 训练过程的PPO指标
├── best_model.zip               最佳模型（基于reward）
├── final_model.zip              最终模型
├── best_episode_details_XX.csv  最佳episode的详细数据
├── config_used.yml              使用的配置
├── events.out.tfevents.XXX      TensorBoard日志
└── analysis/                    分析脚本生成的图表
    ├── reward_comparison.png
    ├── energy_comparison.png
    ├── training_curves.png
    └── training_report.txt
```

---

## 2. 快速分析方法

### 方法A：使用自动分析脚本（推荐）

```bash
# 在 drl-manager 目录运行
cd drl-manager

# 使用官方分析脚本
.venv/Scripts/python.exe analyze_training.py --log_dir ../logs/SPEC_Authentic/exp10_spec_real

# 或使用教程脚本（更详细）
.venv/Scripts/python.exe tutorial_analyze.py ../logs/SPEC_Authentic/exp10_spec_real
```

**生成的文件**：
- `analysis/training_report.txt` - 文本报告
- `analysis/reward_comparison.png` - Reward对比图
- `analysis/energy_comparison.png` - 能源对比图
- `analysis/training_curves.png` - 训练曲线图

### 方法B：直接查看图表

打开生成的PNG文件：
1. `analysis/training_curves.png` - 查看整体训练趋势
2. `analysis/reward_comparison.png` - First 10 vs Last 10对比
3. `analysis/energy_comparison.png` - 能源指标对比

---

## 3. 手动分析CSV

### 使用PowerShell快速查看

#### 查看最后5个episodes
```powershell
cd logs/SPEC_Authentic/exp10_spec_real
Import-Csv monitor.csv | Select-Object -Last 5 | Format-Table -AutoSize
```

#### 统计总能耗
```powershell
Import-Csv monitor.csv | Measure-Object -Property cumulative_energy_wh -Average -Min -Max
```

#### 统计episode长度
```powershell
Import-Csv monitor.csv | Measure-Object -Property l -Average -Min -Max
```

#### 检查truncation比例
```powershell
$data = Import-Csv monitor.csv
$truncated = ($data | Where-Object { [int]$_.l -eq 600 }).Count
$total = $data.Count
Write-Host "Truncated: $truncated / $total ($([math]::Round($truncated/$total*100, 1))%)"
```

### 使用Excel分析

1. **打开monitor.csv**
   - 用Excel打开 `logs/SPEC_Authentic/exp10_spec_real/monitor.csv`
   - 删除第一行（metadata）

2. **创建趋势图**
   - 选择 `r` 列（Total Reward）
   - 插入 → 折线图
   - 添加趋势线：右键 → 添加趋势线 → 移动平均（期间=10）

3. **关键列说明**
   | 列名 | 含义 | 理想值 |
   |------|------|--------|
   | r | 总奖励 | 越高越好，应该上升 |
   | l | Episode长度 | 应该<max_episode_length |
   | cumulative_energy_wh | 累计能耗 | 越低越好，应该下降 |
   | current_power_w | 平均功率 | 取决于负载 |
   | average_host_utilization | CPU利用率 | 50-80%最佳 |

4. **计算改善率**
   ```excel
   # 在新列计算前10和后10的平均值
   =AVERAGE(C2:C11)      # First 10
   =AVERAGE(C192:C201)   # Last 10

   # 计算改善百分比
   =(Last10 - First10) / First10 * 100
   ```

---

## 4. 使用Python自定义分析

### 示例1：基本统计

```python
import pandas as pd

# 读取数据
df = pd.read_csv("logs/SPEC_Authentic/exp10_spec_real/monitor.csv", skiprows=1)

# 基本统计
print("Total Episodes:", len(df))
print("\nReward Statistics:")
print(df['r'].describe())

print("\nEnergy Statistics:")
print(df['cumulative_energy_wh'].describe())

# First 10 vs Last 10
first_10 = df.head(10)
last_10 = df.tail(10)

print(f"\nReward: {first_10['r'].mean():.2f} -> {last_10['r'].mean():.2f}")
print(f"Energy: {first_10['cumulative_energy_wh'].mean():.2f} -> {last_10['cumulative_energy_wh'].mean():.2f} Wh")
```

### 示例2：检测常见问题

```python
import pandas as pd

df = pd.read_csv("logs/SPEC_Authentic/exp10_spec_real/monitor.csv", skiprows=1)

# 检查truncation
max_len = df['l'].max()
truncated = (df['l'] >= max_len * 0.99).sum()
print(f"Truncated: {truncated}/{len(df)} ({truncated/len(df)*100:.1f}%)")

# 检查利用率
avg_util = df['average_host_utilization'].mean() * 100
print(f"Average Utilization: {avg_util:.2f}%")
if avg_util < 10:
    print("WARNING: Very low utilization!")

# 检查学习进度
first_reward = df['r'].head(10).mean()
last_reward = df['r'].tail(10).mean()
improvement = (last_reward - first_reward) / abs(first_reward) * 100
print(f"Reward Improvement: {improvement:.2f}%")
if abs(improvement) < 1:
    print("WARNING: Minimal learning!")
```

### 示例3：绘制自定义图表

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/SPEC_Authentic/exp10_spec_real/monitor.csv", skiprows=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Total Reward
axes[0, 0].plot(df.index, df['r'], alpha=0.3)
axes[0, 0].plot(df.index, df['r'].rolling(10).mean(), linewidth=2)
axes[0, 0].set_title('Total Reward')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].grid(True)

# 2. Energy Consumption
axes[0, 1].plot(df.index, df['cumulative_energy_wh'])
axes[0, 1].set_title('Cumulative Energy (Wh)')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].grid(True)

# 3. Episode Length
axes[1, 0].plot(df.index, df['l'])
axes[1, 0].axhline(y=600, color='r', linestyle='--', label='Max Length')
axes[1, 0].set_title('Episode Length')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Host Utilization
axes[1, 1].plot(df.index, df['average_host_utilization'] * 100)
axes[1, 1].set_title('Host Utilization (%)')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('custom_analysis.png', dpi=150)
print("Saved to: custom_analysis.png")
```

---

## 5. 如何判断实验成功

### ✅ 成功的指标

#### 1. Episodes能自然完成
```
Episode长度: 80-200步 (不是600!)
Truncated episodes: <10%
```
**如何检查**：
- 查看 `analysis/training_report.txt` 中的 "Episode length"
- 或运行：`df['l'].describe()`

#### 2. Reward上升
```
First 10 -> Last 10: 改善 >10%
总体趋势：上升
```
**如何检查**：
- 查看 `analysis/reward_comparison.png`
- 或运行教程脚本的 "Learning Progress Comparison" 部分

#### 3. 能源消耗下降
```
First 10 -> Last 10: 降低 >20%
```
**如何检查**：
- 查看 `analysis/energy_comparison.png`
- 查看 monitor.csv 中的 `cumulative_energy_wh` 列

#### 4. 资源利用率合理
```
Host utilization: 50-80%
Power: 3000-5000W (视workload而定)
```
**如何检查**：
- 查看 monitor.csv 中的 `average_host_utilization` 列
- 应该远高于1%

#### 5. 训练收敛
```
Reward曲线：趋于平稳
Reward方差：逐渐减小
```
**如何检查**：
- 查看 `analysis/training_curves.png` 中的 Total Reward 图
- 后期曲线应该平稳

### ❌ 失败的指标

#### 1. Episodes全部truncate
```
症状: 99%+ episodes长度 = max_episode_length
原因: 任务无法完成，CloudSim卡死或VM不足
```
**解决**：
- 增加VM数量
- 增加max_episode_length
- 检查workload是否过重

#### 2. 能源不降反升
```
症状: Last 10 energy > First 10 energy
原因: Episode长度固定，无法通过缩短时间来省电
```
**解决**：
- 先解决truncation问题
- 检查energy reward coefficient是否太小

#### 3. 利用率极低
```
症状: average_host_utilization < 10%
原因: VM太多或cloudlets无法执行
```
**解决**：
- 减少初始VM数量
- 检查CloudSim日志确认cloudlets状态

#### 4. Reward无改善
```
症状: First 10 vs Last 10 变化 < 1%
原因: Agent没有学习到有效策略
```
**解决**：
- 增加训练timesteps
- 调整reward weights
- 检查observation space是否合理

---

## 6. 实际案例分析

### 案例：Experiment 10 (exp10_spec_real)

#### 运行分析
```bash
cd drl-manager
.venv/Scripts/python.exe tutorial_analyze.py ../logs/SPEC_Authentic/exp10_spec_real
```

#### 输出结果

```
Total Reward Statistics
  Mean:   -1271.20
  Min:    -1325.21
  Max:    -722.30

Episode Length Statistics
  Mean:   595.7 steps
  [WARNING] Truncated episodes: 199/201 (99.0%)

Energy Consumption Statistics
  Mean:   123.70 Wh

Host Utilization Statistics
  Mean: 0.53%

Learning Progress Comparison
  Total Reward        :   -1276.33    -1279.44 ( -0.24%)
  Energy (Wh)         :     124.22      125.02 ( +0.65%)
```

#### 诊断

🔴 **严重问题**：
1. **99% episodes truncated** - 任务从未完成
2. **利用率0.53%** - 资源严重浪费
3. **能源上升+0.65%** - 优化失败
4. **Reward几乎无改善** - 没有学习

#### 根本原因
查看CloudSim日志发现：
```
Step 200: Waiting=200, Finished=0, SimRunning=144
Step 600: Waiting=600, Finished=0, SimRunning=144
```
→ 144个cloudlets卡在running状态，永不完成！

#### 建议修复
1. **增加VM数量**：26 → 55 VMs
2. **增加episode长度**：600 → 800
3. **使用更简单workload先测试**

---

## 7. 常用命令速查

### PowerShell命令

```powershell
# 查看最后10个episodes的关键指标
cd logs/SPEC_Authentic/exp10_spec_real
Import-Csv monitor.csv |
    Select-Object -Last 10 |
    Select-Object r, l, cumulative_energy_wh, average_host_utilization |
    Format-Table -AutoSize

# 计算能源改善率
$data = Import-Csv monitor.csv
$first10 = ($data | Select-Object -First 10 | Measure-Object -Property cumulative_energy_wh -Average).Average
$last10 = ($data | Select-Object -Last 10 | Measure-Object -Property cumulative_energy_wh -Average).Average
$improvement = ($last10 - $first10) / $first10 * 100
Write-Host "Energy change: $([math]::Round($improvement, 2))%"
```

### Python命令

```python
# 快速统计
import pandas as pd
df = pd.read_csv("monitor.csv", skiprows=1)
print(df[['r', 'l', 'cumulative_energy_wh']].describe())

# 检查truncation
print(f"Truncated: {(df['l'] >= 600).sum()} / {len(df)}")

# First vs Last
print("Reward:", df['r'].head(10).mean(), "->", df['r'].tail(10).mean())
print("Energy:", df['cumulative_energy_wh'].head(10).mean(), "->",
      df['cumulative_energy_wh'].tail(10).mean())
```

---

## 8. 下一步行动

### 如果实验成功
1. ✅ 保存最佳模型：`best_model.zip`
2. ✅ 记录超参数：保存 `config_used.yml`
3. ✅ 尝试更复杂的workload
4. ✅ 调整reward weights进一步优化

### 如果实验失败
1. 🔍 运行诊断脚本找出问题
2. 📊 查看CloudSim日志确认simulation状态
3. 🛠️ 修改配置（VMs, episode length, workload）
4. 🔄 重新训练

---

## 9. 参考资源

- **分析脚本**：`drl-manager/analyze_training.py`
- **教程脚本**：`drl-manager/tutorial_analyze.py`
- **项目指标文档**：`docs/PROJECT_METRICS_SUMMARY.md`
- **能源优化文档**：`docs/ENERGY_OPTIMIZATION_WHvsPOWER.md`

---

**提示**：
- 始终先运行 `tutorial_analyze.py` 获取完整报告
- 关注 "Issue Detection" 部分的警告
- 对比多个实验时，确保使用相同的workload和max_episode_length
