# 📊 TensorBoard 完整使用指南

## 🎯 已删除 training_progress.csv

`training_progress.csv` 在 local_mode 下不可靠，已从代码中删除。所有数据通过以下方式查看：

1. **Episode-level 数据** → `monitor.csv`（用 Python 脚本可视化）
2. **Loss 曲线和聚合统计** → **TensorBoard**

---

## 🚀 启动 TensorBoard

```bash
cd D:\rl-cloudsimplus-greenscheduling\logs\experiment_multi_dc_5
tensorboard --logdir=. --port=6006
```

浏览器打开：`http://localhost:6006`

---

## 📈 TensorBoard 中的核心指标

### **1️⃣ Policy Loss 和 Value Loss**

#### **Global Agent（全局路由策略）**

搜索：`global_agent`

你会看到：
- **`global_agent/policy_loss`** - 策略网络损失
- **`global_agent/value_loss`** - 价值网络损失
- **`global_agent/entropy`** - 策略熵（探索程度）

#### **Local Agents（本地调度策略）**

搜索：`local_agent`

你会看到：
- **`local_agents_avg/policy_loss`** - 所有本地 Agent 平均策略损失
- **`local_agents_avg/value_loss`** - 所有本地 Agent 平均价值损失
- **`local_agent_dc0/policy_loss`** - DC0 的策略损失
- **`local_agent_dc1/policy_loss`** - DC1 的策略损失
- ... 以此类推

#### **✅ 健康的损失曲线：**

```
  ↑
损|╲
失|  ╲___
  |      ‾‾‾‾
  └──────────→ Training Steps

✓ 前期快速下降
✓ 后期小幅波动（正常）
```

---

### **2️⃣ 各个 Agent 的 Reward**

#### **Episode Reward（总奖励）**

搜索：`episode_reward`

- **`episode_reward_mean`** - 每个 episode 的平均总奖励
- **`episode_reward_max`** - 最大 episode 奖励
- **`episode_reward_min`** - 最小 episode 奖励

#### **Agent-wise Rewards（Agent 分解奖励）**

搜索：`agent_rewards`

- **`agent_rewards/global_agent`** - Global Agent 的平均 episode 奖励
- **`agent_rewards/local_agents_avg`** - Local Agents 的平均 episode 奖励

或者搜索：`custom_metrics/.*agent_reward`

- **`custom_metrics/global_agent_reward_mean`** - Global Agent 奖励
- **`custom_metrics/local_agents_avg_reward_mean`** - Local Agents 平均奖励

#### **✅ 期望趋势：**

```
  ↑
奖|     ╱‾‾‾
励|   ╱
  | ╱
  └──────────→ Training Steps

✓ 持续上升
✓ 最终趋于稳定
```

---

### **3️⃣ Carbon Emission 和 Energy Metrics**

搜索：`carbon` 或 `custom_metrics`

- **`custom_metrics/total_carbon_kg_mean`** - 平均碳排放（kg CO2）
- **`custom_metrics/brown_used_wh_mean`** - 平均棕色能源使用（Wh）
- **`custom_metrics/green_ratio_mean`** - 绿色能源使用比例
- **`custom_metrics/green_waste_wh_mean`** - 绿色能源浪费（Wh）

#### **✅ 期望趋势：**

```
Carbon Emission 应该下降：
  ↑
碳|╲
排|  ╲_____
放|        ‾‾‾
  └──────────→ Training Steps

Green Ratio 应该上升：
  ↑
绿|     ╱‾‾‾
色|   ╱
比| ╱
例└──────────→ Training Steps
```

---

## 🔍 快速查找指标速查表

| 你想看的指标 | TensorBoard 搜索关键词 | 完整路径示例 |
|------------|---------------------|------------|
| **Global Policy Loss** | `global_agent/policy` | `global_agent/policy_loss` |
| **Global Value Loss** | `global_agent/value` | `global_agent/value_loss` |
| **Local Policy Loss (平均)** | `local_agents_avg/policy` | `local_agents_avg/policy_loss` |
| **Local Value Loss (平均)** | `local_agents_avg/value` | `local_agents_avg/value_loss` |
| **DC0 Policy Loss** | `local_agent_dc0` | `local_agent_dc0/policy_loss` |
| **Episode Reward** | `episode_reward_mean` | `episode_reward_mean` |
| **Global Agent Reward** | `agent_rewards/global` | `agent_rewards/global_agent` |
| **Local Agents Reward** | `agent_rewards/local` | `agent_rewards/local_agents_avg` |
| **Carbon Emission** | `carbon` | `custom_metrics/total_carbon_kg_mean` |
| **Green Energy Ratio** | `green_ratio` | `custom_metrics/green_ratio_mean` |

---

## 🎨 TensorBoard 使用技巧

### **1. 使用搜索过滤**

在左侧搜索框输入关键词：

**只看损失：**
```
loss
```

**只看奖励：**
```
reward
```

**只看能源指标：**
```
carbon|green|brown
```

**只看 Global Agent：**
```
global_agent
```

**只看 Local Agents：**
```
local_agent
```

### **2. 调整平滑度**

左侧有 **Smoothing** 滑块：
- 默认 0.6 通常就很好
- 拖到 0.8-0.9 可以看到更平滑的趋势

### **3. 隐藏不需要的指标**

TensorBoard 会显示很多 Ray 内部指标（`ray/tune/counters/*` 等）

**隐藏方法：**
1. 搜索 `counters`
2. 点击每个图表右上角的 ❌ 隐藏
3. 重复步骤隐藏 `connector_metrics`、`done` 等

**最后只保留：**
- `global_agent/*`
- `local_agent*/*`
- `agent_rewards/*`
- `episode_reward_*`
- `custom_metrics/*`

### **4. 对比多个实验**

如果你运行了多次实验：

```bash
tensorboard --logdir=logs/experiment_multi_dc_5
```

TensorBoard 会自动加载所有时间戳的实验，可以对比不同超参数的效果。

---

## 📊 完整的监控工作流

### **训练期间：**

**终端 1 - 运行训练：**
```bash
cd drl-manager
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5 --total-timesteps 100000
```

**终端 2 - 启动 TensorBoard：**
```bash
cd ..\logs\experiment_multi_dc_5
tensorboard --logdir=. --port=6006 --reload_interval=5
```

**浏览器：**
打开 `http://localhost:6006`

实时查看：
1. Policy Loss 和 Value Loss 是否下降 ✅
2. Episode Reward 是否上升 ✅
3. Carbon Emission 是否下降 ✅

---

### **训练完成后：**

**方法 1 - 查看 Episode-Level 详细数据（推荐）：**

```bash
cd drl-manager
python view_core_metrics.py <timestamp>
```

会生成包含所有指标的图表，保存到：
```
logs/experiment_multi_dc_5/<timestamp>/episode_metrics.png
```

**方法 2 - TensorBoard 回顾：**

```bash
cd ..\logs\experiment_multi_dc_5
tensorboard --logdir=.
```

查看完整的训练曲线。

**方法 3 - 直接读取 CSV：**

```python
import pandas as pd
df = pd.read_csv('logs/experiment_multi_dc_5/<timestamp>/monitor.csv')

# 查看所有列
print(df.columns)

# 绘制任意指标
import matplotlib.pyplot as plt
plt.plot(df['episode'], df['brown_used_wh'])
plt.show()
```

---

## ⚠️ 注意事项

### **1. 横轴是 Training Steps，不是 Episode Number**

TensorBoard 的横轴是：
- **num_env_steps_sampled** - 累计环境采样步数

**不是** Episode Number（1, 2, 3...）

如果你想看 **episode-level 的数据**（横轴 = Episode Number），用：
```bash
python view_core_metrics.py <timestamp>
```

### **2. Custom Metrics 是聚合统计**

`custom_metrics/total_carbon_kg_mean` 显示的是：
- 该 iteration 中所有 episode 的**平均值**

**不是**单个 episode 的真实值。

单个 episode 的真实值在 `monitor.csv`。

### **3. 初期数据可能不稳定**

训练最初几个 iterations，统计可能不准确（数据点少）。

等训练运行 10+ iterations 后，曲线会稳定下来。

---

## 🎯 检查列表

训练运行后，在 TensorBoard 确认以下指标可见：

### **损失曲线（必须有）：**

- [ ] `global_agent/policy_loss` - 存在且下降
- [ ] `global_agent/value_loss` - 存在且下降
- [ ] `local_agents_avg/policy_loss` - 存在且下降
- [ ] `local_agents_avg/value_loss` - 存在且下降

### **奖励指标（必须有）：**

- [ ] `episode_reward_mean` - 存在且上升
- [ ] `agent_rewards/global_agent` - 存在
- [ ] `agent_rewards/local_agents_avg` - 存在

### **能源指标（应该有）：**

- [ ] `custom_metrics/total_carbon_kg_mean` - 存在且下降
- [ ] `custom_metrics/green_ratio_mean` - 存在且上升
- [ ] `custom_metrics/brown_used_wh_mean` - 存在且下降

如果以上指标全部 ✓，说明训练正常且指标记录完整！

---

## 🆘 常见问题

### **Q: 看不到 policy_loss 和 value_loss？**

**A:** 等待训练至少 5-10 iterations。最初几个 iteration 可能没有 learner_stats。

检查训练日志中是否有：
```
[Iteration X] Global Agent - Policy Loss: 0.xxxxx, Value Loss: 0.xxxxx
```

如果日志有输出但 TensorBoard 没显示，刷新浏览器（Ctrl+R）。

---

### **Q: custom_metrics 里的值全是 0？**

**A:** 这是 local_mode=True 的限制。但 Policy Loss 和 Value Loss 应该正常显示（不依赖 custom_metrics）。

carbon emission 等指标查看 `monitor.csv`。

---

### **Q: 想看每个数据中心的独立指标？**

**A:** 搜索：`local_agent_dc0` 或 `dc_0`

会看到：
- `local_agent_dc0/policy_loss`
- `local_agent_dc0/value_loss`
- `custom_metrics/dc_0/green_ratio_mean`

---

### **Q: 训练很慢，TensorBoard 能加速吗？**

**A:** TensorBoard 不会影响训练速度（它只是读取日志文件）。

如果觉得 TensorBoard 加载慢，可以：
1. 只加载单个实验：`tensorboard --logdir=logs/experiment_multi_dc_5/<timestamp>/multidc_training`
2. 减少保留的 checkpoint 数量（在代码中设置 `num_to_keep=1`）

---

## 💾 数据备份和导出

### **从 TensorBoard 导出数据：**

点击图表左下角的 📥 按钮，下载 CSV 格式。

### **直接读取 monitor.csv：**

```python
import pandas as pd

# 读取数据
df = pd.read_csv('logs/experiment_multi_dc_5/<timestamp>/monitor.csv')

# 导出你需要的列
df[['episode', 'episode_reward', 'total_carbon_kg', 'brown_used_wh']].to_csv('my_results.csv', index=False)
```

---

## 📚 总结

### **TensorBoard 适合看：**
✅ Policy Loss 和 Value Loss
✅ 训练过程中的实时趋势
✅ 多个实验的对比

### **monitor.csv 适合看：**
✅ 每个 episode 的详细数据
✅ Episode-level 的精确分析
✅ 用 Python/Excel 进行自定义分析

### **推荐工作流：**
1. 训练时用 TensorBoard 监控 Loss 和 Reward 趋势
2. 训练完成后用 `view_core_metrics.py` 分析 episode-level 数据
3. 需要更深入分析时直接读取 `monitor.csv`

---

**Happy Training! 🚀**

