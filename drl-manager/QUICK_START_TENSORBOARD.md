# 🚀 快速开始 - 查看核心训练指标

## 📋 TL;DR - 下次实验只看这个

### **步骤 1: 运行训练**

```bash
cd drl-manager
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5 --total-timesteps 50000
```

### **步骤 2: 启动 TensorBoard**

训练开始后，打开**新的终端**：

```bash
cd D:\rl-cloudsimplus-greenscheduling\logs\experiment_multi_dc_5
tensorboard --logdir=. --port=6006
```

浏览器打开：`http://localhost:6006`

### **步骤 3: 查看三个核心指标**

在 TensorBoard 左侧搜索框**依次输入**以下关键词（每次只输入一个）：

#### **1️⃣ Policy Loss & Value Loss**

搜索：`loss`

会看到：
- `custom_metrics/global_policy_loss_mean` ← Global Agent 策略损失
- `custom_metrics/global_value_loss_mean` ← Global Agent 价值损失
- `custom_metrics/local_agents_avg_policy_loss_mean` ← Local Agents 平均策略损失
- `custom_metrics/local_agents_avg_value_loss_mean` ← Local Agents 平均价值损失

**✅ 期望：曲线下降并趋于稳定**

#### **2️⃣ Carbon Emission**

搜索：`carbon`

会看到：
- `custom_metrics/total_carbon_kg_mean` ← 每个 episode 的碳排放

**✅ 期望：曲线下降（说明绿色能源利用率提高）**

#### **3️⃣ Episode Reward**

搜索：`episode_reward_mean`

会看到：
- `episode_reward_mean` ← 每个 episode 的总奖励

**✅ 期望：曲线上升并趋于稳定**

---

## 🎯 核心指标位置速查表

| 你想看的指标 | TensorBoard 搜索关键词 | 完整路径 |
|------------|---------------------|---------|
| **Policy Loss** | `loss` | `ray/tune/env_runners/custom_metrics/global_policy_loss_mean` |
| **Value Loss** | `loss` | `ray/tune/env_runners/custom_metrics/global_value_loss_mean` |
| **Carbon Emission** | `carbon` | `ray/tune/env_runners/custom_metrics/total_carbon_kg_mean` |
| **Episode Reward** | `episode_reward` | `episode_reward_mean` |
| **Green Energy Ratio** | `green_ratio` | `ray/tune/env_runners/custom_metrics/green_ratio_mean` |

---

## 🔍 如何隐藏不需要的指标

TensorBoard 默认显示很多 Ray 内部指标，隐藏它们：

1. **搜索 `counters`**，点击每个图表右上角的 ❌ 隐藏
2. **搜索 `connector`**，同样隐藏
3. **搜索 `done`**，隐藏

最后只保留：
- `episode_reward_*`
- `custom_metrics/*`

---

## 📊 方法 2: 直接查看 CSV 文件（更简单）

如果 TensorBoard 太复杂，直接用 Python 脚本查看：

```bash
cd drl-manager
python view_core_metrics.py 20251122_203819
```

会自动生成图表，包含所有核心指标！

---

## ⚠️ 常见问题

### **Q1: TensorBoard 里只有 ray/tune 开头的指标？**

**A:** 你看的是旧实验数据。需要用**修改后的代码重新运行训练**。

```bash
# 确认代码已更新
git pull  # 或者手动确认 rllib_green_energy_logger.py 已修改

# 重新训练
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5
```

### **Q2: custom_metrics 里没有 policy_loss 和 value_loss？**

**A:** 训练时间太短（< 10 iterations）。等待训练运行更长时间，或者：

```bash
# 检查日志中是否有 "[Global Agent] Policy Loss" 输出
tail -f <训练日志>
```

如果日志中有输出但 TensorBoard 没显示，刷新页面（Ctrl+R）。

### **Q3: 想看每个数据中心的独立指标？**

搜索：`local_dc0` 或 `local_dc1` ...

会看到：
- `custom_metrics/local_dc0_policy_loss_mean`
- `custom_metrics/local_dc0_value_loss_mean`
- ... 以此类推

---

## 📈 健康的训练曲线特征

### ✅ Policy Loss & Value Loss

```
  ↑
损|╲
失|  ╲___
  |      ‾‾‾‾
  └──────────→ 训练步数

✓ 前期快速下降
✓ 后期小幅波动（正常）
```

### ✅ Carbon Emission

```
  ↑
碳|╲
排|  ╲_____
放|        ‾‾‾
  └──────────→ 训练步数

✓ 持续下降
✓ 说明绿色能源利用率提高
```

### ✅ Episode Reward

```
  ↑
奖|     ╱‾‾‾
励|   ╱
  | ╱
  └──────────→ 训练步数

✓ 持续上升
✓ 最终趋于稳定
```

---

## 💾 数据备份

所有指标也保存在 CSV 文件：

```
logs/experiment_multi_dc_5/<timestamp>/
├── training_progress.csv      ← 每次迭代的汇总（推荐）
├── monitor.csv                ← 每个 episode 的详细数据
└── best_episode_details.csv   ← 最佳 episode
```

用 Excel 或 Python pandas 直接打开查看。

---

## 🎓 完整文档

需要更详细的说明？查看：
- `RLLIB_TENSORBOARD_GUIDE.md` - 完整 TensorBoard 使用指南
- `view_core_metrics.py` - Python 脚本直接可视化 CSV 数据

---

**记住：下次实验只需搜索这三个关键词**
1. `loss` - 查看损失
2. `carbon` - 查看碳排放
3. `episode_reward` - 查看奖励

就这么简单！🎉

