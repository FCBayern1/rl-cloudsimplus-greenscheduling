# TensorBoard 指标查看指南

## 📊 概述

您的多数据中心强化学习系统现在会记录详细的训练指标，包括：

1. **每个 Agent 的 Episode Reward**
   - Global Agent 奖励
   - Local Agent 奖励
   - 总奖励

2. **PPO 神经网络损失**
   - Policy Loss (策略网络损失)
   - Value Loss (价值网络损失)
   - Entropy Loss (熵损失)
   - KL Divergence (KL 散度)
   - Clip Fraction (裁剪比例)

3. **训练统计**
   - Episode 长度
   - 滚动平均奖励
   - 学习率变化

---

## 🚀 启动 TensorBoard

### 1. 训练完成后，找到日志目录

训练日志保存在：
```
logs/joint_training/<timestamp>/tensorboard/
├── global/    # Global Agent 的日志
└── local/     # Local Agent 的日志
```

### 2. 启动 TensorBoard

在 `drl-manager` 目录下运行：

```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
# 或
source .venv/bin/activate  # Linux/Mac

# 启动 TensorBoard，指向您的实验目录
tensorboard --logdir=../logs/joint_training/<timestamp>/tensorboard

# 或者查看所有实验
tensorboard --logdir=../logs/joint_training
```

### 3. 在浏览器中打开

TensorBoard 会输出一个 URL，通常是：
```
http://localhost:6006
```

在浏览器中打开这个地址。

---

## 📈 查看指标详解

### **1. Episode Rewards (Episode 奖励)**

在 TensorBoard 左侧选择 **"SCALARS"** 标签。

#### **单独的 Agent 奖励**

查找以下指标：

```
global_agent/episode/reward          # Global Agent 的单个 episode 总奖励
global_agent/episode/mean_reward_100 # Global Agent 最近100个 episode 的平均奖励

local_agent/episode/reward           # Local Agent 的单个 episode 总奖励
local_agent/episode/mean_reward_100  # Local Agent 最近100个 episode 的平均奖励
```

**如何对比两个 Agent？**
1. 在左侧的搜索框中输入 `episode/reward`
2. 会显示 `global_agent` 和 `local_agent` 的两条曲线
3. 可以用不同颜色区分

#### **组合奖励**

```
episode/reward              # 总的 episode 奖励 (global + local)
episode/global_reward       # Global Agent 的平均单步奖励
episode/local_reward        # Local Agent 的平均单步奖励
episode/mean_reward         # 滚动平均总奖励 (100 episodes)
```

---

### **2. PPO 损失曲线**

#### **Global Agent 的损失**

```
global/train/policy_loss        # 策略网络损失
global/train/value_loss         # 价值网络损失
global/train/entropy_loss       # 熵损失 (探索 vs 利用)
global/train/loss               # 总损失
```

#### **Local Agent 的损失**

```
local/train/policy_loss         # 策略网络损失
local/train/value_loss          # 价值网络损失
local/train/entropy_loss        # 熵损失
local/train/loss                # 总损失
```

**解读损失曲线：**
- **Policy Loss**: 应该逐渐下降并稳定，表示策略在改进
- **Value Loss**: 应该下降，表示价值函数估计更准确
- **Entropy Loss**: 如果太低，可能过度开发（exploitation）；太高则探索过多

---

### **3. PPO 训练指标**

```
global/train/approx_kl          # KL 散度（新旧策略差异）
global/train/clip_fraction      # 被裁剪的样本比例
global/train/explained_variance # 价值函数的解释方差
global/train/learning_rate      # 学习率（如果使用学习率调度）
global/train/n_updates          # 更新次数

local/train/approx_kl
local/train/clip_fraction
local/train/explained_variance
local/train/learning_rate
local/train/n_updates
```

**关键指标解读：**
- **approx_kl**: 应该保持在一个小的范围（如 < 0.1），过大表示策略变化太快
- **clip_fraction**: PPO 的核心，通常在 0.1-0.3 之间是健康的
- **explained_variance**: 接近 1 表示价值函数很准确，接近 0 表示估计很差

---

### **4. Rollout 统计**

```
global/rollout/mean_ep_reward   # Global rollout 期间的平均 episode 奖励
global/rollout/mean_ep_length   # Global rollout 期间的平均 episode 长度

local/rollout/mean_ep_reward
local/rollout/mean_ep_length
```

---

## 🎨 TensorBoard 高级功能

### **对比多个实验**

如果您运行了多次训练：

```bash
tensorboard --logdir=../logs/joint_training
```

TensorBoard 会自动加载所有子目录的实验，您可以：
1. 在左侧看到所有实验的列表
2. 选择/取消选择某些实验
3. 对比不同配置的效果

### **平滑曲线**

左侧有一个 **"Smoothing"** 滑块：
- 拖动它可以平滑曲线，更容易看出趋势
- 默认值 0.6 通常就很好

### **下载数据**

每个图表左下角有三个按钮：
- 📥 **Download**: 下载 CSV 格式的数据
- 🔍 **Toggle Y-Axis**: 切换 Y 轴刻度（线性/对数）
- 📌 **Pin**: 固定图表

---

## 📊 典型的健康训练曲线

### **Episode Reward**
```
时间 →
  ↑
奖|     ╱‾‾‾‾‾
励|   ╱
  | ╱
  |╱___________
  └──────────→ 训练步数
```
- 初期上升
- 中期波动但整体向上
- 后期趋于稳定

### **Policy Loss**
```
  ↑
损|╲
失|  ╲__
  |     ‾‾‾‾‾‾
  └──────────→ 训练步数
```
- 快速下降
- 然后趋于稳定（小幅波动是正常的）

### **Value Loss**
```
  ↑
损|╲
失|  ╲_
  |    ‾‾‾‾‾
  └──────────→ 训练步数
```
- 类似 Policy Loss
- 可能下降更平滑

---

## 🔍 常见问题诊断

### **1. Reward 不增长或下降**

可能原因：
- 学习率太高 → 检查 `train/learning_rate`
- KL 散度太大 → 检查 `train/approx_kl`
- 价值函数估计差 → 检查 `train/explained_variance`

### **2. Loss 震荡剧烈**

可能原因：
- 学习率太高 → 降低 `learning_rate`
- Batch size 太小 → 增大 `batch_size`
- Clip range 不合适 → 调整 `clip_range`

### **3. Entropy 快速降至 0**

问题：
- Agent 过早收敛到次优策略

解决：
- 增大 `ent_coef`（熵系数）
- 增加探索

---

## 💾 导出训练曲线

### **方法 1: 直接在 TensorBoard 下载**
1. 点击图表左下角的 📥 按钮
2. 下载 CSV 文件

### **方法 2: 使用 Python 脚本读取**

```python
from tensorboard.backend.event_processing import event_accumulator

# 加载 TensorBoard 日志
ea = event_accumulator.EventAccumulator('path/to/tensorboard/global')
ea.Reload()

# 获取指标
policy_loss = ea.Scalars('global/train/policy_loss')
rewards = ea.Scalars('global_agent/episode/reward')

# 转换为 DataFrame
import pandas as pd
df = pd.DataFrame(policy_loss)
```

### **方法 3: 使用 CSV 日志**

训练脚本已经自动保存 CSV：
```
logs/joint_training/<timestamp>/training_progress.csv
```

包含：
- timestep
- episode
- episode_reward
- episode_global_reward
- episode_local_reward
- mean_reward
- best_mean_reward

直接用 Excel 或 Python 打开即可。

---

## 📝 示例：完整查看流程

1. **启动 TensorBoard**
   ```bash
   cd drl-manager
   .venv\Scripts\activate
   tensorboard --logdir=../logs/joint_training
   ```

2. **打开浏览器** → `http://localhost:6006`

3. **查看 Global Agent 表现**
   - 左侧搜索：`global_agent/episode/reward`
   - 观察曲线是否上升

4. **查看 Local Agent 表现**
   - 左侧搜索：`local_agent/episode/reward`
   - 对比与 Global Agent 的差异

5. **检查损失函数**
   - 搜索：`global/train/policy_loss`
   - 搜索：`global/train/value_loss`
   - 确保都在下降

6. **检查 PPO 健康指标**
   - `train/approx_kl` < 0.1 ✓
   - `train/clip_fraction` 在 0.1-0.3 ✓
   - `train/explained_variance` 接近 1 ✓

7. **调整平滑度**
   - 拖动左侧 Smoothing 滑块到 0.7-0.8
   - 更清晰地看出趋势

---

## 🎯 训练成功的标志

✅ **Global Agent**
- Episode reward 稳定上升
- Policy loss 下降并趋于稳定
- Value loss 下降
- Explained variance > 0.5

✅ **Local Agent**
- Episode reward 上升
- Policy loss 下降
- Clip fraction 在 0.1-0.3

✅ **整体**
- 总 reward = global_reward + local_reward 持续改进
- 没有突然的崩溃或震荡
- KL 散度保持在合理范围

---

## 🛠️ 进阶技巧

### **1. 实时监控训练**

训练时同时打开 TensorBoard：
```bash
# Terminal 1: 运行训练
python entrypoint_multidc.py

# Terminal 2: 启动 TensorBoard
tensorboard --logdir=../logs/joint_training
```

TensorBoard 会自动刷新，实时显示最新数据。

### **2. 对比不同配置**

修改 `config.yml` 中的参数后重新训练，TensorBoard 会显示所有实验：
- 不同学习率
- 不同 batch size
- 不同奖励权重

### **3. 使用 TensorBoard.dev 分享**

将您的实验上传到云端分享：
```bash
tensorboard dev upload --logdir ../logs/joint_training/<timestamp>/tensorboard
```

会生成一个公开链接，可以分享给他人。

---

## 📚 相关文档

- [Stable-Baselines3 TensorBoard Integration](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)
- [TensorBoard 官方指南](https://www.tensorflow.org/tensorboard/get_started)
- [PPO 算法详解](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

---

## 🆘 需要帮助？

如果您在查看 TensorBoard 时遇到问题：
1. 检查日志目录是否存在
2. 确保训练脚本正确运行
3. 查看 `training_progress.csv` 备用数据
4. 检查防火墙是否阻止了端口 6006

---

**Happy Training! 🚀**

