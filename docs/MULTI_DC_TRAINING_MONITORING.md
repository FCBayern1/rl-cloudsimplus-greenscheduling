# Multi-DC Hierarchical MARL Training Monitoring

本文档说明如何在训练多数据中心分层 MARL 系统时监控训练进度、查看奖励值并保存最佳模型。

## 功能特性

1. **实时训练进度显示**：显示每个 episode 的奖励值
2. **分层奖励追踪**：分别追踪 Global Agent 和 Local Agent 的奖励
3. **自动保存最佳模型**：当奖励值提升时自动保存模型
4. **TensorBoard 可视化**：实时可视化训练曲线
5. **CSV 日志记录**：将训练数据保存为 CSV 文件以便后续分析

## 新增文件

### 1. Callback 类

`drl-manager/src/callbacks/save_on_best_reward_hierarchical.py`

- 追踪 Global 和 Local 智能体的奖励
- 在奖励提升时自动保存最佳模型
- 记录训练进度到 CSV
- 记录最佳 episode 的详细数据

### 2. 更新的训练脚本

`drl-manager/src/training/train_hierarchical_multidc_joint.py`

- 集成了 Monitor wrapper 用于 episode 统计
- 集成了新的 hierarchical callback
- 添加了 TensorBoard 日志支持
- 添加了进度条显示

### 3. 更新的环境

`drl-manager/gym_cloudsimplus/envs/joint_training_env.py`

- 在 `info` 字典中添加了奖励信息
- 添加了能源统计信息（green energy ratio, brown energy, wasted green energy）

## 使用方法

### 1. 运行训练

```bash
cd drl-manager

# 使用 alternating 策略（推荐）
python -m src.training.train_hierarchical_multidc_joint --config ../config.yml --experiment experiment_multi_dc_3 --strategy alternating --output_dir ../logs/multi_dc_training
```

### 2. 训练过程中的输出

训练时你会看到类似以下的输出：

```
======================================================================
  Cycle 1/10
======================================================================
Training Global Agent...

Episode 1 completed (Timestep: 512)
  Episode Reward: 142.350
  Global Agent Reward: 85.120
  Local Agent Reward: 57.230
  Mean Reward (last 1 eps): 142.350
  Best Mean Reward: -inf
============================================================
🎉 New best mean reward! Saving models...
  ✅ Saved best global model to logs/multi_dc_training/best_global_model.zip
  ✅ Saved best local model to logs/multi_dc_training/best_local_model.zip
  ✅ Saved best episode details to logs/multi_dc_training/best_episode_details.csv
============================================================

Episode 2 completed (Timestep: 1024)
  Episode Reward: 158.750
  Global Agent Reward: 92.340
  Local Agent Reward: 66.410
  Mean Reward (last 2 eps): 150.550
  Best Mean Reward: 142.350
============================================================
🎉 New best mean reward! Saving models...
...
```

### 3. 输出文件结构

训练完成后，输出目录包含以下文件：

```
logs/multi_dc_training/
├── monitor/                          # Monitor 日志
│   └── 0.monitor.csv                # Episode 统计
├── tensorboard/                      # TensorBoard 日志
│   ├── global/                      # Global Agent 训练曲线
│   └── local/                       # Local Agent 训练曲线
├── checkpoints/                      # 定期检查点
│   ├── model_5000_steps.zip
│   ├── model_10000_steps.zip
│   └── ...
├── training_progress.csv            # 训练进度 CSV
├── best_episode_details.csv         # 最佳 episode 详细数据
├── best_global_model.zip            # 最佳 Global 模型
├── best_local_model.zip             # 最佳 Local 模型
├── global_cycle_1.zip               # 每个 cycle 的检查点
├── local_cycle_1.zip
├── global_cycle_2.zip
├── local_cycle_2.zip
├── ...
├── final_global_model.zip           # 最终模型
└── final_local_model.zip
```

### 4. 查看 TensorBoard

在训练过程中或训练后，可以使用 TensorBoard 查看训练曲线：

```bash
# 在新终端中运行
tensorboard --logdir=logs/multi_dc_training/tensorboard

# 然后在浏览器中访问：http://localhost:6006
```

你可以看到：
- **Global Agent**：全局路由智能体的学习曲线
- **Local Agent**：本地调度智能体的学习曲线
- **Episode reward**：每个 episode 的总奖励
- **Episode length**：每个 episode 的长度

### 5. 分析训练数据

#### 5.1 使用 Pandas 分析 CSV

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练进度
df = pd.read_csv("logs/multi_dc_training/training_progress.csv")

# 绘制奖励曲线
plt.figure(figsize=(12, 6))
plt.plot(df['timestep'], df['mean_reward'], label='Mean Reward')
plt.plot(df['timestep'], df['mean_global_reward'], label='Global Reward')
plt.plot(df['timestep'], df['mean_local_reward'], label='Local Reward')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.legend()
plt.title('Training Progress')
plt.grid(True)
plt.show()
```

#### 5.2 查看最佳 Episode

```python
# 读取最佳 episode 详细数据
best_episode = pd.read_csv("logs/multi_dc_training/best_episode_details.csv")

print(f"Best episode had {len(best_episode)} steps")
print(f"Total reward: {best_episode['reward'].sum():.3f}")
print(f"Average global reward: {best_episode['global_reward'].mean():.3f}")
print(f"Average local reward: {best_episode['local_reward'].mean():.3f}")
```

### 6. 加载最佳模型

```python
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

# 加载最佳模型
best_global_model = PPO.load("logs/multi_dc_training/best_global_model")
best_local_model = MaskablePPO.load("logs/multi_dc_training/best_local_model")

# 使用模型进行推理
# ...
```

## CSV 文件格式

### training_progress.csv

| 列名 | 描述 |
|------|------|
| `timestep` | 当前训练步数 |
| `episode` | Episode 编号 |
| `episode_reward` | 该 episode 的总奖励 |
| `episode_global_reward` | 该 episode 的 Global Agent 平均奖励 |
| `episode_local_reward` | 该 episode 的 Local Agent 平均奖励 |
| `mean_reward` | 最近 100 个 episodes 的平均总奖励 |
| `mean_global_reward` | 最近 100 个 episodes 的平均 Global 奖励 |
| `mean_local_reward` | 最近 100 个 episodes 的平均 Local 奖励 |
| `best_mean_reward` | 到目前为止的最佳平均奖励 |

### best_episode_details.csv

| 列名 | 描述 |
|------|------|
| `timestep` | 训练步数 |
| `reward` | 该步的总奖励 |
| `global_reward` | 该步的 Global Agent 奖励 |
| `local_reward` | 该步的 Local Agent 平均奖励 |

### monitor/0.monitor.csv

Stable-Baselines3 Monitor 生成的标准格式：

| 列名 | 描述 |
|------|------|
| `r` | Episode 总奖励 |
| `l` | Episode 长度（步数）|
| `t` | 训练时间（秒）|
| `global_reward` | Global Agent 奖励（如果在 info 中）|
| `local_reward` | Local Agent 奖励（如果在 info 中）|
| `total_reward` | 总奖励（如果在 info 中）|
| `green_energy_ratio` | 绿色能源比例（如果可用）|
| `brown_energy_wh` | 褐色能源消耗（如果可用）|
| `wasted_green_wh` | 浪费的绿色能源（如果可用）|

## 配置选项

在 `config.yml` 中的 `hierarchical_multi_dc.training` 部分：

```yaml
hierarchical_multi_dc:
  training:
    strategy: "alternating"           # "alternating" 或 "simultaneous"
    global_steps_per_cycle: 10000    # 每个 cycle 训练 Global Agent 的步数
    local_steps_per_cycle: 10000     # 每个 cycle 训练 Local Agent 的步数
    num_cycles: 10                    # 总共多少个训练 cycle
```

## 提示和技巧

### 1. 调整 Callback 保存频率

在 `train_hierarchical_multidc_joint.py` 中修改：

```python
save_best_callback = SaveOnBestRewardHierarchicalCallback(
    log_dir=str(self.output_dir),
    global_model=self.global_model,
    local_model=self.local_model,
    save_freq=1000,  # 修改此值以改变检查频率
    verbose=1
)
```

### 2. 调整检查点保存频率

```python
checkpoint_callback = CheckpointCallback(
    save_freq=5000,  # 每 5000 步保存一次检查点
    save_path=str(self.output_dir / "checkpoints"),
    name_prefix="model",
    verbose=1
)
```

### 3. 监控特定指标

在 `_create_environment` 中修改 `info_keywords` 以记录更多指标：

```python
info_keywords = (
    "global_reward", "local_reward", "total_reward",
    "cloudlets_routed", "cloudlets_completed",
    "green_energy_ratio", "brown_energy_wh", "wasted_green_wh",
    # 添加更多你关心的指标
)
```

## 故障排除

### 问题 1: 看不到训练进度

**解决方案**：确保你的环境正确返回 `info` 字典，并且包含必要的奖励信息。

### 问题 2: TensorBoard 没有显示数据

**解决方案**：
1. 检查 `tensorboard_log` 路径是否正确
2. 确保训练已经进行了至少一个 episode
3. 刷新 TensorBoard 浏览器页面

### 问题 3: 模型没有保存

**解决方案**：
1. 检查输出目录是否有写入权限
2. 确认奖励确实在提升（只有当奖励提升时才保存）
3. 查看日志中是否有错误信息

### 问题 4: CSV 文件为空

**解决方案**：
1. 确保训练至少完成了一个 episode
2. 检查 Monitor wrapper 是否正确初始化
3. 查看是否有文件写入权限问题

## 总结

通过这些改进，multi-DC 训练现在具有与 single-DC 训练相同的监控和日志功能：

✅ 实时显示训练进度
✅ 分别追踪 Global 和 Local Agent 奖励
✅ 自动保存最佳模型
✅ TensorBoard 可视化
✅ 详细的 CSV 日志
✅ 最佳 episode 详细记录

这些功能使得训练过程更加透明，便于调试和优化。
