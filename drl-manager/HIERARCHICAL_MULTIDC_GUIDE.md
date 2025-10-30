# Hierarchical Multi-Datacenter MARL Guide

## 概述

这个指南说明如何使用Hierarchical Multi-Datacenter环境进行训练。

## 架构设计

### 两层决策架构

```
┌────────────────────────────────────────────────────┐
│              Global Agent (Level 1)                │
│  决策: Cloudlet → 哪个Datacenter?                   │
│  输入: 所有DC的聚合状态（队列、利用率、绿色能源）      │
│  输出: [DC_0, DC_1, DC_2, ...] 路由决策             │
└───────────────┬────────────────────────────────────┘
                │ 路由cloudlets到各个DC
                ↓
┌────────────────────────────────────────────────────┐
│           Local Agents (Level 2)                   │
│  决策: Cloudlet → DC内的哪个VM?                     │
│  输入: 本DC的详细状态（VM负载、Host状态）            │
│  输出: [VM_0, VM_1, VM_2, ...] 调度决策            │
│                                                     │
│  DC_0: Local Agent 0 → VM调度                      │
│  DC_1: Local Agent 1 → VM调度                      │
│  DC_2: Local Agent 2 → VM调度                      │
└────────────────────────────────────────────────────┘
```

### Observation Spaces

#### Global Observation（全局观测）
```python
{
    "dc_green_power": [123.5, 456.2, 789.1],        # 每个DC的绿色能源供应
    "dc_queue_sizes": [5, 12, 3],                    # 每个DC的队列长度
    "dc_utilizations": [0.7, 0.85, 0.6],            # 每个DC的利用率
    "dc_available_pes": [120, 80, 150],             # 每个DC的可用PEs
    "dc_ram_utilizations": [0.65, 0.78, 0.55],      # 每个DC的RAM利用率
    "upcoming_cloudlets_count": 8,                   # 即将到达的任务数
    "next_cloudlet_pes": 4,                         # 下一个任务需要的PEs
    "next_cloudlet_mi": 50000,                      # 下一个任务的MI
    "upcoming_pes_distribution": [3, 2, 3],         # 即将到达任务的PEs分布
    "load_imbalance": 0.15,                         # 负载不平衡度
    "recent_completed": 125,                         # 最近完成的任务数
}
```

#### Local Observation（本地观测，每个DC独立）
```python
{
    "host_loads": [0.7, 0.8, 0.6, ...],             # 每个Host的负载
    "host_ram_usage": [0.65, 0.72, 0.58, ...],      # 每个Host的RAM使用
    "vm_loads": [0.9, 0.5, 0.7, ...],               # 每个VM的负载
    "vm_types": [1, 2, 3, 1, ...],                  # VM类型 (1=S, 2=M, 3=L)
    "vm_available_pes": [2, 4, 8, 2, ...],          # 每个VM的可用PEs
    "waiting_cloudlets": 5,                          # 等待队列中的任务数
    "next_cloudlet_pes": 4,                         # 下一个任务需要的PEs
}
```

### Action Spaces

#### Global Action（全局动作）
```python
# 为每个到达的cloudlet选择目标DC
global_action = [0, 1, 2, 0, 1]  # 5个cloudlets分别路由到DC_0, DC_1, DC_2, DC_0, DC_1
```

#### Local Action（本地动作，每个DC独立）
```python
# 为每个DC的队首cloudlet选择目标VM
local_actions = {
    0: 5,   # DC_0的队首任务分配给VM_5
    1: 12,  # DC_1的队首任务分配给VM_12
    2: 3,   # DC_2的队首任务分配给VM_3
}
```

## 配置文件设置

在`config.yml`中配置multi-DC实验（参考`experiment_multi_dc_3`）：

```yaml
experiment_multi_dc_3:
  # 启用多数据中心模式
  multi_datacenter_enabled: true
  py4j_port: 25333
  max_arriving_cloudlets: 50

  # 定义数据中心配置
  datacenters:
    - datacenter_id: 0
      name: "DC_HighPerformance"
      hosts_count: 20
      host_pes: 24
      initial_s_vm_count: 20
      initial_m_vm_count: 10
      initial_l_vm_count: 6
      green_energy_enabled: true
      turbine_id: 57

    - datacenter_id: 1
      name: "DC_EnergyEfficient"
      hosts_count: 24
      host_pes: 16
      initial_s_vm_count: 18
      initial_m_vm_count: 8
      initial_l_vm_count: 4
      green_energy_enabled: true
      turbine_id: 58

    - datacenter_id: 2
      name: "DC_Edge"
      hosts_count: 12
      host_pes: 12
      initial_s_vm_count: 12
      initial_m_vm_count: 6
      initial_l_vm_count: 2
      green_energy_enabled: true
      turbine_id: 59

  # Global Agent配置
  global_agent:
    algorithm: "PPO"
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99

    # Global reward权重
    reward_total_energy_coef: 2.0      # 最小化总能耗
    reward_green_ratio_coef: 3.0       # 最大化绿色能源使用率
    reward_load_balance_coef: 1.5      # 负载均衡
    reward_global_queue_coef: 1.0      # 最小化全局等待队列

  # Local Agents配置
  local_agents:
    algorithm: "MaskablePPO"
    parameter_sharing: true             # 所有DC的local agent共享参数
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99

    # Local reward权重
    reward_wait_time_coef: 1.0
    reward_unutilization_coef: 0.8
    reward_queue_penalty_coef: 0.5
```

## 运行训练

### 方法1: 通过entrypoint.py

```bash
# 设置实验ID
export EXPERIMENT_ID="experiment_multi_dc_3"

# 运行训练（会自动检测multi_datacenter_enabled并使用相应模块）
python drl-manager/mnt/entrypoint.py
```

### 方法2: 直接运行训练脚本

```bash
cd drl-manager/mnt

# 直接运行hierarchical训练
python train_hierarchical_multidc.py
```

### 方法3: 在Python代码中使用

```python
import gymnasium as gym
import gym_cloudsimplus

# 加载配置
from utils.config_loader import load_config
params = load_config("config.yml", "experiment_multi_dc_3")

# 创建环境
env = gym.make("HierarchicalMultiDC-v0", config=params)

# Reset环境
observations, info = env.reset()

# observations包含两层:
# - observations["global"]: 全局观测
# - observations["local"]: {0: dc0_obs, 1: dc1_obs, 2: dc2_obs}

# 执行step
action = {
    "global": [0, 1, 2],  # 路由决策
    "local": {0: 5, 1: 12, 2: 3}  # 每个DC的调度决策
}

observations, rewards, terminated, truncated, info = env.step(action)

# rewards也包含两层:
# - rewards["global"]: 全局奖励（能源优化、负载均衡）
# - rewards["local"]: {0: dc0_reward, 1: dc1_reward, 2: dc2_reward}
```

## 训练策略

### 策略1: 独立训练（推荐用于初学）

```python
# Phase 1: 先训练Local Agents（固定路由策略）
# - 所有cloudlets均匀分配到各DC
# - Local agents学习VM调度

# Phase 2: 再训练Global Agent（使用训练好的Local Agents）
# - Local agents固定（或缓慢更新）
# - Global agent学习DC路由策略
```

**优点：**
- ✅ 简单易实现
- ✅ 训练稳定
- ✅ 容易调试

**缺点：**
- ❌ 可能不是全局最优解

### 策略2: 联合训练（更高级）

```python
# 同时训练Global和Local Agents
# - 使用共享的replay buffer
# - 交替更新两层agents
```

**优点：**
- ✅ 可能达到更优性能

**缺点：**
- ❌ 训练不稳定
- ❌ 需要careful hyperparameter tuning

## 训练输出

训练完成后，会生成以下文件：

```
logs/hierarchical_3dc/
├── local_agent.zip          # 训练好的Local Agent模型
├── global_agent.zip         # 训练好的Global Agent模型
├── progress.csv             # 训练进度日志
├── monitor.csv              # Episode统计
└── config_used.yml          # 使用的配置
```

## 评估模型

```python
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

# 加载训练好的agents
local_agent = MaskablePPO.load("logs/hierarchical_3dc/local_agent.zip")
global_agent = PPO.load("logs/hierarchical_3dc/global_agent.zip")

# 创建评估环境
eval_env = gym.make("HierarchicalMultiDC-v0", config=params)

# 运行评估episode
obs, info = eval_env.reset()
done = False
episode_reward = 0

while not done:
    # Global agent决策
    global_obs = obs["global"]
    global_action, _ = global_agent.predict(global_obs, deterministic=True)

    # Local agents决策
    local_actions = {}
    for dc_id, local_obs in obs["local"].items():
        local_action, _ = local_agent.predict(local_obs, deterministic=True)
        local_actions[dc_id] = local_action

    # 组合action
    action = {
        "global": global_action.tolist(),
        "local": local_actions
    }

    # 执行step
    obs, rewards, terminated, truncated, info = eval_env.step(action)
    episode_reward += rewards["global"]
    done = terminated or truncated

print(f"Episode reward: {episode_reward}")
```

## 常见问题

### Q1: 为什么需要两层决策？

**A:** 跨数据中心调度是典型的hierarchical决策问题：
- **Global层**: 考虑宏观因素（绿色能源、网络延迟、负载均衡）
- **Local层**: 考虑微观因素（VM适配性、Host负载、资源碎片）

### Q2: 能否只用单层agent？

**A:** 可以，但会有问题：
- ❌ Action space巨大（所有DC的所有VM）
- ❌ 难以学习跨DC协调策略
- ❌ 不符合真实系统的决策层次

### Q3: Parameter sharing是什么意思？

**A:** 所有DC的Local Agents共享同一个神经网络参数。

**优点：**
- ✅ 减少参数量
- ✅ 跨DC知识共享
- ✅ 训练更快

**适用条件：**
- DC配置相似
- 任务分布相似

### Q4: 如何处理异构DC？

**A:** 如果DC配置差异很大：
- 方法1: 独立训练每个DC的Local Agent（不共享参数）
- 方法2: 使用DC ID作为observation的一部分（conditional policy）

## 下一步

1. **实验调参**: 调整reward权重平衡能源和性能
2. **绿色能源优化**: 利用风能预测优化调度
3. **Multi-agent通信**: 实现DC间信息交换机制
4. **Transfer learning**: 用单DC训练的agent初始化multi-DC的Local Agents

## 参考资料

- [CloudSim Plus文档](https://cloudsimplus.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Hierarchical RL论文](https://arxiv.org/abs/1604.06057)
