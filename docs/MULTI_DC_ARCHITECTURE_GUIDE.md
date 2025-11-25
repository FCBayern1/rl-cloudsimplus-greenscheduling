# Multi-Datacenter Hierarchical RL 架构完整指南

本文档详细介绍基于CloudSim Plus的多数据中心（Multi-DC）分层强化学习系统的完整架构、执行流程和文件说明。

---

## 📋 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [关键文件说明](#3-关键文件说明)
4. [实验流程](#4-实验流程)
5. [数据流向](#5-数据流向)
6. [配置说明](#6-配置说明)
7. [训练策略](#7-训练策略)

---

## 1. 系统概述

### 1.1 什么是Multi-DC Hierarchical RL？

这是一个**两层分层强化学习（Hierarchical RL）系统**，用于优化多数据中心环境下的任务调度：

```
┌─────────────────────────────────────────────────────────┐
│              Global Agent (高层)                         │
│  决策: 新到达的Cloudlets → 路由到哪个Datacenter           │
│  目标: 负载均衡 + 绿色能源利用                            │
└────────────┬────────────────────────────────────────────┘
             │ 将cloudlets路由到各个DC
             ↓
┌────────────┴────────────────────────────────────────────┐
│          Local Agents (低层，每个DC各1个)                │
│  决策: DC内部的Cloudlet → 分配到哪个VM                    │
│  目标: 完成时间最小化 + 资源利用率最大化                   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心特性

- ✅ **固定批量路由（Fixed Batch Routing）**: Global Agent每个timestep处理固定数量的cloudlets（batch_size=5）
- ✅ **全局等待队列（Global Waiting Queue）**: 未处理的cloudlets排队等待
- ✅ **绿色能源感知（Green Energy Aware）**: 优化可再生能源使用
- ✅ **MaskablePPO**: Local Agents使用动作掩码避免无效动作
- ✅ **交替训练（Alternating Training）**: Global和Local Agents轮流训练

---

## 2. 架构设计

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python (DRL Manager)                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  JointTrainingManager                                   │    │
│  │  - 管理训练循环                                          │    │
│  │  - 协调Global和Local Agents                             │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                      │
│  ┌────────┴──────────────┬──────────────────────────────┐      │
│  │   Global Agent        │    Local Agents              │      │
│  │   (PPO)               │    (MaskablePPO x N)         │      │
│  └───────┬───────────────┴──────┬───────────────────────┘      │
│          │                      │                               │
│  ┌───────┴──────────────────────┴────────────────────────┐     │
│  │  JointTrainingEnv (Gymnasium)                          │     │
│  │  - 统一观察空间和动作空间                                │     │
│  │  - 协调global和local actions                           │     │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                      │
│  ┌────────┴───────────────────────────────────────────────┐    │
│  │  HierarchicalMultiDCEnv (Gymnasium)                     │    │
│  │  - 实现observation/action处理                           │    │
│  │  - 批量路由逻辑                                          │    │
│  │  - 动作掩码生成                                          │    │
│  └────────┬───────────────────────────────────────────────┘    │
└───────────┼────────────────────────────────────────────────────┘
            │ Py4J Gateway (Java-Python Bridge)
            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Java (CloudSim Plus Gateway)                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  HierarchicalMultiDCGateway                              │   │
│  │  - 暴露API给Python                                        │   │
│  │  - 管理仿真生命周期                                        │   │
│  └─────┬───────────────────────────────────────────────────┘   │
│        │                                                         │
│  ┌─────┴───────────────────────────────────────────────────┐   │
│  │  MultiDatacenterSimulationCore                           │   │
│  │  - 核心仿真逻辑                                           │   │
│  │  - 分层step执行                                           │   │
│  │  - 观察状态和奖励计算                                      │   │
│  └─────┬─────────────────────┬───────────────────────────┘    │
│        │                     │                                  │
│  ┌─────┴──────────┐    ┌─────┴──────────────────────┐         │
│  │  GlobalBroker  │    │  DatacenterInstance x N     │         │
│  │  - 全局队列     │    │  - LoadBalancingBroker      │         │
│  │  - 批量路由     │    │  - Datacenter              │         │
│  │                │    │  - VMs + Hosts             │         │
│  └────────────────┘    └────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 两层决策流程

#### 2.2.1 Global Agent决策（DC路由）

```python
# 每个timestep
observation = {
    "global": {
        "dc_cpu_load": [0.6, 0.3, 0.8],           # 3个DC的CPU负载
        "dc_ram_load": [0.5, 0.4, 0.7],           # 3个DC的RAM负载
        "dc_green_ratio": [0.8, 0.5, 0.2],        # 绿色能源比例
        "global_waiting_count": 12,               # 全局队列中cloudlet数量
        "next_cloudlet_pes": 4,                   # 下一个cloudlet的PE需求
        # ... 更多特征
    }
}

# Global Agent输出（MultiDiscrete action space）
action = [2, 0, 1, 2, 0]  # batch_size=5
# 含义: 
#   - Cloudlet 1 → DC2
#   - Cloudlet 2 → DC0
#   - Cloudlet 3 → DC1
#   - Cloudlet 4 → DC2
#   - Cloudlet 5 → DC0
```

#### 2.2.2 Local Agents决策（VM调度）

```python
# 每个DC的Local Agent各自决策
observation = {
    "local": {
        0: {  # DC0的观察
            "vm_cpu_usage": [0.2, 0.8, 0.5, ...],  # DC0所有VM的CPU使用率
            "vm_ram_usage": [0.3, 0.6, 0.4, ...],  # RAM使用率
            "local_queue_size": 5,                 # DC0本地队列大小
            # ... 更多特征
        },
        1: { ... },  # DC1的观察
        2: { ... },  # DC2的观察
    }
}

# Local Agents输出（每个DC一个action）
actions = {
    0: 12,   # DC0: 将queue头部的cloudlet分配到VM 12
    1: -1,   # DC1: NoAssign（不分配，等待更好时机）
    2: 5,    # DC2: 将queue头部的cloudlet分配到VM 5
}
```

### 2.3 Action Space设计

#### Global Agent
```python
# MultiDiscrete: 每个cloudlet选择一个DC
action_space = spaces.MultiDiscrete([num_datacenters] * batch_size)
# 例如: 3个DC，batch_size=5
# action_space = MultiDiscrete([3, 3, 3, 3, 3])
```

#### Local Agents
```python
# Discrete: 选择一个VM（或NoAssign）
action_space = spaces.Discrete(num_vms_in_dc + 1)
# +1 是因为有NoAssign选项（action=0）
```

---

## 3. 关键文件说明

### 3.1 Python文件（DRL Manager）

#### 🎯 训练脚本

| 文件路径 | 作用 | 何时使用 |
|---------|------|---------|
| **`drl-manager/src/training/train_hierarchical_multidc_joint.py`** | **联合训练主脚本**<br>- 管理Global和Local Agents的训练循环<br>- 实现交替训练策略<br>- 包含GlobalAgentEnv和LocalAgentEnv包装器 | **这是你运行multi-dc实验的主要入口** |
| `drl-manager/src/training/train_hierarchical_multidc.py` | 独立训练脚本（只训练Global或只训练Local） | 单独测试某一层agent时使用 |
| `drl-manager/src/training/train_single_dc.py` | 单数据中心训练（对比基准） | 作为baseline对比 |

#### 🏋️ 环境文件

| 文件路径 | 作用 | 核心功能 |
|---------|------|---------|
| **`drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py`** | **核心Multi-DC环境**<br>- 实现Gymnasium接口<br>- 处理global/local观察和动作<br>- 固定批量路由逻辑<br>- 动作掩码生成 | - `step()`: 执行一步仿真<br>- `reset()`: 重置环境<br>- `_get_observation()`: 构造观察<br>- `get_action_masks()`: 生成valid动作 |
| **`drl-manager/gym_cloudsimplus/envs/joint_training_env.py`** | **联合训练环境包装器**<br>- 为Global和Local Agents提供统一接口<br>- 管理联合观察/动作空间<br>- 包含ParameterSharingWrapper | - 统一obs/action格式<br>- 简化多agent交互 |

#### 📊 回调和工具

| 文件路径 | 作用 |
|---------|------|
| `drl-manager/src/callbacks/save_on_best_reward_hierarchical.py` | 保存最佳模型（同时保存Global和Local） |
| `drl-manager/src/callbacks/monitoring.py` | TensorBoard监控和日志记录 |
| `drl-manager/scripts/analyze_training.py` | 训练结果分析脚本 |

### 3.2 Java文件（CloudSim Plus Gateway）

#### 🌐 Gateway和核心仿真

| 文件路径 | 作用 | 核心方法 |
|---------|------|---------|
| **`cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/HierarchicalMultiDCGateway.java`** | **Python-Java接口**<br>- 暴露API给Python<br>- 管理仿真生命周期 | - `reset()`: 重置仿真<br>- `step()`: 执行一步<br>- `getObservation()`: 获取观察<br>- `getActionMasks()`: 获取动作掩码 |
| **`cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/MultiDatacenterSimulationCore.java`** | **核心仿真引擎**<br>- 实现分层step逻辑<br>- 计算观察和奖励<br>- 管理DC实例 | - `executeHierarchicalStep()`: 分层执行<br>- `executeGlobalRouting()`: 全局路由<br>- `executeLocalScheduling()`: 本地调度<br>- `getGlobalObservation()`: 全局观察<br>- `getLocalObservation()`: 局部观察 |

#### 🏢 Datacenter组件

| 文件路径 | 作用 | 核心功能 |
|---------|------|---------|
| **`cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/GlobalBroker.java`** | **全局路由代理**<br>- 管理全局等待队列<br>- 批量路由cloudlets | - `processArrivingCloudlets()`: 处理新到达<br>- `getBatchForRouting()`: 获取批次<br>- `getGlobalWaitingCloudletsCount()`: 队列大小<br>- `routeCloudlets()`: 执行路由 |
| **`cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/DatacenterInstance.java`** | **单个DC实例**<br>- 封装Datacenter<br>- 管理Local Broker<br>- 统计指标 | - `getLocalObservation()`: DC观察<br>- `getLocalActionMask()`: 动作掩码<br>- `executeLocalAction()`: 执行调度 |
| **`cloudsimplus-gateway/src/main/java/giu/edu/cspg/singledc/LoadBalancingBroker.java`** | **Local DC的Broker**<br>- 管理VM创建<br>- Cloudlet调度<br>- 统计收集 | - `receiveCloudlet()`: 接收cloudlet<br>- `assignCloudletToVm()`: 分配到VM<br>- 监听cloudlet完成事件 |

#### ⚡ 绿色能源

| 文件路径 | 作用 |
|---------|------|
| **`cloudsimplus-gateway/src/main/java/giu/edu/cspg/energy/GreenEnergyProvider.java`** | 绿色能源模拟（太阳能/风能） |

#### 🛠️ 工具类

| 文件路径 | 作用 |
|---------|------|
| `cloudsimplus-gateway/src/main/java/giu/edu/cspg/common/DatacenterSetup.java` | 创建Datacenter、Host、VM的工厂类 |
| `cloudsimplus-gateway/src/main/java/giu/edu/cspg/common/SimulationSettings.java` | 配置解析和管理 |
| `cloudsimplus-gateway/src/main/java/giu/edu/cspg/workload/WorkloadGenerator.java` | Cloudlet生成（CSV或随机） |

---

## 4. 实验流程

### 4.1 完整训练流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 启动训练                                                   │
│    python train_hierarchical_multidc_joint.py \              │
│      --config config.yml \                                   │
│      --experiment experiment_multi_dc_3 \                    │
│      --strategy alternating                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 2. 初始化                                                     │
│    - 加载config.yml                                          │
│    - 设置随机种子（reproducibility）                          │
│    - 创建输出目录（logs/joint_training/timestamp/）           │
│    - 初始化Java Gateway（Py4J）                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 3. 创建环境                                                   │
│    JointTrainingEnv (包装 HierarchicalMultiDCEnv)            │
│    → 创建Java仿真（MultiDatacenterSimulationCore）           │
│    → 创建3个DatacenterInstances                              │
│    → 初始化GlobalBroker                                      │
│    → 加载workload CSV                                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 4. 创建Agents                                                │
│    - Global Agent: PPO (策略网络)                            │
│    - Local Agents: MaskablePPO (共享策略网络)                │
│    - 配置学习率、gamma、buffer等超参数                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 5. 交替训练循环 (Alternating Training)                        │
│                                                              │
│    For cycle in range(num_cycles):                          │
│                                                              │
│      ┌──────────────────────────────────────────────┐       │
│      │ 5.1 训练Global Agent                          │       │
│      │     - 环境返回global observation              │       │
│      │     - Global Agent选择DC routing actions     │       │
│      │     - Local Agents使用random masked actions  │       │
│      │     - 执行global_steps次step                  │       │
│      │     - 更新Global Agent策略                    │       │
│      └──────────────────┬───────────────────────────┘       │
│                         │                                    │
│      ┌──────────────────▼───────────────────────────┐       │
│      │ 5.2 训练Local Agents                          │       │
│      │     - 环境返回local observations (3个DC)      │       │
│      │     - Global Agent使用固定策略                │       │
│      │     - Local Agents学习VM调度策略              │       │
│      │     - 执行local_steps次step                   │       │
│      │     - 更新Local Agents策略（参数共享）         │       │
│      └──────────────────┬───────────────────────────┘       │
│                         │                                    │
│      ┌──────────────────▼───────────────────────────┐       │
│      │ 5.3 保存Checkpoint                            │       │
│      │     - global_cycle_N.zip                     │       │
│      │     - local_cycle_N.zip                      │       │
│      └──────────────────────────────────────────────┘       │
│                                                              │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│ 6. 保存最终模型                                               │
│    - final_global_model.zip                                 │
│    - final_local_model.zip                                  │
│    - training_metrics.csv                                   │
│    - seed_used.txt                                          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 单个Episode流程

```
Episode开始
    │
    ├─→ reset() 
    │   ├─ Java: 重置CloudSim仿真
    │   ├─ 清空全局队列
    │   ├─ 重置所有DC状态
    │   └─ 返回初始observation
    │
    ├─→ Loop: 直到episode结束
    │   │
    │   ├─→ step(global_actions, local_actions)
    │   │   │
    │   │   ├─ [1] Global Routing Phase
    │   │   │   ├─ 获取新到达的cloudlets
    │   │   │   ├─ 加入全局等待队列
    │   │   │   ├─ 从队列取batch_size个cloudlets
    │   │   │   ├─ 根据global_actions路由到各DC
    │   │   │   └─ 剩余cloudlets留在队列
    │   │   │
    │   │   ├─ [2] Local Scheduling Phase
    │   │   │   ├─ 每个DC执行local_action
    │   │   │   ├─ 将DC queue头部cloudlet分配到指定VM
    │   │   │   └─ 或NoAssign（action=0）
    │   │   │
    │   │   ├─ [3] Simulation Advance
    │   │   │   ├─ CloudSim时钟前进timestep秒
    │   │   │   ├─ VM执行cloudlets
    │   │   │   ├─ 完成的cloudlets被记录
    │   │   │   └─ 更新能源消耗
    │   │   │
    │   │   ├─ [4] Observation Collection
    │   │   │   ├─ 收集global observation
    │   │   │   │   - DC负载、能源状态
    │   │   │   │   - 全局队列大小
    │   │   │   │   - 下一个cloudlet特征
    │   │   │   │
    │   │   │   └─ 收集local observations (每个DC)
    │   │   │       - VM使用率
    │   │   │       - 本地队列状态
    │   │   │       - 绿色能源可用量
    │   │   │
    │   │   ├─ [5] Reward Calculation
    │   │   │   ├─ Global Reward:
    │   │   │   │   - 负载均衡奖励
    │   │   │   │   - 绿色能源利用奖励
    │   │   │   │   - 队列管理惩罚
    │   │   │   │
    │   │   │   └─ Local Rewards (每个DC):
    │   │   │       - Cloudlet完成奖励
    │   │   │       - 等待时间惩罚
    │   │   │       - VM利用率奖励
    │   │   │
    │   │   └─ 返回 (obs, rewards, terminated, truncated, info)
    │   │
    │   └─ 检查终止条件
    │       ├─ 时间到达max_time
    │       ├─ 所有cloudlets完成
    │       └─ 或episode被截断
    │
    └─→ Episode结束，统计指标
```

### 4.3 关键执行细节

#### 4.3.1 固定批量路由（Fixed Batch Routing）

```java
// GlobalBroker.java

// 每个timestep开始
public void processArrivingCloudlets(double currentTime, double timestep) {
    // 获取这个timestep内新到达的cloudlets
    List<Cloudlet> arriving = workloadGenerator.getCloudletsInTimeRange(
        currentTime, currentTime + timestep
    );
    
    // 全部加入全局等待队列
    globalWaitingQueue.addAll(arriving);
}

// 获取固定批次用于路由
public List<Cloudlet> getBatchForRouting(int batchSize) {
    List<Cloudlet> batch = new ArrayList<>();
    int toRoute = Math.min(batchSize, globalWaitingQueue.size());
    
    for (int i = 0; i < toRoute; i++) {
        batch.add(globalWaitingQueue.poll());  // 从队列头部取出
    }
    
    return batch;  // 可能 < batchSize（队列不足）
}

// 执行路由
public void routeCloudlets(List<Cloudlet> batch, List<Integer> dcChoices) {
    for (int i = 0; i < batch.size(); i++) {
        Cloudlet cloudlet = batch.get(i);
        int targetDC = dcChoices.get(i);
        
        // 发送到对应DC的LoadBalancingBroker
        datacenters.get(targetDC).getBroker().receiveCloudlet(cloudlet);
    }
}
```

#### 4.3.2 动作掩码生成（Action Masking）

```java
// DatacenterInstance.java

public boolean[] getLocalActionMask() {
    int numVms = broker.getVmCreatedList().size();
    boolean[] mask = new boolean[numVms + 1];
    
    // Action 0: NoAssign，总是有效
    mask[0] = true;
    
    // 检查每个VM是否可用
    for (int i = 0; i < numVms; i++) {
        Vm vm = broker.getVmCreatedList().get(i);
        
        Cloudlet nextCloudlet = localQueue.peek();
        if (nextCloudlet != null) {
            // VM必须有足够的PEs
            boolean hasEnoughPes = vm.getPesNumber() >= nextCloudlet.getPesNumber();
            
            // VM不能overloaded
            boolean notOverloaded = vm.getCpuPercentUtilization() < 0.95;
            
            mask[i + 1] = hasEnoughPes && notOverloaded;
        } else {
            mask[i + 1] = false;  // 没有cloudlet要调度
        }
    }
    
    return mask;
}
```

---

## 5. 数据流向

### 5.1 Observation数据流

```
Java (CloudSim Plus)                    Python (RL Agents)
────────────────────                    ──────────────────

MultiDatacenterSimulationCore
├─ collectGlobalObservation()
│  ├─ DC CPU/RAM loads        ─────┐
│  ├─ Green energy ratios          │
│  ├─ Global queue size             ├─→ Py4J ─→ HierarchicalMultiDCEnv
│  └─ Next cloudlet features        │          ↓
│                                    │       observation dict
├─ collectLocalObservations()        │       {
│  ├─ DC0:                          │         "global": {...},
│  │  ├─ VM usages            ─────┤         "local": {
│  │  ├─ Local queue                │           0: {...},
│  │  └─ Resources                  │           1: {...},
│  ├─ DC1: ...               ──────┤           2: {...}
│  └─ DC2: ...               ──────┘         }
│                                           }
└─ getActionMasks()                         ↓
   ├─ Global mask (always all valid)    JointTrainingEnv
   └─ Local masks for each DC           ↓
                                        ┌─────────┴──────────┐
                                        ↓                    ↓
                                   Global Agent        Local Agents
                                   (PPO)               (MaskablePPO)
```

### 5.2 Action数据流

```
Python (RL Agents)                      Java (CloudSim Plus)
──────────────────                      ────────────────────

Global Agent (PPO)
  action = [2, 0, 1, 2, 0]  ─────┐
                                  │
Local Agents (MaskablePPO)        │
  actions = {                     ├─→ Py4J ─→ MultiDatacenterSimulationCore
    0: 12,                        │            ↓
    1: -1,                        │         executeHierarchicalStep()
    2: 5                   ──────┘         ├─ executeGlobalRouting()
  }                                        │  └─ GlobalBroker.routeCloudlets()
                                           │
                                           └─ executeLocalScheduling()
                                              ├─ DC0: assignCloudletToVm(12)
                                              ├─ DC1: NoAssign
                                              └─ DC2: assignCloudletToVm(5)
```

### 5.3 Reward数据流

```
Java                                    Python
────                                    ──────

MultiDatacenterSimulationCore
├─ calculateGlobalReward()
│  ├─ Load balance score
│  ├─ Green energy utilization   ──────┐
│  └─ Queue management                 │
│                                       ├─→ Py4J ─→ rewards dict
├─ calculateLocalRewards()              │          {
│  ├─ DC0:                              │            "global": 0.75,
│  │  ├─ Cloudlets completed     ──────┤            "local": {
│  │  ├─ Wait time penalty              │              0: 0.82,
│  │  └─ Resource utilization           │              1: 0.65,
│  ├─ DC1: ...                   ──────┤              2: 0.91
│  └─ DC2: ...                   ──────┘            }
│                                                  }
└─ Return to Python                               ↓
                                            Update Agent policies
```

---

## 6. 配置说明

### 6.1 config.yml结构

```yaml
experiment_multi_dc_3:
  # 实验类型
  type: "multi_dc"
  
  # 仿真参数
  simulation:
    timestep: 5.0                    # 每个RL step的仿真时间（秒）
    max_time: 3600.0                 # Episode最大时长（秒）
  
  # Global路由配置
  global_routing_batch_size: 5       # ✅ 固定批量大小
  
  # Workload配置
  workload:
    type: "csv"
    csv_file: "path/to/cloudlets.csv"
  
  # Datacenters配置
  datacenters:
    - id: 0
      name: "DC_USA_West"
      initial_s_vm_count: 15         # Small VMs
      initial_m_vm_count: 15         # Medium VMs
      initial_l_vm_count: 6          # Large VMs
      
      green_energy:
        enabled: true
        solar_capacity: 5000
        wind_capacity: 3000
    
    - id: 1
      name: "DC_Europe"
      initial_s_vm_count: 12
      initial_m_vm_count: 12
      initial_l_vm_count: 6
    
    - id: 2
      name: "DC_Asia"
      initial_s_vm_count: 8
      initial_m_vm_count: 8
      initial_l_vm_count: 4
  
  # 训练配置
  timesteps: 200000
  
  joint_training:
    strategy: "alternating"          # 交替训练
    
    alternating:
      num_cycles: 10                 # 训练周期数
      global_steps_per_cycle: 10000  # 每周期Global训练步数
      local_steps_per_cycle: 10000   # 每周期Local训练步数
    
    checkpoint_freq: 10000           # Checkpoint保存频率
    log_freq: 100                    # 日志记录频率
  
  # Agent超参数
  global_agent:
    policy: "MlpPolicy"
    learning_rate: 0.0003
    gamma: 0.99
    n_steps: 2048
    batch_size: 64
  
  local_agent:
    policy: "MlpPolicy"
    learning_rate: 0.0003
    gamma: 0.99
    n_steps: 2048
    batch_size: 64
  
  # 随机种子
  seed: 42
```

### 6.2 命令行参数

```bash
python drl-manager/src/training/train_hierarchical_multidc_joint.py \
  --config config.yml \
  --experiment experiment_multi_dc_3 \
  --strategy alternating \
  --total-timesteps 200000 \
  --seed 42
```

---

## 7. 训练策略

### 7.1 交替训练（Alternating Training）

这是**推荐的训练策略**，让两层agent轮流学习：

```
Cycle 1:
  Train Global Agent (10000 steps) → Save checkpoint
  Train Local Agents (10000 steps) → Save checkpoint

Cycle 2:
  Train Global Agent (10000 steps) → Save checkpoint
  Train Local Agents (10000 steps) → Save checkpoint

...

Cycle 10:
  Train Global Agent (10000 steps) → Save checkpoint
  Train Local Agents (10000 steps) → Save checkpoint

Total: 200000 steps (10 cycles × 20000 steps/cycle)
```

**优势**：
- ✅ 稳定收敛
- ✅ 避免agent之间相互干扰
- ✅ 容易调试

**训练期间的行为**：
- **训练Global Agent时**: Local Agents使用**random masked actions**（只选择valid的VM）
- **训练Local Agents时**: Global Agent使用**固定策略**（可以是random或已学习的策略）

### 7.2 同时训练（Simultaneous Training）

两个agent同时学习（更复杂，容易不稳定）：

```python
manager._train_simultaneous()  # 不推荐初学者使用
```

---

## 8. 输出和日志

### 8.1 输出目录结构

```
logs/joint_training/20251110_183000/
├── config_used.yml              # 使用的配置副本
├── seed_used.txt                # 随机种子
├── monitor/                     # Monitor日志
│   ├── 0.monitor.csv
│   └── ...
├── checkpoints/                 # 定期checkpoint
│   ├── model_5000_steps.zip
│   ├── model_10000_steps.zip
│   └── ...
├── global_cycle_1.zip           # Global Agent checkpoint（周期1）
├── local_cycle_1.zip            # Local Agent checkpoint（周期1）
├── global_cycle_2.zip
├── local_cycle_2.zip
├── ...
├── final_global_model.zip       # 最终Global模型
├── final_local_model.zip        # 最终Local模型
└── training.log                 # 训练日志
```

### 8.2 Java日志

```
cloudsimplus-gateway/logs/cloudsimplus/2025-11-10_18-30/
├── cspg.log                     # 完整仿真日志
├── cspg.current.log             # 当前运行日志（软链接）
└── episode_*/                   # 每个episode的详细日志
    ├── cloudlets_submitted.csv
    ├── cloudlets_finished.csv
    ├── vm_utilization.csv
    └── energy_consumption.csv
```

---

## 9. 常见问题和调试

### 9.1 如何查看训练进度？

```bash
# TensorBoard
tensorboard --logdir logs/joint_training/

# 查看日志
tail -f logs/joint_training/20251110_183000/training.log

# Java仿真日志
tail -f cloudsimplus-gateway/logs/cloudsimplus/cspg.current.log
```

### 9.2 如何验证batch routing是否工作？

检查日志中的这些行：

```
[INFO] Routing 5 cloudlets (batch_size=5, available=12)
[INFO] Global waiting queue: 7 cloudlets remaining
```

### 9.3 如何确认Local Actions是否正确？

检查日志：

```python
local_actions = {0: 12, 1: -1, 2: 5}
# ✅ 正确: 字典，每个DC一个action

local_actions = [12, -1, 5, 8, 2, ...]
# ❌ 错误: 这不是Local actions的格式
```

### 9.4 Global Actions长度不对？

确保使用固定batch_size：

```python
# ✅ 正确
batch_size = env.global_routing_batch_size  # 5
global_actions = [agent_choice] * batch_size  # [2, 2, 2, 2, 2]

# ❌ 错误
num_arriving = java_env.getArrivingCloudletsCount()  # 动态变化
global_actions = [agent_choice] * num_arriving  # 长度不固定
```

---

## 10. 下一步

1. **运行训练**:
   ```bash
   cd drl-manager
   python src/training/train_hierarchical_multidc_joint.py \
     --config ../config.yml \
     --experiment experiment_multi_dc_3 \
     --strategy alternating
   ```

2. **监控训练**:
   ```bash
   tensorboard --logdir ../logs/joint_training/
   ```

3. **分析结果**:
   ```bash
   python scripts/analyze_training.py \
     --log-dir ../logs/joint_training/latest/
   ```

4. **评估模型**:
   ```bash
   python scripts/evaluate_model.py \
     --global-model final_global_model.zip \
     --local-model final_local_model.zip \
     --config ../config.yml
   ```

---

## 附录: 代码执行追踪示例

### 从Python调用到Java的完整追踪

```
[Python] train_hierarchical_multidc_joint.py
  ↓
[Python] JointTrainingManager.train_alternating()
  ↓ global_model.learn()
  ↓
[Python] GlobalAgentEnv.step(action=2)
  ↓ global_actions = [2, 2, 2, 2, 2]
  ↓ local_actions = {0: 12, 1: -1, 2: 5}
  ↓
[Python] JointTrainingEnv.step({"global": [...], "local": {...}})
  ↓
[Python] HierarchicalMultiDCEnv.step(actions)
  ↓ self.java_env.step(global_actions, local_actions_java)
  ↓
[Py4J Bridge]
  ↓
[Java] HierarchicalMultiDCGateway.step(globalActions, localActionsMap)
  ↓
[Java] MultiDatacenterSimulationCore.executeHierarchicalStep()
  ↓
  ├─→ executeGlobalRouting(globalActions)
  │   ├─ globalBroker.processArrivingCloudlets()
  │   ├─ globalBroker.getBatchForRouting(5)
  │   └─ globalBroker.routeCloudlets(batch, globalActions)
  │       └─ datacenters[i].getBroker().receiveCloudlet(cloudlet)
  │
  ├─→ executeLocalScheduling(localActionsMap)
  │   ├─ datacenters[0].executeLocalAction(12)
  │   │   └─ broker.assignCloudletToVm(cloudlet, vm12)
  │   ├─ datacenters[1].executeLocalAction(-1)  // NoAssign
  │   └─ datacenters[2].executeLocalAction(5)
  │       └─ broker.assignCloudletToVm(cloudlet, vm5)
  │
  ├─→ advanceSimulationTime(timestep=5.0)
  │   └─ simulation.runFor(5.0)
  │       └─ CloudSim仿真执行5秒
  │
  ├─→ getGlobalObservation() → observation["global"]
  ├─→ getLocalObservations() → observation["local"]
  ├─→ calculateGlobalReward() → rewards["global"]
  ├─→ calculateLocalRewards() → rewards["local"]
  └─→ checkTermination() → terminated, info
  ↓
[Py4J Bridge] 返回 (observation, rewards, terminated, info)
  ↓
[Python] HierarchicalMultiDCEnv.step() 返回
  ↓
[Python] JointTrainingEnv.step() 返回
  ↓
[Python] GlobalAgentEnv.step() 返回 global_obs, global_reward
  ↓
[Python] PPO.learn() 更新策略
```

---

**文档版本**: 1.0  
**最后更新**: 2025-11-10  
**适用版本**: Multi-DC Hierarchical RL v2.0 (Fixed Batch Routing)

