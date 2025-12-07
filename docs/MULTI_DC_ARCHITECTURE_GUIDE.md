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

## 10. 本系统的形式化定义（状态 / 动作 / 回报 / 绿色能源特征）

> 本节给出 Multi-DC 分层 RL 中 **Global / Local Agents 的状态、动作和奖励** 以及 **绿色能源未来趋势特征** 的数学形式定义，方便论文/技术文档引用。

### 10.1 记号约定

- \(D\)：数据中心数量（`num_datacenters`）。  
- 第 \(i\) 个 DC 记为 \(\text{DC}_i, \; i \in \{1,\dots,D\}\)。  
- 每步时间为 \(t\)；全局环境观测记为 \(s_t\)，local agent 在 DC \(i\) 的观测记为 \(s_i(t)\)。  
- Global agent 每步处理固定 batch 大小 \(B\)（`global_routing_batch_size`），即每步最多路由 \(B\) 个 cloudlet。  
- 绿色能源时间序列功率为 \(P[k]\)（单位 kW），最大功率为 \(P_{\max}\)。  

---

### 10.2 Local Agent（每个 DC 内部调度）

#### 10.2.1 Local 状态 \(s_i(t)\)

对第 \(i\) 个 DC，设其 host 数量为 \(H_i\)，VM 数量为 \(V_i\)。在时间步 \(t\) 的本地观测为：

\[
s_i(t) \;=\;
\bigl(
\mathbf{h}^{\text{load}}_i(t),
\mathbf{h}^{\text{ram}}_i(t),
\mathbf{v}^{\text{load}}_i(t),
\mathbf{v}^{\text{type}}_i(t),
\mathbf{v}^{\text{avail}}_i(t),
w_i(t),
p_i^{\text{next}}(t)
\bigr)
\]

- Host 相关（维度 \(H_i\)）：
\[
\mathbf{h}^{\text{load}}_i(t) \in [0,1]^{H_i},\quad
h^{\text{load}}_{i,j}(t) = \text{Host}_j \text{ 的 CPU 利用率}
\]
\[
\mathbf{h}^{\text{ram}}_i(t) \in [0,1]^{H_i},\quad
h^{\text{ram}}_{i,j}(t) = \text{Host}_j \text{ 的 RAM 利用率}
\]

- VM 相关（维度 \(V_i\)）：
\[
\mathbf{v}^{\text{load}}_i(t) \in [0,1]^{V_i},\quad
v^{\text{load}}_{i,k}(t) = \text{VM}_k \text{ 的 CPU 利用率}
\]
\[
\mathbf{v}^{\text{type}}_i(t) \in \{0,1,2,3\}^{V_i},\quad
v^{\text{type}}_{i,k}(t) =
\begin{cases}
0 & \text{Off}\\
1 & \text{Small}\\
2 & \text{Medium}\\
3 & \text{Large}
\end{cases}
\]
\[
\mathbf{v}^{\text{avail}}_i(t) \in \mathbb{N}_0^{V_i},\quad
v^{\text{avail}}_{i,k}(t) = \text{VM}_k \text{ 当前可用 PEs 数}
\]

- 队列信息：
\[
w_i(t) \in \mathbb{N}_0 \quad (\text{本 DC 等待队列中的 cloudlet 数})
\]
\[
p_i^{\text{next}}(t) \in \mathbb{N}_0 \quad (\text{队首 cloudlet 所需 PEs，若队列空则为 }0)
\]

在 PettingZoo / RLlib 中，local agent 实际观测为：

\[
\text{obs}^{\text{local}}_i(t)
=\bigl\{
\text{"observation"}: s_i(t),\;
\text{"action\_mask"}: m_i(t)
\bigr\}
\]

其中动作掩码 \(m_i(t) \in \{0,1\}^{V_i+1}\)，由 Python 侧根据队列和 VM 资源计算（1=允许，0=屏蔽）。

#### 10.2.2 Local 动作 \(a_i(t)\)

每个 local agent 的动作空间为：

\[
\mathcal{A}_i = \{0,1,\dots,V_i\}
\]

动作语义：

\[
a_i(t) =
\begin{cases}
0, & \text{NoAssign：本步不从队列取任务}\\[2pt]
k\in\{1,\dots,V_i\}, & \text{将一个 cloudlet 分配给 VM }(k-1)
\end{cases}
\]

Python → Java 映射为：

\[
\text{targetVmId} = a_i(t) - 1
\]

- \(a_i(t)=0 \Rightarrow \text{targetVmId}=-1\)：显式 NoAssign；  
- \(a_i(t)=k>0 \Rightarrow \text{targetVmId}=k-1\)：选择本 DC 第 \(k-1\) 个 VM。

#### 10.2.3 Local 奖励 \(r_i(t)\)

单个 DC 的 local reward 由以下几部分组成：

\[
r_i(t) \;=\;
R^{\text{wait}}_i(t)
 + R^{\text{util}}_i(t)
 + R^{\text{queue}}_i(t)
 + R^{\text{invalid}}_i(t)
 + R^{\text{compl}}_i(t)
\]

其中目前实现中 \(R^{\text{compl}}_i(t)=0\)（completion 奖励暂时禁用）。

**(1) 等待时间惩罚 \(R^{\text{wait}}_i(t)\)**  
设本步在 \(\text{DC}_i\) 完成的 cloudlet 等待时间集合为 \(\mathcal{W}_i(t)\)（秒）：

\[
\bar{w}_i(t) =
\begin{cases}
\dfrac{1}{|\mathcal{W}_i(t)|} \sum\limits_{w\in \mathcal{W}_i(t)} w, & |\mathcal{W}_i(t)|>0\\[4pt]
0, & \text{否则}
\end{cases}
\]

系数 \(\alpha_{\text{wait}} = \texttt{reward\_wait\_time\_coef}\)：

\[
R^{\text{wait}}_i(t) =
\begin{cases}
-\alpha_{\text{wait}} \cdot \log\!\bigl(1 + \bar{w}_i(t)\bigr), & |\mathcal{W}_i(t)|>0\\[4pt]
0, & \text{否则}
\end{cases}
\]

**(2) 利用率与负载均衡惩罚 \(R^{\text{util}}_i(t)\)**  
对 \(\text{DC}_i\) 的活跃 VM 集合 \(\mathcal{V}_i\)，记每个 VM 的 CPU 利用率为 \(u_v(t)\in[0,1]\)：

\[
\mu_i(t) = \frac{1}{|\mathcal{V}_i|} \sum_{v\in\mathcal{V}_i} u_v(t)
\]
\[
\sigma_i^2(t) = \frac{1}{|\mathcal{V}_i|} \sum_{v\in\mathcal{V}_i} \bigl(u_v(t)-\mu_i(t)\bigr)^2
\]

目标利用率 \(u^* = 0.75\)，系数 \(\alpha_{\text{util}} = \texttt{reward\_unutilization\_coef}\)：

\[
R^{\text{util}}_i(t)
= -\alpha_{\text{util}} \left(
\sqrt{\sigma_i^2(t)} + \bigl|\mu_i(t) - u^*\bigr|
\right)
\]

**(3) 队列长度惩罚 \(R^{\text{queue}}_i(t)\)**  
设本 DC 当前等待队列长度为 \(q_i(t)\)，自 episode 开始以来接收的 cloudlet 数为 \(N^{\text{recv}}_i(t)\)，系数 \(\alpha_{\text{queue}} = \texttt{reward\_queue\_penalty\_coef}\)：

\[
R^{\text{queue}}_i(t) =
\begin{cases}
-\alpha_{\text{queue}} \cdot \dfrac{q_i(t)}{N^{\text{recv}}_i(t)}, & N^{\text{recv}}_i(t) > 0\\[6pt]
0, & \text{否则}
\end{cases}
\]

**(4) 非法动作惩罚 \(R^{\text{invalid}}_i(t)\)**  
设指示变量：

\[
\mathbb{I}^{\text{inv}}_i(t) =
\begin{cases}
1, & \text{本步 local 动作被判定为 invalid}\\
0, & \text{否则}
\end{cases}
\]

例如：队列非空却选 NoAssign、VM index 越界等。系数 \(\alpha_{\text{inv}} = \texttt{reward\_invalid\_action\_coef}\)：

\[
R^{\text{invalid}}_i(t)
= -\alpha_{\text{inv}} \cdot \mathbb{I}^{\text{inv}}_i(t)
\]

**(5) 完成奖励 \(R^{\text{compl}}_i(t)\)**  
设计上：若 \(C_i(t)\) 为本步完成的 cloudlet 数，系数 \(\alpha_{\text{compl}} = \texttt{reward\_completion\_coef}\)，则：

\[
R^{\text{compl}}_i(t) = \alpha_{\text{compl}} \cdot C_i(t)
\]

当前实现中该项被禁用：\(R^{\text{compl}}_i(t) = 0\)。  

---

### 10.3 Global Agent（跨 DC 路由）

#### 10.3.1 Global 状态 \(s^{(g)}_t\)

记全局观测为：

\[
s^{(g)}_t
=
\bigl(
\mathbf{g}_1(t), \dots, \mathbf{g}_D(t),
 C_{\text{up}}(t),
 \mathbf{p}^{\text{batch}}(t),
 \mathbf{m}^{\text{batch}}(t),
 \mathbf{d}^{\text{pes}}(t),
 L_{\text{imb}}(t),
 C_{\text{recent}}(t)
\bigr)
\]

对第 \(i\) 个 DC，其聚合状态为：

\[
\mathbf{g}_i(t)
=
\bigl(
P^{\text{green}}_i(t),
P^{\text{tot}}_i(t),
\rho^{\text{green}}_i(t),
W^{\text{green}}_i(t),
\mu^{\text{short}}_i(t),
\tau^{\text{short}}_i(t),
\mu^{\text{long}}_i(t),
\phi^{\text{peak}}_i(t),
Q_i(t),
U^{\text{cpu}}_i(t),
A^{\text{pes}}_i(t),
U^{\text{ram}}_i(t)
\bigr)
\]

含义：

- \(P^{\text{green}}_i(t)\)：当前绿色电力（W）  
- \(P^{\text{tot}}_i(t)\)：当前总功率（W）  
- \(\rho^{\text{green}}_i(t)\in[0,1]\)：绿色能耗占比  
- \(W^{\text{green}}_i(t)\)：累计浪费绿色电量（Wh）  
- \(\mu^{\text{short}}_i,\tau^{\text{short}}_i,\mu^{\text{long}}_i,\phi^{\text{peak}}_i\)：未来绿色功率趋势特征（见 10.4）  
- \(Q_i(t)\)：该 DC 全局队列中的等待 cloudlet 数  
- \(U^{\text{cpu}}_i(t)\in[0,1]\)：平均 host CPU 利用率  
- \(A^{\text{pes}}_i(t)\in\mathbb{N}_0\)：该 DC 所有 VM 当前可用 PEs 总数  
- \(U^{\text{ram}}_i(t)\in[0,1]\)：平均 host RAM 利用率  

全局队列与 batch 信息：

- \(C_{\text{up}}(t)\in\mathbb{N}_0\)：全局等待队列中的 cloudlet 总数；  
- Batch 大小 \(B\)：
\[
\mathbf{p}^{\text{batch}}(t) = (p^{\text{batch}}_1(t),\dots,p^{\text{batch}}_B(t)),
\quad p^{\text{batch}}_j(t)\in\mathbb{N}_0
\]
\[
\mathbf{m}^{\text{batch}}(t) = (m^{\text{batch}}_1(t),\dots,m^{\text{batch}}_B(t)),
\quad m^{\text{batch}}_j(t)\in\mathbb{N}_0
\]
分别为 batch 中每个位置的 PEs 和 MI（没有 cloudlet 时为 0）。  

- PES 分布（小/中/大任务数）：
\[
\mathbf{d}^{\text{pes}}(t) =
\bigl(d^{\text{small}}(t), d^{\text{med}}(t), d^{\text{large}}(t)\bigr) \in \mathbb{N}_0^3
\]

- 负载不均衡度（DC CPU 利用率的标准差）：
\[
L_{\text{imb}}(t) = \sqrt{\frac{1}{D}\sum_{i=1}^D \bigl(U^{\text{cpu}}_i(t)-\bar{U}(t)\bigr)^2}
\]
其中 \(\bar{U}(t)\) 为所有 DC 平均 CPU 利用率。  

- 已完成 cloudlet 计数（近似）：
\[
C_{\text{recent}}(t)
= \sum_{i=1}^{D} |\mathcal{C}^{\text{finished}}_i(t)|
\]

#### 10.3.2 Global 动作 \(a^{(g)}_t\)

Global agent 的动作空间为 MultiDiscrete：

\[
a^{(g)}_t = \bigl(a^{(g)}_{t,1}, \dots, a^{(g)}_{t,B}\bigr),
\qquad
a^{(g)}_{t,j} \in \{0,1,\dots,D\}
\]

每个位置 \(j\) 的含义：

\[
a^{(g)}_{t,j} =
\begin{cases}
0, & \text{NoAssign：该位置不路由 cloudlet}\\[2pt]
k\in\{1,\dots,D\}, & \text{将该位置的 cloudlet 路由到 DC }(k-1)
\end{cases}
\]

Python 侧首先过滤掉 NoAssign，得到：

\[
\tilde{L}_t = \bigl\{\,k-1 \;\big|\; a^{(g)}_{t,j}=k>0,\ 1\le j\le B\,\bigr\}
\]

记当前全局等待队列长度为 \(C_{\text{up}}(t)\)，则真正传给 Java 的路由列表为：

\[
L_t = \tilde{L}_t[0:\,C_{\text{up}}(t)]
\]

即：

- 若非 0 动作数大于队列长度，多余部分被截断丢弃；  
- 若队列为空（\(C_{\text{up}}(t)=0\)），则 \(L_t\) 为空列表（等价于本步不路由任何 cloudlet）。  

#### 10.3.3 Global 奖励 \(r^{(g)}_t\)

在每个时间步 \(t\)：

1. 对每个 DC \(i\) 计算 local 奖励 \(r^{\text{loc}}_i(t)\)（见 10.2.3）。  
2. 汇总 local 奖励：
\[
R_{\text{local-sum}}(t) = \sum_{i=1}^{D} r^{\text{loc}}_i(t)
\]
3. 计算碳排放惩罚：  
   - 能耗增量（Wh）：\(\Delta E^{\text{green}}_i(t),\;\Delta E^{\text{brown}}_i(t)\)；  
   - 使用 DC 配置中的碳因子 \(f^{\text{green}}_i,f^{\text{brown}}_i\)：
\[
\Delta C_i(t) =
\frac{\Delta E^{\text{green}}_i(t)}{1000}\cdot f^{\text{green}}_i
 +
\frac{\Delta E^{\text{brown}}_i(t)}{1000}\cdot f^{\text{brown}}_i
\]
\[
\Delta C_{\text{tot}}(t) = \sum_{i=1}^{D} \Delta C_i(t)
\]
   - 碳惩罚（系数 \(\alpha_{\text{carbon}} = \texttt{carbon\_emission\_penalty\_coef}\)）：
\[
P_{\text{carbon}}(t) = \alpha_{\text{carbon}} \cdot \Delta C_{\text{tot}}(t)
\]

4. Global agent 的奖励：

\[
r^{(g)}_t
=
R_{\text{local-sum}}(t)
 - P_{\text{carbon}}(t)
=
\sum_{i=1}^{D} r^{\text{loc}}_i(t)
 - \alpha_{\text{carbon}} \sum_{i=1}^{D} \Delta C_i(t)
\]

---

### 10.4 绿色能源未来趋势特征（God’s Eye）

对某个 `GreenEnergyProvider`，其历史/未来功率时间序列为 \(\{P[k]\}_{k=0}^{N-1}\)（单位 kW），最大功率：

\[
P_{\max} = \max_k P[k]
\]

仿真时间 \(t\) 通过 `simTimeToRowIndex()` 映射到当前行索引 \(\text{currentIdx}\)，并考虑时区偏移 `time_zone_offset_rows` 以及压缩/真实时间模式（COMPRESSED/REAL_TIME）。

#### 10.4.1 短期窗口特征（short-term）

窗口长度为 `short_term_rows`，索引区间：

\[
i \in [\text{currentIdx},\, \text{shortEndIdx}-1],\quad
\text{shortEndIdx} = \min(\text{currentIdx}+\text{shortTermRows}, N)
\]
\[
\text{shortAvailable} = \text{shortEndIdx} - \text{currentIdx}
\]

**短期均值 \(\mu_i^{\text{short}}(t)\)**：

\[
\text{shortMean} =
\frac{1}{\text{shortAvailable}}
\sum_{k=\text{currentIdx}}^{\text{shortEndIdx}-1} P[k]
\]
\[
\mu_i^{\text{short}}(t)
= \min\!\left(1,\; \max\!\left(0,\; \frac{\text{shortMean}}{P_{\max}}\right)\right)
\]

**短期趋势 \(\tau_i^{\text{short}}(t)\)**：

\[
P_{\text{start}} = P[\text{currentIdx}],\quad
P_{\text{end}} = P[\text{shortEndIdx}-1]
\]
\[
\text{shortTrend} = \frac{P_{\text{end}} - P_{\text{start}}}{P_{\max}}
\]
\[
\tau_i^{\text{short}}(t)
= \min\!\left(1,\; \max\!\left(-1,\; \text{shortTrend}\right)\right)
\]

#### 10.4.2 长期窗口特征（long-term）

窗口长度为 `long_term_rows`，索引区间：

\[
i \in [\text{currentIdx},\, \text{longEndIdx}-1],\quad
\text{longEndIdx} = \min(\text{currentIdx}+\text{longTermRows}, N)
\]
\[
\text{longAvailable} = \text{longEndIdx} - \text{currentIdx}
\]

**长期均值 \(\mu_i^{\text{long}}(t)\)**：

\[
\text{longMean} =
\frac{1}{\text{longAvailable}}
\sum_{k=\text{currentIdx}}^{\text{longEndIdx}-1} P[k]
\]
\[
\mu_i^{\text{long}}(t)
= \min\!\left(1,\; \max\!\left(0,\; \frac{\text{longMean}}{P_{\max}}\right)\right)
\]

**峰值时间 \(\phi_i^{\text{peak}}(t)\)**：

\[
k^* = \arg\max_{k \in [\text{currentIdx},\,\text{longEndIdx}-1]} P[k]
\]
\[
\phi_i^{\text{peak}}(t)
= \min\!\left(1,\; \max\!\left(0,\;
 \frac{k^* - \text{currentIdx}}{\text{longAvailable}}
\right)\right)
\]

#### 10.4.3 多风机聚合到 DC 级别

若某个 DC 有多台风机（多个 `GreenEnergyProvider`），每个 provider \(p\) 具有特征：

\[
(\mu^{\text{short}}_p,\; \tau^{\text{short}}_p,\;
 \mu^{\text{long}}_p,\; \phi^{\text{peak}}_p)
\]

和最大功率 \(P_{\max,p}\)。DC 级别的聚合特征（`computeAggregatedFutureTrendFeatures`）：

- 加权短期均值：
\[
\mu^{\text{short}}_{\text{DC}}(t)
= \min\!\left(1,\;
  \frac{\sum_p P_{\max,p}\,\mu^{\text{short}}_p(t)}{\sum_p P_{\max,p}}
\right)
\]

- 加权短期趋势：
\[
\tau^{\text{short}}_{\text{DC}}(t)
= \text{clip}_{[-1,1]}\!\left(
  \frac{\sum_p P_{\max,p}\,\tau^{\text{short}}_p(t)}{\sum_p P_{\max,p}}
\right)
\]

- 加权长期均值：
\[
\mu^{\text{long}}_{\text{DC}}(t)
= \min\!\left(1,\;
  \frac{\sum_p P_{\max,p}\,\mu^{\text{long}}_p(t)}{\sum_p P_{\max,p}}
\right)
\]

- 峰值时间取“最早”的一台：
\[
\phi^{\text{peak}}_{\text{DC}}(t)
= \min_p \phi^{\text{peak}}_p(t)
\]

若一个 DC 没有任何 green provider，则使用默认值：\(\mu^{\text{short}}=\mu^{\text{long}}=0.5,\ \tau^{\text{short}}=0,\ \phi^{\text{peak}}=0.5\)。

---

## 11. 下一步

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

