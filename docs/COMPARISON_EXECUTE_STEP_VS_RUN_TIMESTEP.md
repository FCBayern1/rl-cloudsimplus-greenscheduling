# executeHierarchicalStep vs runOneTimestep 对比分析

## 🎯 核心回答

**是的**，`MultiDatacenterSimulationCore.executeHierarchicalStep()` 和 `SimulationCore.runOneTimestep()` 的**核心作用是一致的**：**执行一个仿真时间步**。

但它们在**设计目标**和**接口层次**上有显著差异：

| 特性 | SimulationCore.runOneTimestep() | MultiDatacenterSimulationCore.executeHierarchicalStep() |
|------|--------------------------------|--------------------------------------------------------|
| **核心功能** | 推进仿真一个时间步 ✅ | 推进仿真一个时间步 ✅ |
| **设计层次** | **纯仿真接口** | **强化学习环境接口** |
| **决策方式** | 自动调度 | RL Agent 控制 |
| **返回值** | void | 完整 RL 结果 (obs, reward, done, info) |
| **控制架构** | 单层 | 分层（全局 + 局部） |

---

## 📊 详细对比

### 1️⃣ **SimulationCore.runOneTimestep()** - 纯仿真接口

#### **代码结构** (lines 266-286)

```java
public void runOneTimestep() {
    final double oldClock = getClock();
    final double startTime = Math.max(getClock() - settings.getSimulationTimestep(), 0);
    final double targetTime = calculateTargetTime();

    // 1️⃣ 确保仿真运行中
    ensureSimulationIsRunning();

    // 2️⃣ 清理和准备
    broker.clearFinishedWaitTimes();
    clearListsIfNeeded();

    // 3️⃣ 获取在此时间窗口到达的 cloudlets
    List<Cloudlet> cloudletsToQueueList = broker.getCloudletsToQueueAtThisTimestep(
            targetTime, startTime, this.getClock());

    // 4️⃣ 推进时钟
    proceedClockTo(targetTime);

    // 5️⃣ 将到达的 cloudlets 加入队列
    final double startTimeNew = getClock() - settings.getSimulationTimestep();
    broker.queueCloudletsAtThisTimestep(cloudletsToQueueList, startTimeNew, getClock());

    // 6️⃣ 打印统计信息
    if (shouldPrintStats()) {
        printStats();
    }

    // 7️⃣ 更新状态
    if (firstStep) {
        firstStep = false;
    }

    LOGGER.info("VMs running: {}", broker.getVmExecList().size());
    LOGGER.info("{}: Proceeding clock to {}", oldClock, targetTime);
}
```

#### **特点**

✅ **自动化调度**: Broker 自动处理 cloudlet 到达和排队
✅ **无外部控制**: 没有动作参数，完全内部驱动
✅ **无返回值**: void，不返回观测或奖励
✅ **单一职责**: 只负责推进仿真

**使用场景**:
- 传统的离散事件仿真
- 非 RL 场景（如纯性能评估）
- Broker 内部有调度策略

---

### 2️⃣ **MultiDatacenterSimulationCore.executeHierarchicalStep()** - RL 环境接口

#### **代码结构** (lines 243-281)

```java
public HierarchicalStepResult executeHierarchicalStep(
        List<Integer> globalActions,      // ← 全局 Agent 的动作
        Map<Integer, Integer> localActions) { // ← 局部 Agents 的动作

    currentStep++;
    LOGGER.debug("=== Step {} starting (Clock: {}s) ===", currentStep, currentClock);

    // === Phase 1: Global Routing (由全局 Agent 控制) ===
    int cloudletsRouted = executeGlobalRouting(globalActions);

    // === Phase 2: Local Scheduling (由局部 Agents 控制) ===
    Map<Integer, Boolean> localResults = executeLocalScheduling(localActions);

    // === Phase 3: Advance Simulation Time (推进时钟) ===
    advanceSimulationTime();

    // === Phase 4: Collect Observations (采集观测) ===
    ObservationState globalObs = getGlobalObservation();
    Map<Integer, ObservationState> localObs = getLocalObservations();

    // === Phase 5: Calculate Rewards (计算奖励) ===
    double globalReward = calculateGlobalReward();
    Map<Integer, Double> localRewards = calculateLocalRewards();

    // === Phase 6: Check Termination (检查终止) ===
    boolean allCloudletsCompleted = !hasUnfinishedCloudlets();
    boolean simulationEnded = !simulation.isRunning();
    boolean terminated = allCloudletsCompleted && simulationEnded;
    boolean truncated = currentStep >= settings.getMaxEpisodeLength();

    // === Phase 7: Build Info (构建信息字典) ===
    Map<String, Object> info = buildStepInfo(cloudletsRouted, localResults);
    info.put("all_cloudlets_completed", allCloudletsCompleted);
    info.put("simulation_ended", simulationEnded);

    LOGGER.debug("=== Step {} completed (Clock: {}s, Terminated: {}, Truncated: {}) ===",
                 currentStep, currentClock, terminated, truncated);

    // 返回完整的 RL step 结果
    return new HierarchicalStepResult(
            globalObs, localObs,          // 观测
            globalReward, localRewards,   // 奖励
            terminated, truncated,         // 终止标志
            info                          // 额外信息
    );
}
```

#### **特点**

✅ **RL 标准接口**: 接收动作，返回 (obs, reward, done, info)
✅ **分层决策**: 全局路由 + 局部调度，两级 Agent 控制
✅ **手动控制**: 调度完全由 RL Agent 决定
✅ **完整返回**: 返回训练所需的全部信息

**使用场景**:
- 强化学习训练
- 分层多智能体系统
- 需要学习最优策略

---

## 🔄 执行流程对比

### **SimulationCore.runOneTimestep()**

```
┌─────────────────────────────────────┐
│   Python/Java 调用                   │
│   gateway.runOneTimestep()          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  1. 确保仿真运行中                   │
│  2. 清理上一步的数据                 │
│  3. Broker 自动获取到达的 cloudlets  │
│  4. proceedClockTo(targetTime)      │
│  5. Broker 自动将 cloudlets 入队    │
│  6. 打印统计信息                     │
└─────────────────────────────────────┘
               │
               ▼
           返回 void
```

**关键**: Broker 自动处理所有调度决策

---

### **MultiDatacenterSimulationCore.executeHierarchicalStep()**

```
┌─────────────────────────────────────────┐
│   Python 强化学习环境                    │
│   result = gateway.step(global_actions,  │
│                        local_actions)    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Phase 1: 全局路由 (globalActions)      │
│    - GlobalBroker 根据动作路由 cloudlets │
│                                          │
│  Phase 2: 局部调度 (localActions)       │
│    - 每个 LocalBroker 根据动作分配 VM   │
│                                          │
│  Phase 3: 推进时钟                       │
│    - advanceSimulationTime()            │
│      → proceedClockTo(currentClock + dt)│
│                                          │
│  Phase 4: 采集观测                       │
│    - getGlobalObservation()             │
│    - getLocalObservations()             │
│                                          │
│  Phase 5: 计算奖励                       │
│    - calculateGlobalReward()            │
│    - calculateLocalRewards()            │
│                                          │
│  Phase 6: 检查终止                       │
│    - terminated, truncated              │
│                                          │
│  Phase 7: 构建信息字典                   │
└─────────────────────────────────────────┘
               │
               ▼
    返回 HierarchicalStepResult
    {
      globalObs, localObs,
      globalReward, localRewards,
      terminated, truncated, info
    }
```

**关键**: RL Agent 控制所有调度决策

---

## 🧩 核心差异总结

### **1. 接口层次**

```
SimulationCore.runOneTimestep()
  ↓
  纯仿真层 (Simulation Layer)
  - 只关心时间推进
  - 内部自动化调度


MultiDatacenterSimulationCore.executeHierarchicalStep()
  ↓
  强化学习环境层 (RL Environment Layer)
  - 实现 Gym-style 接口
  - 接收动作 → 返回 (obs, reward, done, info)
  ↓
  仿真层 (Simulation Layer)
  - 底层仍然是推进时间
```

### **2. 调度控制方式**

| 方法 | 调度方式 | 决策者 |
|------|---------|--------|
| runOneTimestep() | **自动调度** | Broker 内部策略 |
| executeHierarchicalStep() | **手动调度** | RL Agent |

**SimulationCore**:
```java
// Broker 自动调度
List<Cloudlet> cloudletsToQueueList =
    broker.getCloudletsToQueueAtThisTimestep(...);
broker.queueCloudletsAtThisTimestep(...);
```

**MultiDatacenterSimulationCore**:
```java
// Agent 控制调度
executeGlobalRouting(globalActions);    // ← 全局 Agent 决定
executeLocalScheduling(localActions);   // ← 局部 Agents 决定
```

### **3. 返回值**

**SimulationCore**:
```java
public void runOneTimestep() {
    // 无返回值
}
```

**MultiDatacenterSimulationCore**:
```java
public HierarchicalStepResult executeHierarchicalStep(...) {
    return new HierarchicalStepResult(
        globalObs, localObs,        // 观测状态
        globalReward, localRewards, // 奖励信号
        terminated, truncated,      // 终止标志
        info                        // 诊断信息
    );
}
```

---

## 🎓 类比理解

### **SimulationCore.runOneTimestep()**
```
就像一个自动驾驶的汽车：
- 按下"前进"按钮
- 汽车自己决定油门、刹车、转向
- 你只能看到车走了一段路
```

### **MultiDatacenterSimulationCore.executeHierarchicalStep()**
```
就像一个手动驾驶的汽车（带 RL Agent）：
- 你（RL Agent）决定油门踩多少、方向盘打多少
- 汽车执行你的操作
- 你看到仪表盘数据（观测）、油耗指标（奖励）
- 根据反馈学习如何更好地驾驶
```

---

## 📈 架构演进

### **阶段1: 传统仿真**
```
┌────────────────┐
│ SimulationCore │
│  runOneTimestep│
└────────────────┘
      ↓
  纯仿真驱动
  内部自动调度
```

### **阶段2: 强化学习改造**
```
┌─────────────────────────────┐
│ MultiDatacenterSimulationCore│
│  executeHierarchicalStep    │
└─────────────────────────────┘
      ↓
  RL 环境接口
  Agent 控制调度
  返回 (obs, reward, done, info)
```

---

## 🔗 相似方法映射

| 功能 | SimulationCore | MultiDatacenterSimulationCore |
|------|---------------|------------------------------|
| **执行一步** | `runOneTimestep()` | `executeHierarchicalStep()` |
| **推进时钟** | `proceedClockTo()` | `proceedClockTo()` (现已添加 ✅) |
| **Cloudlet 调度** | Broker 自动 | Agent 手动 |
| **获取观测** | 无 | `getGlobalObservation()`, `getLocalObservations()` |
| **计算奖励** | 无 | `calculateGlobalReward()`, `calculateLocalRewards()` |
| **检查终止** | `isRunning()` | `hasUnfinishedCloudlets()` + `isRunning()` |
| **重置环境** | `resetSimulation()` | `resetSimulation()` |

---

## 💡 最佳实践建议

### **何时使用 runOneTimestep()**
- 纯性能评估
- 基准测试
- 调试仿真逻辑
- 不需要 RL 的场景

### **何时使用 executeHierarchicalStep()**
- 强化学习训练
- 多智能体协作
- 需要学习最优策略
- 分层决策场景

---

## 🎯 结论

**核心一致**: 两者都执行"一个仿真时间步"

**层次不同**:
- `runOneTimestep()` = **仿真层接口**（纯时间推进）
- `executeHierarchicalStep()` = **RL 环境层接口**（时间推进 + RL 机制）

**关系**:
```
executeHierarchicalStep()
  = runOneTimestep() 的功能
  + RL Agent 动作输入
  + 观测采集
  + 奖励计算
  + 终止检查
  + 完整返回值
```

可以说 `executeHierarchicalStep()` 是 `runOneTimestep()` 的 **RL 增强版本**！🚀
