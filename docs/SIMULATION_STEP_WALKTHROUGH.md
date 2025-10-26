# 训练过程中 Simulation Step 详解

本文档详细讲解在强化学习训练过程中，每个 `step()` 调用时仿真系统执行的完整流程。

---

## 概述

在这个负载均衡项目中，**Agent 在每个 step 决定是否将等待队列中的下一个 cloudlet 出列并分配到某个 VM**。这是一个关键点：

- **不是所有 cloudlets 都会自动执行**
- **Agent 必须主动选择将 cloudlet 从等待队列取出并分配**
- **如果 Agent 选择 action 0（不分配），cloudlet 会继续留在队列中等待**

---

## 完整 Step 流程

### 阶段 0: 初始状态
```
┌─────────────────────────────────────┐
│ 等待队列 (Waiting Queue)              │
│ [Cloudlet 1, Cloudlet 2, Cloudlet 3]│
├─────────────────────────────────────┤
│ VM 0: 负载 30%, 可用 PEs: 4          │
│ VM 1: 负载 60%, 可用 PEs: 2          │
│ VM 2: 负载 10%, 可用 PEs: 8          │
└─────────────────────────────────────┘
```

---

### 阶段 1: Agent 接收观测 (Observation)

**位置**: `loadbalancing_env.py:136-166`

Agent 接收当前状态的观测信息（来自上一个 step 或 reset）:

```python
observation = {
    "vm_loads": [0.3, 0.6, 0.1],           # 每个 VM 的 CPU 负载 (0-1)
    "vm_available_pes": [4, 2, 8],         # 每个 VM 的可用处理元件
    "waiting_cloudlets": [3.0],            # 等待队列中的 cloudlet 数量
    "next_cloudlet_pes": [2.0]             # 队列头部 cloudlet 需要的 PEs
}
```

**关键字段含义**:
- `waiting_cloudlets`: 当前有 **3 个** cloudlets 在等待队列中
- `next_cloudlet_pes`: 队列**最前面**的 cloudlet 需要 **2 个 PEs**
- `vm_available_pes`: 告诉 Agent 哪些 VM 有足够资源接受这个任务

---

### 阶段 2: Action Masking (可选，使用 MaskablePPO 时)

**位置**: `loadbalancing_env.py:329-388`

系统生成动作掩码，限制 Agent 只能选择有效动作：

```python
def action_masks(self) -> list[bool]:
    # 检查是否有 cloudlets 在等待
    waiting_cloudlets = 3
    next_cloudlet_pes = 2

    if waiting_cloudlets == 0:
        # 队列为空，只允许 action 0 (不分配)
        return [True, False, False, False]  # 只有索引 0 为 True

    # 有 cloudlets 等待，检查每个 VM 是否有足够 PEs
    mask = [False] * 4  # [action_0, action_1, action_2, action_3]

    # Action 0 (不分配) - 禁止，因为有任务等待
    mask[0] = False

    # Action 1 (分配到 VM 0) - VM 0 有 4 PEs >= 2 PEs
    mask[1] = True

    # Action 2 (分配到 VM 1) - VM 1 有 2 PEs >= 2 PEs
    mask[2] = True

    # Action 3 (分配到 VM 2) - VM 2 有 8 PEs >= 2 PEs
    mask[3] = True

    return [False, True, True, True]
```

**掩码规则**:
1. **队列为空** → 只允许 action 0（不分配）
2. **队列有任务 + 有 VM 满足要求** → 只允许分配到满足要求的 VM
3. **队列有任务 + 没有 VM 满足要求** → 允许所有 VM（让 Java 端处理无效分配并惩罚）

---

### 阶段 3: Agent 选择 Action

**位置**: Agent 策略网络

假设 Agent 选择:
```python
action = 3  # 分配到 VM 2 (VM ID = 2)
```

**Action 映射**:
```
Agent 输出 → 映射后的 target_vm_id
─────────────────────────────────
action 0   → -1 (不分配，跳过)
action 1   →  0 (分配到 VM 0)
action 2   →  1 (分配到 VM 1)
action 3   →  2 (分配到 VM 2)  ← 当前选择
```

---

### 阶段 4: Python 端处理 Action

**位置**: `loadbalancing_env.py:195-207`

```python
def step(self, action: int):
    # 映射 action
    target_vm_id = int(action) - 1  # action=3 → target_vm_id=2

    # 调用 Java 端
    step_result_java = self.loadbalancer_gateway.step(target_vm_id)
```

此时控制权转移到 **Java 端**。

---

### 阶段 5: Java 端执行 Action

**位置**: `LoadBalancerGateway.java:241-275`

```java
public SimulationStepResult step(int targetVmId) {
    currentStep++;

    boolean assignSuccess = false;
    boolean wasInvalidAction = false;

    // === 子阶段 5.1: 执行 Action ===
    if (targetVmId == -1) {
        // Agent 选择不分配
        if (simulationCore.getBroker().hasWaitingCloudlets()) {
            // 但队列有任务等待 → 无效动作
            wasInvalidAction = true;
        }
    } else {
        // Agent 选择分配 (targetVmId = 2)
        if (simulationCore.getBroker().hasWaitingCloudlets()) {
            // 队列有任务，尝试分配
            assignSuccess = simulationCore.getBroker()
                .assignCloudletToVm(targetVmId);

            if (!assignSuccess) {
                // 分配失败（VM 不存在或资源不足）
                wasInvalidAction = true;
            }
        } else {
            // 队列为空但 Agent 想分配 → 无效动作
            wasInvalidAction = true;
        }
    }

    // ... 继续到阶段 6
}
```

---

### 阶段 6: Cloudlet 分配逻辑（关键！）

**位置**: `LoadBalancingBroker.java:159-224`

这是决定 cloudlet **是否出列并分配**的关键代码：

```java
public boolean assignCloudletToVm(int vmId) {
    // 6.1: 检查队列是否为空
    if (cloudletWaitingQueue.isEmpty()) {
        return false;  // 队列空，分配失败
    }

    // 6.2: 检查目标 VM 是否存在且可用
    Vm vm = getVmFromCreatedList(vmId);  // 获取 VM 2
    if (vm == Vm.NULL || !vm.isCreated() || vm.isFailed()) {
        return false;  // VM 不存在或不可用，分配失败
    }

    // 6.3: 从队列头部取出 cloudlet（关键步骤！）
    Cloudlet cloudlet = cloudletWaitingQueue.poll();  // 出列！

    // 6.4: 检查 VM 是否有足够资源
    if (!vm.isSuitableForCloudlet(cloudlet)) {
        // VM 资源不足，cloudlet 重新入队
        cloudletWaitingQueue.offer(cloudlet);  // 重新加入队列
        return false;  // 分配失败
    }

    // 6.5: 设置 cloudlet 属性
    cloudlet.setVm(vm);              // 绑定到 VM 2
    cloudlet.setBroker(this);        // 设置 broker
    cloudlet.setSubmissionDelay(0);  // 立即提交

    // 6.6: 提交到 CloudSim Plus 事件队列
    getCloudletSubmittedList().add(cloudlet);
    send(getDatacenter(vm), 0, CloudSimTag.CLOUDLET_SUBMIT, cloudlet);

    return true;  // 分配成功
}
```

**状态变化**:
```
执行前:
等待队列: [Cloudlet 1, Cloudlet 2, Cloudlet 3]

执行 cloudletWaitingQueue.poll():
等待队列: [Cloudlet 2, Cloudlet 3]  ← Cloudlet 1 已出列
Cloudlet 1: 被绑定到 VM 2，提交到 Datacenter

CloudSim 事件队列:
[CLOUDLET_SUBMIT(Cloudlet 1, VM 2, delay=0)]
```

---

### 阶段 7: CloudSim Plus 推进仿真时间

**位置**: `SimulationCore.java:225-251` → `LoadBalancerGateway.java:278`

```java
// 推进 1 秒仿真时间
simulationCore.runOneTimestep();  // 默认推进 1.0 秒
```

**CloudSim Plus 内部发生的事情**:

```java
public void runOneTimestep() {
    double targetTime = currentClock + timestepSize;  // 例如: 100.0 + 1.0 = 101.0

    // 处理所有时间戳 <= 101.0 的事件
    while (hasMoreEvents() && nextEventTime() <= targetTime) {
        CloudSimEvent event = eventQueue.poll();

        if (event.getTag() == CLOUDLET_SUBMIT) {
            Cloudlet cloudlet = (Cloudlet) event.getData();
            Vm vm = cloudlet.getVm();

            // 调用 VM 的调度器来执行 cloudlet
            vm.getCloudletScheduler().cloudletSubmit(cloudlet);
        }

        // 处理其他事件...
    }

    // 推进时钟到目标时间
    currentClock = targetTime;
}
```

**OptimizedCloudletScheduler 执行**:

**位置**: `OptimizedCloudletScheduler.java:31-43`

```java
@Override
public double updateProcessing(double currentTime, List<Double> mipsShare) {
    // 每次时钟推进时被调用
    // 遍历所有正在执行的 cloudlets
    for (CloudletExecution execution : getCloudletExecList()) {
        Cloudlet cloudlet = execution.getCloudlet();

        // 计算自上次更新以来的执行进度
        double elapsedTime = currentTime - execution.getLastProcessedTime();
        double mips = mipsShare.get(0);  // VM 的 MIPS
        double processedMI = mips * elapsedTime;  // 处理的百万指令数

        // 更新 cloudlet 完成进度
        cloudlet.addFinishedLengthSoFar(processedMI);

        // 检查是否完成
        if (cloudlet.isFinished()) {
            // 发送 CLOUDLET_RETURN 事件
            send(broker, 0, CLOUDLET_RETURN, cloudlet);
        }

        execution.setLastProcessedTime(currentTime);
    }

    return nextFinishTime;  // 返回下一个 cloudlet 完成的时间
}
```

**时间推进示例**:

假设:
- Cloudlet 1 长度: 10000 MI (百万指令)
- VM 2 MIPS: 1000 MIPS
- 完成时间: 10000 / 1000 = 10 秒

```
Step 1 (仿真时间 100s): Cloudlet 1 提交到 VM 2
                       执行进度: 0 MI

Step 2 (仿真时间 101s): updateProcessing() 被调用
                       执行进度: 1000 MI (10%)

Step 3 (仿真时间 102s): 执行进度: 2000 MI (20%)

...

Step 11 (仿真时间 110s): 执行进度: 10000 MI (100%)
                        Cloudlet 1 完成！
                        发送 CLOUDLET_RETURN 事件
```

**重要**: 在这 10 个 steps 期间，Agent 仍然可以继续分配队列中的其他 cloudlets！

---

### 阶段 8: 计算奖励

**位置**: `LoadBalancerGateway.java:296`

```java
double totalReward = calculateReward(wasInvalidAction);
```

奖励由多个组件组成（在 `calculateReward()` 方法中计算）:

```java
private double calculateReward(boolean wasInvalidAction) {
    // 1. Wait Time 奖励（越小越好）
    rewardWaitTimeComponent = -waitTime * waitTimeCoef;

    // 2. Unutilization 惩罚（资源浪费）
    rewardUnutilizationComponent = -unutilization * unutilizationCoef;

    // 3. Queue Penalty（队列积压）
    int queueSize = broker.getWaitingCloudletCount();
    rewardQueuePenaltyComponent = -queueSize * queuePenaltyCoef;

    // 4. Invalid Action 惩罚
    rewardInvalidActionComponent = wasInvalidAction ?
        -invalidActionCoef : 0.0;

    // 5. Energy 奖励/惩罚
    rewardEnergyComponent = -currentPowerW * energyCoef;

    // 总奖励
    return rewardWaitTimeComponent
         + rewardUnutilizationComponent
         + rewardQueuePenaltyComponent
         + rewardInvalidActionComponent
         + rewardEnergyComponent;
}
```

**奖励示例**:
```
假设当前状态:
- 等待时间: 5s
- 资源利用率: 70% (unutilization = 30%)
- 队列大小: 2
- 当前功耗: 150W
- 无效动作: false

计算:
rewardWaitTimeComponent      = -5 * 0.1   = -0.5
rewardUnutilizationComponent = -30 * 0.01 = -0.3
rewardQueuePenaltyComponent  = -2 * 0.5   = -1.0
rewardInvalidActionComponent = 0.0
rewardEnergyComponent        = -150 * 0.001 = -0.15

totalReward = -0.5 - 0.3 - 1.0 - 0.0 - 0.15 = -1.95
```

---

### 阶段 9: 获取新的观测

**位置**: `LoadBalancerGateway.java:294`

```java
ObservationState newState = getCurrentState();
```

更新后的状态（Cloudlet 1 已经从队列移除）:

```python
observation = {
    "vm_loads": [0.3, 0.6, 0.25],          # VM 2 负载增加
    "vm_available_pes": [4, 2, 6],         # VM 2 可用 PEs 减少
    "waiting_cloudlets": [2.0],            # 队列减少到 2 个
    "next_cloudlet_pes": [4.0]             # 新的队列头部 cloudlet
}
```

---

### 阶段 10: 检查终止条件

**位置**: `LoadBalancerGateway.java:299-313`

```java
// 终止条件: CloudSim 仿真结束（所有 cloudlets 完成）
boolean terminated = !simulationCore.isRunning();

// 截断条件: 达到最大 episode 长度
boolean truncated = !terminated && (currentStep >= maxEpisodeLength);
```

**终止场景**:

1. **正常终止** (`terminated = true`):
   - 所有 cloudlets 都已完成执行
   - 等待队列为空
   - 没有 cloudlets 在 VM 上执行

2. **强制截断** (`truncated = true`):
   - Episode 达到最大步数（例如 10000 步）
   - 仍有 cloudlets 未完成

---

### 阶段 11: 返回结果到 Python

**位置**: `LoadBalancerGateway.java:341` → `loadbalancing_env.py:209-243`

Java 返回:
```java
return new SimulationStepResult(
    newState,       // 新观测
    totalReward,    // 总奖励
    terminated,     // 是否终止
    truncated,      // 是否截断
    stepInfo        // 详细信息
);
```

Python 接收:
```python
observation = newState
reward = totalReward
terminated = terminated
truncated = truncated
info = stepInfo

return (observation, reward, terminated, truncated, info)
```

Agent 使用这些信息更新策略网络，然后进行下一个 step。

---

## 关键决策点总结

### 1. Cloudlet 何时出列？

**只有当 Agent 选择一个有效的 VM ID 时，cloudlet 才会从等待队列出列。**

```
Action 0 (-1, 不分配) → Cloudlet 留在队列中
Action 1-N (分配到某个 VM) → Cloudlet 出列（如果分配成功）
```

### 2. Cloudlet 分配失败会怎样？

如果分配失败（例如 VM 资源不足），cloudlet 会**重新入队**：

```java
if (!vm.isSuitableForCloudlet(cloudlet)) {
    cloudletWaitingQueue.offer(cloudlet);  // 重新入队
    return false;
}
```

### 3. 为什么要有 Action Masking？

**避免 Agent 选择无效动作，提高学习效率**:

- 队列为空时，禁止分配动作
- VM 资源不足时，禁止分配到该 VM
- 减少无效动作惩罚，加速收敛

### 4. 一个 Step 推进多少时间？

**默认推进 1 秒仿真时间**（可在配置中修改）:

```yaml
# config.yml
timestep_size: 1.0  # 秒
```

**重要**: 这是**仿真时间**，不是真实时间！
- 一个 step 可能在实际中只需要 0.1 秒执行
- 但在仿真中推进了 1 秒虚拟时间

---

## 多个 Cloudlets 并发执行示例

```
时间线（仿真时间）:

t=100s: Agent 分配 Cloudlet 1 (10s) 到 VM 0
        等待队列: [C2, C3, C4]
        VM 0: [C1 执行中]

t=101s: Agent 分配 Cloudlet 2 (15s) 到 VM 1
        等待队列: [C3, C4]
        VM 0: [C1 执行中 - 进度 10%]
        VM 1: [C2 执行中]

t=102s: Agent 分配 Cloudlet 3 (8s) 到 VM 2
        等待队列: [C4]
        VM 0: [C1 执行中 - 进度 20%]
        VM 1: [C2 执行中 - 进度 6.7%]
        VM 2: [C3 执行中]

...

t=110s: Cloudlet 1 完成！ ← CLOUDLET_RETURN 事件
        等待队列: [C4]
        VM 0: [空闲]
        VM 1: [C2 执行中 - 进度 60%]
        VM 2: [C3 执行中 - 进度 100%] → Cloudlet 3 也完成

t=111s: Agent 分配 Cloudlet 4 到 VM 0
        等待队列: []
        VM 0: [C4 执行中]
        VM 1: [C2 执行中 - 进度 66.7%]
```

---

## 训练循环伪代码

```python
# 完整的训练循环
for episode in range(num_episodes):
    observation, info = env.reset()

    for step in range(max_steps):
        # 1. Agent 根据观测选择动作
        action, _ = model.predict(observation, action_masks=env.action_masks())

        # 2. 执行动作（包含上述所有 11 个阶段）
        observation, reward, terminated, truncated, info = env.step(action)

        # 3. 学习（更新策略）
        model.learn(observation, action, reward, next_observation)

        # 4. 检查是否结束
        if terminated or truncated:
            break

    # Episode 结束，开始下一个 episode
```

---

## 配置参数对 Step 的影响

**相关配置文件**: `config.yml`

```yaml
# 时间相关
timestep_size: 1.0              # 每个 step 推进的仿真时间（秒）
max_episode_length: 10000       # 最大 step 数

# 奖励权重
reward_wait_time_coef: 0.1      # 等待时间权重
reward_unutilization_coef: 0.01 # 资源利用率权重
reward_queue_penalty_coef: 0.5  # 队列惩罚权重
reward_invalid_action_coef: 2.0 # 无效动作惩罚
reward_energy_coef: 0.001       # 能耗权重

# 基础设施
initial_s_vm_count: 20          # 小型 VM 数量
initial_m_vm_count: 10          # 中型 VM 数量
initial_l_vm_count: 5           # 大型 VM 数量
# 总共 35 个 VM → action_space = Discrete(36)
```

---

## 常见问题 (FAQ)

### Q1: 为什么有时候队列一直不减少？

**A**: Agent 可能一直选择 action 0（不分配），或者选择的 VM 资源不足导致分配失败。检查:
- Action masking 是否正常工作
- 奖励函数是否正确惩罚队列积压
- Agent 是否学到了正确的分配策略

### Q2: Cloudlet 执行需要多久？

**A**: 取决于 cloudlet 的长度和 VM 的 MIPS:
```
完成时间 = cloudlet.length / vm.mips
```
例如: 10000 MI cloudlet 在 1000 MIPS VM 上需要 10 秒仿真时间 = 10 个 steps。

### Q3: 一个 VM 可以同时执行多个 cloudlets 吗？

**A**: 可以！只要 VM 有足够的 PEs（处理元件）。例如:
- VM 有 8 个 PEs
- Cloudlet A 需要 2 PEs
- Cloudlet B 需要 3 PEs
- 可以同时执行 A 和 B（使用 5/8 PEs）

### Q4: 为什么会有 "Invalid Action"？

**A**: 以下情况会触发无效动作惩罚:
1. 队列为空但 Agent 选择分配动作
2. 队列有任务但 Agent 选择不分配（action 0）
3. 选择的 VM 不存在或资源不足

使用 MaskablePPO 可以大幅减少无效动作。

### Q5: Episode 何时结束？

**A**: 两种情况:
1. **自然终止**: 所有 cloudlets 都完成执行，队列为空
2. **强制截断**: 达到 `max_episode_length` 步数

---

## 相关文件索引

| 组件 | 文件路径 | 关键方法 |
|------|---------|---------|
| **Python 环境** | `drl-manager/gym_cloudsimplus/envs/loadbalancing_env.py` | `step()` (L195), `action_masks()` (L329) |
| **Java 网关** | `cloudsimplus-gateway/.../LoadBalancerGateway.java` | `step()` (L241), `calculateReward()` |
| **Cloudlet 分配** | `cloudsimplus-gateway/.../LoadBalancingBroker.java` | `assignCloudletToVm()` (L159) |
| **Cloudlet 执行** | `cloudsimplus-gateway/.../OptimizedCloudletScheduler.java` | `updateProcessing()` (L31) |
| **仿真核心** | `cloudsimplus-gateway/.../SimulationCore.java` | `runOneTimestep()` (L266) |
| **配置文件** | `config.yml` | 所有奖励权重和环境参数 |

---

## 流程图总结

```
┌─────────────────────────────────────────────────────────────────┐
│                     Simulation Step 完整流程                      │
└─────────────────────────────────────────────────────────────────┘

Python 端:
  ┌──────────────────────┐
  │ 1. 获取 Observation   │
  │    - vm_loads        │
  │    - waiting_queue   │
  │    - next_cloudlet   │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │ 2. Action Masking    │
  │    (MaskablePPO)     │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────┐
  │ 3. Agent 选择 Action │
  │    action = 3        │
  │    (分配到 VM 2)     │
  └──────────┬───────────┘
             │ Py4J RPC
             ▼
Java 端:
  ┌──────────────────────┐
  │ 4. 映射 Action       │
  │    target_vm_id = 2  │
  └──────────┬───────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 5. 检查队列 & VM                      │
  │    - 队列非空? ✓                      │
  │    - VM 存在? ✓                       │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 6. Cloudlet 出列并分配 (关键!)        │
  │    cloudlet = queue.poll()           │
  │    cloudlet.setVm(vm2)               │
  │    send(CLOUDLET_SUBMIT)             │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 7. CloudSim 推进仿真时间              │
  │    runOneTimestep() → +1秒           │
  │    - 处理 CLOUDLET_SUBMIT 事件       │
  │    - 调用 updateProcessing()         │
  │    - 计算执行进度                    │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 8. 计算奖励                          │
  │    reward = wait + util + queue      │
  │           + invalid + energy         │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 9. 获取新观测                        │
  │    newState = getCurrentState()      │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 10. 检查终止                         │
  │     terminated / truncated           │
  └──────────┬───────────────────────────┘
             │ Py4J RPC
             ▼
Python 端:
  ┌──────────────────────────────────────┐
  │ 11. 返回结果                         │
  │     return (obs, reward, done, info) │
  └──────────┬───────────────────────────┘
             │
  ┌──────────▼───────────────────────────┐
  │ 12. Agent 学习                       │
  │     model.learn(...)                 │
  └──────────────────────────────────────┘
```

---

**文档版本**: v1.0
**最后更新**: 2025-10-26
**作者**: Claude Code Analysis
