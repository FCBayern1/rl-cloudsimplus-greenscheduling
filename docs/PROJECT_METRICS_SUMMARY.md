# 项目衡量指标全面总结

## 📊 指标体系概览

本项目的指标体系分为4个层次：
1. **RL训练指标** - 用于训练过程的reward计算和策略优化
2. **绿色调度核心指标** - 能耗和效率相关
3. **性能指标** - 任务完成和QoS相关
4. **CloudSim结果指标** - Episode结束后的详细统计

---

## 1️⃣ RL训练指标（Reward Components）

这些指标在每个step计算，用于指导agent学习。

### 📍 数据来源
- **计算位置**: `LoadBalancerGateway.java:calculateReward()`
- **记录位置**: `monitor.csv` (每个episode一行)
- **代码行数**: LoadBalancerGateway.java:420-483

### 📈 具体指标

#### 1.1 Total Reward
```java
// LoadBalancerGateway.java:484-488
double totalReward = rewardWaitTimeComponent +
                     rewardUnutilizationComponent +
                     rewardQueuePenaltyComponent +
                     rewardInvalidActionComponent +
                     rewardEnergyComponent;
```

**含义**: 所有reward组件的总和
**目标**: 最大化（RL的优化目标）
**典型值**: -900 到 -950（负值越小越好）

#### 1.2 Wait Time Penalty
```java
// LoadBalancerGateway.java:423-426
this.rewardWaitTimeComponent = -settings.getRewardWaitTimeCoef() *
                                getAverageWaitTime();
```

**公式**: `-reward_wait_time_coef × average_wait_time`
**配置**: `reward_wait_time_coef: 1.0`
**含义**: 惩罚任务等待时间
**目标**: 最大化（接近0最好）
**典型值**: -0.0 到 -5.0

**当前问题**: 在训练中始终为0，可能未正确记录wait time

#### 1.3 Unutilization Penalty
```java
// LoadBalancerGateway.java:428-431
double unutilizationRatio = 1.0 - getAverageUtilization();
this.rewardUnutilizationComponent = -settings.getRewardUnutilizationCoef() *
                                     unutilizationRatio;
```

**公式**: `-reward_unutilization_coef × (1 - average_utilization)`
**配置**: `reward_unutilization_coef: 1.0`
**含义**: 惩罚资源未充分利用
**目标**: 最大化（高利用率 → 接近0）
**典型值**: -1.04 到 -1.05（表示平均利用率约5%）

**问题**: 当前利用率很低（~5%），说明资源浪费严重

#### 1.4 Queue Penalty
```java
// LoadBalancerGateway.java:436-437
this.rewardQueuePenaltyComponent = -settings.getRewardQueuePenaltyCoef() *
                                    getWaitingCloudletsRatio();
```

**公式**: `-reward_queue_penalty_coef × (waiting_cloudlets / total_cloudlets)`
**配置**: `reward_queue_penalty_coef: 0.8`
**含义**: 惩罚任务积压在队列中
**目标**: 最大化（无积压 → 0）
**典型值**: -0.08 到 -0.14

**观察**: 训练后恶化了82%（从-0.08到-0.14），说明调度效率下降

#### 1.5 Invalid Action Penalty
```java
// LoadBalancerGateway.java:439-440
this.rewardInvalidActionComponent = -settings.getRewardInvalidActionCoef() *
                                     (wasInvalidAction ? 1.0 : 0.0);
```

**公式**: `-reward_invalid_action_coef × (1 if invalid else 0)`
**配置**: `reward_invalid_action_coef: 2.0`
**含义**: 惩罚无效动作（如分配到已满的VM）
**目标**: 0（不产生无效动作）
**典型值**: 0.0（训练中没有无效动作）

#### 1.6 Energy Penalty ⚡ **核心指标**
```java
// LoadBalancerGateway.java:442-467
double timeDeltaHours = (currentClock - previousClock) / 3600.0;
double previousEnergyWh = this.cumulativeEnergyWh;
this.currentPowerW = calculateAndUpdateEnergy(currentClock);
double stepEnergyWh = this.cumulativeEnergyWh - previousEnergyWh;
double maxStepEnergyWh = this.maxTotalPowerW * timeDeltaHours;
double normalizedStepEnergy = maxStepEnergyWh > 0 ?
                              stepEnergyWh / maxStepEnergyWh : 0;
this.rewardEnergyComponent = -settings.getRewardEnergyCoef() * normalizedStepEnergy;
```

**公式**: `-reward_energy_coef × (step_energy_Wh / max_step_energy_Wh)`
**配置**: `reward_energy_coef: 2.5`
**含义**: 惩罚步骤能耗增量（Wh），考虑时间和功率
**目标**: 最大化（低能耗 → 接近0）
**典型值**: -0.21 到 -0.22

**重要性**: 这是绿色调度的核心优化目标！

---

## 2️⃣ 绿色调度核心指标（Energy Metrics）

这些指标直接反映能源消耗，是绿色调度的终极目标。

### 📍 数据来源
- **计算位置**: `LoadBalancerGateway.java:calculateAndUpdateEnergy()`
- **记录位置**:
  - `monitor.csv` (训练过程，每episode)
  - `energy_consumption.csv` (episode结束后，详细统计)
- **代码行数**: LoadBalancerGateway.java:796-821

### 📈 具体指标

#### 2.1 Cumulative Energy (Wh) ⭐ **最重要**
```java
// LoadBalancerGateway.java:814-817
if (timeDeltaHours > 0) {
    cumulativeEnergyWh += currentPowerW * timeDeltaHours;
}
```

**公式**: `∫ Power(t) dt` = Σ(功率 × 时间间隔)
**单位**: 瓦时(Wh)
**含义**: Episode内累积的总能耗
**目标**: 最小化
**典型值**: 71-76 Wh (当前)
**理论最优**: 30-45 Wh (如果episode能在80秒完成)

**关键点**:
- 这是绿色调度的**终极优化目标**
- 当前变化很小（73.50Wh → 73.73Wh），说明优化空间未释放
- 原因：所有episode都运行固定时间（350秒）

#### 2.2 Current Power (W)
```java
// LoadBalancerGateway.java:797-812
for (var host : hostList) {
    if (host.getPowerModel() != null) {
        double utilization = host.getCpuPercentUtilization();
        double power = host.getPowerModel().getPower(utilization);
        currentPowerW += power;
    }
}
```

**公式**: `Σ(Pidle + utilization × (Ppeak - Pidle))` for all hosts
**单位**: 瓦特(W)
**含义**: 当前时刻的瞬时功率
**目标**: 降低（但不是主要目标）
**典型值**: 730-770 W

**组成**（基于28台SPEC服务器）:
```
理论最小功率（所有host idle）:
  4×155 + 6×69.4 + 8×48.2 + 8×51.4 + 2×106 = 1,862W

理论最大功率（所有host peak）:
  4×269 + 6×315 + 8×385 + 8×214 + 2×430 = 8,618W

当前实际：730-770W
说明：大部分host处于idle或低负载状态
```

#### 2.3 Average Host Utilization
```java
// LoadBalancerGateway.java:828-845
var hostList = datacenter.getHostList();
double totalUtilization = 0;
int activeHostCount = 0;

for (var host : hostList) {
    double util = host.getCpuPercentUtilization();
    if (util > 0) {
        totalUtilization += util;
        activeHostCount++;
    }
}

return activeHostCount > 0 ? totalUtilization / activeHostCount : 0;
```

**公式**: `Σ(host_utilization) / active_host_count`
**单位**: 百分比（0-1）
**含义**: 活跃主机的平均CPU利用率
**目标**: 提高（集中负载，减少活跃主机数）
**典型值**: 0.005-0.01 (0.5%-1%)

**严重问题**:
- 平均利用率仅**0.5%-1%**，极低！
- 说明26个VM分散在28台host上，但负载很轻
- 优化方向：集中VM到少数高效host，关闭其他host

#### 2.4 Maximum Total Power (W)
```java
// LoadBalancerGateway.java:854-882
private double calculateMaxTotalPower() {
    double totalMaxPower = 0;
    for (var host : hostList) {
        if (host.getPowerModel() != null) {
            double maxPower = host.getPowerModel().getPower(1.0);
            totalMaxPower += maxPower;
        }
    }
    return totalMaxPower;
}
```

**公式**: `Σ host.getPower(1.0)` for all hosts
**单位**: 瓦特(W)
**含义**: 所有host满载时的理论最大功率
**用途**: 能耗归一化的分母
**当前值**: 8,618W (基于SPEC服务器配置)

**重要性**: 用于正确归一化能耗reward

---

## 3️⃣ 性能指标（Performance Metrics）

### 📍 数据来源
- **实时**: `LoadBalancerGateway.java` 和 `SimulationCore.java`
- **汇总**: `cloudlets.csv`, `training_report.txt`

### 📈 具体指标

#### 3.1 Episode Length
**单位**: Steps
**含义**: Episode运行的步数
**目标**: 自然完成（所有cloudlet执行完），而不是truncated
**当前值**: 350（所有episode都被truncated）
**理论最优**: 80-150 steps (如果调度高效)

**关键问题**:
- 所有344个episode都恰好运行350步
- 说明没有一个episode自然完成
- 需要调查原因

#### 3.2 Cloudlets Finished
**单位**: 个数
**含义**: Episode内完成的任务数量
**目标**: 164（全部完成）
**当前值**: 未知（需要诊断日志确认）

#### 3.3 Cloudlets Waiting
**单位**: 个数
**含义**: 队列中等待的任务数量
**目标**: 0（无积压）
**观察**: Queue penalty在增加，说明积压变严重

#### 3.4 Average Wait Time
**单位**: 秒
**含义**: 任务从到达到开始执行的平均等待时间
**目标**: 最小化
**当前值**: 0（可能未正确记录）

#### 3.5 Completion Time
**单位**: 秒
**含义**: Episode的实际仿真时间（clock）
**目标**: 最小化（快速完成所有任务）
**当前值**: 350秒（所有episode）
**理论最优**: 35-80秒

**计算**：
```
Total MI = 25,300,000
Total MIPS = 104 PEs × 50,000 = 5,200,000
Ideal time = 25,300,000 / 5,200,000 = 4.87秒（100%利用率）
+ 到达时间 (0-30秒) = 30秒
+ 调度效率损失 (×2) = 60-80秒
```

---

## 4️⃣ 资源利用率指标

### 📍 数据来源
- **VM**: `vms.csv`
- **Host**: `host{id}.csv`, `energy_consumption.csv`

### 📈 具体指标

#### 4.1 VM CPU Utilization
**来源**: `vms.csv`
**公式**: `vm.getCpuUtilizationStats().getMean()`
**单位**: 百分比(%)
**含义**: 每个VM的平均CPU利用率
**目标**: 高利用率（80%+）说明资源充分使用
**典型范围**: 5%-95%

**用途**:
- 识别低效VM（利用率<20%应考虑合并或删除）
- 识别高效VM（利用率>80%，资源使用充分）

#### 4.2 Host CPU Utilization
**来源**: `host{id}.csv`, `energy_consumption.csv`
**公式**: `host.getCpuPercentUtilization()`
**单位**: 百分比(%)
**含义**: 每个物理主机的CPU利用率
**目标**:
- 活跃host：高利用率（>60%）
- 非活跃host：关闭（0%）

**当前问题**: 平均仅0.5%-1%，极度浪费

#### 4.3 VM Count
**来源**: `monitor.csv` (actual_vm_count)
**单位**: 个数
**含义**: Episode中的VM数量
**当前值**: 26（固定）
**潜在优化**: 动态调整（16-26）

#### 4.4 Assignment Success Rate
**来源**: `monitor.csv` (assignment_success)
**单位**: 布尔值
**含义**: Cloudlet分配是否成功
**目标**: True (高成功率)

---

## 5️⃣ CloudSim结果指标

这些指标在episode结束后生成，用于详细分析。

### 📍 数据来源
- `results/{experiment_name}/`目录下的CSV文件

### 📈 具体指标

#### 5.1 VM Cost ($)
**来源**: `vms.csv`
**公式**: `VmCost.getTotalCost()` = 运行时间 × 固定费率
**含义**: 传统云计算费用（按时间计费）
**重要性**: ❌ 不重要（不反映能耗，且当前所有episode的cost相同）

#### 5.2 Cloudlet Cost ($)
**来源**: `cloudlets.csv`
**公式**: 基于cloudlet运行时间
**含义**: 任务执行费用
**重要性**: ❌ 不重要（同样不反映能耗）

#### 5.3 Energy Consumption per Host
**来源**: `energy_consumption.csv`
**包含**:
- Host ID
- Energy (Wh)
- Average Power (W)
- Max Power (W)
- CPU Utilization (%)

**用途**:
- 识别高能耗host
- 分析不同host类型的效率
- 验证是否优先使用高效服务器

#### 5.4 CO2 Emissions
**来源**: `energy_consumption.csv`
**公式**: `Energy (kWh) × 0.5 kg/kWh`
**单位**: 千克(kg)
**含义**: 估算的碳排放
**用途**: 绿色调度的环境影响评估

---

## 📊 指标优先级总结

### 🔴 **最高优先级**（绿色调度核心）

1. **Cumulative Energy (Wh)** - 终极优化目标
   - 当前: 73.5 Wh
   - 目标: <45 Wh
   - 改进空间: 40%+

2. **Episode Completion Time** - 影响总能耗
   - 当前: 350秒（固定）
   - 目标: 80-150秒（自然完成）
   - 改进空间: 60%+

3. **Cloudlets Finished** - 任务完成率
   - 当前: 未知
   - 目标: 164/164 (100%)

### 🟡 **高优先级**（效率优化）

4. **Average Host Utilization** - 资源效率
   - 当前: 0.5%-1%（极低）
   - 目标: 60%+
   - 改进空间: 巨大

5. **Queue Penalty** - 调度效率
   - 当前: 恶化82%
   - 目标: 接近0
   - 需要: 改进调度策略

6. **Current Power (W)** - 瞬时功率
   - 当前: 730-770W
   - 目标: 根据负载动态调整
   - 优化: 集中到高效服务器

### 🟢 **中等优先级**（辅助分析）

7. **Unutilization Penalty** - 资源浪费
8. **Wait Time** - QoS指标
9. **Invalid Action Penalty** - 策略正确性
10. **VM/Cloudlet Cost** - 传统云计费（参考）

---

## 🎯 当前问题诊断

基于现有指标，发现以下关键问题：

### 问题1: Episode从不自然完成 ⚠️
**证据**: 所有344个episode都在第350步truncated
**影响**: 无法优化episode时长 → 总能耗固定
**根因**: 未知（需要诊断日志）

### 问题2: 资源利用率极低 🔴
**证据**: Average host utilization = 0.5%-1%
**影响**: 大量idle功耗浪费
**根因**: 26个VM分散在28台host，负载很轻

### 问题3: 能耗优化停滞 🔴
**证据**: Cumulative energy几乎不变（73.50 → 73.73 Wh）
**影响**: RL无法学到有效的节能策略
**根因**:
1. Episode时长固定（350秒）
2. 资源利用率固定（~1%）
3. 缺少动态VM管理

### 问题4: 调度效率下降 ⚠️
**证据**: Queue penalty恶化82%
**影响**: 任务积压增加
**根因**: Agent学到了次优策略

---

## 📈 指标监控建议

### 训练过程中

**monitor.csv** - 每个episode记录一次：
```csv
r,                      # Total reward
l,                      # Episode length
t,                      # Wall time
reward_energy,          # ⭐ 关键
cumulative_energy_wh,   # ⭐⭐⭐ 最重要
current_power_w,        # ⭐ 关键
average_host_utilization, # ⭐ 关键
reward_queue_penalty,   # 监控调度效率
actual_vm_count,        # 监控VM数量
current_clock           # 监控episode时长
```

### Episode结束后

**analysis/training_report.txt** - 自动生成：
```
First 10 vs Last 10 episodes对比:
- Total reward改善 %
- Cumulative energy改善 % ⭐⭐⭐
- Average power变化 %
- Host utilization变化 %
```

### 详细分析

**energy_consumption.csv** - 每个episode结束：
```csv
Host ID, Energy(Wh), Avg Power(W), Max Power(W), Utilization(%)
```

用于分析：
- 哪些host消耗最多能量
- 是否优先使用高效服务器
- 低效服务器是否被关闭

---

## 🔧 指标计算工具

### 1. 训练过程分析
```bash
python drl-manager/analyze_training.py --log_dir logs/SPEC_Authentic/exp10_spec_real
```

生成：
- `reward_comparison.png` - Reward组件对比
- `energy_comparison.png` - **能耗指标对比** ⭐
- `training_curves.png` - 训练曲线
- `training_report.txt` - 详细报告

### 2. 单episode诊断
```powershell
.\test_single_episode.ps1
```

输出：
```
Step 1: Clock=1.00s, Waiting=X, Finished=0, SimRunning=true
Step 50: Clock=50.00s, Waiting=Y, Finished=Z, SimRunning=...
...
Episode TERMINATED/TRUNCATED at step X
```

### 3. Java日志（SimulationResultUtils.java）

自动生成每个episode的详细结果：
- `cloudlets.csv` - 任务级统计
- `vms.csv` - VM级统计
- `host{id}.csv` - 主机历史记录
- `energy_consumption.csv` - 能耗详细统计

---

## 📋 快速参考表

| 指标 | 当前值 | 目标值 | 优先级 | 数据源 |
|------|--------|--------|--------|--------|
| **Cumulative Energy** | 73.5 Wh | <45 Wh | 🔴 最高 | monitor.csv |
| **Episode Length** | 350 steps | 80-150 | 🔴 最高 | monitor.csv |
| **Host Utilization** | 0.5%-1% | >60% | 🔴 最高 | monitor.csv |
| **Cloudlets Finished** | ? | 164 | 🔴 最高 | 诊断日志 |
| **Current Power** | 730-770W | 动态 | 🟡 高 | monitor.csv |
| **Queue Penalty** | -0.14 | ~0 | 🟡 高 | monitor.csv |
| **Total Reward** | -920 | >-800 | 🟡 高 | monitor.csv |
| **Completion Time** | 350s | 80-150s | 🟡 高 | monitor.csv |
| **Energy Penalty** | -0.22 | >-0.1 | 🟢 中 | monitor.csv |
| **VM Cost** | 固定 | N/A | ⚪ 低 | vms.csv |

---

## 🎯 总结

### 核心指标（必须优化）
1. ✅ **Cumulative Energy (Wh)** - 绿色调度的终极目标
2. ✅ **Episode Completion Time** - 影响总能耗
3. ✅ **Cloudlets Finished Rate** - 确保任务完成

### 效率指标（重要辅助）
4. ✅ **Host Utilization** - 资源效率
5. ✅ **Current Power** - 功率优化
6. ✅ **Queue Length** - 调度效率

### 传统指标（参考）
7. ⚠️ **VM/Cloudlet Cost** - 不反映能耗，仅供参考

### 下一步行动
1. **诊断**: 运行 `.\test_single_episode.ps1` 了解episode为何不能完成
2. **优化**: 根据诊断结果调整max_episode_length
3. **重训**: 启用episode自然完成，释放优化空间
4. **分析**: 使用 `analyze_training.py` 验证改进效果
