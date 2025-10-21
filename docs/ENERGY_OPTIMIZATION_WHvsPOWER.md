# Energy Optimization: Total Energy (Wh) vs Instantaneous Power (W)

## 问题背景

**用户目标**：最小化整个simulation过程中的**总能耗（Wh）**，而不仅仅是降低瞬时功率（W）。

## 关键区别

### 能耗 (Energy) vs 功率 (Power)

- **功率 (Power, W)**: 瞬时能量消耗速率
- **能耗 (Energy, Wh)**: 一段时间内消耗的总能量 = ∫ Power × dt

**例子**：
```
场景A: 200W × 0.1小时 = 20 Wh  ✅ 能耗低
场景B: 50W  × 1.0小时 = 50 Wh  ❌ 能耗高（虽然功率低）
```

## 之前的实现问题

**原代码** (LoadBalancerGateway.java:447-451):
```java
// ❌ 只惩罚瞬时功率，忽略了时间因素
double normalizedPower = this.currentPowerW / this.maxTotalPowerW;
this.rewardEnergyComponent = -settings.getRewardEnergyCoef() * normalizedPower;
```

**问题**：
1. CloudSim是事件驱动的，每一步的时间间隔（Δt）不固定
2. 只看功率无法反映实际能耗：
   - 短时间高功率 可能比 长时间低功率 消耗更少能量
   - Agent可能学会"拖延"策略（低功率但长时间运行）

## 新的实现

**修改后的代码** (LoadBalancerGateway.java:442-467):

```java
// ✅ 惩罚每一步实际消耗的能耗（Wh），考虑时间因素
double currentClock = simulationCore.getClock();

// 1. 计算时间间隔（必须在调用calculateAndUpdateEnergy之前）
double timeDeltaHours = (currentClock - previousClock) / 3600.0;

// 2. 记录更新前的累积能耗
double previousEnergyWh = this.cumulativeEnergyWh;

// 3. 更新功率和累积能耗
this.currentPowerW = calculateAndUpdateEnergy(currentClock);
this.averageHostUtilization = calculateAverageHostUtilization();

// 4. 计算本步骤消耗的能耗增量（Wh）
double stepEnergyWh = this.cumulativeEnergyWh - previousEnergyWh;

// 5. 计算最大可能的步骤能耗（所有host满载）
double maxStepEnergyWh = this.maxTotalPowerW * timeDeltaHours;

// 6. 归一化步骤能耗（0-1范围）
double normalizedStepEnergy = maxStepEnergyWh > 0
    ? stepEnergyWh / maxStepEnergyWh
    : 0;

// 7. 能耗惩罚 = 系数 × 归一化的步骤能耗
this.rewardEnergyComponent = -settings.getRewardEnergyCoef() * normalizedStepEnergy;
```

## 优化效果

### 数学推导

整个episode的总reward（能耗部分）：
```
Total_Energy_Reward = Σ(每一步的能耗惩罚)
                    = Σ[-coef × (step_energy_i / max_step_energy_i)]
                    = -coef × Σ(step_energy_i / max_step_energy_i)
```

当agent最大化total_reward时，等价于**最小化总能耗**：
```
Minimize: Σ step_energy_i = cumulative_energy_wh (整个episode的总能耗)
```

### Agent学习行为改变

**之前（只看功率）**：
- ❌ 可能选择低功率但长时间运行的策略
- ❌ 无法正确权衡"快速高功率 vs 缓慢低功率"

**现在（优化总能耗）**：
- ✅ 学习最小化 E = P × t 的组合
- ✅ 可能选择：高效服务器快速完成 > 低效服务器长时间运行
- ✅ 真正优化总能耗，而不仅仅是降低功率

## 实验配置建议

在 `config.yml` 中调整能耗系数：

```yaml
reward_energy_coef: 2.5  # 推荐值：2.0-4.0

# 该系数现在表示：每单位归一化能耗增量的惩罚
# 更大的值 → Agent更激进地优化能耗
# 建议：根据与其他reward组件的平衡来调整
```

## 验证方法

训练后检查：
1. **cumulative_energy_wh** 应该显著降低（从第一批episodes到最后批）
2. **current_power_w** 可能不会显著降低（因为优化的是总能耗）
3. **episode_length** 可能变化（agent学会了时间-能耗权衡）

使用分析脚本：
```bash
python drl-manager/analyze_training.py --log_dir logs/SPEC_Authentic/exp10_spec_real
```

查看：
- `energy_comparison.png` - 能耗指标对比
- `training_report.txt` - 累积能耗的改善百分比

## 技术细节

### 为什么要在 calculateAndUpdateEnergy 之前计算 timeDelta？

```java
// ✅ 正确顺序
double timeDeltaHours = (currentClock - previousClock) / 3600.0;  // Step 1
this.currentPowerW = calculateAndUpdateEnergy(currentClock);      // Step 2 (更新previousClock)

// ❌ 错误顺序
this.currentPowerW = calculateAndUpdateEnergy(currentClock);      // previousClock被更新！
double timeDeltaHours = (currentClock - previousClock) / 3600.0;  // = 0!
```

因为 `calculateAndUpdateEnergy()` 内部会更新 `previousClock = currentClock`（line 819），必须在调用前获取时间差。

### 归一化方法

```
normalizedStepEnergy = stepEnergyWh / maxStepEnergyWh
                     = (actual_power × Δt) / (max_power × Δt)
                     = actual_power / max_power × (Δt / Δt)
                     = actual_power / max_power
```

虽然数学上等价于功率归一化，但**语义不同**：
- 我们明确表达的是"能耗归一化"
- 代码更清晰地反映了优化目标（最小化能耗）
- 未来如果时间步长变化逻辑改变，代码仍然正确

## 修改历史

- **2025-10-21**: 修改能耗惩罚从"瞬时功率"改为"步骤能耗增量"
- **原因**: 用户目标是降低总能耗（Wh），而非仅降低功率（W）
- **文件**: LoadBalancerGateway.java:442-467

---

**总结**: 这次修改确保agent真正优化的是**整个episode的总能耗**，而不仅仅是瞬时功率水平。这更符合绿色调度的实际目标。
