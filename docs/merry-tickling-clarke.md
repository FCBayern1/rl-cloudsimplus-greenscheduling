# God's Eye 未来绿色能源预测实现计划

### 设计原则

1. **当前能源已有**：`dc_current_green_power_w` 已提供当前状态，新特征专注于**未来**
2. **短期用于即时决策**：未来30分钟，决定"现在路由到哪个DC"
3. **长期用于规划决策**：未来24小时，决定"是否延迟任务等待更好时机"
4. **每DC独立**：异构DC需要独立特征，不做跨DC比较

### 特征设计 (每DC 4个，总计12个新特征)

***HEADSUP： max_power在初始化扫描CSV的时候就先找到***

  double **maxPower** = 0.0;
  for (double power : allPowerData) {
      maxPower = Math.max(maxPower, power);
  }
  this.maxPowerKw = **maxPower**;

#### 短期特征 (未来30分钟 = 3个CSV行)

| 特征 | 字段名 | 计算方式 | 范围 | 决策含义 |
|------|--------|---------|------|---------|
| **短期均值** | `dc_future_short_mean` | `mean(P[t:t+3]) / max_power` | [0,1] | 未来30分钟平均能有多少绿电？值高=优先路由到这里 |
| **短期趋势** | `dc_future_short_trend` | `(P[t+3] - P[t]) / max_power` | [-1,1] | 能源在上升(+)还是下降(-)？上升=可以多分配任务 |

**示例解读**：
- DC_A: short_mean=0.8, short_trend=+0.2 → "DC_A未来30分钟绿电充足且还在涨，优先路由"
- DC_B: short_mean=0.3, short_trend=-0.1 → "DC_B绿电不足且在跌，尽量避免"

#### 长期特征 (未来24小时 = 144个CSV行)

| 特征 | 字段名 | 计算方式 | 范围 | 决策含义 |
|------|--------|---------|------|---------|
| **长期均值** | `dc_future_long_mean` | `mean(P[t:t+144]) / max_power` | [0,1] | 未来一天整体绿电水平，用于评估DC的长期价值 |
| **峰值时机** | `dc_future_long_peak_timing` | `argmax(P[t:t+144]) / 144` | [0,1] | 最佳绿电时刻在何时？0=马上，0.5=12h后，1=24h后 |

**`long_peak_timing` 详细解释**：

```
假设当前时刻 t=0，未来24小时的绿电曲线：

功率 ^
     |      峰值在这里 (t=4h)
     |         *
     |       *   *
     |     *       *
     |   *           *    *
     | *               *  *  *
     +-------------------------> 时间
     0    4h   12h   18h   24h

long_peak_timing = 4/24 = 0.17

含义：峰值在未来4小时后到来
决策：如果有可延迟的大任务，等4小时后再调度到这个DC
```

**示例解读**：
- DC_A: long_mean=0.6, peak_timing=0.1 → "DC_A整体绿电好，且峰值很快(2.4h后)，大任务稍等一下"
- DC_B: long_mean=0.4, peak_timing=0.8 → "DC_B整体一般，峰值要等19小时，不值得等"

### 为什么选这4个特征？

| 特征 | 回答的问题 | RL决策价值 |
|------|-----------|-----------|
| short_mean | "未来30分钟这个DC绿电够不够？" | 即时路由选择 |
| short_trend | "绿电在变好还是变差？" | 是否加大/减少分配 |
| long_mean | "这个DC一天下来绿电表现如何？" | 整体DC偏好 |
| long_peak_timing | "最佳时机什么时候来？" | 大任务延迟决策 |

### 实现位置：Java端

**理由**：
1. `GreenEnergyProvider` 已加载CSV并有spline插值能力
2. 避免Python重复加载CSV
3. 保持所有观测从Java单一来源的架构一致性

### 数据流架构

```
GreenEnergyProvider.java (已有CSV数据)
    │
    ├── getCurrentPowerW(simTime)     // 现有方法
    │
    └── computeFutureTrendFeatures(simTime)  // 新增方法
            │
            ├── 读取 P[t : t+3]   计算 short_mean, short_trend
            └── 读取 P[t : t+144] 计算 long_mean, long_peak_timing
            │
            └── 返回 double[4]

GlobalObservationState.java
    │
    ├── double[] dcCurrentGreenPowerW;      // 现有
    │
    └── 新增字段:
        ├── double[] dcFutureShortMean;     // shape: (num_dc,)
        ├── double[] dcFutureShortTrend;    // shape: (num_dc,)
        ├── double[] dcFutureLongMean;      // shape: (num_dc,)
        └── double[] dcFutureLongPeakTiming;// shape: (num_dc,)

MultiDatacenterSimulationCore.java
    │
    └── buildGlobalObservation(clock)
            │
            └── for each datacenter:
                    features = dc.greenEnergyProvider.computeFutureTrendFeatures(clock)
                    obs.dcFutureShortMean[i] = features[0]
                    obs.dcFutureShortTrend[i] = features[1]
                    obs.dcFutureLongMean[i] = features[2]
                    obs.dcFutureLongPeakTiming[i] = features[3]
```

### Python端更新

```python
# hierarchical_multidc_env.py - 更新 global_observation_space

global_observation_space = Dict({
    # ... 现有字段 ...

    # 新增：未来能源预测特征
    "dc_future_short_mean": Box(low=0.0, high=1.0, shape=(num_dc,), dtype=np.float32),
    "dc_future_short_trend": Box(low=-1.0, high=1.0, shape=(num_dc,), dtype=np.float32),
    "dc_future_long_mean": Box(low=0.0, high=1.0, shape=(num_dc,), dtype=np.float32),
    "dc_future_long_peak_timing": Box(low=0.0, high=1.0, shape=(num_dc,), dtype=np.float32),
})
```

### 文件修改清单

| 文件 | 修改类型 | 具体内容 |
|------|---------|---------|
| `GreenEnergyProvider.java` | 修改 | 添加 `computeFutureTrendFeatures(double simTime)` 方法 |
| `GlobalObservationState.java` | 修改 | 添加4个新的 double[] 字段和对应 getter |
| `MultiDatacenterSimulationCore.java` | 修改 | 在 `buildGlobalObservation()` 中调用新方法填充字段 |
| `hierarchical_multidc_env.py` | 修改 | 更新 `global_observation_space` 定义 |
| `hierarchical_multidc_pettingzoo.py` | 修改 | 同步更新观测空间（如果需要） |
| `config.yml` | 修改 | 添加配置项（short_horizon, long_horizon, max_power等） |

### 配置参数

```yaml
# config.yml 新增配置节
future_energy_forecast:
  enabled: true
  short_term_rows: 3        # 30分钟 = 3 × 10min
  long_term_rows: 144       # 24小时 = 144 × 10min
  max_power_kw: 1500.0      # 归一化用的最大功率
```

### 边界情况处理

| 情况 | 处理方式 |
|------|---------|
| CSV数据不足144行 | 用可用数据计算，或返回默认值(0.5) |
| 功率为负值 | clip到0 |
| 功率超过max_power | clip到1.0 |
| simTime超出CSV范围 | 返回最后可用数据的特征 |

### Agent决策示例

| 场景 | 观测信号 | 建议决策 |
|------|---------|---------|
| DC_A短期好且上升 | short_mean[A]=0.8, trend[A]=+0.2 | 立即路由任务到DC_A |
| DC_B长期好但峰值远 | long_mean[B]=0.7, peak[B]=0.7 | 小任务现在做，大任务考虑延迟 |
| 所有DC短期都差 | short_mean[*]<0.3 | 接受使用棕色能源，优先完成任务 |
| DC_C峰值即将到来 | peak[C]=0.05 (约1小时后) | 可延迟任务等一等再分配给DC_C |

