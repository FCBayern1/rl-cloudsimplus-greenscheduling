# Green Energy Datacenter Implementation Analysis

## Executive Summary

This green energy datacenter implementation integrates real-world wind power data into CloudSim Plus simulations to enable renewable-aware scheduling research. The system tracks green vs brown energy consumption, supports optional wind power prediction, and enables the RL agent to optimize for energy sustainability.

**Key Features:**
- Real wind turbine data integration with spline interpolation (10-minute → 1-second resolution)
- Green-first energy allocation strategy (prioritize renewable energy)
- Wasted green energy tracking (excess not stored)
- Historical data buffering for prediction models
- Py4J integration for Python-based wind forecasting
- Energy-aware reward function component

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoadBalancerGateway                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Energy Tracking Variables:                              │  │
│  │  - cumulativeGreenEnergyWh                               │  │
│  │  - cumulativeBrownEnergyWh                               │  │
│  │  - totalWastedGreenWh                                    │  │
│  │  - currentGreenPowerW                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         GreenEnergyProvider (if enabled)                 │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  Data Loading & Interpolation                       │ │  │
│  │  │  - Load CSV (turbine_id, timestamp, 18 features)   │ │  │
│  │  │  - Apache Commons SplineInterpolator              │ │  │
│  │  │  - timePoints[], powerValues[]                     │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  Historical Buffer (CircularBuffer)                │ │  │
│  │  │  - Capacity: 12 frames                             │ │  │
│  │  │  - Each frame: WindDataFrame (16 features)         │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  Energy Allocation                                  │ │  │
│  │  │  - allocateEnergy(demand, time, delta)             │ │  │
│  │  │  - Returns: EnergyAllocation                       │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  Prediction (Optional, via Py4J)                   │ │  │
│  │  │  - windPowerPredictor object                       │ │  │
│  │  │  - Cache: 600s (10 minutes)                        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Data Structures                               │
│  ┌──────────────────────┐  ┌───────────────────────────────┐   │
│  │  WindDataFrame       │  │  EnergyAllocation (Result)    │   │
│  │  - timestamp         │  │  - greenEnergyWh              │   │
│  │  - features[16]      │  │  - brownEnergyWh              │   │
│  │    (Wspd, Wdir,      │  │  - wastedGreenWh              │   │
│  │     Etmp, ..., Patv) │  │  - greenPowerW                │   │
│  └──────────────────────┘  │  - demandPowerW               │   │
│                            │  + getGreenRatio()            │   │
│  ┌──────────────────────┐  │  + isFullyGreen()             │   │
│  │  CircularBuffer<T>   │  └───────────────────────────────┘   │
│  │  - capacity: 12      │                                      │
│  │  - head, size        │                                      │
│  │  + add(element)      │                                      │
│  │  + getLast(n)        │                                      │
│  └──────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. GreenEnergyProvider.java

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/energy/GreenEnergyProvider.java`

**Purpose:** Manages wind power data, interpolation, historical buffering, and energy allocation.

#### Key Responsibilities:

1. **Data Loading (loadAndBuildSpline)**
   - Loads CSV with format: `turbine_id, timestamp, Wspd, Wdir, Etmp, ..., Patv (18 columns total)`
   - Filters rows by `turbineId` parameter
   - Converts absolute timestamps to relative seconds (from first record)
   - Extracts power column (Patv, column 17) in kW

2. **Spline Interpolation**
   - Uses Apache Commons Math `SplineInterpolator`
   - Converts 10-minute wind data to continuous function
   - Enables 1-second resolution queries: `getCurrentPowerW(simulationTime)`
   - Handles out-of-range gracefully (returns 0)

3. **Energy Allocation Strategy** (allocateEnergy method)
   ```java
   double greenAvailableWh = greenPowerW * timeDeltaHours;
   double greenUsedWh = Math.min(demandWh, greenAvailableWh);  // Prioritize green
   double brownUsedWh = demandWh - greenUsedWh;                // Fill gap with brown
   double wastedGreenWh = greenAvailableWh - greenUsedWh;      // Excess is WASTED
   ```

   **Key Insight:** No energy storage modeled - excess green energy is wasted, not saved for later.

4. **Historical Buffer Management**
   - Maintains last 12 timesteps of wind data (WindDataFrame)
   - Updated every simulation step via `updateHistoryBuffer(time, powerW)`
   - Used for prediction model input (12-step lookback)

5. **Prediction Integration (Stub Implementation)**
   - Method: `getPredictedPowerW(windPredictor, horizon, currentTime)`
   - Cache duration: 600 seconds (10 minutes)
   - Requires Py4J object registered via `LoadBalancerGateway.registerWindPowerPredictor()`
   - Current status: **Not implemented** (returns zeros, placeholder for future work)

#### Configuration Parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `turbineId` | int | Which turbine to load from CSV | `1` (turbine T01) |
| `csvFilePath` | String | Path to wind data | `windProduction/sdwpf_2001_2112_full.csv` |

#### Statistics Tracking:

| Metric | Description |
|--------|-------------|
| `cumulativeGreenEnergyWh` | Total green energy consumed (Wh) |
| `cumulativeBrownEnergyWh` | Total brown energy consumed (Wh) |
| `totalWastedGreenWh` | Total green energy wasted (Wh) |
| `getOverallGreenRatio()` | Green / (Green + Brown) ratio |

---

### 2. EnergyAllocation.java

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/energy/EnergyAllocation.java`

**Purpose:** Immutable result object returned by `GreenEnergyProvider.allocateEnergy()`.

#### Fields:

```java
private final double greenEnergyWh;    // Green energy consumed this step
private final double brownEnergyWh;    // Brown energy consumed this step
private final double wastedGreenWh;    // Excess green energy (not stored)
private final double greenPowerW;      // Current green power availability
private final double demandPowerW;     // Total power demand
```

#### Utility Methods:

```java
public double getTotalEnergyWh()         // Green + Brown
public double getGreenRatio()            // Green / Total (0.0 to 1.0)
public boolean isFullyGreen()            // True if brownEnergyWh == 0
public boolean hasWaste()                // True if wastedGreenWh > 0
```

#### Example Usage:

```java
EnergyAllocation allocation = greenEnergyProvider.allocateEnergy(
    500.0,    // demandPowerW (current datacenter power)
    120.0,    // currentTime (simulation clock)
    1.0       // timeDelta (1 second)
);

System.out.println("Green ratio: " + allocation.getGreenRatio() * 100 + "%");
// Output: "Green ratio: 68.5%" (if green covered 342.5Wh out of 500Wh demand)
```

---

### 3. WindDataFrame.java

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/energy/WindDataFrame.java`

**Purpose:** Stores a single timestep of wind turbine data with all 16 features.

#### Feature Schema:

| Index | Name | Description |
|-------|------|-------------|
| 0 | Wspd | Wind speed (m/s) |
| 1 | Wdir | Wind direction (°) |
| 2 | Etmp | Environment temperature (°C) |
| 3 | Itmp | Internal temperature (°C) |
| 4 | Ndir | Nacelle direction (°) |
| 5-7 | Pab1, Pab2, Pab3 | Pitch angles of blades (°) |
| 8 | Prtv | Reactive power (kVar) |
| 9 | T2m | Temperature at 2m height (°C) |
| 10 | Sp | Surface pressure (Pa) |
| 11 | RelH | Relative humidity (%) |
| 12 | Wspd_w | Wind speed at hub height (m/s) |
| 13 | Wdir_w | Wind direction at hub height (°) |
| 14 | Tp | Total precipitation (mm) |
| **15** | **Patv** | **Active power output (kW)** |

**Key Constant:**
```java
public static final int POWER_INDEX = 15;  // Patv (active power)
public static final int NUM_FEATURES = 16;
```

#### Usage in Prediction:

When calling a prediction model, the historical buffer provides 12 consecutive `WindDataFrame` objects, giving the model 12 × 16 = 192 features as input.

---

### 4. CircularBuffer.java

**Location:** `cloudsimplus-gateway/src/main/java/giu/edu/cspg/energy/CircularBuffer.java`

**Purpose:** Generic fixed-capacity buffer that overwrites oldest elements when full.

#### Implementation Details:

```java
private final Object[] buffer;
private int head = 0;    // Index of oldest element
private int size = 0;    // Current number of elements
private final int capacity;
```

**Add Operation:**
```
Initial: [A, B, C, _, _]  (capacity=5, size=3, head=0)
add(D):  [A, B, C, D, _]  (size=4)
add(E):  [A, B, C, D, E]  (size=5, buffer full)
add(F):  [F, B, C, D, E]  (size=5, head=1, overwrote A)
```

**getLast(n) Operation:**
Returns the n most recent elements (newest first or oldest first depending on use case).

**Why Circular Buffer?**
- Efficient O(1) add/remove operations
- Fixed memory footprint (12 frames = ~1.5 KB)
- No dynamic allocation during simulation (performance)

---

## Integration with LoadBalancerGateway

### Initialization (reset method)

```java
// LoadBalancerGateway.java:163-177
if (settings.isGreenEnergyEnabled()) {
    try {
        this.greenEnergyProvider = new GreenEnergyProvider(
            settings.getTurbineId(),
            settings.getWindDataFile()
        );
        LOGGER.info("Green Energy Provider initialized for turbine {}", settings.getTurbineId());
    } catch (Exception e) {
        LOGGER.error("Failed to initialize Green Energy Provider: {}", e.getMessage(), e);
        this.greenEnergyProvider = null;
    }
} else {
    this.greenEnergyProvider = null;
    LOGGER.info("Green Energy Provider disabled");
}
```

### Energy Calculation (calculateAndUpdateEnergy method)

```java
// LoadBalancerGateway.java:880-930
private double calculateAndUpdateEnergy(double currentClock) {
    // 1. Calculate total power demand from all hosts
    double currentPowerW = 0;
    for (var host : datacenter.getHostList()) {
        if (host.getPowerModel() != null) {
            double utilization = host.getCpuPercentUtilization();
            double power = host.getPowerModel().getPower(utilization);
            currentPowerW += power;
        }
    }

    // 2. Calculate time interval
    double timeDelta = currentClock - previousClock;
    double timeDeltaHours = timeDelta / 3600.0;

    // 3. Allocate energy (green vs brown)
    if (timeDeltaHours > 0) {
        if (greenEnergyProvider != null) {
            // GREEN ENERGY MODE
            EnergyAllocation allocation = greenEnergyProvider.allocateEnergy(
                currentPowerW, currentClock, timeDelta
            );

            // Update tracking variables
            cumulativeGreenEnergyWh += allocation.getGreenEnergyWh();
            cumulativeBrownEnergyWh += allocation.getBrownEnergyWh();
            totalWastedGreenWh += allocation.getWastedGreenWh();
            currentGreenPowerW = allocation.getGreenPowerW();
            cumulativeEnergyWh += allocation.getTotalEnergyWh();

            LOGGER.debug("Green Energy - Green: {:.2f}Wh({:.1f}%), Brown: {:.2f}Wh, ...",
                allocation.getGreenEnergyWh(), allocation.getGreenRatio() * 100, ...);
        } else {
            // BROWN ENERGY ONLY MODE
            cumulativeEnergyWh += currentPowerW * timeDeltaHours;
            cumulativeBrownEnergyWh += currentPowerW * timeDeltaHours;
        }
    }

    previousClock = currentClock;
    return currentPowerW;
}
```

**BUG NOTICE:** Line 917 has Python format string `{:.2f}` instead of Java format. This was identified in previous analysis but not yet fixed.

### Reward Function Integration (calculateReward method)

```java
// LoadBalancerGateway.java:519-548
// 5. Energy Consumption Penalty
double timeDeltaHours = (currentClock - previousClock) / 3600.0;
double previousEnergyWh = this.cumulativeEnergyWh;

// Update power and cumulative energy (calls calculateAndUpdateEnergy)
this.currentPowerW = calculateAndUpdateEnergy(currentClock);
this.averageHostUtilization = calculateAverageHostUtilization();

// Calculate energy consumed in this step (Wh)
double stepEnergyWh = this.cumulativeEnergyWh - previousEnergyWh;

// Calculate maximum possible step energy (all hosts at 100% utilization)
double maxStepEnergyWh = this.maxTotalPowerW * timeDeltaHours;

// Normalize step energy consumption (0 to 1 range)
double normalizedStepEnergy = maxStepEnergyWh > 0 ? stepEnergyWh / maxStepEnergyWh : 0;

// Energy penalty based on actual energy consumed (Wh), not instantaneous power (W)
this.rewardEnergyComponent = -settings.getRewardEnergyCoef() * normalizedStepEnergy;
```

**Key Design Choice:**
- Reward penalizes **total energy (Wh)**, not just **power (W)**
- Formula: `Energy (Wh) = Power (W) × Time (hours)`
- This accounts for BOTH power level AND duration
- Normalized by `maxTotalPowerW` (calculated during reset)

---

## Configuration (config.yml)

### Green Energy Section

```yaml
green_energy:
  enabled: true                                      # Enable green energy tracking
  turbine_id: 1                                      # Which turbine to load (T01, T02, etc.)
  wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"  # Path to wind data CSV

  prediction:                                        # Optional wind power prediction
    enabled: false                                   # Not yet implemented
    model_path: ""                                   # Path to trained model (future)
    cache_duration_seconds: 600.0                    # Cache predictions for 10 minutes
    horizon: 8                                       # Predict 8 steps ahead
```

### Energy Reward Coefficient

```yaml
reward_energy_coef: 0.5  # Weight for energy penalty in reward function
                         # 0.0 = disabled (default)
                         # Higher values = stronger energy optimization
```

### SimulationSettings.java Loading

```java
// SimulationSettings.java:214-242
@SuppressWarnings("unchecked")
Map<String, Object> greenEnergyConfig = (Map<String, Object>) params.getOrDefault("green_energy", Map.of());
this.greenEnergyEnabled = getBoolParam(greenEnergyConfig, "enabled", false);
this.turbineId = getIntParam(greenEnergyConfig, "turbine_id", 1);
this.windDataFile = getStringParam(greenEnergyConfig, "wind_data_file",
    "windProduction/sdwpf_2001_2112_full.csv");

@SuppressWarnings("unchecked")
Map<String, Object> predictionConfig = (Map<String, Object>) greenEnergyConfig.getOrDefault("prediction", Map.of());
this.greenPredictionEnabled = getBoolParam(predictionConfig, "enabled", false);
this.predictionModelPath = getStringParam(predictionConfig, "model_path", "");
this.predictionCacheDuration = getDoubleParam(predictionConfig, "cache_duration_seconds", 600.0);
this.predictionHorizon = getIntParam(predictionConfig, "horizon", 8);
```

---

## Data Flow Example

### Scenario: One Simulation Step

**Initial State:**
- Simulation time: 120.0 seconds
- Previous clock: 119.0 seconds (1 second elapsed)
- Datacenter power demand: 450W (from 3 active hosts)
- Wind turbine T01 currently producing: 300W (from spline interpolation)

**Step 1: Calculate Energy Allocation**
```
Time delta: 120.0 - 119.0 = 1.0 second = 1/3600 hours
Demand energy: 450W × (1/3600)h = 0.125 Wh

Green available: 300W × (1/3600)h = 0.0833 Wh
Green used: min(0.125, 0.0833) = 0.0833 Wh  (limited by availability)
Brown used: 0.125 - 0.0833 = 0.0417 Wh      (fill the gap)
Wasted green: 0.0833 - 0.0833 = 0 Wh        (fully utilized)

Green ratio: 0.0833 / 0.125 = 66.6%
```

**Step 2: Update Cumulative Statistics**
```
cumulativeGreenEnergyWh += 0.0833   → 15.234 Wh (total so far)
cumulativeBrownEnergyWh += 0.0417   → 7.891 Wh  (total so far)
totalWastedGreenWh += 0              → 2.456 Wh  (from previous steps)
cumulativeEnergyWh += 0.125          → 23.125 Wh (total)
```

**Step 3: Update Historical Buffer**
```
WindDataFrame frame = new WindDataFrame(120.0, features);
// features[15] = 300W / 1000 = 0.3 kW (power)
// features[0] = wind speed, features[1] = wind direction, etc.

historyBuffer.add(frame);  // Now contains last 12 frames (108s - 120s)
```

**Step 4: Calculate Reward Component**
```
stepEnergyWh = 0.125 Wh
maxStepEnergyWh = maxTotalPowerW × timeDeltaHours
                = 1200W × (1/3600)h = 0.333 Wh

normalizedStepEnergy = 0.125 / 0.333 = 0.375

rewardEnergyComponent = -0.5 × 0.375 = -0.1875  (reward_energy_coef = 0.5)
```

**Step 5: Log Output**
```
DEBUG: Green Energy - Green: 0.08Wh(66.6%), Brown: 0.04Wh, Wasted: 0.00Wh, GreenPower: 300.0W
DEBUG: Energy - Step: 0.125Wh, Power: 450W, Cumulative: 23.125Wh, TimeDelta: 0.000278h, Normalized: 0.375, Reward: -0.1875
```

---

## Statistics and Monitoring

### Available Statistics (getGreenEnergyStats method)

```java
public Map<String, Double> getGreenEnergyStats() {
    Map<String, Double> stats = new HashMap<>();
    stats.put("cumulative_green_wh", cumulativeGreenEnergyWh);
    stats.put("cumulative_brown_wh", cumulativeBrownEnergyWh);
    stats.put("total_wasted_green_wh", totalWastedGreenWh);
    stats.put("current_green_power_w", currentGreenPowerW);
    stats.put("green_ratio", cumulativeEnergyWh > 0 ? cumulativeGreenEnergyWh / cumulativeEnergyWh : 0.0);

    if (greenEnergyProvider != null) {
        stats.put("overall_green_ratio", greenEnergyProvider.getOverallGreenRatio());
    }

    return stats;
}
```

### Logging Levels

| Level | Location | Information |
|-------|----------|-------------|
| INFO | GreenEnergyProvider | CSV loading progress, spline build, initialization |
| DEBUG | LoadBalancerGateway | Per-step green/brown allocation, energy consumption |
| WARN | GreenEnergyProvider | Out-of-range queries, prediction failures |
| ERROR | LoadBalancerGateway | Initialization failures |

---

## Key Design Decisions

### 1. Why Spline Interpolation?

**Problem:** Wind data is recorded at 10-minute intervals, but simulation runs at 1-second timesteps.

**Options Considered:**
- **Nearest neighbor:** Too coarse, creates sudden power jumps
- **Linear interpolation:** Better, but unrealistic for wind (not smooth)
- **Spline interpolation:** ✓ Smooth, continuous, physically plausible

**Implementation:**
```java
SplineInterpolator interpolator = new SplineInterpolator();
powerSpline = interpolator.interpolate(timePoints, powerValues);

// Later queries are O(log n) with cubic spline evaluation
double powerKW = powerSpline.value(simulationTime);
```

**Trade-offs:**
- ✓ Smooth, realistic power curves
- ✓ Fast queries (logarithmic search + cubic eval)
- ✗ Higher memory (stores knots + coefficients)
- ✗ Initial build time (one-time cost during reset)

### 2. Why No Energy Storage?

**Current Design:**
```java
double wastedGreenWh = greenAvailableWh - greenUsedWh;  // Excess is wasted
```

**Justification:**
- Simplifies initial implementation
- Reflects datacenters without battery storage
- Future work: add battery model with capacity, charge/discharge rates, efficiency

**Impact on Scheduling:**
- Agent cannot "save" excess green energy for later
- Creates time-pressure: use green energy now or lose it
- Realistic for many real datacenters (limited storage)

### 3. Why Green-First Allocation?

**Strategy:**
```java
double greenUsedWh = Math.min(demandWh, greenAvailableWh);  // Use all available green first
double brownUsedWh = demandWh - greenUsedWh;                 // Fill remainder with brown
```

**Alternative Approaches:**
- **Brown-first:** Never makes sense (renewable is free)
- **Proportional mix:** Wastes renewable potential
- **Load shifting:** Requires prediction + storage (future work)

### 4. Why 12-Frame History Buffer?

**Rationale:**
- Common in time series models (LSTM, Transformer)
- 12 steps × 10-minute data = 2-hour context window
- Sufficient for short-term wind patterns
- Small enough to avoid memory bloat (12 × 16 features = 192 values)

---

## Future Work and TODOs

### 1. Implement Wind Power Prediction

**Current Status:** Stub implementation (returns zeros)

**Location:** `GreenEnergyProvider.java:399-404`
```java
private double[] invokePythonPredictor(Object predictor, List<WindDataFrame> history, int horizon) {
    // TODO: Implement actual Py4J call
    LOGGER.warn("Python predictor invocation not yet implemented, returning zeros");
    return new double[horizon];
}
```

**What's Needed:**
1. Python service with Py4J gateway
2. Trained prediction model (LSTM/Transformer)
3. Interface definition:
   ```python
   class WindPowerPredictor:
       def predict(self, history: List[List[float]], horizon: int) -> List[float]:
           """
           Args:
               history: 12 × 16 array (12 timesteps, 16 features each)
               horizon: Number of future steps to predict
           Returns:
               Predicted power values in kW
           """
   ```

### 2. Add Energy Storage Model

**Design Proposal:**
```java
public class BatteryStorage {
    private final double capacityWh;
    private final double maxChargeRateW;
    private final double maxDischargeRateW;
    private final double efficiency;  // Round-trip efficiency (e.g., 0.9)

    private double currentChargeWh;

    public double charge(double availableWh, double timeDelta) { ... }
    public double discharge(double demandWh, double timeDelta) { ... }
}
```

**Integration Point:**
```java
// Modified allocation strategy
double greenUsedWh = Math.min(demandWh, greenAvailableWh);
double excessGreenWh = greenAvailableWh - greenUsedWh;

// Store excess instead of wasting
double storedWh = battery.charge(excessGreenWh, timeDelta);
double wastedGreenWh = excessGreenWh - storedWh;

// If green insufficient, discharge battery before using brown
if (greenUsedWh < demandWh) {
    double batteryWh = battery.discharge(demandWh - greenUsedWh, timeDelta);
    greenUsedWh += batteryWh;  // Count battery as "green" (was stored green)
}
double brownUsedWh = demandWh - greenUsedWh;
```

### 3. Multi-Turbine Support

**Current Limitation:** Single turbine per datacenter

**Enhancement:**
```java
private List<GreenEnergyProvider> turbines;

public double getTotalGreenPowerW(double currentTime) {
    return turbines.stream()
        .mapToDouble(t -> t.getCurrentPowerW(currentTime))
        .sum();
}
```

### 4. Carbon Intensity Tracking

**Proposal:** Track CO₂ emissions based on energy mix

```java
private static final double BROWN_CO2_PER_KWH = 0.5;  // kg CO₂ per kWh (grid average)

public double getCumulativeCO2Kg() {
    return cumulativeBrownEnergyWh / 1000.0 * BROWN_CO2_PER_KWH;
}
```

### 5. Time-of-Day Grid Pricing

**Proposal:** Variable brown energy cost by time

```java
public double getBrownEnergyCost(double currentTime) {
    int hourOfDay = (int) (currentTime / 3600) % 24;
    return HOURLY_RATES[hourOfDay];  // $/kWh by hour
}
```

---

## Testing and Validation

### Unit Test Checklist

- [ ] `CircularBuffer` overflow behavior
- [ ] `WindDataFrame` feature indexing
- [ ] `EnergyAllocation` ratio calculations
- [ ] `GreenEnergyProvider` CSV loading with missing data
- [ ] `GreenEnergyProvider` spline interpolation accuracy
- [ ] `GreenEnergyProvider` energy allocation edge cases (zero demand, zero supply)
- [ ] `LoadBalancerGateway` green energy statistics

### Integration Test Checklist

- [ ] Full simulation with green energy enabled
- [ ] Verify cumulative stats match step-by-step calculations
- [ ] Test with multiple turbine IDs
- [ ] Test with CSV containing gaps/invalid data
- [ ] Verify reward component calculation
- [ ] Test reset behavior (cumulative stats zeroed)

### Validation Criteria

1. **Energy Conservation:** `cumulativeEnergyWh == cumulativeGreenEnergyWh + cumulativeBrownEnergyWh`
2. **No Negative Energy:** All energy values ≥ 0
3. **Green Ratio Bounds:** `0 ≤ greenRatio ≤ 1`
4. **Waste Only When Surplus:** `wastedGreenWh > 0` only if `greenAvailableWh > demandWh`
5. **Spline Accuracy:** Interpolated values close to original data points (within 5%)

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Frequency | Impact |
|-----------|------------|-----------|--------|
| CSV loading | O(n log n) | Once per reset | Low (one-time) |
| Spline build | O(n) | Once per reset | Low (one-time) |
| Power query | O(log n) | Every step | Medium (frequent) |
| Energy allocation | O(1) | Every step | Low (simple math) |
| History update | O(1) | Every step | Low (circular buffer) |

**Optimization:**
- Spline queries are cached by Apache Commons Math
- Circular buffer avoids dynamic allocation

### Memory Footprint

| Component | Size | Notes |
|-----------|------|-------|
| timePoints[] | 8n bytes | n = number of data points (~200k for full year) |
| powerValues[] | 8n bytes | Same as timePoints |
| Spline coefficients | ~32n bytes | 4 coefficients per interval |
| History buffer | 768 bytes | 12 frames × 16 features × 8 bytes |
| **Total** | **~10 MB** | For full year of 10-min data |

**Note:** Acceptable for most systems, but consider downsampling for multi-year datasets.

---

## References

### Data Source
- **SDWPF Dataset:** Spatial Dynamic Wind Power Forecasting Challenge
- **Format:** 134 turbines (T01-T134), 245 days, 10-minute resolution
- **Features:** 16 meteorological + operational features
- **File:** `windProduction/sdwpf_2001_2112_full.csv`

### Related Literature
1. SPEC power_ssj2008 - Server power modeling (host profiles)
2. CloudSim Plus Power Model - `getPower(utilization)` interface
3. Apache Commons Math Spline - Cubic spline interpolation

### Code References
- `GreenEnergyProvider.java` - Lines 262-293 (spline build)
- `GreenEnergyProvider.java` - Lines 179-205 (energy allocation)
- `LoadBalancerGateway.java` - Lines 880-930 (energy calculation)
- `LoadBalancerGateway.java` - Lines 519-548 (reward integration)

---

## Summary

This green energy implementation provides a **solid foundation** for renewable-aware datacenter scheduling research. The key strengths are:

✓ **Real-world data integration** (SDWPF turbine dataset)
✓ **Smooth interpolation** (10-minute → 1-second via splines)
✓ **Green-first allocation** (prioritize renewables)
✓ **Comprehensive statistics** (green/brown/waste tracking)
✓ **RL integration** (energy component in reward function)
✓ **Extensible design** (prediction stub, battery ready)

**Limitations to be aware of:**
- No energy storage (excess is wasted)
- Prediction not implemented (stub only)
- Single turbine per simulation
- No grid pricing or carbon tracking

**Next steps for research:**
1. Implement battery storage model
2. Enable wind power prediction (Py4J + trained model)
3. Add carbon intensity/cost tracking
4. Validate against real datacenter energy traces

This implementation enables experiments like:
- Workload shifting to high-wind periods
- VM migration to maximize green energy usage
- Trade-off analysis: performance vs sustainability
- Prediction-based proactive scheduling
