// ========================================================================
// REFERENCE: New Reward Implementation for MultiDatacenterSimulationCore.java
// Location: Replace calculateGlobalReward() method (around line 681)
// ========================================================================

/**
 * Calculate global reward for the Global Agent.
 *
 * Design: 50% Green Energy + 50% Performance
 *
 * Global objectives:
 * - Maximize green energy usage and minimize waste
 * - Reduce carbon emissions (brown energy)
 * - Maintain high task completion rate
 * - Minimize waiting time
 * - Balance load across datacenters
 */
private double calculateGlobalReward() {
    // === Component 1: Green Energy Reward (50% weight) ===
    double greenReward = calculateGreenEnergyReward();

    // === Component 2: Performance Reward (50% weight) ===
    double performanceReward = calculatePerformanceReward();

    // === Final Reward (50%-50% balance) ===
    double reward = 0.5 * greenReward + 0.5 * performanceReward;

    LOGGER.debug("Global Reward: total={} (green={}, performance={})",
            reward, greenReward, performanceReward);

    return reward;
}

/**
 * Calculate green energy component of global reward.
 *
 * Components:
 * - 50%: Green energy usage ratio (encourages using green power)
 * - 30%: Waste penalty (penalizes wasting available green energy)
 * - 20%: Carbon emission penalty (penalizes using brown energy)
 *
 * @return Green energy reward [0, 1]
 */
private double calculateGreenEnergyReward() {
    double totalEnergy = datacenterInstances.stream()
            .mapToDouble(DatacenterInstance::getTotalEnergyWh)
            .sum();

    if (totalEnergy == 0) {
        return 0.0;
    }

    // 1.1 Green Energy Usage Ratio (weighted average across DCs)
    double greenEnergy = datacenterInstances.stream()
            .mapToDouble(DatacenterInstance::getCumulativeGreenEnergyWh)
            .sum();
    double greenRatioReward = greenEnergy / totalEnergy;  // [0, 1]

    // 1.2 Waste Penalty
    // Penalize situations where green power is available but load is low
    double wastePenalty = 0.0;
    for (DatacenterInstance dc : datacenterInstances) {
        double greenPower = dc.getCurrentGreenPowerW(currentClock);
        double currentPower = dc.getCurrentPowerW();

        // If green power exceeds consumption by >10W, penalize the waste
        if (greenPower > currentPower + 10.0) {
            double wasted = greenPower - currentPower;
            wastePenalty += wasted / Math.max(greenPower, 1.0);
        }
    }
    wastePenalty /= Math.max(datacenterInstances.size(), 1);  // Normalize
    wastePenalty = Math.min(wastePenalty, 1.0);  // Clip to [0, 1]

    // 1.3 Carbon Emission Penalty
    double brownEnergy = datacenterInstances.stream()
            .mapToDouble(DatacenterInstance::getCumulativeBrownEnergyWh)
            .sum();
    double carbonPenalty = brownEnergy / Math.max(totalEnergy, 1.0);  // [0, 1]

    // Combine green energy components
    double greenReward = 0.5 * greenRatioReward +      // 50%: Use green energy
                        0.3 * (1.0 - wastePenalty) +   // 30%: Reduce waste
                        0.2 * (1.0 - carbonPenalty);   // 20%: Reduce carbon

    LOGGER.trace("  Green Energy - Ratio: {}, Waste: {}, Carbon: {} -> Reward: {}",
            greenRatioReward, wastePenalty, carbonPenalty, greenReward);

    return greenReward;  // [0, 1]
}

/**
 * Calculate performance component of global reward.
 *
 * Components:
 * - 40%: Task completion rate (encourage finishing cloudlets)
 * - 30%: Waiting time penalty (minimize queue lengths)
 * - 30%: Load balance reward (even distribution across DCs)
 *
 * @return Performance reward [0, 1]
 */
private double calculatePerformanceReward() {
    // 2.1 Task Completion Rate
    int totalCompleted = datacenterInstances.stream()
            .mapToInt(DatacenterInstance::getCloudletsCompleted)
            .sum();
    int totalReceived = datacenterInstances.stream()
            .mapToInt(DatacenterInstance::getCloudletsReceived)
            .sum();

    double completionReward = totalReceived > 0 ?
            (double) totalCompleted / totalReceived : 0.0;  // [0, 1]

    // 2.2 Waiting Time Penalty
    double avgWaitingCloudlets = datacenterInstances.stream()
            .mapToInt(DatacenterInstance::getWaitingCloudletCount)
            .average()
            .orElse(0.0);

    // Normalize by expected max queue size (assume max 100 cloudlets per DC)
    double waitPenalty = Math.min(avgWaitingCloudlets / 100.0, 1.0);  // [0, 1]

    // 2.3 Load Balance Reward
    double[] utilizations = datacenterInstances.stream()
            .mapToDouble(DatacenterInstance::getAverageHostUtilization)
            .toArray();

    double variance = calculateVariance(utilizations);
    double balanceReward = Math.exp(-variance * 10.0);  // Exp decay, lower variance = higher reward

    // Combine performance components
    double perfReward = 0.4 * completionReward +        // 40%: Complete tasks
                       0.3 * (1.0 - waitPenalty) +      // 30%: Reduce waiting
                       0.3 * balanceReward;             // 30%: Balance load

    LOGGER.trace("  Performance - Completion: {}, Wait: {}, Balance: {} -> Reward: {}",
            completionReward, waitPenalty, balanceReward, perfReward);

    return perfReward;  // [0, 1]
}

/**
 * Calculate variance of an array of values.
 *
 * @param values Array of values
 * @return Variance
 */
private double calculateVariance(double[] values) {
    if (values.length == 0) {
        return 0.0;
    }

    double mean = Arrays.stream(values).average().orElse(0.0);
    double variance = Arrays.stream(values)
            .map(v -> Math.pow(v - mean, 2))
            .average()
            .orElse(0.0);

    return variance;
}

// ========================================================================
// INSTRUCTIONS:
// 1. Open MultiDatacenterSimulationCore.java
// 2. Find calculateGlobalReward() method (around line 681)
// 3. Replace the entire method and calculateLoadBalancePenalty()
//    with the three methods above (calculateGlobalReward,
//    calculateGreenEnergyReward, calculatePerformanceReward)
// 4. Keep or remove calculateLoadBalancePenalty() (it's now integrated
//    into calculatePerformanceReward as the balance component)
// 5. Add calculateVariance() helper method
// ========================================================================
