package giu.edu.cspg.multidc;

import java.util.HashMap;
import java.util.Map;

/**
 * Energy metrics for a single datacenter.
 * Provides energy consumption statistics including green/brown energy split.
 */
public class DatacenterEnergyMetrics {
    private final int datacenterId;
    private final String datacenterName;

    // Cumulative energy metrics
    private final double cumulativeGreenEnergyWh;   // Total green energy consumed (Wh)
    private final double cumulativeBrownEnergyWh;   // Total brown energy consumed (Wh)
    private final double totalWastedGreenWh;        // Total wasted green energy (Wh)
    private final double cumulativeCarbonEmissionKg; // Total carbon emissions (kg CO2)

    // Current state metrics
    private final double currentPowerW;             // Current power consumption (W)
    private final double currentGreenPowerW;        // Current available green power (W)

    /**
     * Create energy metrics for a datacenter.
     *
     * @param datacenterId Datacenter ID
     * @param datacenterName Datacenter name
     * @param cumulativeGreenEnergyWh Total green energy consumed
     * @param cumulativeBrownEnergyWh Total brown energy consumed
     * @param totalWastedGreenWh Total wasted green energy
     * @param cumulativeCarbonEmissionKg Total carbon emissions
     * @param currentPowerW Current power consumption
     * @param currentGreenPowerW Current available green power
     */
    public DatacenterEnergyMetrics(int datacenterId, String datacenterName,
                                  double cumulativeGreenEnergyWh, double cumulativeBrownEnergyWh,
                                  double totalWastedGreenWh, double cumulativeCarbonEmissionKg,
                                  double currentPowerW, double currentGreenPowerW) {
        this.datacenterId = datacenterId;
        this.datacenterName = datacenterName;
        this.cumulativeGreenEnergyWh = cumulativeGreenEnergyWh;
        this.cumulativeBrownEnergyWh = cumulativeBrownEnergyWh;
        this.totalWastedGreenWh = totalWastedGreenWh;
        this.cumulativeCarbonEmissionKg = cumulativeCarbonEmissionKg;
        this.currentPowerW = currentPowerW;
        this.currentGreenPowerW = currentGreenPowerW;
    }

    public int getDatacenterId() {
        return datacenterId;
    }

    public String getDatacenterName() {
        return datacenterName;
    }

    public double getCumulativeGreenEnergyWh() {
        return cumulativeGreenEnergyWh;
    }

    public double getCumulativeBrownEnergyWh() {
        return cumulativeBrownEnergyWh;
    }

    public double getTotalWastedGreenWh() {
        return totalWastedGreenWh;
    }

    public double getCurrentPowerW() {
        return currentPowerW;
    }

    public double getCurrentGreenPowerW() {
        return currentGreenPowerW;
    }

    public double getCumulativeCarbonEmissionKg() {
        return cumulativeCarbonEmissionKg;
    }

    /**
     * Get total energy consumed (green + brown).
     */
    public double getTotalEnergyWh() {
        return cumulativeGreenEnergyWh + cumulativeBrownEnergyWh;
    }

    /**
     * Get green energy ratio (0-1).
     */
    public double getGreenEnergyRatio() {
        double total = getTotalEnergyWh();
        return total > 0 ? cumulativeGreenEnergyWh / total : 0.0;
    }

    /**
     * Get carbon intensity (kg CO2 per kWh).
     */
    public double getCarbonIntensity() {
        double totalEnergyKWh = getTotalEnergyWh() / 1000.0;  // Wh to kWh
        return totalEnergyKWh > 0 ? cumulativeCarbonEmissionKg / totalEnergyKWh : 0.0;
    }

    /**
     * Convert to Map for Py4J transfer to Python.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("datacenter_id", datacenterId);
        map.put("datacenter_name", datacenterName);
        map.put("cumulative_green_wh", cumulativeGreenEnergyWh);
        map.put("cumulative_brown_wh", cumulativeBrownEnergyWh);
        map.put("total_wasted_green_wh", totalWastedGreenWh);
        map.put("cumulative_carbon_kg", cumulativeCarbonEmissionKg);
        map.put("current_power_w", currentPowerW);
        map.put("current_green_power_w", currentGreenPowerW);
        map.put("total_energy_wh", getTotalEnergyWh());
        map.put("green_energy_ratio", getGreenEnergyRatio());
        map.put("carbon_intensity_kg_per_kwh", getCarbonIntensity());
        return map;
    }

    @Override
    public String toString() {
        return String.format(
            "DatacenterEnergyMetrics{id=%d, name='%s', green=%.2fWh(%.1f%%), brown=%.2fWh, wasted=%.2fWh, power=%.1fW}",
            datacenterId, datacenterName, cumulativeGreenEnergyWh, getGreenEnergyRatio() * 100,
            cumulativeBrownEnergyWh, totalWastedGreenWh, currentPowerW
        );
    }
}
