package giu.edu.cspg.multidc;

import lombok.Builder;
import lombok.Getter;

/**
 * Energy metrics delta for a single timestep.
 * Tracks the incremental energy consumption and waste for reward calculation.
 */
@Getter
@Builder
public class EnergyMetricsDelta {

    // Energy consumed in this timestep (Wh)
    private final double deltaGreenEnergyUsedWh;     // Green energy actually used
    private final double deltaBrownEnergyUsedWh;     // Brown energy used (to penalize)
    private final double deltaGreenEnergyWastedWh;   // Green energy available but not used (to penalize)

    // Carbon emissions in this timestep (kg CO2)
    private final double deltaCarbonEmissionKg;      // Carbon emissions from energy use

    // Power metrics at end of timestep (W)
    private final double currentPowerW;              // Current power consumption
    private final double availableGreenPowerW;       // Available green power

    // Utilization metrics
    private final double greenUtilizationRatio;      // deltaGreenUsed / availableGreenEnergy
    private final double timestepDurationHours;      // Duration of this timestep

    /**
     * Get total energy consumed in this timestep.
     */
    public double getDeltaTotalEnergyWh() {
        return deltaGreenEnergyUsedWh + deltaBrownEnergyUsedWh;
    }

    /**
     * Get green energy ratio for this timestep.
     */
    public double getGreenRatio() {
        double total = getDeltaTotalEnergyWh();
        return total > 0 ? deltaGreenEnergyUsedWh / total : 0.0;
    }

    /**
     * Get waste ratio (wasted green / available green).
     */
    public double getWasteRatio() {
        double availableGreen = deltaGreenEnergyUsedWh + deltaGreenEnergyWastedWh;
        return availableGreen > 0 ? deltaGreenEnergyWastedWh / availableGreen : 0.0;
    }

    /**
     * Calculate normalized brown energy penalty.
     * Returns value in [0, 1] where 1 is worst (all brown energy).
     */
    public double getNormalizedBrownPenalty() {
        double total = getDeltaTotalEnergyWh();
        return total > 0 ? deltaBrownEnergyUsedWh / total : 0.0;
    }

    /**
     * Calculate normalized waste penalty.
     * Returns value in [0, 1] where 1 is worst (all available green wasted).
     */
    public double getNormalizedWastePenalty() {
        double availableGreen = deltaGreenEnergyUsedWh + deltaGreenEnergyWastedWh;
        return availableGreen > 0 ? deltaGreenEnergyWastedWh / availableGreen : 0.0;
    }

    /**
     * Get carbon intensity (kg CO2 per kWh) for this timestep.
     * Indicates how clean the energy mix was.
     */
    public double getCarbonIntensity() {
        double totalEnergyKWh = getDeltaTotalEnergyWh() / 1000.0;  // Wh to kWh
        return totalEnergyKWh > 0 ? deltaCarbonEmissionKg / totalEnergyKWh : 0.0;
    }

    @Override
    public String toString() {
        return String.format(
            "EnergyDelta{green=%.2fWh, brown=%.2fWh, wasted=%.2fWh, carbon=%.3fkg, greenRatio=%.1f%%, wasteRatio=%.1f%%, intensity=%.3fkg/kWh}",
            deltaGreenEnergyUsedWh, deltaBrownEnergyUsedWh, deltaGreenEnergyWastedWh, deltaCarbonEmissionKg,
            getGreenRatio() * 100, getWasteRatio() * 100, getCarbonIntensity()
        );
    }
}