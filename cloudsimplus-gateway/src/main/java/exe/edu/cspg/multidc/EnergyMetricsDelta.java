package exe.edu.cspg.multidc;

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
    private final double greenUtilizationRatio;      // deltaGreenUsedWh / (deltaGreenUsedWh + deltaGreenWastedWh)
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

    // ============================================================================
    // Pure analytical helpers for counterfactual reward evaluation (CRD framework).
    //
    // These mirror the per-DC formula applied in
    // DatacenterInstance.updateEnergyMetrics() — kept side-effect-free so Python
    // callbacks can re-evaluate carbon / waste under hypothetical wind power
    // without re-simulating.
    // ============================================================================

    /**
     * Split power demand over a step into green-used and green-wasted (Wh).
     * Green energy is consumed first; the remainder of the demand becomes brown
     * (= demandWh - greenUsedWh), and surplus green is wasted.
     */
    public static double[] computeGreenUsedWastedWh(
            double availableGreenW, double demandW, double durationHours) {
        double demandWh = Math.max(0.0, demandW) * durationHours;
        double greenAvailableWh = Math.max(0.0, availableGreenW) * durationHours;
        double greenUsedWh = Math.min(demandWh, greenAvailableWh);
        double greenWastedWh = greenAvailableWh - greenUsedWh;
        return new double[] { greenUsedWh, greenWastedWh };
    }

    /**
     * Carbon emission (kg CO2) for a single (green, demand) pair over one step.
     * Mirrors the formula in DatacenterInstance.updateEnergyMetrics() exactly.
     */
    public static double computeCarbonKg(
            double availableGreenW, double demandW, double durationHours,
            double greenFactor, double brownFactor) {
        double[] uw = computeGreenUsedWastedWh(availableGreenW, demandW, durationHours);
        double greenKWh = uw[0] / 1000.0;
        double brownKWh = (Math.max(0.0, demandW) * durationHours - uw[0]) / 1000.0;
        return greenKWh * greenFactor + brownKWh * brownFactor;
    }

    /**
     * Per-step waste ratio: wasted / (used + wasted), 0 when no green is available.
     */
    public static double computeWasteRatio(
            double availableGreenW, double demandW, double durationHours) {
        double[] uw = computeGreenUsedWastedWh(availableGreenW, demandW, durationHours);
        double total = uw[0] + uw[1];
        return total > 0 ? uw[1] / total : 0.0;
    }
}