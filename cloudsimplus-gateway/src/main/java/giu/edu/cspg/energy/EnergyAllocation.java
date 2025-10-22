package giu.edu.cspg.energy;

/**
 * Represents the result of energy allocation between green and brown sources.
 * Immutable data class for tracking energy distribution.
 */
public class EnergyAllocation {
    private final double greenEnergyWh;    // Green energy consumed (Wh)
    private final double brownEnergyWh;    // Brown energy consumed (Wh)
    private final double wastedGreenWh;    // Green energy wasted (excess, not stored)
    private final double greenPowerW;      // Available green power (W)
    private final double demandPowerW;     // Total demand power (W)

    /**
     * Create an energy allocation result.
     *
     * @param greenEnergyWh Green energy consumed (Wh)
     * @param brownEnergyWh Brown energy consumed (Wh)
     * @param wastedGreenWh Green energy wasted (Wh)
     * @param greenPowerW   Available green power (W)
     * @param demandPowerW  Total demand power (W)
     */
    public EnergyAllocation(double greenEnergyWh, double brownEnergyWh, double wastedGreenWh,
                           double greenPowerW, double demandPowerW) {
        this.greenEnergyWh = greenEnergyWh;
        this.brownEnergyWh = brownEnergyWh;
        this.wastedGreenWh = wastedGreenWh;
        this.greenPowerW = greenPowerW;
        this.demandPowerW = demandPowerW;
    }

    /**
     * Get green energy consumed.
     *
     * @return Green energy in Wh
     */
    public double getGreenEnergyWh() {
        return greenEnergyWh;
    }

    /**
     * Get brown energy consumed.
     *
     * @return Brown energy in Wh
     */
    public double getBrownEnergyWh() {
        return brownEnergyWh;
    }

    /**
     * Get wasted green energy.
     *
     * @return Wasted energy in Wh
     */
    public double getWastedGreenWh() {
        return wastedGreenWh;
    }

    /**
     * Get available green power.
     *
     * @return Green power in W
     */
    public double getGreenPowerW() {
        return greenPowerW;
    }

    /**
     * Get total demand power.
     *
     * @return Demand power in W
     */
    public double getDemandPowerW() {
        return demandPowerW;
    }

    /**
     * Calculate green energy ratio (0-1).
     *
     * @return Ratio of green energy to total energy consumed
     */
    public double getGreenRatio() {
        double total = greenEnergyWh + brownEnergyWh;
        return total > 0 ? greenEnergyWh / total : 0.0;
    }

    /**
     * Get total energy consumed.
     *
     * @return Total energy in Wh
     */
    public double getTotalEnergyWh() {
        return greenEnergyWh + brownEnergyWh;
    }

    /**
     * Check if demand was fully met by green energy.
     *
     * @return true if 100% green
     */
    public boolean isFullyGreen() {
        return brownEnergyWh == 0 && greenEnergyWh > 0;
    }

    /**
     * Check if any green energy was used.
     *
     * @return true if some green energy was used
     */
    public boolean hasGreenEnergy() {
        return greenEnergyWh > 0;
    }

    @Override
    public String toString() {
        return String.format(
            "EnergyAllocation[green=%.2fWh(%.1f%%), brown=%.2fWh, wasted=%.2fWh, demand=%.1fW]",
            greenEnergyWh, getGreenRatio() * 100, brownEnergyWh, wastedGreenWh, demandPowerW
        );
    }
}
