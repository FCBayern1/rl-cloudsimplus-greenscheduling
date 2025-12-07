package giu.edu.cspg.multidc;
import giu.edu.cspg.common.DatacenterConfig;
import giu.edu.cspg.singledc.LoadBalancingBroker;

import java.util.ArrayList;
import java.util.List;

import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import giu.edu.cspg.energy.GreenEnergyProvider;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;

/**
 * Encapsulates all runtime state and components for a single datacenter
 * in a multi-datacenter simulation environment.
 *
 * Each instance maintains its own:
 * - Datacenter (CloudSim Plus entity)
 * - LocalBroker (manages cloudlet-to-VM assignment within this DC)
 * - Hosts (physical servers)
 * - VMs (virtual machines)
 * - GreenEnergyProvider (wind turbine power supply)
 */
@Getter
@Setter
public class DatacenterInstance {
    private static final Logger LOGGER = LoggerFactory.getLogger(DatacenterInstance.class.getSimpleName());

    // === Configuration ===
    private final DatacenterConfig config;

    // === CloudSim Plus Entities ===
    private Datacenter datacenter;
    private LoadBalancingBroker localBroker;  // Will be renamed to LocalBroker later

    // === Infrastructure ===
    private List<Host> hostList;
    private List<Vm> vmPool;

    // === Green Energy ===
    /**
     * List of green energy providers (one per wind turbine).
     * Multiple turbines can supply power to a single datacenter.
     */
    private List<GreenEnergyProvider> greenEnergyProviders = new ArrayList<>();

    // === Energy Tracking ===
    private double cumulativeGreenEnergyWh = 0.0;   // Total green energy consumed
    private double cumulativeBrownEnergyWh = 0.0;   // Total brown (grid) energy consumed
    private double totalWastedGreenWh = 0.0;        // Total wasted green energy
    private double currentPowerW = 0.0;             // Current power consumption
    private double previousClock = 0.0;             // Previous update time for delta calculation

    // Track previous values for delta calculation
    private double previousGreenEnergyWh = 0.0;     // Previous cumulative green energy
    private double previousBrownEnergyWh = 0.0;     // Previous cumulative brown energy
    private double previousWastedGreenWh = 0.0;     // Previous cumulative wasted green

    // === Carbon Emission Tracking ===
    private double cumulativeCarbonEmissionKg = 0.0;   // Total carbon emissions for episode (kg CO2)
    private double previousCarbonEmissionKg = 0.0;     // Previous cumulative carbon emissions (for delta calculation)

    // Latest timestep delta
    private EnergyMetricsDelta latestEnergyDelta = null;

    // === Runtime State ===
    private int cloudletsReceived = 0;      // Total cloudlets assigned to this DC
    private int cloudletsCompleted = 0;     // Total cloudlets finished
    private int cloudletsWaiting = 0;       // Current waiting cloudlets

    /**
     * Create a new datacenter instance with the specified configuration.
     */
    public DatacenterInstance(DatacenterConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("DatacenterConfig cannot be null");
        }
        this.config = config;
        this.hostList = new ArrayList<>();
        this.vmPool = new ArrayList<>();

        LOGGER.info("Created DatacenterInstance: {} (ID: {})",
                config.getDatacenterName(), config.getDatacenterId());
    }

    /**
     * Get the datacenter ID.
     */
    public int getId() {
        return config.getDatacenterId();
    }

    /**
     * Get the datacenter name.
     */
    public String getName() {
        return config.getDatacenterName();
    }

    /**
     * Check if green energy is enabled for this datacenter.
     */
    public boolean isGreenEnergyEnabled() {
        return config.isGreenEnergyEnabled() && !greenEnergyProviders.isEmpty();
    }

    /**
     * Get current green power availability in Watts.
     * Aggregates power from all turbines.
     */
    public double getCurrentGreenPowerW(double simulationTime) {
        if (!isGreenEnergyEnabled()) {
            return 0.0;
        }
        return greenEnergyProviders.stream()
                .mapToDouble(p -> p.getCurrentPowerW(simulationTime))
                .sum();
    }

    /**
     * Add a green energy provider (turbine) to this datacenter.
     */
    public void addGreenEnergyProvider(GreenEnergyProvider provider) {
        if (provider != null) {
            greenEnergyProviders.add(provider);
        }
    }

    /**
     * Get all green energy providers.
     */
    public List<GreenEnergyProvider> getGreenEnergyProviders() {
        return greenEnergyProviders;
    }

    /**
     * Get single green energy provider (backward compatibility).
     * Returns the first provider or null if none.
     * @deprecated Use getGreenEnergyProviders() for multi-turbine support
     */
    @Deprecated
    public GreenEnergyProvider getGreenEnergyProvider() {
        return greenEnergyProviders.isEmpty() ? null : greenEnergyProviders.get(0);
    }

    /**
     * Set single green energy provider (backward compatibility).
     * Clears existing providers and adds the new one.
     * @deprecated Use addGreenEnergyProvider() for multi-turbine support
     */
    @Deprecated
    public void setGreenEnergyProvider(GreenEnergyProvider provider) {
        greenEnergyProviders.clear();
        if (provider != null) {
            greenEnergyProviders.add(provider);
        }
    }

    /**
     * Get number of turbines supplying this datacenter.
     */
    public int getTurbineCount() {
        return greenEnergyProviders.size();
    }

    /**
     * Get number of waiting cloudlets in this datacenter.
     */
    public int getWaitingCloudletCount() {
        if (localBroker == null) {
            return 0;
        }
        return localBroker.getWaitingCloudletCount();
    }

    /**
     * Get number of active (running) VMs.
     */
    public int getActiveVmCount() {
        if (vmPool == null) {
            return 0;
        }
        return (int) vmPool.stream()
                .filter(vm -> vm.isCreated() && !vm.isFailed())
                .count();
    }

    /**
     * Get total number of VMs (active + inactive).
     */
    public int getTotalVmCount() {
        return vmPool != null ? vmPool.size() : 0;
    }

    /**
     * Get total number of hosts.
     */
    public int getHostCount() {
        return hostList != null ? hostList.size() : 0;
    }

    /**
     * Get average host CPU utilization.
     */
    public double getAverageHostUtilization() {
        if (hostList == null || hostList.isEmpty()) {
            return 0.0;
        }
        return hostList.stream()
                .mapToDouble(Host::getCpuPercentUtilization)
                .average()
                .orElse(0.0);
    }

    /**
     * Get total available PEs across all VMs.
     */
    public int getTotalAvailablePes() {
        if (vmPool == null) {
            return 0;
        }
        return vmPool.stream()
                .filter(vm -> vm.isCreated() && !vm.isFailed())
                .mapToInt(vm -> (int) vm.getFreePesNumber())
                .sum();
    }

    /**
     * Increment the counter for cloudlets received.
     */
    public void incrementCloudletsReceived() {
        cloudletsReceived++;
        LOGGER.debug("{}: Cloudlets received: {}", getName(), cloudletsReceived);
    }

    /**
     * Increment the counter for cloudlets completed.
     */
    public void incrementCloudletsCompleted() {
        cloudletsCompleted++;
        LOGGER.debug("{}: Cloudlets completed: {}", getName(), cloudletsCompleted);
    }

    /**
     * Update waiting cloudlets count.
     */
    public void updateWaitingCloudletsCount(int count) {
        this.cloudletsWaiting = count;
    }

    /**
     * Update energy metrics for this datacenter at current timestep.
     * Calculates power consumption from all hosts and allocates energy between green and brown sources.
     * Also tracks energy deltas for incremental reward calculation.
     *
     * @param currentClock Current simulation time in seconds
     */
    public void updateEnergyMetrics(double currentClock) {
        if (hostList == null || hostList.isEmpty()) {
            return;
        }

        // Calculate time delta
        double timeDelta = currentClock - previousClock;
        if (timeDelta <= 0) {
            return;  // Skip if no time has passed
        }

        // Store previous values for delta calculation
        previousGreenEnergyWh = cumulativeGreenEnergyWh;
        previousBrownEnergyWh = cumulativeBrownEnergyWh;
        previousWastedGreenWh = totalWastedGreenWh;

        // Calculate total power consumption from all hosts
        currentPowerW = hostList.stream()
                .mapToDouble(host -> {
                    if (host.getPowerModel() != null) {
                        double utilization = host.getCpuPercentUtilization();
                        return host.getPowerModel().getPower(utilization);
                    }
                    return 0.0;
                })
                .sum();

        double deltaGreenUsed = 0.0;
        double deltaBrownUsed = 0.0;
        double deltaGreenWasted = 0.0;
        double availableGreenPower = 0.0;

        // Allocate energy if green energy is enabled
        if (isGreenEnergyEnabled()) {
            // Calculate aggregated green power from all turbines
            availableGreenPower = getCurrentGreenPowerW(currentClock);
            double timeDeltaHours = timeDelta / 3600.0;

            // Calculate energy amounts in Wh
            double demandWh = currentPowerW * timeDeltaHours;
            double greenAvailableWh = availableGreenPower * timeDeltaHours;

            // Prioritize green energy, excess is wasted (no storage)
            deltaGreenUsed = Math.min(demandWh, greenAvailableWh);
            deltaBrownUsed = demandWh - deltaGreenUsed;
            deltaGreenWasted = greenAvailableWh - deltaGreenUsed;

            // Update cumulative statistics
            cumulativeGreenEnergyWh += deltaGreenUsed;
            cumulativeBrownEnergyWh += deltaBrownUsed;
            totalWastedGreenWh += deltaGreenWasted;

            // Update statistics in each provider (for internal tracking)
            for (GreenEnergyProvider provider : greenEnergyProviders) {
                provider.updateCumulativeStatistics(deltaGreenUsed / greenEnergyProviders.size(),
                        deltaGreenWasted / greenEnergyProviders.size());
            }

            double greenRatio = demandWh > 0 ? deltaGreenUsed / demandWh * 100 : 0;
            LOGGER.debug("{}: Energy allocated (multi-turbine) - Green: {:.2f}Wh ({:.1f}%), Brown: {:.2f}Wh, Wasted: {:.2f}Wh, Turbines: {}",
                    getName(), deltaGreenUsed, greenRatio, deltaBrownUsed, deltaGreenWasted, getTurbineCount());
        } else {
            // No green energy - all brown
            double timeDeltaHours = timeDelta / 3600.0;
            double energyWh = currentPowerW * timeDeltaHours;
            cumulativeBrownEnergyWh += energyWh;
            deltaBrownUsed = energyWh;
        }

        // Calculate carbon emissions for this timestep
        // Carbon emission = green_energy × green_factor + brown_energy × brown_factor
        double deltaGreenKWh = deltaGreenUsed / 1000.0;  // Wh to kWh
        double deltaBrownKWh = deltaBrownUsed / 1000.0;  // Wh to kWh
        double deltaCarbonKg = (deltaGreenKWh * config.getGreenCarbonFactor())
                             + (deltaBrownKWh * config.getBrownCarbonFactor());

        // Update cumulative carbon emissions
        previousCarbonEmissionKg = cumulativeCarbonEmissionKg;
        cumulativeCarbonEmissionKg += deltaCarbonKg;

        // Create and store energy delta for this timestep
        latestEnergyDelta = EnergyMetricsDelta.builder()
                .deltaGreenEnergyUsedWh(deltaGreenUsed)
                .deltaBrownEnergyUsedWh(deltaBrownUsed)
                .deltaGreenEnergyWastedWh(deltaGreenWasted)
                .deltaCarbonEmissionKg(deltaCarbonKg)
                .currentPowerW(currentPowerW)
                .availableGreenPowerW(availableGreenPower)
                .greenUtilizationRatio(availableGreenPower > 0 ? deltaGreenUsed / (deltaGreenUsed + deltaGreenWasted) : 0.0)
                .timestepDurationHours(timeDelta / 3600.0)
                .build();

        LOGGER.debug("{}: Carbon emission - Delta: {:.3f} kg CO2, Cumulative: {:.3f} kg CO2, Intensity: {:.3f} kg/kWh",
                getName(), deltaCarbonKg, cumulativeCarbonEmissionKg, latestEnergyDelta.getCarbonIntensity());

        previousClock = currentClock;
    }

    /**
     * Get green energy ratio (0-1).
     */
    public double getGreenEnergyRatio() {
        double total = cumulativeGreenEnergyWh + cumulativeBrownEnergyWh;
        return total > 0 ? cumulativeGreenEnergyWh / total : 0.0;
    }

    /**
     * Get total energy consumed (Wh).
     */
    public double getTotalEnergyWh() {
        return cumulativeGreenEnergyWh + cumulativeBrownEnergyWh;
    }

    /**
     * Get cumulative carbon emissions (kg CO2).
     */
    public double getCumulativeCarbonEmissionKg() {
        return cumulativeCarbonEmissionKg;
    }

    /**
     * Get carbon intensity (kg CO2 per kWh) for the entire episode.
     */
    public double getCarbonIntensity() {
        double totalEnergyKWh = getTotalEnergyWh() / 1000.0;
        return totalEnergyKWh > 0 ? cumulativeCarbonEmissionKg / totalEnergyKWh : 0.0;
    }

    /**
     * Reset statistics for this datacenter instance.
     */
    public void resetStatistics() {
        cloudletsReceived = 0;
        cloudletsCompleted = 0;
        cloudletsWaiting = 0;

        // Reset energy statistics
        cumulativeGreenEnergyWh = 0.0;
        cumulativeBrownEnergyWh = 0.0;
        totalWastedGreenWh = 0.0;
        currentPowerW = 0.0;
        previousClock = 0.0;

        // Reset delta tracking
        previousGreenEnergyWh = 0.0;
        previousBrownEnergyWh = 0.0;
        previousWastedGreenWh = 0.0;
        latestEnergyDelta = null;

        // Reset carbon emissions
        cumulativeCarbonEmissionKg = 0.0;
        previousCarbonEmissionKg = 0.0;

        if (isGreenEnergyEnabled()) {
            for (GreenEnergyProvider provider : greenEnergyProviders) {
                provider.resetStatistics();
            }
        }
        LOGGER.info("{}: Statistics reset (turbines: {})", getName(), getTurbineCount());
    }

    /**
     * Get the latest energy delta for the most recent timestep.
     * Returns null if no timestep has been processed yet.
     */
    public EnergyMetricsDelta getLatestEnergyDelta() {
        return latestEnergyDelta;
    }

    /**
     * Get list of current host CPU utilizations.
     */
    public List<Double> getHostUtilizations() {
        List<Double> utils = new ArrayList<>();
        if (hostList != null) {
            for (Host host : hostList) {
                utils.add(host.getCpuPercentUtilization());
            }
        }
        return utils;
    }

    /**
     * Get count of cloudlets received by this datacenter.
     */
    public int getCloudletsReceived() {
        return cloudletsReceived;
    }

    /**
     * Get count of cloudlets completed by this datacenter.
     * Uses the broker's finished list for accurate count.
     */
    public int getCloudletsCompleted() {
        if (localBroker != null) {
            return localBroker.getCloudletFinishedList().size();
        }
        return cloudletsCompleted;  // Fallback to manual counter
    }

    /**
     * Get the local broker for this datacenter.
     * Used for accessing cloudlet statistics like finished list.
     */
    public LoadBalancingBroker getLocalBroker() {
        return localBroker;
    }

    @Override
    public String toString() {
        return String.format("DatacenterInstance{id=%d, name='%s', hosts=%d, vms=%d, active_vms=%d, waiting=%d}",
                getId(), getName(), getHostCount(), getTotalVmCount(), getActiveVmCount(), getWaitingCloudletCount());
    }
}
