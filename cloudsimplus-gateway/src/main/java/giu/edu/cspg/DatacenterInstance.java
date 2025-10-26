package giu.edu.cspg;

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
    private GreenEnergyProvider greenEnergyProvider;

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
        return config.isGreenEnergyEnabled() && greenEnergyProvider != null;
    }

    /**
     * Get current green power availability in Watts.
     */
    public double getCurrentGreenPowerW(double simulationTime) {
        if (!isGreenEnergyEnabled()) {
            return 0.0;
        }
        return greenEnergyProvider.getCurrentPowerW(simulationTime);
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
     * Reset statistics for this datacenter instance.
     */
    public void resetStatistics() {
        cloudletsReceived = 0;
        cloudletsCompleted = 0;
        cloudletsWaiting = 0;
        if (isGreenEnergyEnabled()) {
            greenEnergyProvider.resetStatistics();
        }
        LOGGER.info("{}: Statistics reset", getName());
    }

    @Override
    public String toString() {
        return String.format("DatacenterInstance{id=%d, name='%s', hosts=%d, vms=%d, active_vms=%d, waiting=%d}",
                getId(), getName(), getHostCount(), getTotalVmCount(), getActiveVmCount(), getWaitingCloudletCount());
    }
}
