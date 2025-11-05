package giu.edu.cspg.common;
import giu.edu.cspg.common.DatacenterConfig;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

/**
 * Configuration class for a single datacenter in multi-datacenter setup.
 * Contains infrastructure specs, VM fleet, and green energy settings.
 */
@Getter
@Builder
@ToString
public class DatacenterConfig {
    // === Datacenter Identification ===
    private final int datacenterId;
    private final String datacenterName;

    // === Host Configuration ===
    private final int hostsCount;
    private final int hostPes;             // Processing Elements (cores) per host
    private final long hostPeMips;         // MIPS capacity per PE
    private final long hostRam;            // RAM in MB
    private final long hostBw;             // Bandwidth in Mbps
    private final long hostStorage;        // Storage in MB

    // === VM Configuration ===
    // Small VM (base tier)
    private final int smallVmPes;
    private final long smallVmRam;
    private final long smallVmBw;
    private final long smallVmStorage;

    // VM multipliers for Medium and Large VMs
    private final int mediumVmMultiplier;  // Medium = Small * multiplier
    private final int largeVmMultiplier;   // Large = Small * multiplier

    // === Initial VM Fleet ===
    private final int initialSmallVmCount;
    private final int initialMediumVmCount;
    private final int initialLargeVmCount;

    // === Green Energy Configuration ===
    private final boolean greenEnergyEnabled;
    private final int turbineId;           // Wind turbine ID for this datacenter
    private final String windDataFile;     // Path to wind power CSV file

    // === VM Lifecycle Delays ===
    private final double vmStartupDelay;   // Time (sec) for VM to boot
    private final double vmShutdownDelay;  // Time (sec) before idle VM is destroyed

    /**
     * Get total number of VMs in initial fleet.
     */
    public int getTotalInitialVmCount() {
        return initialSmallVmCount + initialMediumVmCount + initialLargeVmCount;
    }

    /**
     * Get total cores (PEs) in this datacenter.
     */
    public int getTotalCores() {
        return hostsCount * hostPes;
    }

    /**
     * Calculate PEs for medium VM.
     */
    public int getMediumVmPes() {
        return smallVmPes * mediumVmMultiplier;
    }

    /**
     * Calculate PEs for large VM.
     */
    public int getLargeVmPes() {
        return smallVmPes * largeVmMultiplier;
    }

    /**
     * Calculate RAM for medium VM.
     */
    public long getMediumVmRam() {
        return smallVmRam * mediumVmMultiplier;
    }

    /**
     * Calculate RAM for large VM.
     */
    public long getLargeVmRam() {
        return smallVmRam * largeVmMultiplier;
    }

    /**
     * Create a default datacenter configuration for testing.
     */
    public static DatacenterConfig createDefault(int datacenterId) {
        return DatacenterConfig.builder()
                .datacenterId(datacenterId)
                .datacenterName("DC_" + datacenterId)
                .hostsCount(16)
                .hostPes(16)
                .hostPeMips(50000)
                .hostRam(65536)
                .hostBw(50000)
                .hostStorage(100000)
                .smallVmPes(2)
                .smallVmRam(8192)
                .smallVmBw(1000)
                .smallVmStorage(4000)
                .mediumVmMultiplier(2)
                .largeVmMultiplier(4)
                .initialSmallVmCount(10)
                .initialMediumVmCount(5)
                .initialLargeVmCount(3)
                .greenEnergyEnabled(true)
                .turbineId(57 + datacenterId)  // Turbine 57, 58, 59, ...
                .windDataFile("windProduction/sdwpf_2001_2112_full.csv")
                .vmStartupDelay(0.0)
                .vmShutdownDelay(0.0)
                .build();
    }
}
