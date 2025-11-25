package giu.edu.cspg.common;
import giu.edu.cspg.common.DatacenterConfig;
import giu.edu.cspg.energy.TimeScalingMode;

import lombok.Builder;
import lombok.Getter;
import lombok.Singular;
import lombok.ToString;

import java.util.Map;

/**
 * Configuration class for a single datacenter in multi-datacenter setup.
 * Supports heterogeneous host composition via Map<String, Integer>.
 */
@Getter
@Builder
@ToString
public class DatacenterConfig {
    // === Datacenter Identification ===
    private final int datacenterId;
    private final String datacenterName;

    // === Host Configuration (NEW: Heterogeneous support via Map) ===

    /**
     * Host composition: profileName -> count
     * Example: {"MEDIUM": 10, "HIGH_PERFORMANCE": 5, "LOW_POWER": 3}
     * Use @Singular to enable builder.hostProfile("MEDIUM", 10)
     */
    @Singular("hostProfile")
    private final Map<String, Integer> hostProfiles;

    // === Legacy: Single host configuration (deprecated) ===
    @Deprecated
    private final Integer hostsCount;
    @Deprecated
    private final Integer hostPes;             // Processing Elements (cores) per host
    @Deprecated
    private final Long hostPeMips;         // MIPS capacity per PE
    @Deprecated
    private final Long hostRam;            // RAM in MB
    @Deprecated
    private final Long hostBw;             // Bandwidth in Mbps
    @Deprecated
    private final Long hostStorage;        // Storage in MB

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
    private final int turbineId;                    // Wind turbine ID for this datacenter
    private final String windDataFile;              // Path to wind power CSV file
    @Builder.Default
    private final TimeScalingMode timeScalingMode = TimeScalingMode.REAL_TIME;  // Time scaling mode (default: REAL_TIME)

    // === Carbon Emission Factors ===
    /**
     * Carbon emission factor for brown energy (kg CO2 per kWh).
     * Represents the carbon intensity of the grid electricity at this datacenter's location.
     * Typical values:
     * - Coal-heavy grid: ~0.8 kg/kWh
     * - Natural gas grid: ~0.5 kg/kWh
     * - Clean grid (hydro/nuclear): ~0.3 kg/kWh
     */
    @Builder.Default
    private final double brownCarbonFactor = 0.5;  // Default: 0.5 kg CO2/kWh (natural gas grid)

    /**
     * Carbon emission factor for green energy (kg CO2 per kWh).
     * Represents lifecycle emissions from wind turbine manufacturing and maintenance.
     * Typical value: ~0.01 kg/kWh
     */
    @Builder.Default
    private final double greenCarbonFactor = 0.01;  // Default: 0.01 kg CO2/kWh (lifecycle emissions)

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
     * Get total number of hosts across all profiles.
     */
    public int getTotalHostCount() {
        if (hostProfiles != null && !hostProfiles.isEmpty()) {
            return hostProfiles.values().stream()
                    .mapToInt(Integer::intValue)
                    .sum();
        }
        // Fallback to legacy mode
        return hostsCount != null ? hostsCount : 0;
    }

    /**
     * Get total cores (PEs) across all hosts.
     */
    public int getTotalCores() {
        if (hostProfiles != null && !hostProfiles.isEmpty()) {
            return hostProfiles.entrySet().stream()
                    .mapToInt(entry -> {
                        HostProfile profile = HostProfile.fromName(entry.getKey());
                        return profile.getPes() * entry.getValue();
                    })
                    .sum();
        }
        // Fallback to legacy mode
        if (hostsCount != null && hostPes != null) {
            return hostsCount * hostPes;
        }
        return 0;
    }

    /**
     * Validate this configuration.
     */
    public void validate() {
        if (datacenterId < 0) {
            throw new IllegalArgumentException("Datacenter ID cannot be negative");
        }

        // Validate host configurations
        if (hostProfiles != null && !hostProfiles.isEmpty()) {
            // New mode: validate each profile
            for (Map.Entry<String, Integer> entry : hostProfiles.entrySet()) {
                String profileName = entry.getKey();
                Integer count = entry.getValue();

                if (count == null || count < 0) {
                    throw new IllegalArgumentException(
                        "Invalid host count for profile '" + profileName + "': " + count);
                }

                // Validate profile name exists
                try {
                    HostProfile.fromName(profileName);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                        "Unknown host profile in datacenter config: '" + profileName + "'. " + e.getMessage());
                }
            }

            if (getTotalHostCount() == 0) {
                throw new IllegalArgumentException("Total host count cannot be zero");
            }
        } else if (hostsCount == null || hostsCount <= 0) {
            throw new IllegalArgumentException("No valid host configuration provided");
        }
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
     * Create a default datacenter with heterogeneous hosts.
     */
    public static DatacenterConfig createDefaultHeterogeneous(int datacenterId) {
        return DatacenterConfig.builder()
                .datacenterId(datacenterId)
                .datacenterName("DC_" + datacenterId)
                // Define mixed host composition using @Singular
                .hostProfile("MEDIUM", 10)
                .hostProfile("HIGH_PERFORMANCE", 5)
                .hostProfile("LOW_POWER", 3)
                // VM config
                .smallVmPes(2)
                .smallVmRam(8192)
                .smallVmBw(1000)
                .smallVmStorage(4000)
                .mediumVmMultiplier(2)
                .largeVmMultiplier(4)
                .initialSmallVmCount(10)
                .initialMediumVmCount(5)
                .initialLargeVmCount(3)
                // Green energy
                .greenEnergyEnabled(true)
                .turbineId(57 + datacenterId)  // Turbine 57, 58, 59, ...
                .windDataFile("windProduction/sdwpf_2001_2112_full.csv")
                .vmStartupDelay(0.0)
                .vmShutdownDelay(0.0)
                .build();
    }

    /**
     * Create a default datacenter configuration for testing (legacy, homogeneous).
     * @deprecated Use createDefaultHeterogeneous() instead
     */
    @Deprecated
    public static DatacenterConfig createDefault(int datacenterId) {
        return DatacenterConfig.builder()
                .datacenterId(datacenterId)
                .datacenterName("DC_" + datacenterId)
                .hostsCount(16)
                .hostPes(16)
                .hostPeMips(50000L)
                .hostRam(65536L)
                .hostBw(50000L)
                .hostStorage(100000L)
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
