package giu.edu.cspg;

import java.util.Map;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Holds simulation configuration parameters, loaded from a Map
 * (typically originating from a YAML config file via Python).
 */
public class SimulationSettings {
    private static final Logger LOGGER = LoggerFactory.getLogger(SimulationSettings.class.getSimpleName());

    public final String simulationName;

    // --- Constants for VM Types ---
    public static final String SMALL = "S";
    public static final String MEDIUM = "M";
    public static final String LARGE = "L";
    public static final String[] VM_TYPES = { SMALL, MEDIUM, LARGE };

    // --- Parameter Fields ---
    private final int hostsCount;
    private final int hostPes;
    private final long hostPeMips;
    private final long hostRam;
    private final long hostBw;
    private final long hostStorage;

    private final int smallVmPes;
    private final long smallVmRam;
    private final long smallVmBw;
    private final long smallVmStorage;
    private final int mediumVmMultiplier;
    private final int largeVmMultiplier;

    private final int initialSVmCount;
    private final int initialMVmCount;
    private final int initialLVmCount;
    private final int[] initialVmCounts;
    private final int maxVms;

    private final String workloadMode; // "SWF" or "CSV"
    private final String cloudletTraceFile; // Path for trace file
    private final int maxCloudletsToCreateFromWorkloadFile; // Limit for SWF mode
    private final int workloadReaderMips; // MIPS ref for SWF runtime calculation
    private final boolean splitLargeCloudlets;
    private final int maxCloudletPes;

    private final double simulationTimestep; // RL agent step interval
    private final double minTimeBetweenEvents; // CloudSim internal granularity
    private final boolean clearCreatedLists; // Clear created lists after each episode

    private final double vmStartupDelay; // Time for a VM to become available after creation request
    private final double vmShutdownDelay; // Time before broker actually destroys an idle VM

    private final double smallVmHourlyCost; // Base cost for billing
    private final boolean payingForTheFullHour; // Billing model

    private final int maxEpisodeLength; // For truncation

    // Reward Weights
    private final double rewardWaitTimeCoef;
    private final double rewardThroughputCoef;
    private final double rewardUnutilizationCoef;
    private final double rewardCostCoef;
    private final double rewardQueuePenaltyCoef;
    private final double rewardAssignmentCoef;
    private final double rewardInvalidActionCoef;

    /**
     * Constructor that populates settings from a Map, providing defaults.
     * 
     * @param params Map typically loaded from config.yml via Python.
     */
    public SimulationSettings(Map<String, Object> params) {
        LOGGER.info("Loading Simulation Settings from parameters...");

        // Simulation Name
        this.simulationName = getStringParam(params, "simulation_name", "DefaultSimulationName");

        // Host Configuration
        this.hostsCount = getIntParam(params, "hosts_count", 10);
        this.hostPes = getIntParam(params, "host_pes", 16);
        this.hostPeMips = getLongParam(params, "host_pe_mips", 2000);
        this.hostRam = getLongParam(params, "host_ram", 65536); // 64 GB
        this.hostBw = getLongParam(params, "host_bw", 10000); // 10 Gbps
        this.hostStorage = getLongParam(params, "host_storage", 1000000); // 1 TB

        // Base (Small) VM Configuration
        this.smallVmPes = getIntParam(params, "small_vm_pes", 2); // e.g., AWS m5a.large
        this.smallVmRam = getLongParam(params, "small_vm_ram", 8192); // 8 GB
        this.smallVmBw = getLongParam(params, "small_vm_bw", 1000); // 1 Gbps - adjust as needed
        this.smallVmStorage = getLongParam(params, "small_vm_storage", 20000); // 20 GB

        // VM Size Multipliers
        this.mediumVmMultiplier = getIntParam(params, "medium_vm_multiplier", 2); // -> 4 PEs
        this.largeVmMultiplier = getIntParam(params, "large_vm_multiplier", 4); // -> 8 PEs

        // Initial VM Fleet
        this.initialSVmCount = getIntParam(params, "initial_s_vm_count", 2);
        this.initialMVmCount = getIntParam(params, "initial_m_vm_count", 1);
        this.initialLVmCount = getIntParam(params, "initial_l_vm_count", 1);
        this.initialVmCounts = new int[] { this.initialSVmCount, this.initialMVmCount, this.initialLVmCount };
        this.maxVms = this.initialSVmCount + this.initialMVmCount + this.initialLVmCount;

        // Workload Configuration
        this.workloadMode = getStringParam(params, "workload_mode", "SWF");
        this.cloudletTraceFile = getStringParam(params, "cloudlet_trace_file",
                "traces/LLNL-Atlas-2006-2.1-cln-test.swf");
        this.maxCloudletsToCreateFromWorkloadFile = getIntParam(params, "max_cloudlets_to_create_from_workload_file",
                Integer.MAX_VALUE);
        this.workloadReaderMips = getIntParam(params, "workload_reader_mips", (int) this.hostPeMips);

        this.splitLargeCloudlets = getBoolParam(params, "split_large_cloudlets", true);
        // Default maxCloudletPes to the largest VM's PE count if not specified
        int defaultMaxCloudletPes = this.smallVmPes * this.largeVmMultiplier;
        this.maxCloudletPes = getIntParam(params, "max_cloudlet_pes", defaultMaxCloudletPes);

        // Simulation Control
        this.simulationTimestep = getDoubleParam(params, "simulation_timestep", 1.0); // e.g., 1 second RL step
        this.minTimeBetweenEvents = getDoubleParam(params, "min_time_between_events", 0.1);
        this.clearCreatedLists = getBoolParam(params, "clear_created_lists", true); // Clear created lists after each
                                                                                    // episode

        // VM Control
        // assuming average startup delay is 56s as in 10.48550/arXiv.2107.03467
        this.vmStartupDelay = getDoubleParam(params, "vm_startup_delay", 56.0);
        this.vmShutdownDelay = getDoubleParam(params, "vm_shutdown_delay", 10.0);

        // Costing
        this.smallVmHourlyCost = getDoubleParam(params, "small_vm_hourly_cost", 0.086);
        this.payingForTheFullHour = getBoolParam(params, "paying_for_the_full_hour", false);

        // RL Control
        this.maxEpisodeLength = getIntParam(params, "max_episode_length", 1000); // Timesteps before truncation

        // Reward Weights
        this.rewardWaitTimeCoef = getDoubleParam(params, "reward_wait_time_coef", 0.1);
        this.rewardThroughputCoef = getDoubleParam(params, "reward_throughput_coef", 0.1);
        this.rewardUnutilizationCoef = getDoubleParam(params, "reward_unutilization_coef", 0.85);
        this.rewardCostCoef = getDoubleParam(params, "reward_cost_coef", 0.5);
        this.rewardQueuePenaltyCoef = getDoubleParam(params, "reward_queue_penalty_coef", 0.05);
        this.rewardAssignmentCoef = getDoubleParam(params, "reward_assignment_coef", 0.05);
        this.rewardInvalidActionCoef = getDoubleParam(params, "reward_invalid_action_coef", 1.0);

        LOGGER.info("SimulationSettings loaded successfully.");
    }

    public String printSettings() {
        return """
                SimulationSettings {
                hostsCount=""" + hostsCount + ",\n" +
                "hostPes=" + hostPes + ",\n" +
                "hostPeMips=" + hostPeMips + ",\n" +
                "hostRam=" + hostRam + ",\n" +
                "hostBw=" + hostBw + ",\n" +
                "hostStorage=" + hostStorage + ",\n" +
                "smallVmPes=" + smallVmPes + ",\n" +
                "smallVmRam=" + smallVmRam + ",\n" +
                "smallVmBw=" + smallVmBw + ",\n" +
                "smallVmStorage=" + smallVmStorage + ",\n" +
                "mediumVmMultiplier=" + mediumVmMultiplier + ",\n" +
                "largeVmMultiplier=" + largeVmMultiplier + ",\n" +
                "initialSVmCount=" + initialSVmCount + ",\n" +
                "initialMVmCount=" + initialMVmCount + ",\n" +
                "initialLVmCount=" + initialLVmCount + ",\n" +
                "maxVms=" + maxVms + ",\n" +
                "workloadMode='" + workloadMode + '\'' + ",\n" +
                "cloudletTraceFile='" + cloudletTraceFile + '\'' + ",\n" +
                "maxCloudletsToCreateFromWorkloadFile=" + maxCloudletsToCreateFromWorkloadFile + ",\n" +
                "workloadReaderMips=" + workloadReaderMips + ",\n" +
                "splitLargeCloudlets=" + splitLargeCloudlets + ",\n" +
                "maxCloudletPes=" + maxCloudletPes + ",\n" +
                "simulationTimestep=" + simulationTimestep + ",\n" +
                "minTimeBetweenEvents=" + minTimeBetweenEvents + ",\n" +
                "vmStartupDelay=" + vmStartupDelay + ",\n" +
                "vmShutdownDelay=" + vmShutdownDelay + ",\n" +
                "smallVmHourlyCost=" + smallVmHourlyCost + ",\n" +
                "payingForTheFullHour=" + payingForTheFullHour + ",\n" +
                "maxEpisodeLength=" + maxEpisodeLength + ",\n" +
                "rewardWaitTimeCoef=" + rewardWaitTimeCoef + ",\n" +
                "rewardThroughputCoef=" + rewardThroughputCoef + ",\n" +
                "rewardUnutilizationCoef=" + rewardUnutilizationCoef + ",\n" +
                "rewardCostCoef=" + rewardCostCoef + ",\n" +
                "rewardQueuePenaltyCoef=" + rewardQueuePenaltyCoef + ",\n" +
                "rewardAssignmentCoef=" + rewardAssignmentCoef + ",\n" +
                "rewardInvalidActionCoef=" + rewardInvalidActionCoef + "\n" +
                "}";
    }

    // --- Helper methods for safe parameter extraction ---

    private String getStringParam(Map<String, Object> params, String key, String defaultValue) {
        return Objects.toString(params.getOrDefault(key, defaultValue), defaultValue);
    }

    private int getIntParam(Map<String, Object> params, String key, int defaultValue) {
        Object value = params.get(key);
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return Integer.parseInt(Objects.toString(value, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            LOGGER.warn("Could not parse int for key '{}', using default: {}", key, defaultValue);
            return defaultValue;
        }
    }

    private long getLongParam(Map<String, Object> params, String key, long defaultValue) {
        Object value = params.get(key);
        if (value instanceof Number number) {
            return number.longValue();
        }
        try {
            return Long.parseLong(Objects.toString(value, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            LOGGER.warn("Could not parse long for key '{}', using default: {}", key, defaultValue);
            return defaultValue;
        }
    }

    private double getDoubleParam(Map<String, Object> params, String key, double defaultValue) {
        Object value = params.get(key);
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        try {
            return Double.parseDouble(Objects.toString(value, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            LOGGER.warn("Could not parse double for key '{}', using default: {}", key, defaultValue);
            return defaultValue;
        }
    }

    private boolean getBoolParam(Map<String, Object> params, String key, boolean defaultValue) {
        Object value = params.get(key);
        if (value instanceof Boolean aBoolean) {
            return aBoolean;
        }
        return Boolean.parseBoolean(Objects.toString(value, String.valueOf(defaultValue)));
    }

    // --- Getters for all parameters ---

    public String getSimulationName() {
        return simulationName;
    }

    public int getHostsCount() {
        return hostsCount;
    }

    public int getHostPes() {
        return hostPes;
    }

    public long getHostPeMips() {
        return hostPeMips;
    }

    public long getHostRam() {
        return hostRam;
    }

    public long getHostBw() {
        return hostBw;
    }

    public long getHostStorage() {
        return hostStorage;
    }

    public int getSmallVmPes() {
        return smallVmPes;
    }

    public long getSmallVmRam() {
        return smallVmRam;
    }

    public long getSmallVmBw() {
        return smallVmBw;
    }

    public long getSmallVmStorage() {
        return smallVmStorage;
    }

    public int getMediumVmMultiplier() {
        return mediumVmMultiplier;
    }

    public int getLargeVmMultiplier() {
        return largeVmMultiplier;
    }

    public int getInitialSVmCount() {
        return initialSVmCount;
    }

    public int getInitialMVmCount() {
        return initialMVmCount;
    }

    public int getInitialLVmCount() {
        return initialLVmCount;
    }

    public int[] getInitialVmCounts() {
        return initialVmCounts.clone();
    } // Return copy

    public String getWorkloadMode() {
        return workloadMode;
    }

    public String getCloudletTraceFile() {
        return cloudletTraceFile;
    }

    public int getWorkloadReaderMips() {
        return workloadReaderMips;
    }

    public int getMaxCloudletsToCreateFromWorkloadFile() {
        return maxCloudletsToCreateFromWorkloadFile;
    }

    public boolean isSplitLargeCloudlets() {
        return splitLargeCloudlets;
    }

    public int getMaxCloudletPes() {
        return maxCloudletPes;
    }

    public double getSimulationTimestep() {
        return simulationTimestep;
    }

    public double getMinTimeBetweenEvents() {
        return minTimeBetweenEvents;
    }

    public double getVmStartupDelay() {
        return vmStartupDelay;
    }

    public double getVmShutdownDelay() {
        return vmShutdownDelay;
    }

    public double getSmallVmHourlyCost() {
        return smallVmHourlyCost;
    }

    public boolean isPayingForTheFullHour() {
        return payingForTheFullHour;
    }

    public int getMaxEpisodeLength() {
        return maxEpisodeLength;
    }

    public double getRewardWaitTimeCoef() {
        return rewardWaitTimeCoef;
    }

    public double getRewardThroughputCoef() {
        return rewardThroughputCoef;
    }

    public double getRewardUnutilizationCoef() {
        return rewardUnutilizationCoef;
    }

    public double getRewardCostCoef() {
        return rewardCostCoef;
    }

    public double getRewardQueuePenaltyCoef() {
        return rewardQueuePenaltyCoef;
    }

    public double getRewardInvalidActionCoef() {
        return rewardInvalidActionCoef;
    }

    public boolean isClearCreatedLists() {
        return clearCreatedLists;
    }

    public long getTotalHostCores() {
        return hostsCount * hostPes;
    }

    public int getMaxVms() {
        return maxVms;
    }

    /**
     * Gets the PE multiplier for a given VM type string (S, M, L).
     * 
     * @param type VM type ("S", "M", or "L")
     * @return The multiplier (1, 2, or 4)
     * @throws IllegalArgumentException if type is invalid
     */
    public int getSizeMultiplier(final String type) {
        return switch (type) {
            case LARGE -> largeVmMultiplier; // AWS m5a.2xlarge
            case MEDIUM -> mediumVmMultiplier; // AWS m5a.xlarge
            case SMALL -> 1; // AWS m5a.large
            default -> {
                LOGGER.error("Invalid VM type requested for multiplier: {}", type);
                throw new IllegalArgumentException("Unexpected VM type: " + type);
            }
        };
    }
}
