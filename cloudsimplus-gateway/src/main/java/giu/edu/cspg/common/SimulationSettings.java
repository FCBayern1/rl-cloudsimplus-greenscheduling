package giu.edu.cspg.common;
import giu.edu.cspg.common.SimulationSettings;

import java.util.Map;
import java.util.Objects;

import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Holds simulation configuration parameters, loaded from a Map
 * (typically originating from a YAML config file via Python).
 */
@Getter
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

    // Heterogeneous Hosts Configuration
    private final boolean enableHeterogeneousHosts;
    private final int hostCountLowPower;
    private final int hostCountMedium;
    private final int hostCountHighPerf;
    private final int hostCountUltra;

    // SPEC power_ssj2008 Real Server Profiles
    private final int hostCountSpecAcerR520;      // Legacy inefficient (8 cores, 57.6% idle)
    private final int hostCountSpecAcerAR360;     // Medium older generation (16 cores, 22.0% idle)
    private final int hostCountSpecAsusRS720E9;   // Modern efficient (56 cores, 12.5% idle)
    private final int hostCountSpecAsusRS500A;    // Large AMD EPYC (64 cores, 24.0% idle)
    private final int hostCountSpecAsusRS700A;    // Ultra dual-socket (128 cores, 24.7% idle)

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
    private final int globalRoutingBatchSize; // Number of cloudlets to route per step in batch mode

    // Reward Weights
    private final double rewardWaitTimeCoef;
    private final double rewardThroughputCoef;
    private final double rewardUnutilizationCoef;
    private final double rewardCostCoef;
    private final double rewardQueuePenaltyCoef;
    private final double rewardAssignmentCoef;
    private final double rewardInvalidActionCoef;
    private final double rewardEnergyCoef;
    private final double greenWastePenaltyCoef;  // Base coefficient for green waste penalty (scaled by DC count)
    private final double carbonEmissionPenaltyCoef;  // Coefficient for carbon emission penalty

    // Green Energy Configuration
    private final boolean greenEnergyEnabled;
    private final int turbineId;
    private final String windDataFile;
    private final boolean greenPredictionEnabled;
    private final String predictionModelPath;
    private final double predictionCacheDuration;
    private final int predictionHorizon;

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

        // Heterogeneous Hosts Configuration (based on SPEC power_ssj2008 real servers)
        this.enableHeterogeneousHosts = getBoolParam(params, "enable_heterogeneous_hosts", false);
        this.hostCountLowPower = getIntParam(params, "host_count_low_power", 0);
        this.hostCountMedium = getIntParam(params, "host_count_medium", 0);
        this.hostCountHighPerf = getIntParam(params, "host_count_high_perf", 0);
        this.hostCountUltra = getIntParam(params, "host_count_ultra", 0);

        // SPEC power_ssj2008 Real Server Profiles Configuration
        this.hostCountSpecAcerR520 = getIntParam(params, "host_count_spec_acer_r520", 0);
        this.hostCountSpecAcerAR360 = getIntParam(params, "host_count_spec_acer_ar360", 0);
        this.hostCountSpecAsusRS720E9 = getIntParam(params, "host_count_spec_asus_rs720_e9", 0);
        this.hostCountSpecAsusRS500A = getIntParam(params, "host_count_spec_asus_rs500a", 0);
        this.hostCountSpecAsusRS700A = getIntParam(params, "host_count_spec_asus_rs700a", 0);

        // If heterogeneous hosts enabled, validate that counts match total
        if (this.enableHeterogeneousHosts) {
            int heterogeneousTotal = this.hostCountLowPower + this.hostCountMedium +
                                      this.hostCountHighPerf + this.hostCountUltra +
                                      this.hostCountSpecAcerR520 + this.hostCountSpecAcerAR360 +
                                      this.hostCountSpecAsusRS720E9 + this.hostCountSpecAsusRS500A +
                                      this.hostCountSpecAsusRS700A;
            if (heterogeneousTotal > 0 && heterogeneousTotal != this.hostsCount) {
                LOGGER.warn("Heterogeneous host counts ({}) don't match hosts_count ({}). Using heterogeneous total.",
                    heterogeneousTotal, this.hostsCount);
            }
            LOGGER.info("Heterogeneous Hosts: LowPower={}, Medium={}, HighPerf={}, Ultra={}",
                this.hostCountLowPower, this.hostCountMedium, this.hostCountHighPerf, this.hostCountUltra);
            if (this.hostCountSpecAcerR520 + this.hostCountSpecAcerAR360 +
                this.hostCountSpecAsusRS720E9 + this.hostCountSpecAsusRS500A +
                this.hostCountSpecAsusRS700A > 0) {
                LOGGER.info("SPEC Servers: AcerR520={}, AcerAR360={}, AsusRS720={}, AsusRS500A={}, AsusRS700A={}",
                    this.hostCountSpecAcerR520, this.hostCountSpecAcerAR360,
                    this.hostCountSpecAsusRS720E9, this.hostCountSpecAsusRS500A,
                    this.hostCountSpecAsusRS700A);
            }
        }

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
        this.globalRoutingBatchSize = getIntParam(params, "global_routing_batch_size", 5); // Batch size for routing

        // Reward Weights
        this.rewardWaitTimeCoef = getDoubleParam(params, "reward_wait_time_coef", 0.1);
        this.rewardThroughputCoef = getDoubleParam(params, "reward_throughput_coef", 0.1);
        this.rewardUnutilizationCoef = getDoubleParam(params, "reward_unutilization_coef", 0.85);
        this.rewardCostCoef = getDoubleParam(params, "reward_cost_coef", 0.5);
        this.rewardQueuePenaltyCoef = getDoubleParam(params, "reward_queue_penalty_coef", 0.05);
        this.rewardAssignmentCoef = getDoubleParam(params, "reward_assignment_coef", 0.05);
        this.rewardInvalidActionCoef = getDoubleParam(params, "reward_invalid_action_coef", 1.0);
        this.rewardEnergyCoef = getDoubleParam(params, "reward_energy_coef", 0.0); // Default 0 = disabled
        this.greenWastePenaltyCoef = getDoubleParam(params, "green_waste_penalty_coef", 10.0); // Base coef per DC
        this.carbonEmissionPenaltyCoef = getDoubleParam(params, "carbon_emission_penalty_coef", 1.0); // Default 1.0, adjust based on data

        // Green Energy Configuration
        @SuppressWarnings("unchecked")
        Map<String, Object> greenEnergyConfig = (Map<String, Object>) params.getOrDefault("green_energy", Map.of());
        this.greenEnergyEnabled = getBoolParam(greenEnergyConfig, "enabled", false);
        this.turbineId = getIntParam(greenEnergyConfig, "turbine_id", 1);
        this.windDataFile = getStringParam(greenEnergyConfig, "wind_data_file",
            "windProduction/sdwpf_2001_2112_full.csv");

        // Prediction sub-configuration
        @SuppressWarnings("unchecked")
        Map<String, Object> predictionConfig = (Map<String, Object>) greenEnergyConfig.getOrDefault("prediction", Map.of());
        this.greenPredictionEnabled = getBoolParam(predictionConfig, "enabled", false);
        this.predictionModelPath = getStringParam(predictionConfig, "model_path", "");
        this.predictionCacheDuration = getDoubleParam(predictionConfig, "cache_duration_seconds", 600.0);
        this.predictionHorizon = getIntParam(predictionConfig, "horizon", 8);

        if (this.greenEnergyEnabled) {
            LOGGER.info("Green Energy: enabled, turbine_id={}, data_file={}",
                this.turbineId, this.windDataFile);
            if (this.greenPredictionEnabled) {
                LOGGER.info("Green Prediction: enabled, horizon={}, cache={}s, model={}",
                    this.predictionHorizon, this.predictionCacheDuration,
                    this.predictionModelPath.isEmpty() ? "(not specified)" : this.predictionModelPath);
            } else {
                LOGGER.info("Green Prediction: disabled");
            }
        } else {
            LOGGER.info("Green Energy: disabled");
        }

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
                "rewardInvalidActionCoef=" + rewardInvalidActionCoef + ",\n" +
                "rewardEnergyCoef=" + rewardEnergyCoef + ",\n" +
                "greenEnergyEnabled=" + greenEnergyEnabled + ",\n" +
                "turbineId=" + turbineId + ",\n" +
                "windDataFile='" + windDataFile + '\'' + ",\n" +
                "greenPredictionEnabled=" + greenPredictionEnabled + ",\n" +
                "predictionModelPath='" + predictionModelPath + '\'' + ",\n" +
                "predictionCacheDuration=" + predictionCacheDuration + ",\n" +
                "predictionHorizon=" + predictionHorizon + "\n" +
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

    public int[] getInitialVmCounts() {
        return initialVmCounts.clone();
    } // Return copy

    public long getTotalHostCores() {
        return hostsCount * hostPes;
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

    /**
     * Gets the base green waste penalty coefficient (per datacenter).
     * This will be scaled by the number of datacenters in the simulation.
     *
     * @return Base coefficient for green waste penalty
     */
    public double getGreenWastePenaltyCoef() {
        return greenWastePenaltyCoef;
    }

    /**
     * Gets the carbon emission penalty coefficient.
     * Penalty = coefficient × total carbon emissions (kg CO2).
     *
     * @return Coefficient for carbon emission penalty
     */
    public double getCarbonEmissionPenaltyCoef() {
        return carbonEmissionPenaltyCoef;
    }
}
