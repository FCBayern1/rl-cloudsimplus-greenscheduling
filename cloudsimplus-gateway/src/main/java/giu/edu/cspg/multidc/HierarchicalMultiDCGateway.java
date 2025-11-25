package giu.edu.cspg.multidc;
import giu.edu.cspg.common.DatacenterConfig;
import giu.edu.cspg.energy.TimeScalingMode;
import giu.edu.cspg.singledc.ObservationState;
import giu.edu.cspg.common.SimulationSettings;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import py4j.GatewayServer;

/**
 * Py4J Gateway for Hierarchical Multi-Datacenter RL Environment.
 *
 * This class serves as the entry point for Python code to interact with
 * the multi-datacenter CloudSim Plus simulation.
 *
 * Main responsibilities:
 * - Configure and initialize multi-datacenter simulation
 * - Execute hierarchical steps (global routing + local scheduling)
 * - Provide observations for global and local agents
 * - Calculate and return rewards
 */
public class HierarchicalMultiDCGateway {
    private static final Logger LOGGER = LoggerFactory.getLogger(HierarchicalMultiDCGateway.class.getSimpleName());
    private static final HierarchicalMultiDCGateway INSTANCE = new HierarchicalMultiDCGateway();

    private MultiDatacenterSimulationCore simulationCore;
    private SimulationSettings settings;
    private List<DatacenterConfig> datacenterConfigs;

    private int currentStep = 0;
    private boolean configured = false;

    public HierarchicalMultiDCGateway() {
        LOGGER.info("HierarchicalMultiDCGateway created");
    }

    /**
     * Singleton accessor used by MainMultiDC/py4j bootstrap.
     */
    public static HierarchicalMultiDCGateway getInstance() {
        return INSTANCE;
    }

    /**
     * Configure the simulation with parameters from Python.
     *
     * @param params Configuration parameters (converted from Python dict)
     */
    public void configureSimulation(Map<String, Object> params) {
        LOGGER.info("Configuring multi-datacenter simulation...");

        // Parse general settings
        this.settings = new SimulationSettings(params);

        // Parse datacenter configurations
        this.datacenterConfigs = parseDatacenterConfigs(params);

        LOGGER.info("Configuration complete: {} datacenters", datacenterConfigs.size());
        configured = true;
    }

    /**
     * Parse datacenter configurations from parameters.
     */
    @SuppressWarnings("unchecked")
    private List<DatacenterConfig> parseDatacenterConfigs(Map<String, Object> params) {
        List<DatacenterConfig> configs = new ArrayList<>();

        // Check if multi_datacenter is enabled
        Object multiDcObj = params.get("multi_datacenter_enabled");
        boolean enabled = multiDcObj != null && Boolean.parseBoolean(multiDcObj.toString());

        if (!enabled) {
            // Single datacenter mode - create one config from common params
            DatacenterConfig config = DatacenterConfig.builder()
                    .datacenterId(0)
                    .datacenterName("DC_0")
                    .hostsCount(getIntParam(params, "hosts_count", 16))
                    .hostPes(getIntParam(params, "host_pes", 16))
                    .hostPeMips(getLongParam(params, "host_pe_mips", 50000))
                    .hostRam(getLongParam(params, "host_ram", 65536))
                    .hostBw(getLongParam(params, "host_bw", 50000))
                    .hostStorage(getLongParam(params, "host_storage", 100000))
                    .smallVmPes(getIntParam(params, "small_vm_pes", 2))
                    .smallVmRam(getLongParam(params, "small_vm_ram", 8192))
                    .smallVmBw(getLongParam(params, "small_vm_bw", 1000))
                    .smallVmStorage(getLongParam(params, "small_vm_storage", 4000))
                    .mediumVmMultiplier(getIntParam(params, "medium_vm_multiplier", 2))
                    .largeVmMultiplier(getIntParam(params, "large_vm_multiplier", 4))
                    .initialSmallVmCount(getIntParam(params, "initial_s_vm_count", 10))
                    .initialMediumVmCount(getIntParam(params, "initial_m_vm_count", 5))
                    .initialLargeVmCount(getIntParam(params, "initial_l_vm_count", 3))
                    .greenEnergyEnabled(getBooleanParam(params, "green_energy_enabled", true))
                    .turbineId(getIntParam(params, "turbine_id", 57))
                    .windDataFile(getStringParam(params, "wind_data_file",
                            "windProduction/sdwpf_2001_2112_full.csv"))
                    .vmStartupDelay(getDoubleParam(params, "vm_startup_delay", 0.0))
                    .vmShutdownDelay(getDoubleParam(params, "vm_shutdown_delay", 0.0))
                    .build();
            configs.add(config);
            return configs;
        }

        // Multi-datacenter mode - parse list of DC configs
        Object dcListObj = params.get("datacenters");
        if (dcListObj instanceof List) {
            List<Map<String, Object>> dcList = (List<Map<String, Object>>) dcListObj;
            for (Map<String, Object> dcParams : dcList) {
                DatacenterConfig config = parseDatacenterConfig(dcParams);
                configs.add(config);
            }
        }

        return configs;
    }

    /**
     * Parse a single datacenter configuration.
     */
    private DatacenterConfig parseDatacenterConfig(Map<String, Object> dcParams) {
        return DatacenterConfig.builder()
                .datacenterId(getIntParam(dcParams, "datacenter_id", 0))
                .datacenterName(getStringParam(dcParams, "name", "DC_0"))
                .hostsCount(getIntParam(dcParams, "hosts_count", 16))
                .hostPes(getIntParam(dcParams, "host_pes", 16))
                .hostPeMips(getLongParam(dcParams, "host_pe_mips", 50000))
                .hostRam(getLongParam(dcParams, "host_ram", 65536))
                .hostBw(getLongParam(dcParams, "host_bw", 50000))
                .hostStorage(getLongParam(dcParams, "host_storage", 100000))
                .smallVmPes(getIntParam(dcParams, "small_vm_pes", 2))
                .smallVmRam(getLongParam(dcParams, "small_vm_ram", 8192))
                .smallVmBw(getLongParam(dcParams, "small_vm_bw", 1000))
                .smallVmStorage(getLongParam(dcParams, "small_vm_storage", 4000))
                .mediumVmMultiplier(getIntParam(dcParams, "medium_vm_multiplier", 2))
                .largeVmMultiplier(getIntParam(dcParams, "large_vm_multiplier", 4))
                .initialSmallVmCount(getIntParam(dcParams, "initial_s_vm_count", 10))
                .initialMediumVmCount(getIntParam(dcParams, "initial_m_vm_count", 5))
                .initialLargeVmCount(getIntParam(dcParams, "initial_l_vm_count", 3))
                .greenEnergyEnabled(getBooleanParam(dcParams, "green_energy_enabled", true))
                .turbineId(getIntParam(dcParams, "turbine_id", 57))
                .windDataFile(getStringParam(dcParams, "wind_data_file",
                        "windProduction/sdwpf_2001_2112_full.csv"))
                .timeScalingMode(parseTimeScalingMode(
                        getStringParam(dcParams, "time_scaling_mode", "REAL_TIME")))
                .vmStartupDelay(getDoubleParam(dcParams, "vm_startup_delay", 0.0))
                .vmShutdownDelay(getDoubleParam(dcParams, "vm_shutdown_delay", 0.0))
                .build();
    }

    /**
     * Reset the simulation environment.
     * Following Gymnasium convention, reset() returns only observations and info,
     * without rewards or termination flags.
     *
     * @param seed Random seed
     * @return Initial observation state
     */
    public HierarchicalResetResult reset(int seed) {
        if (!configured) {
            throw new IllegalStateException("Simulation not configured. Call configureSimulation() first.");
        }

        LOGGER.info("Resetting simulation with seed: {}", seed);
        currentStep = 0;

        // Create simulation core if not exists
        if (simulationCore == null) {
            simulationCore = new MultiDatacenterSimulationCore(settings, datacenterConfigs);
        }

        // Reset simulation
        simulationCore.resetSimulation();

        // Return initial observation
        GlobalObservationState globalObs = simulationCore.getGlobalObservation();
        Map<Integer, ObservationState> localObs = simulationCore.getLocalObservations();

        // Create info dict with initial metadata
        Map<String, Object> info = new HashMap<>();
        info.put("num_datacenters", datacenterConfigs.size());
        info.put("seed", seed);
        info.put("episode_start", true);

        return new HierarchicalResetResult(globalObs, localObs, info);
    }

    /**
     * Execute one hierarchical simulation step.
     *
     * @param globalActions List of datacenter IDs for arriving cloudlets
     * @param localActionsMap Map of DC_ID -> VM_ID
     * @return Step result with observations, rewards, etc.
     */
    public HierarchicalStepResult step(List<Integer> globalActions, Map<Integer, Integer> localActionsMap) {
        if (simulationCore == null) {
            throw new IllegalStateException("Simulation not initialized. Call reset() first.");
        }

        currentStep++;
        return simulationCore.executeHierarchicalStep(globalActions, localActionsMap);
    }

    /**
     * Get the number of cloudlets arriving in the current time window.
     * Python needs this to know how many global actions to generate.
     * @deprecated Use getGlobalWaitingCloudletsCount() instead for batch routing mode.
     *
     * @return Number of arriving cloudlets
     */
    @Deprecated
    public int getArrivingCloudletsCount() {
        if (simulationCore == null) {
            return 0;
        }
        List<org.cloudsimplus.cloudlets.Cloudlet> arriving =
                simulationCore.getGlobalBroker().getArrivingCloudlets(
                        simulationCore.getCurrentClock(),
                        simulationCore.getTimestepSize()
                );
        return arriving.size();
    }

    /**
     * Get the number of cloudlets currently in the global waiting queue.
     * This is the total number of cloudlets available for routing.
     *
     * @return Number of cloudlets in global waiting queue
     */
    public int getGlobalWaitingCloudletsCount() {
        if (simulationCore == null) {
            return 0;
        }
        return simulationCore.getGlobalBroker().getGlobalWaitingCloudletsCount();
    }

    /**
     * Get global observation.
     */
    public GlobalObservationState getGlobalObservation() {
        if (simulationCore == null) {
            // Return empty observation
            return GlobalObservationState.createEmpty(1);
        }
        return simulationCore.getGlobalObservation();
    }

    /**
     * Get all local observations.
     */
    public Map<Integer, ObservationState> getLocalObservations() {
        if (simulationCore == null) {
            return new HashMap<>();
        }
        return simulationCore.getLocalObservations();
    }

    /**
     * Close the simulation.
     */
    public void close() {
        LOGGER.info("Closing HierarchicalMultiDCGateway");
        simulationCore = null;
        configured = false;
    }

    // === Helper methods for parameter parsing ===

    private int getIntParam(Map<String, Object> map, String key, int defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        return value instanceof Integer ? (Integer) value : Integer.parseInt(value.toString());
    }

    private long getLongParam(Map<String, Object> map, String key, long defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        return value instanceof Long ? (Long) value : Long.parseLong(value.toString());
    }

    private double getDoubleParam(Map<String, Object> map, String key, double defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        return value instanceof Double ? (Double) value : Double.parseDouble(value.toString());
    }

    private boolean getBooleanParam(Map<String, Object> map, String key, boolean defaultValue) {
        Object value = map.get(key);
        if (value == null) return defaultValue;
        return value instanceof Boolean ? (Boolean) value : Boolean.parseBoolean(value.toString());
    }

    private String getStringParam(Map<String, Object> map, String key, String defaultValue) {
        Object value = map.get(key);
        return value != null ? value.toString() : defaultValue;
    }

    /**
     * Parse TimeScalingMode from string value.
     *
     * @param modeStr Mode string ("REAL_TIME" or "COMPRESSED")
     * @return TimeScalingMode enum value
     */
    private TimeScalingMode parseTimeScalingMode(String modeStr) {
        if (modeStr == null || modeStr.isEmpty()) {
            return TimeScalingMode.REAL_TIME;  // Default
        }

        String upperMode = modeStr.trim().toUpperCase();
        switch (upperMode) {
            case "COMPRESSED":
                return TimeScalingMode.COMPRESSED;
            case "REAL_TIME":
                return TimeScalingMode.REAL_TIME;
            default:
                LOGGER.warn("Unknown time_scaling_mode '{}', defaulting to REAL_TIME", modeStr);
                return TimeScalingMode.REAL_TIME;
        }
    }

    // === Main method to start Py4J server ===

    public static void main(String[] args) {
        HierarchicalMultiDCGateway gateway = HierarchicalMultiDCGateway.getInstance();
        GatewayServer server = new GatewayServer(gateway);
        server.start();
        LOGGER.info("Hierarchical MultiDC Gateway Server started on port 25333");
        System.out.println("Gateway Server Ready");
    }
}
