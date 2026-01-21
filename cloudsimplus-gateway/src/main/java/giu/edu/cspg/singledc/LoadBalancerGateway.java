package giu.edu.cspg.singledc;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import lombok.Getter;
import lombok.Setter;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.Gson;

import giu.edu.cspg.Main;
import giu.edu.cspg.common.SimulationSettings;
import giu.edu.cspg.energy.GreenEnergyProvider;
import giu.edu.cspg.utils.SimulationResultUtils;
import py4j.GatewayServer;

@Getter
@Setter
public class LoadBalancerGateway {
    private static final Logger LOGGER = LoggerFactory.getLogger(LoadBalancerGateway.class.getSimpleName());
    private static final DecimalFormat df = new DecimalFormat("#.###");
    private static final Gson gson = new Gson();

    private SimulationCore simulationCore;
    private SimulationSettings settings;
    private String simName;
    private GatewayServer gatewayServer;
    private long currentSeed = 0; // Store seed for resets
    private int currentStep = 0;
    private int maxPotentialVms = 0; // Calculated based on max capacity and Size of the observation arrays for VMs
    private int episodeCounter = 0; // Track episode number for saving results

    // To store reward components calculated in step for SimulationStepInfo
    private double rewardWaitTimeComponent = 0;
    private double rewardUnutilizationComponent = 0;
    private double rewardQueuePenaltyComponent = 0;
    private double rewardInvalidActionComponent = 0;
    private double rewardEnergyComponent = 0;
    private double rewardCompletionComponent = 0;  // NEW: Positive reward for completing tasks

    // Energy tracking
    private double cumulativeEnergyWh = 0;  // Cumulative energy consumption in Watt-hours
    private double previousClock = 0;        // Track previous clock time for energy calculation
    private double currentPowerW = 0;        // Current power consumption in Watts
    private double averageHostUtilization = 0; // Average host CPU utilization
    private double maxTotalPowerW = 0;       // Cached maximum total power of all hosts (calculated in reset)

    // Green Energy Providers (multi-turbine) and tracking
    private final List<GreenEnergyProvider> greenEnergyProviders = new ArrayList<>();
    private double cumulativeGreenEnergyWh = 0;
    private double cumulativeBrownEnergyWh = 0;
    private double totalWastedGreenWh = 0;
    private double currentGreenPowerW = 0;
    private double cumulativeCarbonEmissionKg = 0.0;

    // Wind power predictor (Py4J interface for optional prediction service)
    private Object windPowerPredictor = null;

    // Historical statistics for enhanced observation
    private static final int HISTORY_WINDOW = 10;
    private final List<Integer> completedCloudletsHistory = new ArrayList<>();  // Completed cloudlets per step
    private int previousFinishedCount = 0;  // Track finished cloudlets to calculate delta

    public LoadBalancerGateway() {
        // Settings will be properly initialized by configureSimulation
        LOGGER.info("LoadBalancerGateway created. Waiting for configuration...");
    }

    /**
     * Configures the simulation settings. Called once by Python after
     * initialization.
     *
     * @param params Map containing simulation parameters.
     */
    public void configureSimulation(Map<String, Object> params) {
        Map<String, Object> effectiveParams = normalizeSingleDcParams(params);
        this.settings = new SimulationSettings(effectiveParams);
        this.simName = settings.getSimulationName();
        LOGGER.info("Simulation Name: {}", this.simName);
        LOGGER.info("Simulation settings dump\n{}", settings.printSettings());

        // Clean up previous results for this simulation to avoid accumulation
        cleanupPreviousResults();

        // Reset episode counter for new training run
        this.episodeCounter = 0;

        LOGGER.info("Simulation configured. Waiting for reset request...");
    }

    /**
     * Deletes previous episode results for the current simulation name.
     * This ensures each training run starts with a clean results directory.
     */
    private void cleanupPreviousResults() {
        String resultsPath = String.format("results/%s", settings.getSimulationName());
        Path resultsDirPath = Paths.get(resultsPath);

        if (!Files.exists(resultsDirPath)) {
            LOGGER.info("No previous results found at: {}", resultsPath);
            return;
        }

        try {
            // Delete all episode directories and files under this simulation's results directory
            Files.walk(resultsDirPath)
                .sorted(Comparator.reverseOrder())  // Delete files before directories
                .map(Path::toFile)
                .forEach(File::delete);

            LOGGER.info("Cleaned up previous results at: {}", resultsPath);
        } catch (IOException e) {
            LOGGER.warn("Failed to clean up previous results at {}: {}", resultsPath, e.getMessage());
            LOGGER.warn("Continuing with new training run, but old episodes may remain mixed with new ones.");
        }
    }

    /**
     * Calculates the theoretical maximum number of smallest VMs the infrastructure
     * could hold.
     */
    private int calculateMaxPotentialVms(SimulationSettings settings) {
        if (settings.getMaxVms() > 0) {
            return settings.getMaxVms(); // Use fixed max if set
        }

        if (settings.getHostsCount() <= 0 || settings.getHostPes() <= 0 || settings.getSmallVmPes() <= 0) {
            return 0; // Avoid division by zero
        }
        int totalHostCores = (int) simulationCore.getTotalHostCores();
        int smallestVmCores = settings.getSmallVmPes();
        LOGGER.info("Total host cores: {}, Smallest VM cores: {}", totalHostCores, smallestVmCores);
        // Simple calculation: total host cores / smallest VM cores
        // Could be refined based on RAM/BW if those are more limiting
        // Theoretical max + a buffer (e.g., 10%) in case of fragmentation or many small
        // VMs created
        return (int) Math.ceil((double) (totalHostCores / smallestVmCores) * 1.1);
        // Alternative: Use a large fixed number from config if preferred
        // return settings.getIntParam("max_observation_vms", 100); // Example
    }

    /**
     * Resets the simulation environment. Will create a new simulation.
     * 
     * @param seed Random seed for reproducibility.
     * @return A SimulationResetResult containing the initial state and info.
     */
    public SimulationResetResult reset(long seed) {
        if (settings == null) {
            throw new IllegalStateException("Simulation not configured. Call configureSimulation first.");
        }
        LOGGER.info("Received reset request from Python with seed {}.", seed);
        this.currentSeed = seed; // Store seed if needed later
        this.currentStep = 0;
        this.cumulativeEnergyWh = 0;  // Reset energy tracking
        this.previousClock = 0;        // Reset clock tracking
        this.cumulativeGreenEnergyWh = 0;  // Reset green energy tracking
        this.cumulativeBrownEnergyWh = 0;   // Reset brown energy tracking
        this.totalWastedGreenWh = 0;         // Reset wasted green energy tracking
        this.currentGreenPowerW = 0;         // Reset current green power
        this.cumulativeCarbonEmissionKg = 0.0; // Reset carbon tracking
        this.greenEnergyProviders.clear();
        // resetEpisodeStats(); // Reset any episode-specific stats if added later

        // Create/Reset core components
        if (this.simulationCore != null) {
            // Save results from previous episode BEFORE stopping simulation
            try {
                String episodeResultName = String.format("%s/episode_%d", settings.getSimulationName(), episodeCounter);
                LOGGER.info("Saving results from previous episode as: {}", episodeResultName);
                SimulationResultUtils.printAndSaveResults(this.simulationCore, episodeResultName);

                // Save green energy statistics if enabled
                if (!greenEnergyProviders.isEmpty()) {
                    SimulationResultUtils.saveGreenEnergyStats(getGreenEnergyStats(), episodeResultName);
                }

                episodeCounter++; // Increment episode counter
            } catch (Exception e) {
                LOGGER.error("Failed to save previous episode results: {}", e.getMessage(), e);
            }

            this.simulationCore.stopSimulation(); // Ensure previous run is fully stopped
        }
        this.simulationCore = new SimulationCore(this.settings); // Calls internal reset

        // Calculate actual maximum total power from all hosts (for accurate energy normalization)
        this.maxTotalPowerW = calculateMaxTotalPower();
        LOGGER.info("Calculated max total power: {}W from {} hosts", this.maxTotalPowerW,
                    simulationCore.getDatacenter().getHostList().size());

        // Calculate max potential VMs for padding observation arrays
        this.maxPotentialVms = calculateMaxPotentialVms(settings);
        if (this.maxPotentialVms <= 0) {
            LOGGER.warn(
                    "Calculated maxPotentialVms is <= 0. Observation padding might be incorrect. Check host/vm PE settings.");
            // Set a minimum size to avoid zero-length arrays
            this.maxPotentialVms = Math.max(10,
                    settings.getInitialSVmCount() + settings.getInitialMVmCount() + settings.getInitialLVmCount());
        }

        LOGGER.info("Max potential VMs calculated: {}", this.maxPotentialVms);

        // Initialize Green Energy Provider(s) if enabled
        if (settings.isGreenEnergyEnabled()) {
            try {
                List<Integer> turbineIds = settings.getTurbineIds();
                for (int turbineId : turbineIds) {
                    GreenEnergyProvider provider = new GreenEnergyProvider(
                        turbineId,
                        settings.getWindDataFile(),
                        settings.getTimeScalingMode(),
                        settings.getShortTermRows(),
                        settings.getLongTermRows(),
                        settings.getTimeZoneOffsetRows()
                    );
                    greenEnergyProviders.add(provider);
                }
                LOGGER.info("Green Energy Providers initialized: turbines={}, file={}, mode={}, tzOffsetRows={}",
                    turbineIds, settings.getWindDataFile(),
                    settings.getTimeScalingMode().getDescription(),
                    settings.getTimeZoneOffsetRows());
            } catch (Exception e) {
                LOGGER.error("Failed to initialize Green Energy Provider: {}", e.getMessage(), e);
                this.greenEnergyProviders.clear();
            }
        } else {
            this.greenEnergyProviders.clear();
            LOGGER.info("Green Energy Provider disabled");
        }

        ObservationState initialState = getCurrentState();
        // Info at reset: clock is 0, no actions taken yet
        SimulationStepInfo initialInfo = new SimulationStepInfo(0.0);

        LOGGER.info("Reset complete. Initial State: {}", initialState);
        return new SimulationResetResult(initialState, initialInfo);
    }

    /**
     * Executes one simulation step based on the agent's action.
     * Action format: [target_vm_id]
     * 
     * @param targetVmId the target VM ID for assigning the cloudlet.
     * @return A StepResult object containing the new state, reward, done flags, and
     *         info.
     */
    public SimulationStepResult step(int targetVmId) {
        if (simulationCore == null || settings == null) {
            throw new IllegalStateException("Simulation not initialized/configured. Call reset() first.");
        }

        // --- 0. Early termination check ---
        // If simulation already finished (all cloudlets done), return terminated immediately
        if (!simulationCore.isRunning()) {
            LOGGER.info("Step called but simulation already finished. Returning terminated=true.");
            ObservationState finalState = getCurrentState();
            SimulationStepInfo finalInfo = new SimulationStepInfo(simulationCore.getClock());
            return new SimulationStepResult(finalState, 0.0, true, false, finalInfo);
        }

        currentStep++;
        LOGGER.debug("Step {} starting. Target VM ID: {}", currentStep, targetVmId);

        boolean assignSuccess = false;
        boolean wasInvalidAction = false; // Flag if the action logic itself deems it invalid

        // --- 1. Execute Action ---
        if (targetVmId == -1) {
            // Agent explicitly chose NoOp/NoAssign
            LOGGER.trace("Action: No assignment requested (targetVmId = -1).");
            if (simulationCore.getBroker().hasWaitingCloudlets()) {
                LOGGER.warn("Agent chose NoAssign (-1) but cloudlets are waiting. Considered invalid action.");
                wasInvalidAction = true; // Penalise choosing NoOp when work exists
            }
        }
        else {
            // Assign Cloudlet
            // Agent chose a specific VM ID (0 to N-1)
            if (simulationCore.getBroker().hasWaitingCloudlets()) {
                assignSuccess = simulationCore.getBroker().assignCloudletToVm(targetVmId);
                if (!assignSuccess) {
                    LOGGER.warn(
                            "Assign Cloudlet to VM {} failed (VM likely invalid or full). Invalid action taken.",
                            targetVmId);
                    wasInvalidAction = true; // Treat failed assignment as invalid action
                }
            }
            else {
                LOGGER.warn("Assign Cloudlet action taken, but queue is empty. Invalid action.");
                wasInvalidAction = true; // Invalid action if queue is empty
            }
        }

        // --- 2. Advance Simulation Time ---
        simulationCore.runOneTimestep(); // Run one timestep (default: 1 second)
        double currentClock = simulationCore.getClock();

        // 🔍 DIAGNOSTIC: Log progress every 50 steps to understand why episodes don't finish
        if (currentStep % 50 == 0 || currentStep == 1) {
            int waiting = simulationCore.getBroker() != null ?
                         simulationCore.getBroker().getWaitingCloudletCount() : 0;
            int finished = simulationCore.getBroker() != null ?
                          simulationCore.getBroker().getCloudletFinishedList().size() : 0;
            boolean running = simulationCore.isRunning();

            LOGGER.info("Step {}: Clock={}s, Waiting={}, Finished={}, SimRunning={}",
                       currentStep, String.format("%.2f", currentClock), waiting, finished, running);
        }

        // --- 3. Get New State ---
        ObservationState newState = getCurrentState();

        // --- 4. Calculate Reward ---
        double totalReward = calculateReward(wasInvalidAction);

        // --- 5. Check Termination/Truncation ---
        boolean terminated = !simulationCore.isRunning();
        boolean truncated = !terminated && (currentStep >= settings.getMaxEpisodeLength());
        if (truncated) {
            int waiting = simulationCore.getBroker() != null ?
                         simulationCore.getBroker().getWaitingCloudletCount() : 0;
            int finished = simulationCore.getBroker() != null ?
                          simulationCore.getBroker().getCloudletFinishedList().size() : 0;
            LOGGER.warn("Episode TRUNCATED at step {}, clock={}s, waiting={}, finished={}",
                       currentStep, currentClock, waiting, finished);
        }
        if (terminated) {
            LOGGER.info("Episode TERMINATED naturally at step {}, clock={}s, all_cloudlets_completed",
                       currentStep, currentClock);
        }

        // --- 6. Create Info Object ---
        // Calculate current step green ratio
        double stepGreenRatio = (this.cumulativeEnergyWh > 0)
            ? (this.cumulativeGreenEnergyWh / this.cumulativeEnergyWh)
            : 0.0;

        // Calculate episode statistics (only meaningful at episode end)
        double episodeDuration = 0;
        int episodeCompletedCloudlets = 0;
        int episodeTotalCloudlets = 0;
        double episodeCompletionRate = 0;

        if (terminated || truncated) {
            episodeDuration = currentClock;
            episodeCompletedCloudlets = simulationCore.getBroker() != null
                ? simulationCore.getBroker().getCloudletFinishedList().size()
                : 0;
            episodeTotalCloudlets = simulationCore.getBroker() != null
                ? simulationCore.getBroker().getInputCloudlets().size()
                : 0;
            episodeCompletionRate = episodeTotalCloudlets > 0
                ? (double) episodeCompletedCloudlets / episodeTotalCloudlets
                : 0.0;
        }

        double carbonEmissionKg = computeCarbonEmissionKg();
        double carbonIntensityKgPerKwh = computeCarbonIntensityKgPerKwh();
        double greenGenerationWh = cumulativeGreenEnergyWh + totalWastedGreenWh;
        double episodeDurationHours = currentClock > 0 ? currentClock / 3600.0 : 0.0;
        double avgPowerW = episodeDurationHours > 0 ? cumulativeEnergyWh / episodeDurationHours : currentPowerW;
        double greenPowerW = episodeDurationHours > 0 ? greenGenerationWh / episodeDurationHours : currentGreenPowerW;

        SimulationStepInfo stepInfo = new SimulationStepInfo(
                assignSuccess, false, false,
                false, false, wasInvalidAction,
                -1, 0,
                currentClock,
                this.rewardWaitTimeComponent,
                this.rewardUnutilizationComponent,
                this.rewardQueuePenaltyComponent, this.rewardInvalidActionComponent,
                this.rewardEnergyComponent,
                getInfrastructureObservation(),
                simulationCore.getBroker()
                        .getFinishedWaitTimesLastStep(simulationCore.getClock()),
                this.currentPowerW, this.cumulativeEnergyWh, this.averageHostUtilization,
                avgPowerW, carbonEmissionKg, carbonIntensityKgPerKwh,
                this.cumulativeGreenEnergyWh, this.cumulativeBrownEnergyWh,
                this.totalWastedGreenWh, this.currentGreenPowerW, stepGreenRatio,
                greenGenerationWh, greenPowerW,
                episodeDuration, episodeCompletedCloudlets, episodeTotalCloudlets, episodeCompletionRate);

        LOGGER.debug("Step {} finished. Reward: {}, Term: {}, Trunc: {}, Info: {}", currentStep, totalReward,
                terminated, truncated, stepInfo);

        // --- 7. Return Result ---
        return new SimulationStepResult(newState, totalReward, terminated, truncated, stepInfo);
    }

    /**
     * Executes one simulation step based on the agent's action list.
     * Action format: [action_type, target_vm_id, target_host_id, vm_type_index]
     * 
     * @param actionList Java List of Integers representing the action.
     * @return A StepResult object containing the new state, reward, done flags, and
     *         info.
     */
    public SimulationStepResult step(List<Integer> actionList) {
        if (simulationCore == null || settings == null) {
            throw new IllegalStateException("Simulation not initialized/configured. Call reset() first.");
        }
        if (actionList == null || actionList.size() < 4) {
            LOGGER.error("Invalid action list received: {}", actionList);
            throw new IllegalArgumentException("Action list must have 4 elements.");
        }

        currentStep++;
        LOGGER.debug("Step {} starting. Action: {}", currentStep, actionList);

        // --- 1. Execute Action ---
        int actionType = actionList.get(0);
        int targetVmId = actionList.get(1);
        int targetHostId = actionList.get(2);
        int vmTypeIndex = actionList.get(3);

        boolean assignSuccess = false;
        boolean createAttempted = false;
        boolean createSuccess = false;
        boolean destroyAttempted = false;
        boolean destroySuccess = false;
        boolean wasInvalidAction = false; // Flag if the action logic itself deems it invalid
        int hostAffectedId = -1; // Default: no host affected
        int coresChanged = 0; // Default: no cores changed

        switch (actionType) {
            case 0 -> {
                // No-op
                LOGGER.trace("Action Type 0: No operation.");
                break;
            }
            case 1 -> {
                // Assign Cloudlet
                if (simulationCore.getBroker().hasWaitingCloudlets()) {
                    assignSuccess = simulationCore.getBroker().assignCloudletToVm(targetVmId);
                    if (!assignSuccess) {
                        LOGGER.warn(
                                "Assign Cloudlet to VM {} failed (VM likely invalid or full). Invalid action taken.",
                                targetVmId);
                        wasInvalidAction = true; // Treat failed assignment as invalid action
                    }
                } else {
                    LOGGER.warn("Assign Cloudlet action taken, but queue is empty. Invalid action.");
                    wasInvalidAction = true; // Invalid action if queue is empty
                }
            }
            case 2 -> {
                // Create VM
                createAttempted = true;
                if (vmTypeIndex >= 0 && vmTypeIndex < SimulationSettings.VM_TYPES.length) {
                    String vmType = SimulationSettings.VM_TYPES[vmTypeIndex];
                    // The method returns a boolean if submission failed internally (rare)
                    if (simulationCore.createVmOnHost(vmType, targetHostId)) {
                        createSuccess = true; // Request submitted
                        hostAffectedId = targetHostId; // Host where creation was requested
                        coresChanged = settings.getSmallVmPes() * settings.getSizeMultiplier(vmType); // Cores added
                    } else {
                        // Should ideally not happen if host ID is valid, allocation policy might fail
                        // later
                        LOGGER.error("Core VM creation/submission failed unexpectedly for type {} on host {}.", vmType,
                                targetHostId);
                        wasInvalidAction = true;
                    }
                } else {
                    LOGGER.warn("Invalid VM type index {} in Create VM action.", vmTypeIndex);
                    wasInvalidAction = true;
                }
            }
            case 3 -> {
                // Destroy VM
                destroyAttempted = true;
                if (simulationCore.getBroker().getVmExecList().isEmpty()) {
                    LOGGER.warn("Destroy VM action taken, but no VMs are running. Invalid action.");
                    wasInvalidAction = true;
                }
                // Need to find the VM first to get its details before destroying
                try {
                    Vm vmToDestroy = simulationCore.getBroker().getVmExecList().get(targetVmId);
                    if (vmToDestroy != Vm.NULL && vmToDestroy.isCreated() && !vmToDestroy.isFailed()) {
                        hostAffectedId = (int) vmToDestroy.getHost().getId(); // Get host before destroy request
                        coresChanged = -(int) vmToDestroy.getPesNumber(); // Cores removed (negative)
                        destroySuccess = simulationCore.destroyVmById(targetVmId); // Initiate destruction
                        if (!destroySuccess) { // Should ideally not happen if checks pass
                            LOGGER.warn("Destroy VM {} initiation failed unexpectedly.", targetVmId);
                            wasInvalidAction = true;
                            hostAffectedId = -1; // Reset if initiation failed
                            coresChanged = 0;
                        }
                    } else {
                        LOGGER.warn(
                                "Destroy VM {} failed (VM likely doesn't exist or already stopped). Invalid action taken.",
                                targetVmId);
                        wasInvalidAction = true;
                    }
                } catch (IndexOutOfBoundsException e) {
                    // Handle case where targetVmId is out of bounds for the VM list
                    LOGGER.warn("Destroy VM action taken with invalid VM ID {}. Invalid action taken.", targetVmId);
                    wasInvalidAction = true;
                } catch (Exception e) {
                    LOGGER.error("Unexpected error during Destroy VM action: {}", e.getMessage(), e);
                    wasInvalidAction = true; // Treat as invalid if any unexpected error occurs
                }
            }
            default -> {
                LOGGER.warn("Invalid action type received: {}", actionType);
                wasInvalidAction = true;
            }
        }

        // --- 2. Advance Simulation Time ---
        simulationCore.runOneTimestep(); // Run one timestep (default: 1 second)
        double currentClock = simulationCore.getClock();

        //  DIAGNOSTIC: Log progress every 50 steps to understand why episodes don't finish
        if (currentStep % 50 == 0 || currentStep == 1) {
            int waiting = simulationCore.getBroker() != null ?
                         simulationCore.getBroker().getWaitingCloudletCount() : 0;
            int finished = simulationCore.getBroker() != null ?
                          simulationCore.getBroker().getCloudletFinishedList().size() : 0;
            boolean running = simulationCore.isRunning();

            LOGGER.info("Step {}: Clock={}s, Waiting={}, Finished={}, SimRunning={}",
                       currentStep, String.format("%.2f", currentClock), waiting, finished, running);
        }

        // --- 3. Get New State ---
        ObservationState newState = getCurrentState();

        // --- 4. Calculate Reward ---
        double totalReward = calculateReward(wasInvalidAction);

        // --- 5. Check Termination/Truncation ---
        boolean terminated = !simulationCore.isRunning();
        boolean truncated = !terminated && (currentStep >= settings.getMaxEpisodeLength());
        if (truncated) {
            int waiting = simulationCore.getBroker() != null ?
                         simulationCore.getBroker().getWaitingCloudletCount() : 0;
            int finished = simulationCore.getBroker() != null ?
                          simulationCore.getBroker().getCloudletFinishedList().size() : 0;
            LOGGER.warn("Episode TRUNCATED at step {}, clock={}s, waiting={}, finished={}",
                       currentStep, currentClock, waiting, finished);
        }
        if (terminated) {
            LOGGER.info("Episode TERMINATED naturally at step {}, clock={}s, all_cloudlets_completed",
                       currentStep, currentClock);
        }

        // --- 6. Create Info Object ---
        // Calculate current step green ratio
        double stepGreenRatio = (this.cumulativeEnergyWh > 0)
            ? (this.cumulativeGreenEnergyWh / this.cumulativeEnergyWh)
            : 0.0;

        // Calculate episode statistics (only meaningful at episode end)
        double episodeDuration = 0;
        int episodeCompletedCloudlets = 0;
        int episodeTotalCloudlets = 0;
        double episodeCompletionRate = 0;

        if (terminated || truncated) {
            episodeDuration = currentClock;
            episodeCompletedCloudlets = simulationCore.getBroker() != null
                ? simulationCore.getBroker().getCloudletFinishedList().size()
                : 0;
            episodeTotalCloudlets = simulationCore.getBroker() != null
                ? simulationCore.getBroker().getInputCloudlets().size()
                : 0;
            episodeCompletionRate = episodeTotalCloudlets > 0
                ? (double) episodeCompletedCloudlets / episodeTotalCloudlets
                : 0.0;
        }

        double carbonEmissionKg = computeCarbonEmissionKg();
        double carbonIntensityKgPerKwh = computeCarbonIntensityKgPerKwh();
        double greenGenerationWh = cumulativeGreenEnergyWh + totalWastedGreenWh;
        double episodeDurationHours = currentClock > 0 ? currentClock / 3600.0 : 0.0;
        double avgPowerW = episodeDurationHours > 0 ? cumulativeEnergyWh / episodeDurationHours : currentPowerW;
        double greenPowerW = episodeDurationHours > 0 ? greenGenerationWh / episodeDurationHours : currentGreenPowerW;

        SimulationStepInfo stepInfo = new SimulationStepInfo(
                assignSuccess, createAttempted, createSuccess,
                destroyAttempted, destroySuccess, wasInvalidAction,
                hostAffectedId, coresChanged,
                currentClock,
                this.rewardWaitTimeComponent,
                this.rewardUnutilizationComponent,
                this.rewardQueuePenaltyComponent, this.rewardInvalidActionComponent,
                this.rewardEnergyComponent,
                getInfrastructureObservation(),
                simulationCore.getBroker()
                        .getFinishedWaitTimesLastStep(simulationCore.getClock()),
                this.currentPowerW, this.cumulativeEnergyWh, this.averageHostUtilization,
                avgPowerW, carbonEmissionKg, carbonIntensityKgPerKwh,
                this.cumulativeGreenEnergyWh, this.cumulativeBrownEnergyWh,
                this.totalWastedGreenWh, this.currentGreenPowerW, stepGreenRatio,
                greenGenerationWh, greenPowerW,
                episodeDuration, episodeCompletedCloudlets, episodeTotalCloudlets, episodeCompletionRate);

        LOGGER.debug("Step {} finished. Reward: {}, Term: {}, Trunc: {}, Info: {}", currentStep, totalReward,
                terminated, truncated, stepInfo);

        // --- 7. Return Result ---
        return new SimulationStepResult(newState, totalReward, terminated, truncated, stepInfo);
    }

    /**
     * Calculates the reward for the current state and action outcome.
     * Also stores individual components for the SimulationStepInfo object.
     *
     * @param wasInvalidAction true if the action taken in this step was logically invalid.
     * @return The total calculated reward.
     */
    private double calculateReward(boolean wasInvalidAction) {
        if (simulationCore == null || settings == null)
            return 0.0;

        LoadBalancingBroker broker = simulationCore.getBroker();
        List<Vm> runningVms = simulationCore.getVmPool();

        // --- Components ---

        // 1. Wait Time Penalty (Avg wait of finished cloudlets)
        List<Double> finishedWaitTimes = broker.getFinishedWaitTimesLastStep(simulationCore.getClock());
        double avgFinishedWaitTime = finishedWaitTimes.isEmpty() ? 0
                : finishedWaitTimes.stream().mapToDouble(d -> d).average().orElse(0.0);
        // Scale wait time penalty - e.g., shorter waits are less penalized
        this.rewardWaitTimeComponent = -settings.getRewardWaitTimeCoef() * Math.log1p(avgFinishedWaitTime);

        // 2. Load Balance Penalty (Only penalize variance across VMs)
        // Rationale: In multi-DC setup, local agent cannot control total load amount
        // (that's global agent's decision). It can only control load DISTRIBUTION.
        // So we only penalize imbalance (variance), not deviation from fixed 95% target.
        this.rewardUnutilizationComponent = 0;
        long waitingCount = simulationCore.getNotYetRunningCloudletsCount();
        if (!runningVms.isEmpty() && waitingCount > 0) {
            // Only penalize imbalance when there are tasks waiting
            // (if queue is empty, load balance doesn't matter)
            final double avgUtilization = runningVms.stream()
                    .mapToDouble(Vm::getCpuPercentUtilization)
                    .average().orElse(0.0);

            final double utilizationVariance = runningVms.stream()
                    .mapToDouble(vm -> Math.pow(vm.getCpuPercentUtilization() - avgUtilization, 2))
                    .average().orElse(0.0);

            // Only penalize variance (load imbalance), NOT deviation from fixed target
            this.rewardUnutilizationComponent = -settings.getRewardUnutilizationCoef()
                    * Math.sqrt(utilizationVariance);
        }
        // When queue is empty, no penalty - low utilization is not local agent's fault

        // 3. Queue Length Penalty (Ratio relative to arrived)
        this.rewardQueuePenaltyComponent = -settings.getRewardQueuePenaltyCoef() * getWaitingCloudletsRatio();

        // 4. Invalid Action Penalty
        this.rewardInvalidActionComponent = -settings.getRewardInvalidActionCoef() * (wasInvalidAction ? 1.0 : 0.0);

        // 5. Completion Reward (NEW: positive reward for completing tasks)
        // Use log1p scaling to avoid reward explosion when many tasks complete at once
        // log1p(0) = 0, log1p(1) ≈ 0.69, log1p(5) ≈ 1.79, log1p(10) ≈ 2.40
        List<Cloudlet> completedList = broker.getCloudletsFinishedLastStep(simulationCore.getClock());
        int completedThisStep = completedList.size();
        this.rewardCompletionComponent = settings.getRewardCompletionCoef() * Math.log1p(completedThisStep);

        // Energy tracking (for logging purposes only, not part of reward)
        double currentClock = simulationCore.getClock();
        this.currentPowerW = calculateAndUpdateEnergy(currentClock);
        this.averageHostUtilization = calculateAverageHostUtilization();
        this.rewardEnergyComponent = 0.0;  // Not used in original reward function

        // --- Total Reward ---
        // Include positive completion reward to incentivize task completion.
        double totalReward = this.rewardCompletionComponent +
            this.rewardWaitTimeComponent +
            this.rewardUnutilizationComponent +
            this.rewardQueuePenaltyComponent +
            this.rewardInvalidActionComponent;

        LOGGER.debug("Reward Calc: Completion={}, Wait={}, UtilBal={}, Queue={}, Invalid={}, Total={}",
                String.format("%.3f", this.rewardCompletionComponent),
                String.format("%.3f", this.rewardWaitTimeComponent),
                String.format("%.3f", this.rewardUnutilizationComponent),
                String.format("%.3f", this.rewardQueuePenaltyComponent),
                String.format("%.3f", this.rewardInvalidActionComponent),
                String.format("%.3f", totalReward));

        // IMPORTANT: Clear per-step finished lists so "last step" metrics don't accumulate across the episode.
        // Otherwise completion/wait-time calculations become non-stationary and can distort learning.
        try {
            broker.clearFinishedWaitTimes();
        } catch (Exception ignored) {
            // Non-critical; reward already computed.
        }

        return totalReward;
    }

    /**
     * Calculates the ratio of waiting cloudlets to total arrived cloudlets.
     * Used for queue penalty calculation.
     * 
     * @return The ratio of waiting cloudlets to total arrived cloudlets.
     */
    private double getWaitingCloudletsRatio() {
        final long arrivedCloudletsCount = simulationCore.getArrivedCloudletsCount();

        return arrivedCloudletsCount > 0
                ? (double) simulationCore.getNotYetRunningCloudletsCount() / (double) arrivedCloudletsCount
                : 0.0;
    }

    /**
     * Collects the current simulation state and formats it into an ObservationState
     * object,
     * including padding for dynamic VM lists.
     */
    private ObservationState getCurrentState() {
        if (simulationCore == null || settings == null) {
            LOGGER.warn("Attempting to get state before simulation core/settings is initialized.");
            // Return a default empty/zero state matching the expected dimensions
            int maxHosts = settings != null ? settings.getHostsCount() : 10; // Use default if settings is null somehow
            int maxVms = maxPotentialVms > 0 ? maxPotentialVms : 50; // Use calculated max or default
            return new ObservationState(
                    new double[maxHosts],
                    new double[maxHosts],
                    new double[maxVms],
                    new int[maxVms],
                    new int[maxVms],
                    getInfrastructureObservation(),
                    0,
                    0,
                    new int[maxVms],
                    0,
                    0,
                    0L,
                    0.0,
                    new int[3],
                    0);  // New fields with default values
        }

        // Initialize padded arrays
        int numHosts = settings.getHostsCount();
        double[] hostLoads = new double[numHosts];
        double[] hostRamUsageRatio = new double[numHosts];

        double[] vmLoads = new double[maxPotentialVms]; // Padded (0=off)
        int[] vmAvailablePes = new int[maxPotentialVms]; // Initialize with 0
        int[] vmTypes = new int[maxPotentialVms]; // Padded (0=off)
        int[] vmHostMap = new int[maxPotentialVms]; // Padded (-1=off)
        Arrays.fill(vmHostMap, -1); // Default to -1

        // Populate Host data (assuming host IDs are 0 to numHosts-1)
        List<Host> currentHosts = simulationCore.getDatacenter().getHostList();
        for (int i = 0; i < numHosts; i++) {
            if (i < currentHosts.size()) { // Check bounds
                Host host = currentHosts.get(i);
                if (host != null && host != Host.NULL && host.isActive()) { // Check if host is valid and active
                    hostLoads[i] = host.getCpuPercentUtilization();
                    hostRamUsageRatio[i] = host.getRam().getPercentUtilization();
                } else { // Handle inactive or null hosts if possible in future scenarios
                    hostLoads[i] = 0.0;
                    hostRamUsageRatio[i] = 0.0;
                }
            } else {
                // Should not happen with fixed hosts, but handle defensively
                hostLoads[i] = 0.0;
                hostRamUsageRatio[i] = 0.0;
            }
        }

        // Populate VM data into padded arrays
        List<Vm> currentVms = simulationCore.getVmPool(); // Gets broker's exec list
        int actualVmCount = 0;
        for (Vm vm : currentVms) {
            if (vm != null && vm.isCreated() && !vm.isFailed()) {
                int vmId = (int) vm.getId();
                // Ensure vmId is within the bounds of our observation arrays
                if (vmId >= 0 && vmId < maxPotentialVms) {
                    vmLoads[vmId] = vm.getCpuPercentUtilization();
                    // IMPORTANT: use remaining/free PEs, not total PEs.
                    // This is required for meaningful action masking and scheduling decisions.
                    vmAvailablePes[vmId] = (int) vm.getFreePesNumber();
                    vmTypes[vmId] = vmTypeStringToIndex(vm.getDescription());
                    vmHostMap[vmId] = (int) vm.getHost().getId(); // Store host ID
                    actualVmCount++;
                } else {
                    LOGGER.warn("VM ID {} is out of bounds for observation array size {}. Ignoring VM.", vmId,
                            maxPotentialVms);
                }
            }
            // Ignore VMs that are not created, failed, or null
        }

        // Get Queue state
        int waitingCloudlets = (simulationCore.getBroker() != null)
                ? simulationCore.getBroker().getWaitingCloudletCount()
                : 0;
        Cloudlet nextCloudlet = (simulationCore.getBroker() != null) ? simulationCore.getBroker().peekWaitingCloudlet()
                : null;
        int nextCloudletPes = (int) ((nextCloudlet != null) ? nextCloudlet.getPesNumber() : 0);

        // ===== Enhanced Cloudlet Information =====
        // 1. Next cloudlet MI
        long nextCloudletMi = (nextCloudlet != null) ? nextCloudlet.getLength() : 0;

        // 2. Next cloudlet wait time (how long it has been waiting)
        double nextCloudletWaitTime = 0.0;
        if (nextCloudlet != null && simulationCore.getBroker() != null) {
            Double arrivalTime = simulationCore.getBroker().getCloudletArrivalTimeMap().get(nextCloudlet.getId());
            if (arrivalTime != null) {
                nextCloudletWaitTime = simulationCore.getClock() - arrivalTime;
            }
        }

        // 3. Queue PEs distribution [small (1-2 PEs), medium (3-4 PEs), large (5+ PEs)]
        int[] queuePesDistribution = new int[3]; // [small, medium, large]
        if (simulationCore.getBroker() != null) {
            List<Cloudlet> waitingQueue = simulationCore.getBroker().getCloudletWaitingList();
            for (Cloudlet cloudlet : waitingQueue) {
                int pes = (int) cloudlet.getPesNumber();
                if (pes <= 2) {
                    queuePesDistribution[0]++;  // small
                } else if (pes <= 4) {
                    queuePesDistribution[1]++;  // medium
                } else {
                    queuePesDistribution[2]++;  // large
                }
            }
        }

        // ===== Historical Statistics =====
        // 4. Track completed cloudlets in this step
        int currentFinishedCount = (simulationCore.getBroker() != null)
                ? simulationCore.getBroker().getCloudletFinishedList().size()
                : 0;
        int completedThisStep = currentFinishedCount - previousFinishedCount;
        previousFinishedCount = currentFinishedCount;

        completedCloudletsHistory.add(completedThisStep);
        if (completedCloudletsHistory.size() > HISTORY_WINDOW) {
            completedCloudletsHistory.remove(0);  // Circular buffer
        }

        // Calculate total completed in last N steps
        int completedCloudletsLast10Steps = 0;
        for (int count : completedCloudletsHistory) {
            completedCloudletsLast10Steps += count;
        }

        return new ObservationState(
                hostLoads, hostRamUsageRatio, vmLoads, vmTypes, vmHostMap, getInfrastructureObservation(),
                waitingCloudlets, nextCloudletPes, vmAvailablePes, actualVmCount, numHosts,
                nextCloudletMi, nextCloudletWaitTime, queuePesDistribution,
                completedCloudletsLast10Steps);
    }

    /** Helper to convert VM type string (S, M, L) to integer index (1, 2, 3). */
    private int vmTypeStringToIndex(String type) {
        // Handle potential temporary description like "S-host5"
        String actualType = type.contains("-") ? type.substring(0, type.indexOf('-')) : type;
        return switch (actualType) {
            case SimulationSettings.SMALL -> 1;
            case SimulationSettings.MEDIUM -> 2;
            case SimulationSettings.LARGE -> 3;
            default -> {
                LOGGER.warn("Unrecognized VM type in description: '{}'. Mapping to 0 (Off).", type);
                yield 0; // Map unknown/null to 0
            }
        };
    }

    /**
     * Gets a string representation of the current simulation state for rendering.
     */
    public String getRenderInfo() {
        if (simulationCore == null) {
            return "Simulation not initialized.";
        }
        ObservationState state = getCurrentState();
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Time: %s | Step: %d\n",
                df.format(simulationCore.getClock()), currentStep));
        sb.append(String.format("Hosts (%d): ", state.getActualHostCount()));
        for (int i = 0; i < state.getActualHostCount(); i++) {
            sb.append(String.format("H%d[CPU:%.1f%% RAM:%.1f%%] ",
                    i,
                    state.getHostLoads()[i] * 100,
                    state.getHostRamUsageRatio()[i] * 100));
        }
        sb.append("\n");
        sb.append(String.format("VMs (%d / %d potential): ",
                state.getActualVmCount(), maxPotentialVms));
        for (int i = 0; i < maxPotentialVms; i++) {
            if (state.getVmTypes()[i] > 0) {
                String type = switch (state.getVmTypes()[i]) {
                    case 1 -> "S";
                    case 2 -> "M";
                    case 3 -> "L";
                    default -> "?";
                };
                sb.append(String.format("V%d(%s@H%d)[CPU:%.1f%%] ",
                        i,
                        type,
                        state.getVmHostMap()[i],
                        state.getVmLoads()[i] * 100));
            }
        }
        if (state.getActualVmCount() == 0) {
            sb.append("(None)");
        }
        sb.append("\n");
        sb.append(String.format("Queue: %d waiting | Next PEs: %d\n",
                state.getWaitingCloudlets(),
                state.getNextCloudletPes()));
        sb.append("Infrastructure tree:\n");
        appendInfrastructureTree(sb, state.getInfrastructureObservation());
        sb.append("\n");
        sb.append("--------------------\n");

        return sb.toString();
    }

    /**
     * Collects current state information suitable for rendering and returns it as a
     * JSON string.
     * 
     * @return JSON string representing the current renderable state.
     */
    public String getRenderInfoAsJson() {
        if (simulationCore == null) {
            // Return empty JSON object or error message
            return "{}"; // Or: "{\"error\": \"Simulation not initialized.\"}"
        }
        try {
            ObservationState state = getCurrentState();
            // Create a Map to hold the data for JSON conversion
            Map<String, Object> renderData = new HashMap<>();
            renderData.put("time", simulationCore.getClock());
            renderData.put("step", this.currentStep);
            renderData.put("actual_hosts", state.getActualHostCount());
            renderData.put("actual_vms", state.getActualVmCount());
            renderData.put("waiting_cloudlets", state.getWaitingCloudlets());
            renderData.put("next_cloudlet_pes", state.getNextCloudletPes());

            // Add host data (only for actual hosts)
            List<Map<String, Object>> hostInfoList = new ArrayList<>();
            for (int i = 0; i < state.getActualHostCount(); i++) {
                Map<String, Object> hostData = new HashMap<>();
                hostData.put("id", i);
                hostData.put("cpu_load_percent", state.getHostLoads()[i]);
                hostData.put("ram_usage_ratio", state.getHostRamUsageRatio()[i]);
                hostInfoList.add(hostData);
            }
            renderData.put("hosts", hostInfoList);

            // Add VM data (only for active VMs, using ID as key or in a list)
            List<Map<String, Object>> vmInfoList = new ArrayList<>();
            for (int i = 0; i < maxPotentialVms; i++) { // Iterate through all potential slots
                if (state.getVmTypes()[i] > 0) { // If VM exists in this slot
                    Map<String, Object> vmData = new HashMap<>();
                    vmData.put("id", i); // Use the index as the VM ID
                    vmData.put("type_code", state.getVmTypes()[i]); // 1=S, 2=M, 3=L
                    vmData.put("host_id", state.getVmHostMap()[i]);
                    vmData.put("cpu_load_percent", state.getVmLoads()[i]);
                    vmInfoList.add(vmData);
                }
            }
            renderData.put("vms", vmInfoList);

            // Add Tree Array if desired
            renderData.put("infrastructure_tree", state.getInfrastructureObservation());

            return gson.toJson(renderData); // Convert map to JSON

        } catch (Exception e) {
            LOGGER.error("Error generating render info JSON", e);
            return "{\"error\": \"Failed to generate render info: " + e.getMessage().replace("\"", "'") + "\"}";
        }
    }

    /**
     * Reads the flat infrastructureObservation array and appends
     * a human-friendly, indented tree to the StringBuilder.
     */
    private void appendInfrastructureTree(StringBuilder sb, int[] obs) {
        int idx = 0;
        int totalCores = obs[idx++];
        int hostsNum = obs[idx++];
        sb.append(String.format("  Total cores: %d\n", totalCores));
        sb.append(String.format("  Hosts: %d\n", hostsNum));

        for (int h = 0; h < hostsNum; h++) {
            int hostPes = obs[idx++];
            int vmCount = obs[idx++];
            sb.append(String.format("    Host[%d]: PEs=%d  VMs=%d\n",
                    h, hostPes, vmCount));

            for (int v = 0; v < vmCount; v++) {
                int vmPes = obs[idx++];
                int cloudletCount = obs[idx++];
                sb.append(String.format("      VM[%d]: PEs=%d  Cloudlets=%d\n",
                        v, vmPes, cloudletCount));

                for (int j = 0; j < cloudletCount; j++) {
                    int cloudletPes = obs[idx++];
                    idx++; // skip the “0” child-count
                    sb.append(String.format("        Cloudlet[%d]: PEs=%d\n",
                            j, cloudletPes));
                }
            }
        }
    }

    private int[] getInfrastructureObservation() {
        // 1) Fetch hosts
        List<Host> hosts = simulationCore.getDatacenter().getHostList();
        int hostsNum = hosts.size();

        // 2) Use a dynamic list
        List<Integer> treeList = new ArrayList<>();

        // 3) Header: total cores & number of hosts
        treeList.add((int) simulationCore.getTotalHostCores());
        treeList.add(hostsNum);

        // 4) For each host, record its cores & VM count, then each VM, then each
        // cloudlet
        for (Host host : hosts) {
            List<Vm> vmList = host.getVmList();
            treeList.add((int) host.getPesNumber());
            treeList.add(vmList.size());

            for (Vm vm : vmList) {
                if (vm != null && vm.isCreated() && !vm.isFailed()) {
                    List<Cloudlet> cloudletList = vm.getCloudletScheduler().getCloudletList();
                    treeList.add((int) vm.getPesNumber());
                    treeList.add(cloudletList.size());

                    for (Cloudlet cloudlet : cloudletList) {
                        treeList.add((int) cloudlet.getPesNumber());
                        treeList.add(0); // cloudlets have no children
                    }
                }
            }
        }

        // 5) Convert to primitive int[]
        return treeList.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Allows Python to cleanly shut down the simulation and gateway. */
    public void close() {
        LOGGER.info("Received close request from Python.");
        if (simulationCore != null) {
            // Print results before stopping (save final episode)
            String finalEpisodeResultName = String.format("%s/episode_%d", settings.getSimulationName(), episodeCounter);
            SimulationResultUtils.printAndSaveResults(simulationCore, finalEpisodeResultName);

            // Save green energy statistics if enabled
            if (!greenEnergyProviders.isEmpty()) {
                SimulationResultUtils.saveGreenEnergyStats(getGreenEnergyStats(), finalEpisodeResultName);
            }

            simulationCore.stopSimulation();
        }
        // Trigger JVM shutdown
        if (gatewayServer != null) {
            LOGGER.info("Initiating gateway shutdown via Main.initiateShutdown.");
            Main.initiateShutdown(this.gatewayServer);
        }
    }

    /**
     * Calculates current power consumption and updates cumulative energy.
     * Integrates green energy allocation if enabled.
     *
     * @param currentClock Current simulation time in seconds
     * @return Current total power consumption in Watts
     */
    private double calculateAndUpdateEnergy(double currentClock) {
        double currentPowerW = 0;
        double timeDelta = currentClock - previousClock;
        double timeDeltaHours = timeDelta / 3600.0;

        var datacenter = simulationCore.getDatacenter();
        if (datacenter == null) {
            return 0;
        }

        // Calculate total power demand from all hosts
        var hostList = datacenter.getHostList();
        for (var host : hostList) {
            if (host.getPowerModel() != null) {
                double utilization = host.getCpuPercentUtilization();
                double power = host.getPowerModel().getPower(utilization);
                currentPowerW += power;
            }
        }

        // Update cumulative energy
        if (timeDeltaHours > 0) {
            if (!greenEnergyProviders.isEmpty()) {
                // Multi-turbine: aggregate available green power and allocate energy
                double availableGreenPowerW = 0.0;
                for (GreenEnergyProvider p : greenEnergyProviders) {
                    availableGreenPowerW += p.getCurrentPowerW(currentClock);
                }

                double demandWh = currentPowerW * timeDeltaHours;
                double greenAvailableWh = availableGreenPowerW * timeDeltaHours;

                double greenUsedWh = Math.min(demandWh, greenAvailableWh);
                double brownUsedWh = demandWh - greenUsedWh;
                double wastedGreenWh = greenAvailableWh - greenUsedWh;

                // Update cumulative statistics
                cumulativeGreenEnergyWh += greenUsedWh;
                cumulativeBrownEnergyWh += brownUsedWh;
                totalWastedGreenWh += wastedGreenWh;
                currentGreenPowerW = availableGreenPowerW;

                // Total cumulative energy (green + brown)
                cumulativeEnergyWh += (greenUsedWh + brownUsedWh);

                // Update provider stats proportionally by their instantaneous power share
                if (availableGreenPowerW > 0) {
                    for (GreenEnergyProvider p : greenEnergyProviders) {
                        double share = p.getCurrentPowerW(currentClock) / availableGreenPowerW;
                        p.updateCumulativeStatistics(greenUsedWh * share, wastedGreenWh * share);
                    }
                }

                // Update carbon (kg): green_kWh*green_factor + brown_kWh*brown_factor
                double deltaGreenKwh = greenUsedWh / 1000.0;
                double deltaBrownKwh = brownUsedWh / 1000.0;
                cumulativeCarbonEmissionKg += deltaGreenKwh * settings.getGreenCarbonFactor()
                                           + deltaBrownKwh * settings.getBrownCarbonFactor();
            } else {
                // Traditional brown energy only
                cumulativeEnergyWh += currentPowerW * timeDeltaHours;
                cumulativeBrownEnergyWh += currentPowerW * timeDeltaHours;

                // Carbon: treat all as brown
                cumulativeCarbonEmissionKg += (currentPowerW * timeDeltaHours / 1000.0) * settings.getBrownCarbonFactor();
            }
        }

        previousClock = currentClock;
        return currentPowerW;
    }

    /**
     * Calculates average host CPU utilization across all hosts.
     *
     * @return Average utilization (0.0 to 1.0)
     */
    private double calculateAverageHostUtilization() {
        var datacenter = simulationCore.getDatacenter();
        if (datacenter == null) {
            return 0;
        }

        var hostList = datacenter.getHostList();
        if (hostList.isEmpty()) {
            return 0;
        }

        double totalUtilization = 0;
        for (var host : hostList) {
            totalUtilization += host.getCpuPercentUtilization();
        }

        return totalUtilization / hostList.size();
    }

    /**
     * Calculates the actual maximum total power consumption of all hosts.
     * This accounts for heterogeneous hosts with different power characteristics.
     * Called once during reset() and cached for accurate energy normalization.
     *
     * @return Maximum total power in Watts when all hosts are at 100% utilization
     */
    private double calculateMaxTotalPower() {
        var datacenter = simulationCore.getDatacenter();
        if (datacenter == null) {
            LOGGER.warn("Datacenter is null, cannot calculate max power. Using fallback.");
            return settings.getHostsCount() * 250.0; // Fallback to old behavior
        }

        var hostList = datacenter.getHostList();
        if (hostList.isEmpty()) {
            LOGGER.warn("Host list is empty, cannot calculate max power. Using fallback.");
            return settings.getHostsCount() * 250.0; // Fallback
        }

        double totalMaxPower = 0;
        for (var host : hostList) {
            if (host.getPowerModel() != null) {
                // Get max power at 100% utilization
                double maxPower = host.getPowerModel().getPower(1.0);
                totalMaxPower += maxPower;
                LOGGER.debug("Host {}: max power = {}W", host.getId(), maxPower);
            } else {
                LOGGER.warn("Host {} has no power model, using default 250W", host.getId());
                totalMaxPower += 250.0; // Default fallback for hosts without power model
            }
        }

        LOGGER.info("Total maximum power across {} hosts: {}W", hostList.size(), totalMaxPower);
        return totalMaxPower;
    }

    private double computeCarbonEmissionKg() {
        // Prefer per-source factors aligned with multi-DC configs.
        // This is cumulative over the episode.
        if (cumulativeCarbonEmissionKg > 0) {
            return cumulativeCarbonEmissionKg;
        }
        // Backward compatibility fallback (older single-DC configs only had a single factor for brown).
        double carbonFactor = settings.getCarbonEmissionFactorKgPerKwh();
        return (this.cumulativeBrownEnergyWh / 1000.0) * carbonFactor;
    }

    private double computeCarbonIntensityKgPerKwh() {
        double totalEnergyKwh = this.cumulativeEnergyWh / 1000.0;
        if (totalEnergyKwh <= 0) {
            return 0.0;
        }
        double carbonKg = computeCarbonEmissionKg();
        return carbonKg / totalEnergyKwh;
    }

    /**
     * Register a wind power predictor service (via Py4J).
     * This allows the GreenEnergyProvider to call the prediction model for future power forecasting.
     *
     * @param predictor Python predictor object accessible via Py4J
     */
    public void registerWindPowerPredictor(Object predictor) {
        this.windPowerPredictor = predictor;
        LOGGER.info("Wind power predictor registered: {}", predictor != null ? "enabled" : "disabled");
    }

    /**
     * Get green energy statistics.
     * Returns cumulative green/brown energy consumption and waste.
     *
     * @return Map containing green energy statistics
     */
    public Map<String, Double> getGreenEnergyStats() {
        Map<String, Double> stats = new HashMap<>();
        stats.put("cumulative_green_wh", cumulativeGreenEnergyWh);
        stats.put("cumulative_brown_wh", cumulativeBrownEnergyWh);
        stats.put("total_wasted_green_wh", totalWastedGreenWh);
        stats.put("current_green_power_w", currentGreenPowerW);
        stats.put("green_ratio", cumulativeEnergyWh > 0 ? cumulativeGreenEnergyWh / cumulativeEnergyWh : 0.0);
        stats.put("overall_green_ratio",
            (cumulativeGreenEnergyWh + cumulativeBrownEnergyWh) > 0
                ? cumulativeGreenEnergyWh / (cumulativeGreenEnergyWh + cumulativeBrownEnergyWh)
                : 0.0);
        stats.put("carbon_emission_kg", computeCarbonEmissionKg());

        return stats;
    }

    /**
     * Normalize parameters for Single-DC mode to support multi-DC-style `datacenters:` configs.
     *
     * If `datacenters` exists, we take the first datacenter entry and map its fields into:
     * - Single-DC flat keys (hosts_count, host_count_spec_*, initial_*_vm_count, etc.)
     * - Single-DC `green_energy` map (enabled, turbine_ids, wind_data_file, time_scaling_mode, tz offset, carbon factors)
     *
     * This enables reusing multi-DC datacenter blocks in single-DC experiments without duplication.
     */
    @SuppressWarnings("unchecked")
    private Map<String, Object> normalizeSingleDcParams(Map<String, Object> params) {
        if (params == null) {
            return Map.of();
        }

        Map<String, Object> out = new HashMap<>(params);

        Object dcListObj = params.get("datacenters");
        if (!(dcListObj instanceof List<?> dcList) || dcList.isEmpty() || !(dcList.get(0) instanceof Map)) {
            return out;
        }

        Map<String, Object> dc0 = (Map<String, Object>) dcList.get(0);
        LOGGER.info("Single-DC: detected multi-DC-style `datacenters:` config. Using datacenters[0]={}",
            dc0.getOrDefault("name", "DC_0"));

        // Copy common infrastructure and VM keys if present in dc0
        copyIfPresent(dc0, out, "hosts_count", "host_pes", "host_pe_mips", "host_ram", "host_bw", "host_storage",
            "small_vm_pes", "small_vm_ram", "small_vm_bw", "small_vm_storage",
            "medium_vm_multiplier", "large_vm_multiplier",
            "initial_s_vm_count", "initial_m_vm_count", "initial_l_vm_count",
            "vm_startup_delay", "vm_shutdown_delay");

        // Copy any heterogeneous host composition keys (host_count_*)
        boolean anyHostCount = false;
        for (Map.Entry<String, Object> e : dc0.entrySet()) {
            String k = e.getKey();
            if (k != null && k.startsWith("host_count_")) {
                out.put(k, e.getValue());
                anyHostCount = true;
            }
        }
        if (anyHostCount) {
            out.put("enable_heterogeneous_hosts", true);
        }

        // Build / merge green_energy map from datacenter block
        Map<String, Object> existingGe = Map.of();
        Object existingGeObj = out.get("green_energy");
        if (existingGeObj instanceof Map) {
            existingGe = (Map<String, Object>) existingGeObj;
        }

        Map<String, Object> ge = new HashMap<>();
        // Preserve prediction config if present (single-DC uses this shape)
        Object predObj = existingGe.get("prediction");
        if (predObj instanceof Map) {
            ge.put("prediction", predObj);
        }

        ge.put("enabled", getBoolean(dc0.get("green_energy_enabled"), false));
        if (dc0.containsKey("turbine_ids")) {
            ge.put("turbine_ids", dc0.get("turbine_ids"));
        }
        if (dc0.containsKey("turbine_id")) {
            ge.put("turbine_id", dc0.get("turbine_id"));
        } else if (dc0.get("turbine_ids") instanceof List<?> ids && !ids.isEmpty()) {
            ge.put("turbine_id", ids.get(0));
        }
        if (dc0.containsKey("wind_data_file")) {
            ge.put("wind_data_file", dc0.get("wind_data_file"));
        }
        if (dc0.containsKey("time_scaling_mode")) {
            ge.put("time_scaling_mode", dc0.get("time_scaling_mode"));
        }
        if (dc0.containsKey("time_zone_offset_rows")) {
            ge.put("time_zone_offset_rows", dc0.get("time_zone_offset_rows"));
        }
        if (dc0.containsKey("brown_carbon_factor")) {
            ge.put("brown_carbon_factor", dc0.get("brown_carbon_factor"));
        }
        if (dc0.containsKey("green_carbon_factor")) {
            ge.put("green_carbon_factor", dc0.get("green_carbon_factor"));
        }

        // Future forecast rows: map dc-level keys into green_energy.future_energy_forecast
        Map<String, Object> forecast = new HashMap<>();
        if (dc0.containsKey("short_term_rows")) {
            forecast.put("short_term_rows", dc0.get("short_term_rows"));
        }
        if (dc0.containsKey("long_term_rows")) {
            forecast.put("long_term_rows", dc0.get("long_term_rows"));
        }
        if (!forecast.isEmpty()) {
            ge.put("future_energy_forecast", forecast);
        }

        out.put("green_energy", ge);
        return out;
    }

    private void copyIfPresent(Map<String, Object> src, Map<String, Object> dst, String... keys) {
        for (String k : keys) {
            if (src.containsKey(k)) {
                dst.put(k, src.get(k));
            }
        }
    }

    private boolean getBoolean(Object value, boolean defaultValue) {
        if (value == null) return defaultValue;
        if (value instanceof Boolean b) return b;
        return Boolean.parseBoolean(value.toString());
    }
}
