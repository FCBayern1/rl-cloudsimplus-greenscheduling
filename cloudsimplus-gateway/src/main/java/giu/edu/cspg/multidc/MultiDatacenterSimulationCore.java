package giu.edu.cspg.multidc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.common.CloudletDescriptor;
import giu.edu.cspg.common.DatacenterConfig;
import giu.edu.cspg.common.DatacenterSetup;
import giu.edu.cspg.common.SimulationSettings;
import giu.edu.cspg.common.VmAllocationPolicyCustom;
import giu.edu.cspg.energy.GreenEnergyProvider;
import giu.edu.cspg.singledc.LoadBalancingBroker;
import giu.edu.cspg.singledc.ObservationState;
import giu.edu.cspg.tables.CloudletsTableBuilderWithDetails;
import giu.edu.cspg.utils.WorkloadFileReader;
import lombok.Getter;

/**
 * Multi-Datacenter Simulation Core for Hierarchical MARL.
 *
 * Manages multiple datacenter instances with independent configurations:
 * - Each datacenter has its own hosts, VMs, and green energy provider
 * - GlobalBroker routes arriving cloudlets to datacenters
 * - LocalBrokers (within each DC) schedule cloudlets to VMs
 *
 * Design: Hierarchical two-level decision making
 * - Level 1 (Global): Cloudlet routing to datacenters
 * - Level 2 (Local): Cloudlet scheduling to VMs within each DC
 */
@Getter
public class MultiDatacenterSimulationCore {
    private static final Logger LOGGER = LoggerFactory.getLogger(MultiDatacenterSimulationCore.class.getSimpleName());

    // === Configuration ===
    private final SimulationSettings settings;
    private final List<DatacenterConfig> datacenterConfigs;

    // === CloudSim Plus Core ===
    private CloudSimPlus simulation;
    private GlobalBroker globalBroker;

    // === Datacenter Instances ===
    private List<DatacenterInstance> datacenterInstances;

    // === Workload ===
    private List<Cloudlet> allCloudlets;

    // === Simulation State ===
    private double currentClock = 0.0;
    private double timestepSize;
    private int currentStep = 0;
    private boolean firstStep = true;
    private boolean firstReset = true;  // Track if this is the first reset

    /**
     * Initialise the multi-datacenter simulation core.
     *
     * @param settings General simulation settings
     * @param datacenterConfigs List of configurations for each datacenter
     */
    public MultiDatacenterSimulationCore(SimulationSettings settings,
                                        List<DatacenterConfig> datacenterConfigs) {
        if (settings == null || datacenterConfigs == null || datacenterConfigs.isEmpty()) {
            throw new IllegalArgumentException("Settings and datacenter configs cannot be null or empty");
        }

        this.settings = settings;
        this.datacenterConfigs = new ArrayList<>(datacenterConfigs);
        this.timestepSize = settings.getSimulationTimestep();
        this.datacenterInstances = new ArrayList<>();

        LOGGER.info("Initializing MultiDatacenterSimulationCore with {} datacenters",
                datacenterConfigs.size());
    }

    /**
     * Reset and initialize the simulation environment.
     * Creates all datacenters, loads workload, and initializes brokers.
     */
    public void resetSimulation() {
        LOGGER.info("Resetting multi-datacenter simulation environment...");

        // Print cloudlet execution summary from previous episode (skip first reset)
        if (!firstReset) {
            printCloudletExecutionSummary();
        }
        firstReset = false;

        stopSimulation();

        // Reset VM ID counter to ensure VM IDs start from 0 in each episode
        DatacenterSetup.resetVmIdCounter();

        // Initialize CloudSim Plus engine
        simulation = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        currentClock = 0.0;
        currentStep = 0;
        firstStep = true;

        // === Step 1: Load Cloudlet Workload ===
        allCloudlets = loadAllCloudlets();
        LOGGER.info("Loaded {} cloudlets from workload trace", allCloudlets.size());

        // === Step 2: Create Datacenter Instances including datacenter, hosts, vms, and localbroker ===
        datacenterInstances = new ArrayList<>();
        for (DatacenterConfig config : datacenterConfigs) {
            DatacenterInstance dcInstance = createDatacenterInstance(config);
            datacenterInstances.add(dcInstance);
            LOGGER.info("Created datacenter: {} (ID: {})", config.getDatacenterName(), config.getDatacenterId());
        }

        // === Step 3: Create Global Broker ===
        globalBroker = new GlobalBroker(simulation, allCloudlets, datacenterInstances);
        LOGGER.info("GlobalBroker created with {} cloudlets", allCloudlets.size());

        // === Step 4: Ensure all cloudlets complete ===
        ensureAllCloudletsCompleteBeforeSimulationEnds();
        LOGGER.info("Configured event listener to ensure all cloudlets complete");

        // === Step 5: Start Simulation (sync mode) ===
        simulation.startSync();

        // === Step 6: Initialize simulation by proceeding clock ===
        proceedClockTo(settings.getMinTimeBetweenEvents());
        LOGGER.info("Simulation clock initialized to {}", currentClock);

        LOGGER.info("Multi-datacenter simulation initialized successfully");
    }

    /**
     * Load all cloudlets from the workload trace.
     */
    private List<Cloudlet> loadAllCloudlets() {
        // Load cloudlet descriptors using WorkloadFileReader
        List<CloudletDescriptor> descriptors = loadCloudletDescriptors();
        LOGGER.info("Loaded {} cloudlet descriptors from trace", descriptors.size());

        // Optionally split large cloudlets
        if (settings.isSplitLargeCloudlets()) {
            LOGGER.info("Splitting large cloudlets (maxCloudletPes = {})", settings.getMaxCloudletPes());
            descriptors = splitLargeCloudletDescriptors(descriptors, settings.getMaxCloudletPes());
            LOGGER.info("Total descriptors after splitting: {}", descriptors.size());
        }

        // Convert descriptors to Cloudlet objects
        return descriptors.stream()
                .map(CloudletDescriptor::toCloudlet)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
    }

    /**
     * Load cloudlet descriptors from workload file.
     */
    private List<CloudletDescriptor> loadCloudletDescriptors() {
        WorkloadFileReader reader = new WorkloadFileReader(
                settings.getCloudletTraceFile(),
                settings.getWorkloadMode(),
                settings.getWorkloadReaderMips()
        );

        List<CloudletDescriptor> descriptors = reader.generateDescriptors();

        // Limit to max cloudlets if specified
        int maxCloudlets = settings.getMaxCloudletsToCreateFromWorkloadFile();
        if (maxCloudlets > 0 && descriptors.size() > maxCloudlets) {
            descriptors = descriptors.subList(0, maxCloudlets);
        }

        LOGGER.info("WorkloadFileReader loaded {} descriptors", descriptors.size());
        return descriptors;
    }

    /**
     * Split large cloudlet descriptors into smaller ones.
     */
    private List<CloudletDescriptor> splitLargeCloudletDescriptors(
            List<CloudletDescriptor> descriptors, int maxPes) {
        List<CloudletDescriptor> result = new ArrayList<>();

        int splitIdCounter = 0;
        for (CloudletDescriptor desc : descriptors) {
            if (desc.getNumberOfCores() > maxPes) {
                // Split into multiple smaller cloudlets
                int numSplits = (int) Math.ceil((double) desc.getNumberOfCores() / maxPes);
                long miPerSplit = desc.getMi() / numSplits;  // Distribute MI across splits

                for (int i = 0; i < numSplits; i++) {
                    int pesForThisSplit = Math.min(maxPes, desc.getNumberOfCores() - i * maxPes);
                    CloudletDescriptor split = new CloudletDescriptor(
                            splitIdCounter++,  // New unique ID for split
                            desc.getSubmissionDelay(),  // Keep same arrival time
                            miPerSplit,  // Distributed MI
                            pesForThisSplit  // Number of cores for this split
                    );
                    result.add(split);
                }
            } else {
                result.add(desc);
            }
        }

        return result;
    }

    /**
     * Create a single datacenter instance from configuration.
     */
    private DatacenterInstance createDatacenterInstance(DatacenterConfig config) {
        DatacenterInstance dcInstance = new DatacenterInstance(config);

        // === Create Hosts ===
        List<Host> hostList = DatacenterSetup.createHostsForDatacenter(config);
        dcInstance.setHostList(hostList);

        // === Create Datacenter ===
        Datacenter datacenter = DatacenterSetup.createDatacenterFromConfig(
                simulation, config, hostList, new VmAllocationPolicyCustom()
        );
        dcInstance.setDatacenter(datacenter);

        // === Create VMs ===
        List<Vm> vmPool = new ArrayList<>();
        DatacenterSetup.createVmFleetForDatacenter(config, vmPool);
        dcInstance.setVmPool(vmPool);

        // === Create Local Broker ===
        LoadBalancingBroker localBroker = new LoadBalancingBroker(
                simulation, new ArrayList<>()  // Empty list, cloudlets come from GlobalBroker
        );
        localBroker.setName("LocalBroker_DC" + config.getDatacenterId());
        dcInstance.setLocalBroker(localBroker);

        // FIX: Set target datacenter for this broker's VMs
        // Use setDatacenterMapper to ensure VMs are created in the correct datacenter
        final Datacenter targetDatacenter = datacenter;
        localBroker.setDatacenterMapper((lastDc, vm) -> targetDatacenter);

        // Submit VMs to this broker
        localBroker.submitVmList(vmPool);

        // === Create Green Energy Providers (if enabled) ===
        // Support multiple turbines per datacenter with timezone offset for geo-distribution
        if (config.isGreenEnergyEnabled()) {
            List<Integer> turbineIds = config.getTurbineIds();
            for (int turbineId : turbineIds) {
                GreenEnergyProvider greenEnergyProvider = new GreenEnergyProvider(
                        turbineId,
                        config.getWindDataFile(),
                        config.getTimeScalingMode(),
                        config.getShortTermRows(),
                        config.getLongTermRows(),
                        config.getTimeZoneOffsetRows()  // Apply timezone offset for geo-distribution
                );
                dcInstance.addGreenEnergyProvider(greenEnergyProvider);
            }
            LOGGER.info("Created {} GreenEnergyProvider(s) for DC {} with turbines {} (mode: {}, " +
                       "forecast: short={}rows, long={}rows, tzOffset={}rows)",
                    turbineIds.size(), config.getDatacenterId(), turbineIds,
                    config.getTimeScalingMode().getDescription(),
                    config.getShortTermRows(), config.getLongTermRows(),
                    config.getTimeZoneOffsetRows());
        }

        return dcInstance;
    }

    /**
     * Execute one hierarchical simulation step.
     *
     * @param globalActions List of datacenter indices for arriving cloudlets
     * @param localActions Map of DC_ID -> VM_ID for local scheduling
     * @return Result containing observations, rewards, and termination flags
     */
    public HierarchicalStepResult executeHierarchicalStep(
            List<Integer> globalActions,
            Map<Integer, Integer> localActions) {

        currentStep++;
        LOGGER.debug("=== Step {} starting (Clock: {}s) ===", currentStep, currentClock);

        // === Phase 1: Global Routing ===
        int cloudletsRouted = executeGlobalRouting(globalActions);

        // === Phase 2: Local Scheduling ===
        Map<Integer, Boolean> localResults = executeLocalScheduling(localActions);

        // === Phase 3: Advance Simulation Time ===
        advanceSimulationTime();

        // === Phase 3.5: Update Energy Metrics for All Datacenters ===
        updateAllDatacenterEnergyMetrics();
        
        // === Phase 3.6: Update Cloudlet Completion Counts ===
        updateCloudletCompletionCounts();

        // === Phase 4: Collect Observations ===
        GlobalObservationState globalObs = getGlobalObservation();
        Map<Integer, ObservationState> localObs = getLocalObservations();

        // === Phase 5: Calculate Rewards ===
        // Calculate local rewards first
        Map<Integer, Double> localRewards = calculateLocalRewards(localResults);
        // Global reward = sum of local rewards - green waste penalty
        double globalReward = calculateGlobalReward(localRewards);
        
        // === Phase 5.5: Clear per-step tracking lists ===
        // Must be done AFTER reward calculation but BEFORE next step
        clearPerStepTrackingLists();

        // === Phase 6: Check Termination ===
        boolean allCloudletsCompleted = !hasUnfinishedCloudlets();
        boolean simulationEnded = !simulation.isRunning();

        // Natural termination: all cloudlets completed (don't require simulation to end)
        // The simulation might still be "running" even when all work is done
        boolean terminated = allCloudletsCompleted;

        // If all cloudlets are completed, proactively terminate the simulation
        if (allCloudletsCompleted && simulation.isRunning()) {
            LOGGER.info("All cloudlets completed at step {}. Terminating simulation.", currentStep);
            simulation.terminate();
            simulationEnded = true;
        }

        // Truncation: reached max steps (may still have unfinished cloudlets)
        boolean truncated = currentStep >= settings.getMaxEpisodeLength();

        // === Phase 7: Build Info ===
        Map<String, Object> info = buildStepInfo(cloudletsRouted, localResults);
        info.put("all_cloudlets_completed", allCloudletsCompleted);
        info.put("simulation_ended", simulationEnded);

        LOGGER.debug("=== Step {} completed (Clock: {}s, Terminated: {}, Truncated: {}) ===",
                     currentStep, currentClock, terminated, truncated);

        return new HierarchicalStepResult(
                globalObs, localObs,
                globalReward, localRewards,
                terminated, truncated, info
        );
    }

    /**
     * Execute global routing phase: process arrivals and route batch to datacenters.
     * 
     * Two-stage process:
     * 1. Process newly arriving cloudlets → add to global waiting queue
     * 2. Get a batch from waiting queue → route to datacenters based on global actions
     */
    private int executeGlobalRouting(List<Integer> globalActions) {
        // Stage 1: Process newly arriving cloudlets (add to waiting queue)
        globalBroker.processArrivingCloudlets(currentClock, timestepSize);
        
        int queueSize = globalBroker.getGlobalWaitingCloudletsCount();
        LOGGER.debug("Global queue size after arrivals: {}", queueSize);

        // Stage 2: Get batch for routing (up to the size of global actions provided)
        int batchSize = globalActions.size();
        List<Cloudlet> cloudletsToRoute = globalBroker.getBatchForRouting(batchSize);

        if (cloudletsToRoute.isEmpty()) {
            LOGGER.debug("No cloudlets available for routing (queue empty)");
            return 0;
        }

        LOGGER.debug("{} cloudlets to route, {} global actions provided, {} remaining in queue",
                cloudletsToRoute.size(), globalActions.size(), 
                globalBroker.getGlobalWaitingCloudletsCount());

        // Validate action count (should match batch size, but double-check)
        int actionCount = Math.min(cloudletsToRoute.size(), globalActions.size());

        // Route each cloudlet to its target datacenter
        int routedCount = 0;
        for (int i = 0; i < actionCount; i++) {
            Cloudlet cloudlet = cloudletsToRoute.get(i);
            int targetDcIndex = globalActions.get(i);

            boolean routed = globalBroker.routeCloudletToDatacenter(cloudlet, targetDcIndex);
            if (routed) {
                routedCount++;
            }
        }

        LOGGER.debug("Routed {}/{} cloudlets to datacenters, {} remain in global queue", 
                routedCount, cloudletsToRoute.size(), globalBroker.getGlobalWaitingCloudletsCount());
        return routedCount;
    }

    /**
     * Execute local scheduling phase: assign cloudlets to VMs within each datacenter.
     */
    private Map<Integer, Boolean> executeLocalScheduling(Map<Integer, Integer> localActions) {
        Map<Integer, Boolean> results = new HashMap<>();

        for (Map.Entry<Integer, Integer> entry : localActions.entrySet()) {
            int dcId = entry.getKey();
            int targetVmId = entry.getValue();

            if (dcId < 0 || dcId >= datacenterInstances.size()) {
                LOGGER.warn("Invalid datacenter ID: {}", dcId);
                results.put(dcId, false);
                continue;
            }

            DatacenterInstance dc = datacenterInstances.get(dcId);
            LoadBalancingBroker localBroker = dc.getLocalBroker();

            // Check if agent chose NoAssign (-1)
            if (targetVmId == -1) {
                // Agent explicitly chose NoAssign
                LOGGER.trace("DC {}: Agent chose NoAssign (targetVmId = -1)", dcId);

                // Check if this is an invalid action (choosing NoAssign when cloudlets are waiting)
                if (localBroker.hasWaitingCloudlets()) {
                    LOGGER.debug("DC {}: Agent chose NoAssign but {} cloudlets are waiting. Invalid action.",
                            dcId, localBroker.getWaitingCloudletCount());
                    results.put(dcId, false);  // Mark as failed action
                } else {
                    LOGGER.trace("DC {}: NoAssign is valid (queue is empty)", dcId);
                    results.put(dcId, true);  // Valid action when no cloudlets waiting
                }
            } else {
                // targetVmId is a local VM index (0-based) within this DC
                // Need to convert to actual VM ID
                List<Vm> vmPool = dc.getVmPool();

                // Validate VM index is within bounds
                if (targetVmId < 0 || targetVmId >= vmPool.size()) {
                    LOGGER.warn("DC {}: Invalid VM index {} (DC has {} VMs). Invalid action.",
                            dcId, targetVmId, vmPool.size());
                    results.put(dcId, false);
                } else {
                    // Get actual VM ID from the VM pool
                    long actualVmId = vmPool.get(targetVmId).getId();

                    // DEBUG: Print VM pool and broker VM list info
                    LOGGER.warn("DC {}: DEBUG - vmPool size: {}, local index: {}, actual VM ID: {}",
                            dcId, vmPool.size(), targetVmId, actualVmId);

                    // DEBUG: Print broker's VM list size
                    int brokerVmCount = localBroker.getVmCreatedList().size();
                    LOGGER.warn("DC {}: Broker has {} VMs created", dcId, brokerVmCount);

                    // DEBUG: Print first few VM IDs in vmPool (for first action of each DC)
                    if (targetVmId < 3) {
                        StringBuilder vmPoolIds = new StringBuilder("DC " + dcId + " vmPool IDs: [");
                        for (int i = 0; i < Math.min(5, vmPool.size()); i++) {
                            vmPoolIds.append(vmPool.get(i).getId());
                            if (i < Math.min(5, vmPool.size()) - 1) vmPoolIds.append(", ");
                        }
                        vmPoolIds.append(", ...]");
                        LOGGER.warn(vmPoolIds.toString());

                        // Also print broker's actual VM IDs
                        StringBuilder brokerVmIds = new StringBuilder("DC " + dcId + " broker VM IDs: [");
                        List<Vm> brokerVms = localBroker.getVmCreatedList();
                        for (int i = 0; i < Math.min(5, brokerVms.size()); i++) {
                            brokerVmIds.append(brokerVms.get(i).getId());
                            if (i < Math.min(5, brokerVms.size()) - 1) brokerVmIds.append(", ");
                        }
                        brokerVmIds.append(", ...]");
                        LOGGER.warn(brokerVmIds.toString());
                    }

                    // Local agent assigns a cloudlet from local queue to a VM
                    boolean success = localBroker.assignCloudletToVm(actualVmId);

                    if (!success) {
                        // This is expected during training - agent chose invalid action (queue empty or VM not found)
                        // The agent will receive negative reward and learn to avoid this
                        LOGGER.debug("DC {}: Invalid local action - could not assign cloudlet to VM {} (local index {}). Broker VM count: {}",
                                dcId, actualVmId, targetVmId, brokerVmCount);

                        // Only print detailed debug info if truly a VM lookup failure (not queue empty)
                        // Check if this was a VM lookup failure by trying to find the VM
                        List<Vm> brokerVms = localBroker.getVmCreatedList();
                        boolean vmExists = brokerVms.stream().anyMatch(vm -> vm.getId() == actualVmId);

                        if (!vmExists && brokerVmCount > 0) {
                            // True VM lookup failure - this is a real bug!
                            LOGGER.error("DC {}: VM LOOKUP FAILURE! VM {} not found in broker's list", dcId, actualVmId);

                            // Print broker's actual VM IDs for debugging
                            StringBuilder brokerVmIds = new StringBuilder("DC " + dcId + " Broker VM IDs: [");
                            for (int i = 0; i < Math.min(10, brokerVms.size()); i++) {
                                brokerVmIds.append(brokerVms.get(i).getId());
                                if (i < Math.min(10, brokerVms.size()) - 1) brokerVmIds.append(", ");
                            }
                            if (brokerVms.size() > 10) brokerVmIds.append(", ...");
                            brokerVmIds.append("]");
                            LOGGER.error(brokerVmIds.toString());

                            // Print vmPool IDs for comparison
                            StringBuilder vmPoolIds = new StringBuilder("DC " + dcId + " vmPool IDs: [");
                            for (int i = 0; i < Math.min(10, vmPool.size()); i++) {
                                vmPoolIds.append(vmPool.get(i).getId());
                                if (i < Math.min(10, vmPool.size()) - 1) vmPoolIds.append(", ");
                            }
                            if (vmPool.size() > 10) vmPoolIds.append(", ...");
                            vmPoolIds.append("]");
                            LOGGER.error(vmPoolIds.toString());
                        }
                    }

                    results.put(dcId, success);
                }
            }
        }

        return results;
    }

    /**
     * Advance simulation time by one timestep.
     * Uses proceedClockTo to ensure precise time advancement.
     */
    private void advanceSimulationTime() {
        double targetTime = currentClock + timestepSize;

        LOGGER.debug("Advancing simulation from {} to {}", currentClock, targetTime);

        // Use proceedClockTo for precise time advancement
        proceedClockTo(targetTime);

        firstStep = false;

        LOGGER.debug("Simulation advanced to {}", currentClock);
    }

    /**
     * Update energy metrics for all datacenters.
     * This should be called after each timestep to track energy consumption.
     */
    private void updateAllDatacenterEnergyMetrics() {
        for (DatacenterInstance dc : datacenterInstances) {
            dc.updateEnergyMetrics(currentClock);
        }
        LOGGER.debug("Energy metrics updated for all {} datacenters at clock={}",
                datacenterInstances.size(), currentClock);
    }
    
    /**
     * Update cloudlet completion counts for all datacenters.
     * This syncs the DatacenterInstance counters with the actual finished cloudlets
     * from the LocalBroker's finished list.
     */
    private void updateCloudletCompletionCounts() {
        for (DatacenterInstance dc : datacenterInstances) {
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker != null) {
                // Get actual finished cloudlet count from broker
                int actualFinished = localBroker.getCloudletFinishedList().size();
                // Update datacenter instance counter to match
                dc.setCloudletsCompleted(actualFinished);
                
                LOGGER.trace("{}: Cloudlets completed updated to {}",
                        dc.getName(), actualFinished);
            }
        }
    }
    
    /**
     * Clear per-step tracking lists for all datacenters.
     * This must be called AFTER reward calculation but BEFORE the next step.
     * 
     * Clears:
     * - cloudletsFinishedLastTimestep: cloudlets that finished in the last step
     * - cloudletsFinishedWaitTimeLastTimestep: wait times of finished cloudlets
     */
    private void clearPerStepTrackingLists() {
        for (DatacenterInstance dc : datacenterInstances) {
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker != null) {
                localBroker.clearFinishedWaitTimes();
            }
        }
        LOGGER.trace("Cleared per-step tracking lists for all {} datacenters", datacenterInstances.size());
    }

    /**
     * Get global observation (aggregated state of all datacenters).
     *
     * Returns datacenter-level aggregated metrics for the Global Agent:
     * - Green energy availability per DC
     * - Future energy trend features (God's Eye) per DC
     * - Queue sizes per DC
     * - CPU/RAM utilization per DC
     * - Available resources per DC
     * - Upcoming cloudlets information
     * - Load balance metrics
     *
     * @return GlobalObservationState with DC-level metrics
     */
    public GlobalObservationState getGlobalObservation() {
        int numDatacenters = datacenterInstances.size();

        // Create arrays for datacenter-level metrics
        double[] dcCurrentGreenPowerW = new double[numDatacenters];
        double[] dcCurrentPowerW = new double[numDatacenters];
        double[] dcGreenRatio = new double[numDatacenters];
        double[] dcCumulativeWastedGreenWh = new double[numDatacenters];

        // Future energy trend features (God's Eye)
        double[] dcFutureShortMean = new double[numDatacenters];
        double[] dcFutureShortTrend = new double[numDatacenters];
        double[] dcFutureLongMean = new double[numDatacenters];
        double[] dcFutureLongPeakTiming = new double[numDatacenters];

        int[] dcQueueSizes = new int[numDatacenters];
        double[] dcUtilizations = new double[numDatacenters];
        int[] dcAvailablePes = new int[numDatacenters];
        double[] dcRamUtilizations = new double[numDatacenters];

        // Collect metrics from each datacenter
        for (int i = 0; i < numDatacenters; i++) {
            DatacenterInstance dc = datacenterInstances.get(i);

            // Green energy metrics (W)
            dcCurrentGreenPowerW[i] = dc.getCurrentGreenPowerW(currentClock);  // W
            dcCurrentPowerW[i] = dc.getCurrentPowerW();  // W
            dcGreenRatio[i] = dc.getGreenEnergyRatio();  // [0, 1]
            dcCumulativeWastedGreenWh[i] = dc.getTotalWastedGreenWh();  // Wh

            // Future energy trend features (God's Eye mode)
            // Use aggregated features from all turbines supplying this datacenter
            List<GreenEnergyProvider> greenProviders = dc.getGreenEnergyProviders();
            if (!greenProviders.isEmpty()) {
                double[] trendFeatures = GreenEnergyProvider.computeAggregatedFutureTrendFeatures(
                        greenProviders, currentClock);
                dcFutureShortMean[i] = trendFeatures[0];
                dcFutureShortTrend[i] = trendFeatures[1];
                dcFutureLongMean[i] = trendFeatures[2];
                dcFutureLongPeakTiming[i] = trendFeatures[3];
            } else {
                // Default values if green energy not enabled
                dcFutureShortMean[i] = 0.5;
                dcFutureShortTrend[i] = 0.0;
                dcFutureLongMean[i] = 0.5;
                dcFutureLongPeakTiming[i] = 0.5;
            }

            // Waiting cloudlets count (exact integer)
            dcQueueSizes[i] = dc.getWaitingCloudletCount();

            // Average CPU utilization [0.0, 1.0]
            dcUtilizations[i] = dc.getAverageHostUtilization();

            // Available PEs across all VMs
            dcAvailablePes[i] = (int) dc.getTotalAvailablePes();

            // Average RAM utilization [0.0, 1.0] - compute from hosts
            double totalRamUtil = 0.0;
            List<Host> hosts = dc.getHostList();
            if (!hosts.isEmpty()) {
                for (Host host : hosts) {
                    totalRamUtil += host.getRam().getPercentUtilization();
                }
                dcRamUtilizations[i] = totalRamUtil / hosts.size();
            } else {
                dcRamUtilizations[i] = 0.0;
            }
        }

        // Get global waiting queue info (for batch routing mode)
        int upcomingCount = globalBroker.getGlobalWaitingCloudletsCount();

        // Get batch cloudlet features from waiting queue
        // Peek at the next batch without removing cloudlets
        List<Cloudlet> batchCloudlets = globalBroker.peekBatch(settings.getGlobalRoutingBatchSize());

        // Create arrays for batch cloudlet information
        int batchSize = settings.getGlobalRoutingBatchSize();
        int[] batchCloudletPes = new int[batchSize];
        long[] batchCloudletMi = new long[batchSize];

        // Fill arrays with actual cloudlet data (0 if no cloudlet at that position)
        for (int i = 0; i < batchCloudlets.size(); i++) {
            Cloudlet cloudlet = batchCloudlets.get(i);
            batchCloudletPes[i] = (int) cloudlet.getPesNumber();
            batchCloudletMi[i] = cloudlet.getLength();
        }
        // Remaining positions stay as 0 (default array values)

        // Calculate REAL PEs distribution from actual queue contents
        int[] upcomingPesDistribution = globalBroker.calculatePesDistribution();

        // Calculate load imbalance (variance in utilization)
        double loadImbalance = calculateLoadImbalance(dcUtilizations);

        // Get recent completed cloudlets (across all DCs)
        // Note: This requires tracking completed counts in DatacenterInstance
        // For now, use broker's finished list size as proxy
        int recentCompleted = datacenterInstances.stream()
                .mapToInt(dc -> dc.getLocalBroker().getCloudletFinishedList().size())
                .sum();

        // Create GlobalObservationState with proper DC-level semantics
        return new GlobalObservationState(
                dcCurrentGreenPowerW,
                dcCurrentPowerW,
                dcGreenRatio,
                dcCumulativeWastedGreenWh,
                dcFutureShortMean,      // Future energy trend features (God's Eye)
                dcFutureShortTrend,
                dcFutureLongMean,
                dcFutureLongPeakTiming,
                dcQueueSizes,
                dcUtilizations,
                dcAvailablePes,
                dcRamUtilizations,
                upcomingCount,
                batchCloudletPes,      // Array of PEs for batch
                batchCloudletMi,       // Array of MI for batch
                upcomingPesDistribution,
                loadImbalance,
                recentCompleted,
                numDatacenters,
                currentClock
        );
    }

    /**
     * Calculate load imbalance metric across datacenters.
     * Uses variance in utilization as the metric.
     *
     * @param dcUtilizations Array of utilization values
     * @return Load imbalance metric (higher = more unbalanced)
     */
    private double calculateLoadImbalance(double[] dcUtilizations) {
        if (dcUtilizations.length == 0) {
            return 0.0;
        }

        double mean = Arrays.stream(dcUtilizations).average().orElse(0.0);
        double variance = Arrays.stream(dcUtilizations)
                .map(u -> Math.pow(u - mean, 2))
                .average()
                .orElse(0.0);

        return Math.sqrt(variance);  // Return standard deviation
    }

    /**
     * Get local observations for all datacenters.
     *
     * For each datacenter's local agent, we provide:
     * - VM loads and availability
     * - Host utilizations
     * - Waiting cloudlets in local queue
     * - Next cloudlet requirements
     */
    public Map<Integer, ObservationState> getLocalObservations() {
        Map<Integer, ObservationState> observations = new HashMap<>();

        for (DatacenterInstance dc : datacenterInstances) {
            int dcId = dc.getId();
            ObservationState localObs = buildLocalObservation(dc);
            observations.put(dcId, localObs);
        }

        return observations;
    }

    /**
     * Build observation state for a specific datacenter (local view).
     */
    private ObservationState buildLocalObservation(DatacenterInstance dc) {
        List<Host> hosts = dc.getHostList();
        List<Vm> vms = dc.getVmPool();
        LoadBalancingBroker localBroker = dc.getLocalBroker();

        int maxHosts = hosts.size();
        int maxVms = vms.size();

        // Host observations
        double[] hostLoads = new double[maxHosts];
        double[] hostRamUsageRatio = new double[maxHosts];

        for (int i = 0; i < hosts.size(); i++) {
            Host host = hosts.get(i);
            hostLoads[i] = host.getCpuPercentUtilization();
            hostRamUsageRatio[i] = host.getRam().getPercentUtilization();
        }

        // VM observations
        double[] vmLoads = new double[maxVms];
        int[] vmTypes = new int[maxVms];
        int[] vmHostMap = new int[maxVms];
        int[] vmAvailablePes = new int[maxVms];

        for (int i = 0; i < vms.size(); i++) {
            Vm vm = vms.get(i);
            if (vm.isCreated() && !vm.isFailed()) {
                vmLoads[i] = vm.getCpuPercentUtilization();
                vmTypes[i] = determineVmType(vm, dc.getConfig());
                vmHostMap[i] = vm.getHost() != null ? (int) vm.getHost().getId() : -1;
                vmAvailablePes[i] = (int) vm.getFreePesNumber();
            } else {
                vmLoads[i] = 0.0;
                vmTypes[i] = 0;  // VM off
                vmHostMap[i] = -1;
                vmAvailablePes[i] = 0;
            }
        }

        // Waiting cloudlets and next cloudlet info
        int waitingCloudlets = localBroker.getWaitingCloudletCount();
        int nextCloudletPes = 0;

        if (waitingCloudlets > 0) {
            Cloudlet nextCloudlet = localBroker.peekWaitingCloudlet();
            if (nextCloudlet != null) {
                nextCloudletPes = (int) nextCloudlet.getPesNumber();
            }
        }

        // Infrastructure observation (DC-specific metrics)
        int[] infrastructureObs = new int[maxHosts];
        for (int i = 0; i < maxHosts; i++) {
            if (i < hosts.size()) {
                Host host = hosts.get(i);
                infrastructureObs[i] = (int) host.getFreePesNumber();
            } else {
                infrastructureObs[i] = 0;
            }
        }

        return new ObservationState(
                hostLoads,
                hostRamUsageRatio,
                vmLoads,
                vmTypes,
                vmHostMap,
                infrastructureObs,
                waitingCloudlets,
                nextCloudletPes,
                vmAvailablePes,
                (int) vms.stream().filter(vm -> vm.isCreated() && !vm.isFailed()).count(),  // actualVmCount
                hosts.size(),  // actualHostCount
                0L,              // nextCloudletMi (not tracked at DC level)
                0.0,             // nextCloudletWaitTime (not tracked at DC level)
                new int[3],      // queuePesDistribution (not tracked at DC level)
                0                // completedCloudletsLast10Steps (not tracked at DC level)
        );
    }

    /**
     * Determine VM type based on PE count.
     * 0 = Off, 1 = Small, 2 = Medium, 3 = Large
     */
    private int determineVmType(Vm vm, DatacenterConfig config) {
        int vmPes = (int) vm.getPesNumber();
        int smallVmPes = config.getSmallVmPes();
        int mediumVmPes = config.getMediumVmPes();
        int largeVmPes = config.getLargeVmPes();

        if (vmPes >= largeVmPes) {
            return 3;  // Large
        } else if (vmPes >= mediumVmPes) {
            return 2;  // Medium
        } else if (vmPes >= smallVmPes) {
            return 1;  // Small
        } else {
            return 0;  // Off or unknown
        }
    }

    /**
     * Calculate global reward as sum of local rewards minus green waste penalty.
     *
     * NEW DESIGN:
     * Global Reward = Σ(Local Rewards) - Carbon Emission Penalty
     *
     * This design encourages:
     * - Global agent to consider local scheduling efficiency (via sum of local rewards)
     * - Minimizing carbon emissions by routing to low-carbon datacenters
     *
     * @param localRewards Map of DC_ID -> local reward
     */
    private double calculateGlobalReward(Map<Integer, Double> localRewards) {
        // === Component 1: Sum of Local Rewards ===
        double sumLocalRewards = localRewards.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        // === Component 2: Carbon Emission Penalty (current timestep) ===
        double carbonEmissionPenalty = calculateCarbonEmissionPenalty();

        // === Final Global Reward ===
        double reward = sumLocalRewards - carbonEmissionPenalty;

        LOGGER.debug("Global Reward: total={} (sumLocal={}, carbonEmissionPenalty={})",
                reward, sumLocalRewards, carbonEmissionPenalty);

        return reward;
    }

    /**
     * Calculate carbon emission penalty for the current timestep.
     *
     * Penalty is based on the total carbon emissions (kg CO2) across all datacenters.
     * Higher carbon emissions = higher penalty.
     *
     * Formula: penalty = carbonEmissionPenaltyCoef × totalCarbonKg
     *
     * @return Carbon emission penalty (non-negative value to be subtracted from reward)
     */
    private double calculateCarbonEmissionPenalty() {
        double totalCarbonKg = 0.0;
        int validDatacenters = 0;

        // Sum carbon emissions from all datacenters for this timestep
        for (DatacenterInstance dc : datacenterInstances) {
            EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
            if (delta == null) {
                continue;
            }
            validDatacenters++;
            totalCarbonKg += delta.getDeltaCarbonEmissionKg();
        }

        if (validDatacenters == 0) {
            return 0.0;
        }

        // Apply penalty coefficient
        double penaltyCoef = settings.getCarbonEmissionPenaltyCoef();  // To be configured
        double penalty = penaltyCoef * totalCarbonKg;

        LOGGER.debug("  Carbon Emission: totalCarbonKg={}, validDCs={}, penaltyCoef={}, penalty={}",
                String.format("%.3f", totalCarbonKg),
                validDatacenters,
                penaltyCoef,
                String.format("%.3f", penalty));

        return penalty;
    }

    /**
     * Calculate green energy component of global reward using INCREMENTAL metrics.
     *
     * Uses energy deltas from the current timestep rather than cumulative values.
     * This provides more immediate feedback on the impact of actions.
     *
     * Reward = Green Usage Reward - Brown Penalty - Waste Penalty
     *
     * Components:
     * - Green Usage Reward: Ratio of green energy used in this timestep
     * - Brown Penalty: Penalty for using brown energy (1.5x amplified)
     * - Waste Penalty: Penalty for wasting available green energy (1.2x amplified)
     *
     * @return Green energy reward in range [-2, 1], can be negative if penalties exceed rewards
     */
    private double calculateGreenEnergyReward() {
        double totalGreenReward = 0.0;
        double totalBrownPenalty = 0.0;
        double totalWastePenalty = 0.0;
        int validDatacenters = 0;

        for (DatacenterInstance dc : datacenterInstances) {
            EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
            if (delta == null) {
                continue;  // Skip if no delta available (e.g., first step)
            }
            validDatacenters++;

            double deltaTotal = delta.getDeltaTotalEnergyWh();

            // === 1. Green Usage Reward ===
            // Reward for using green energy (normalized by total consumption)
            if (deltaTotal > 0) {
                double greenRatio = delta.getGreenRatio();
                totalGreenReward += greenRatio;  // [0, 1]
            }

            // === 2. Brown Energy Penalty ===
            // Direct penalty for brown energy usage
            if (deltaTotal > 0) {
                double brownPenalty = delta.getNormalizedBrownPenalty();
                totalBrownPenalty += brownPenalty * 1.5;  // Amplified penalty
            }

            // === 3. Green Waste Penalty ===
            // Penalty for wasting available green energy
            double wastePenalty = delta.getNormalizedWastePenalty();
            totalWastePenalty += wastePenalty * 1.2;  // Amplified penalty

            LOGGER.trace("  DC{}: Delta Green={}Wh, Brown={}Wh, Wasted={}Wh -> GreenRatio={}, BrownPen={}, WastePen={}",
                    dc.getId(),
                    delta.getDeltaGreenEnergyUsedWh(),
                    delta.getDeltaBrownEnergyUsedWh(),
                    delta.getDeltaGreenEnergyWastedWh(),
                    delta.getGreenRatio(),
                    delta.getNormalizedBrownPenalty(),
                    wastePenalty);
        }

        if (validDatacenters == 0) {
            return 0.0;  // No valid data available
        }

        // Normalize by number of datacenters
        totalGreenReward /= validDatacenters;
        totalBrownPenalty /= validDatacenters;
        totalWastePenalty /= validDatacenters;

        // Combine components: reward minus penalties
        double greenEnergyReward = totalGreenReward - totalBrownPenalty - totalWastePenalty;

        // Clip to reasonable range [-2, 1]
        greenEnergyReward = Math.max(-2.0, Math.min(1.0, greenEnergyReward));

        LOGGER.debug("  Green Energy Reward: green_use={}, brown_pen={}, waste_pen={} -> total={}",
                totalGreenReward, totalBrownPenalty, totalWastePenalty, greenEnergyReward);

        return greenEnergyReward;
    }

    /**
     * Calculate performance component of global reward.
     *
     * Components:
     * - 40%: Task completion rate (encourage finishing cloudlets)
     * - 30%: Waiting time penalty (minimize queue lengths)
     * - 30%: Load balance reward (even distribution across DCs)
     *
     * @return Performance reward [0, 1]
     */
    private double calculatePerformanceReward() {
        // 2.1 Task Completion Rate
        int totalCompleted = datacenterInstances.stream()
                .mapToInt(DatacenterInstance::getCloudletsCompleted)
                .sum();
        int totalReceived = datacenterInstances.stream()
                .mapToInt(DatacenterInstance::getCloudletsReceived)
                .sum();

        double completionReward = totalReceived > 0 ?
                (double) totalCompleted / totalReceived : 0.0;  // [0, 1]

        // 2.2 Waiting Time Penalty
        double avgWaitingCloudlets = datacenterInstances.stream()
                .mapToInt(DatacenterInstance::getWaitingCloudletCount)
                .average()
                .orElse(0.0);

        // Normalize by expected max queue size (assume max 100 cloudlets per DC)
        double waitPenalty = Math.min(avgWaitingCloudlets / 100.0, 1.0);  // [0, 1]

        // 2.3 Load Balance Reward
        double[] utilizations = datacenterInstances.stream()
                .mapToDouble(DatacenterInstance::getAverageHostUtilization)
                .toArray();

        double variance = calculateVariance(utilizations);
        double balanceReward = Math.exp(-variance * 10.0);  // Exp decay, lower variance = higher reward

        // Combine performance components
        double perfReward = 0.4 * completionReward +        // 40%: Complete tasks
                           0.3 * (1.0 - waitPenalty) +      // 30%: Reduce waiting
                           0.3 * balanceReward;             // 30%: Balance load

        LOGGER.trace("  Performance - Completion: {}, Wait: {}, Balance: {} -> Reward: {}",
                completionReward, waitPenalty, balanceReward, perfReward);

        return perfReward;  // [0, 1]
    }

    /**
     * Calculate variance of an array of values.
     *
     * @param values Array of values
     * @return Variance
     */
    private double calculateVariance(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }

        double mean = Arrays.stream(values).average().orElse(0.0);
        double variance = Arrays.stream(values)
                .map(v -> Math.pow(v - mean, 2))
                .average()
                .orElse(0.0);

        return variance;
    }

    /**
     * Calculate local rewards for each datacenter.
     *
     * Local objectives (per DC):
     * - Minimize wait time
     * - Maximize resource utilization
     * - Minimize local queue size
     * - Penalize invalid actions
     *
     * @param localResults Map of DC_ID -> success flag for local scheduling actions
     */
    private Map<Integer, Double> calculateLocalRewards(Map<Integer, Boolean> localResults) {
        Map<Integer, Double> rewards = new HashMap<>();

        for (DatacenterInstance dc : datacenterInstances) {
            int dcId = dc.getId();
            boolean wasInvalidAction = !localResults.getOrDefault(dcId, true);
            double localReward = calculateSingleLocalReward(dc, wasInvalidAction);
            rewards.put(dcId, localReward);
        }

        return rewards;
    }

    /**
     * Calculate reward for a single datacenter (local perspective).
     * ALIGNED WITH LoadBalancerGateway (Single DC) for consistency.
     *
     * Components (same as Single DC):
     * 1. Wait Time Penalty - log scaled
     * 2. Utilization & Balance Penalty - variance + deviation from target
     * 3. Queue Length Penalty - ratio based
     * 4. Invalid Action Penalty - flat penalty
     *
     * @param dc The datacenter instance
     * @param wasInvalidAction Whether the local action was invalid
     * @return Total local reward
     */
    private double calculateSingleLocalReward(DatacenterInstance dc, boolean wasInvalidAction) {
        LoadBalancingBroker localBroker = dc.getLocalBroker();

        // Get coefficients from settings
        double waitTimeCoef = settings.getRewardWaitTimeCoef();
        double utilizationCoef = settings.getRewardUnutilizationCoef();
        double queueCoef = settings.getRewardQueuePenaltyCoef();
        double invalidActionCoef = settings.getRewardInvalidActionCoef();
        double completionCoef = settings.getRewardCompletionCoef();

        // === 1. Wait Time Penalty (exactly like LoadBalancerGateway) ===
        double waitTimePenalty = 0.0;
        if (localBroker != null) {
            List<Double> finishedWaitTimes = localBroker.getFinishedWaitTimesLastStep(currentClock);
            if (!finishedWaitTimes.isEmpty()) {
                double avgWaitTime = finishedWaitTimes.stream()
                    .mapToDouble(d -> d)
                    .average()
                    .orElse(0.0);
                // Use log scaling like LoadBalancerGateway
                waitTimePenalty = -waitTimeCoef * Math.log1p(avgWaitTime);
            }
        }

        // === 2. Utilization & Balance Penalty (use VMs like LoadBalancerGateway) ===
        double utilizationPenalty = 0.0;
        List<Vm> runningVms = dc.getVmPool();
        if (runningVms != null && !runningVms.isEmpty()) {
            // Filter for created and active VMs
            List<Vm> activeVms = runningVms.stream()
                .filter(vm -> vm.isCreated() && !vm.isFailed())
                .collect(java.util.stream.Collectors.toList());

            if (!activeVms.isEmpty()) {
                // Calculate average utilization
                double avgUtilization = activeVms.stream()
                    .mapToDouble(Vm::getCpuPercentUtilization)
                    .average()
                    .orElse(0.0);

                // Calculate variance for load balancing
                double variance = activeVms.stream()
                    .mapToDouble(vm -> Math.pow(vm.getCpuPercentUtilization() - avgUtilization, 2))
                    .average()
                    .orElse(0.0);

                // Target utilization (0.75 for multi-DC, lower than Single DC's 0.95)
                double targetUtil = 0.75;
                double utilDeviationPenalty = Math.abs(avgUtilization - targetUtil);

                // Combine penalties: sqrt(variance) + deviation (same as LoadBalancerGateway)
                utilizationPenalty = -utilizationCoef * (Math.sqrt(variance) + utilDeviationPenalty);
            }
        }

        // === 3. Queue Length Penalty (same logic as LoadBalancerGateway) ===
        double queuePenalty = 0.0;
        int waitingCount = dc.getWaitingCloudletCount();
        int totalReceived = dc.getCloudletsReceived();
        if (totalReceived > 0) {
            // Normalize queue size by total cloudlets received
            double queueRatio = (double) waitingCount / totalReceived;
            queuePenalty = -queueCoef * queueRatio;
        }

        // === 4. Invalid Action Penalty (exactly like LoadBalancerGateway) ===
        double invalidActionPenalty = -invalidActionCoef * (wasInvalidAction ? 1.0 : 0.0);

        // === 5. Completion Rate Reward (POSITIVE reward to prevent sacrificing completion) ===
        // TEMPORARILY DISABLED - uncomment to re-enable
        // Fixed per-completion reward instead of normalized by totalReceived
        // This avoids reward explosion at episode start when totalReceived is small
        //
        // Formula: r_completion = completionCoef × completedThisStep
        // With completionCoef = 0.1, each completion gives +0.1 reward
        // This is similar magnitude to other penalties (~0.3-0.5 per step)
        double completionReward = 0.0;
        int completedThisStep = 0;
        if (localBroker != null) {
            completedThisStep = localBroker.getCloudletsFinishedLastStep(currentClock).size();
            // Simple per-completion reward (no normalization)
            // completionReward = completionCoef * completedThisStep;  // DISABLED
        }

        // === Total Reward (no clipping, same as LoadBalancerGateway) ===
        // Note: completionReward is now always 0.0 (disabled)
        double totalReward = waitTimePenalty + utilizationPenalty + queuePenalty + invalidActionPenalty + completionReward;

        LOGGER.debug("DC {} Local Reward: total={} (wait={}, util={}, queue={}, invalid={}, completion={}[{}])",
                dc.getName(),
                String.format("%.3f", totalReward),
                String.format("%.3f", waitTimePenalty),
                String.format("%.3f", utilizationPenalty),
                String.format("%.3f", queuePenalty),
                String.format("%.3f", invalidActionPenalty),
                String.format("%.3f", completionReward),
                completedThisStep);

        return totalReward;
    }

    /**
     * Build step info dictionary.
     */
    private Map<String, Object> buildStepInfo(int cloudletsRouted, Map<Integer, Boolean> localResults) {
        Map<String, Object> info = new HashMap<>();
        info.put("cloudlets_routed", cloudletsRouted);
        info.put("current_clock", currentClock);
        info.put("current_step", currentStep);
        info.put("total_cloudlets", allCloudlets.size());
        info.put("remaining_cloudlets", globalBroker.getRemainingCloudletCount());

        // Add energy metrics for each datacenter
        Map<Integer, Map<String, Object>> dcEnergyMetrics = collectDatacenterEnergyMetrics();
        info.put("datacenter_energy_metrics", dcEnergyMetrics);

        // Add global energy statistics
        Map<String, Object> globalEnergyStats = calculateGlobalEnergyStatistics();
        info.put("global_energy_stats", globalEnergyStats);

        return info;
    }

    /**
     * Collect energy metrics from all datacenters.
     * Also includes cloudlet completion statistics per DC.
     */
    private Map<Integer, Map<String, Object>> collectDatacenterEnergyMetrics() {
        Map<Integer, Map<String, Object>> metricsMap = new HashMap<>();

        for (DatacenterInstance dc : datacenterInstances) {
            DatacenterEnergyMetrics metrics = new DatacenterEnergyMetrics(
                    dc.getId(),
                    dc.getName(),
                    dc.getCumulativeGreenEnergyWh(),
                    dc.getCumulativeBrownEnergyWh(),
                    dc.getTotalWastedGreenWh(),
                    dc.getCumulativeCarbonEmissionKg(),
                    dc.getCurrentPowerW(),
                    dc.isGreenEnergyEnabled() ? dc.getCurrentGreenPowerW(currentClock) : 0.0
            );

            Map<String, Object> dcMetrics = metrics.toMap();

            // Add cloudlet completion statistics
            dcMetrics.put("cloudlets_received", dc.getCloudletsReceived());
            dcMetrics.put("cloudlets_finished", dc.getCloudletsCompleted());

            // Calculate mean completion time from finished cloudlets
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker != null) {
                List<Cloudlet> finishedCloudlets = localBroker.getCloudletFinishedList();
                if (!finishedCloudlets.isEmpty()) {
                    double totalCompletionTime = 0.0;
                    for (Cloudlet cloudlet : finishedCloudlets) {
                        // Completion time = finishTime - arrivalTime (submissionDelay)
                        double completionTime = cloudlet.getFinishTime() - cloudlet.getSubmissionDelay();
                        totalCompletionTime += completionTime;
                    }
                    double meanCompletionTime = totalCompletionTime / finishedCloudlets.size();
                    dcMetrics.put("mean_completion_time", meanCompletionTime);
                } else {
                    dcMetrics.put("mean_completion_time", 0.0);
                }
            } else {
                dcMetrics.put("mean_completion_time", 0.0);
            }

            metricsMap.put(dc.getId(), dcMetrics);
        }

        return metricsMap;
    }

    /**
     * Calculate global energy statistics (aggregated across all DCs).
     */
    private Map<String, Object> calculateGlobalEnergyStatistics() {
        Map<String, Object> stats = new HashMap<>();

        double totalGreenWh = 0.0;
        double totalBrownWh = 0.0;
        double totalWastedWh = 0.0;
        double totalPowerW = 0.0;
        double totalGreenPowerW = 0.0;
        double totalCarbonKg = 0.0;
        
        // Cloudlet completion tracking
        int totalCloudletsReceived = 0;
        int totalCloudletsCompleted = 0;

        for (DatacenterInstance dc : datacenterInstances) {
            totalGreenWh += dc.getCumulativeGreenEnergyWh();
            totalBrownWh += dc.getCumulativeBrownEnergyWh();
            totalWastedWh += dc.getTotalWastedGreenWh();
            totalPowerW += dc.getCurrentPowerW();
            totalCarbonKg += dc.getCumulativeCarbonEmissionKg();
            if (dc.isGreenEnergyEnabled()) {
                totalGreenPowerW += dc.getCurrentGreenPowerW(currentClock);
            }
            
            // Aggregate cloudlet statistics
            totalCloudletsReceived += dc.getCloudletsReceived();
            totalCloudletsCompleted += dc.getCloudletsCompleted();
        }

        double totalEnergyWh = totalGreenWh + totalBrownWh;
        double greenRatio = totalEnergyWh > 0 ? totalGreenWh / totalEnergyWh : 0.0;
        double totalEnergyKWh = totalEnergyWh / 1000.0;
        double carbonIntensity = totalEnergyKWh > 0 ? totalCarbonKg / totalEnergyKWh : 0.0;

        stats.put("total_green_energy_wh", totalGreenWh);
        stats.put("total_brown_energy_wh", totalBrownWh);
        stats.put("total_wasted_green_wh", totalWastedWh);
        stats.put("total_energy_wh", totalEnergyWh);
        stats.put("green_energy_ratio", greenRatio);
        stats.put("total_power_w", totalPowerW);
        stats.put("total_green_power_w", totalGreenPowerW);
        stats.put("total_carbon_emission_kg", totalCarbonKg);
        stats.put("carbon_intensity_kg_per_kwh", carbonIntensity);
        stats.put("num_datacenters", datacenterInstances.size());
        
        // Add cloudlet completion statistics
        stats.put("total_created_cloudlets", totalCloudletsReceived);
        stats.put("total_finished_cloudlets", totalCloudletsCompleted);

        return stats;
    }

    /**
     * Check if simulation is still running.
     */
    public boolean isRunning() {
        return simulation != null && simulation.isRunning();
    }

    /**
     * Print cloudlet execution summary from all datacenters.
     * Shows which DC, Host, and VM each cloudlet executed on.
     * Only includes successfully executed cloudlets, sorted by cloudlet ID.
     */
    private void printCloudletExecutionSummary() {
        LOGGER.info("========================================");
        LOGGER.info("Cloudlet Execution Summary (Episode End)");
        LOGGER.info("========================================");

        try {
            // Collect all finished cloudlets from all datacenters
            List<Cloudlet> allFinishedCloudlets = new ArrayList<>();
            Map<Long, Double> mergedArrivalTimeMap = new HashMap<>();

            for (DatacenterInstance dcInstance : datacenterInstances) {
                LoadBalancingBroker broker = dcInstance.getLocalBroker();

                // Get finished cloudlets
                List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
                allFinishedCloudlets.addAll(finishedCloudlets);

                // Merge arrival time maps
                Map<Long, Double> arrivalTimeMap = broker.getCloudletArrivalTimeMap();
                mergedArrivalTimeMap.putAll(arrivalTimeMap);

                LOGGER.info("DC {} - Finished cloudlets: {}",
                    dcInstance.getDatacenter().getId(), finishedCloudlets.size());
            }

            // Sort by cloudlet ID (original CSV order)
            allFinishedCloudlets.sort((c1, c2) -> Long.compare(c1.getId(), c2.getId()));

            LOGGER.info("Total finished cloudlets across all DCs: {}", allFinishedCloudlets.size());

            // Print table using CloudletsTableBuilderWithDetails
            if (!allFinishedCloudlets.isEmpty()) {
                new CloudletsTableBuilderWithDetails(allFinishedCloudlets, mergedArrivalTimeMap).build();
            } else {
                LOGGER.warn("No cloudlets were successfully executed in this episode!");
            }

        } catch (Exception e) {
            LOGGER.error("Error printing cloudlet execution summary: {}", e.getMessage(), e);
        }

        LOGGER.info("========================================");
    }

    /**
     * Stop the simulation cleanly.
     */
    private void stopSimulation() {
        if (simulation != null && simulation.isRunning()) {
            simulation.terminate();
            LOGGER.info("Simulation terminated");
        }
    }

    /**
     * Get the number of future events in the simulation.
     */
    private long getNumberOfFutureEvents() {
        return simulation.getNumberOfFutureEvents(info -> true);
    }

    /**
     * Check if there are any unfinished cloudlets across all datacenters.
     *
     * @return true if there are unfinished cloudlets, false otherwise
     */
    private boolean hasUnfinishedCloudlets() {
        // Check GlobalBroker for unrouted cloudlets
        if (globalBroker != null && globalBroker.getRemainingCloudletCount() > 0) {
            return true;
        }

        // Check each LocalBroker for unfinished cloudlets
        for (DatacenterInstance dc : datacenterInstances) {
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker != null && localBroker.hasUnfinishedCloudlets()) {
                return true;
            }
        }

        return false;
    }

    /**
     * Ensure all cloudlets complete before simulation ends.
     *
     * This method adds an event listener that checks if there are unfinished
     * cloudlets when there is only one future event left. If there are unfinished
     * cloudlets, it sends an empty event to keep the simulation running.
     */
    private void ensureAllCloudletsCompleteBeforeSimulationEnds() {
        double interval = settings.getSimulationTimestep();
        simulation.addOnEventProcessingListener(info -> {
            if (getNumberOfFutureEvents() == 1 && hasUnfinishedCloudlets()) {
                LOGGER.trace("Cloudlets not finished. Sending empty event to keep simulation running.");

                // Send event to first datacenter to keep simulation alive
                if (!datacenterInstances.isEmpty()) {
                    Datacenter firstDc = datacenterInstances.get(0).getDatacenter();
                    simulation.send(firstDc, firstDc, interval, CloudSimTag.NONE, null);
                }
            }
        });
    }

    /**
     * Advances the simulation clock to the specified target time.
     *
     * This method runs the simulation in increments until the target time is reached
     * or the maximum number of iterations is exceeded to prevent an infinite loop.
     *
     * @param targetTime The target time to advance the simulation clock to
     */
    private void proceedClockTo(final double targetTime) {
        if (simulation == null) {
            throw new IllegalStateException("Simulation not initialized. Call resetSimulation first.");
        }
        if (!simulation.isRunning()) {
            LOGGER.warn("Attempting to proceed clock on a simulation that is not running.");
        }

        double adjustedInterval = targetTime - currentClock;
        int maxIterations = 1000; // Safety check to prevent infinite loop
        int iterations = 0;

        LOGGER.trace("Proceeding clock from {} to {} (interval: {})", currentClock, targetTime, adjustedInterval);

        // Run the simulation until the target time is reached
        while (simulation.runFor(adjustedInterval) < targetTime) {
            // Update current clock
            currentClock = simulation.clock();

            // Calculate the remaining time to the target
            adjustedInterval = targetTime - currentClock;

            // Use the minimum time between events if the remaining time is non-positive
            adjustedInterval = adjustedInterval <= 0 ? settings.getMinTimeBetweenEvents() : adjustedInterval;

            // Increment the iteration counter and break if it exceeds the maximum allowed iterations
            if (++iterations >= maxIterations) {
                LOGGER.warn("Exceeded maximum iterations ({}) in proceedClockTo. Current clock: {}, Target: {}",
                        maxIterations, currentClock, targetTime);
                break;
            }
        }

        // Final clock update
        currentClock = simulation.clock();

        LOGGER.trace("Clock advanced to {} (target was {})", currentClock, targetTime);
    }

    /**
     * Get the number of datacenters.
     */
    public int getDatacenterCount() {
        return datacenterInstances.size();
    }

    /**
     * Convert double array to int array (helper method).
     */
    private int[] convertDoubleArrayToIntArray(double[] doubleArray) {
        int[] intArray = new int[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++) {
            intArray[i] = (int) doubleArray[i];
        }
        return intArray;
    }
}
