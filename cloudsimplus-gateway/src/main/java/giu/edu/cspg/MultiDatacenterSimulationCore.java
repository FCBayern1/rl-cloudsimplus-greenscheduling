package giu.edu.cspg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.energy.GreenEnergyProvider;
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

    /**
     * Initialize the multi-datacenter simulation core.
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
        stopSimulation();

        // Initialize CloudSim Plus engine
        simulation = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        currentClock = 0.0;
        currentStep = 0;
        firstStep = true;

        // === Step 1: Load Cloudlet Workload ===
        allCloudlets = loadAllCloudlets();
        LOGGER.info("Loaded {} cloudlets from workload trace", allCloudlets.size());

        // === Step 2: Create Datacenter Instances ===
        datacenterInstances = new ArrayList<>();
        for (DatacenterConfig config : datacenterConfigs) {
            DatacenterInstance dcInstance = createDatacenterInstance(config);
            datacenterInstances.add(dcInstance);
            LOGGER.info("Created datacenter: {} (ID: {})", config.getDatacenterName(), config.getDatacenterId());
        }

        // === Step 3: Create Global Broker ===
        globalBroker = new GlobalBroker(simulation, allCloudlets, datacenterInstances);
        LOGGER.info("GlobalBroker created with {} cloudlets", allCloudlets.size());

        // === Step 4: Start Simulation (sync mode) ===
        simulation.startSync();
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

        // Submit VMs to this broker
        localBroker.submitVmList(vmPool);

        // === Create Green Energy Provider (if enabled) ===
        if (config.isGreenEnergyEnabled()) {
            GreenEnergyProvider greenEnergyProvider = new GreenEnergyProvider(
                    config.getTurbineId(),
                    config.getWindDataFile()
            );
            dcInstance.setGreenEnergyProvider(greenEnergyProvider);
            LOGGER.info("GreenEnergyProvider created for DC {} with turbine {}",
                    config.getDatacenterId(), config.getTurbineId());
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

        // === Phase 4: Collect Observations ===
        ObservationState globalObs = getGlobalObservation();
        Map<Integer, ObservationState> localObs = getLocalObservations();

        // === Phase 5: Calculate Rewards ===
        double globalReward = calculateGlobalReward();
        Map<Integer, Double> localRewards = calculateLocalRewards();

        // === Phase 6: Check Termination ===
        boolean terminated = !simulation.isRunning();
        boolean truncated = currentStep >= settings.getMaxEpisodeLength();

        // === Phase 7: Build Info ===
        Map<String, Object> info = buildStepInfo(cloudletsRouted, localResults);

        LOGGER.debug("=== Step {} completed (Clock: {}s) ===", currentStep, currentClock);

        return new HierarchicalStepResult(
                globalObs, localObs,
                globalReward, localRewards,
                terminated, truncated, info
        );
    }

    /**
     * Execute global routing phase: route arriving cloudlets to datacenters.
     */
    private int executeGlobalRouting(List<Integer> globalActions) {
        // Get cloudlets arriving in current time window
        List<Cloudlet> arrivingCloudlets = globalBroker.getArrivingCloudlets(
                currentClock, timestepSize
        );

        if (arrivingCloudlets.isEmpty()) {
            LOGGER.debug("No cloudlets arriving in window [{}, {})",
                    currentClock, currentClock + timestepSize);
            return 0;
        }

        LOGGER.debug("{} cloudlets arriving, {} global actions provided",
                arrivingCloudlets.size(), globalActions.size());

        // Validate action count
        int actionCount = Math.min(arrivingCloudlets.size(), globalActions.size());

        // Route each cloudlet to its target datacenter
        int routedCount = 0;
        for (int i = 0; i < actionCount; i++) {
            Cloudlet cloudlet = arrivingCloudlets.get(i);
            int targetDcIndex = globalActions.get(i);

            boolean routed = globalBroker.routeCloudletToDatacenter(cloudlet, targetDcIndex);
            if (routed) {
                routedCount++;
            }
        }

        LOGGER.debug("Routed {}/{} cloudlets to datacenters", routedCount, arrivingCloudlets.size());
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

            // Local agent assigns a cloudlet from local queue to a VM
            boolean success = localBroker.assignCloudletToVm(targetVmId);
            results.put(dcId, success);
        }

        return results;
    }

    /**
     * Advance simulation time by one timestep.
     */
    private void advanceSimulationTime() {
        simulation.runFor(timestepSize);
        currentClock = simulation.clock();
        firstStep = false;
    }

    /**
     * Get global observation (aggregated state of all datacenters).
     *
     * For global agent, we provide:
     * - dc_green_power[] - Green energy availability per DC
     * - dc_queue_sizes[] - Waiting cloudlets per DC
     * - dc_utilizations[] - CPU usage per DC
     * - dc_available_pes[] - Free resources per DC
     * - upcoming_cloudlets_count - Arriving soon
     * - next_cloudlet_features - Next task info
     */
    public ObservationState getGlobalObservation() {
        int numDatacenters = datacenterInstances.size();

        // Create arrays for datacenter-level metrics
        double[] dcGreenPower = new double[numDatacenters];
        double[] dcQueueSizes = new double[numDatacenters];
        double[] dcUtilizations = new double[numDatacenters];
        double[] dcAvailablePes = new double[numDatacenters];

        // Collect metrics from each datacenter
        for (int i = 0; i < numDatacenters; i++) {
            DatacenterInstance dc = datacenterInstances.get(i);

            // Green power availability (normalized to kW scale)
            dcGreenPower[i] = dc.getCurrentGreenPowerW(currentClock) / 1000.0;

            // Waiting cloudlets count
            dcQueueSizes[i] = dc.getWaitingCloudletCount();

            // Average CPU utilization
            dcUtilizations[i] = dc.getAverageHostUtilization();

            // Available PEs across all VMs
            dcAvailablePes[i] = dc.getTotalAvailablePes();
        }

        // Get upcoming cloudlets info
        List<Cloudlet> upcomingCloudlets = globalBroker.getArrivingCloudlets(
                currentClock, timestepSize * 2);  // Look ahead 2 timesteps
        int upcomingCount = upcomingCloudlets.size();

        // Get next cloudlet features
        int nextCloudletPes = 0;
        if (!upcomingCloudlets.isEmpty()) {
            nextCloudletPes = (int) upcomingCloudlets.get(0).getPesNumber();
        }

        // Pack into infrastructure observation (reusing existing structure)
        int[] infrastructureObs = new int[numDatacenters * 4 + 2];
        for (int i = 0; i < numDatacenters; i++) {
            infrastructureObs[i] = (int) dcGreenPower[i];
            infrastructureObs[numDatacenters + i] = (int) dcQueueSizes[i];
            infrastructureObs[numDatacenters * 2 + i] = (int) (dcUtilizations[i] * 100);
            infrastructureObs[numDatacenters * 3 + i] = (int) dcAvailablePes[i];
        }
        infrastructureObs[numDatacenters * 4] = upcomingCount;
        infrastructureObs[numDatacenters * 4 + 1] = nextCloudletPes;

        // Create observation state (use DC metrics as "host" metrics for global view)
        return new ObservationState(
                dcUtilizations,  // hostLoads
                dcUtilizations,  // hostRamUsageRatio (reuse utilization)
                dcGreenPower,    // vmLoads (reuse for green power)
                new int[numDatacenters],  // vmTypes (not used at global level)
                new int[numDatacenters],  // vmHostMap (not used at global level)
                infrastructureObs,  // infrastructureObservation
                upcomingCount,   // waitingCloudlets
                nextCloudletPes, // nextCloudletPes
                convertDoubleArrayToIntArray(dcAvailablePes),  // vmAvailablePes
                numDatacenters,  // actualVmCount (reuse for DC count)
                numDatacenters,  // actualHostCount (reuse for DC count)
                0L,              // nextCloudletMi (not tracked at global level)
                0.0,             // nextCloudletWaitTime (not tracked at global level)
                new int[3],      // queuePesDistribution (not tracked at global level)
                0                // completedCloudletsLast10Steps (not tracked at global level)
        );
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
     * Calculate global reward (total energy, makespan, etc.).
     *
     * Global objectives:
     * - Minimize total energy consumption
     * - Maximize green energy ratio
     * - Minimize makespan (simulation time)
     * - Balance load across datacenters
     */
    private double calculateGlobalReward() {
        double reward = 0.0;

        // === Energy-based rewards ===
        double totalEnergyWh = 0.0;
        double totalGreenEnergyWh = 0.0;

        for (DatacenterInstance dc : datacenterInstances) {
            // Get datacenter power consumption (simplified)
            double dcPowerW = dc.getHostList().stream()
                    .mapToDouble(host -> {
                        if (host.getPowerModel() != null) {
                            return host.getPowerModel().getPower();
                        }
                        return 0.0;
                    })
                    .sum();

            // Energy consumed in this timestep (Wh)
            double energyWh = dcPowerW * (timestepSize / 3600.0);
            totalEnergyWh += energyWh;

            // Green energy available in this timestep (Wh)
            double greenPowerW = dc.getCurrentGreenPowerW(currentClock);
            double greenEnergyWh = greenPowerW * (timestepSize / 3600.0);
            totalGreenEnergyWh += Math.min(greenEnergyWh, energyWh);  // Actual green energy used
        }

        // Reward for green energy usage (higher ratio = higher reward)
        double greenEnergyRatio = totalEnergyWh > 0 ? totalGreenEnergyWh / totalEnergyWh : 0.0;
        double greenEnergyReward = greenEnergyRatio * 10.0;  // Scale: 0 to 10

        // Penalty for total energy consumption
        double energyPenalty = -totalEnergyWh / 1000.0;  // Convert to kWh and penalize

        // === Load balancing reward ===
        double loadBalancePenalty = calculateLoadBalancePenalty();

        // === Queue size penalty (global perspective) ===
        int totalWaitingCloudlets = datacenterInstances.stream()
                .mapToInt(DatacenterInstance::getWaitingCloudletCount)
                .sum();
        double queuePenalty = -totalWaitingCloudlets * 0.1;

        // === Combine rewards ===
        reward = greenEnergyReward + energyPenalty + loadBalancePenalty + queuePenalty;

        LOGGER.debug("Global Reward: total={:.3f} (green={:.3f}, energy={:.3f}, balance={:.3f}, queue={:.3f})",
                reward, greenEnergyReward, energyPenalty, loadBalancePenalty, queuePenalty);

        return reward;
    }

    /**
     * Calculate load balance penalty across datacenters.
     * Penalizes uneven distribution of work.
     */
    private double calculateLoadBalancePenalty() {
        if (datacenterInstances.isEmpty()) {
            return 0.0;
        }

        // Calculate variance in utilization across DCs
        double[] utilizations = datacenterInstances.stream()
                .mapToDouble(DatacenterInstance::getAverageHostUtilization)
                .toArray();

        double mean = Arrays.stream(utilizations).average().orElse(0.0);
        double variance = Arrays.stream(utilizations)
                .map(u -> Math.pow(u - mean, 2))
                .average()
                .orElse(0.0);

        // Penalize high variance (unbalanced load)
        return -variance * 5.0;
    }

    /**
     * Calculate local rewards for each datacenter.
     *
     * Local objectives (per DC):
     * - Minimize wait time
     * - Maximize resource utilization
     * - Minimize local queue size
     * - Penalize invalid actions
     */
    private Map<Integer, Double> calculateLocalRewards() {
        Map<Integer, Double> rewards = new HashMap<>();

        for (DatacenterInstance dc : datacenterInstances) {
            int dcId = dc.getId();
            double localReward = calculateSingleLocalReward(dc);
            rewards.put(dcId, localReward);
        }

        return rewards;
    }

    /**
     * Calculate reward for a single datacenter (local perspective).
     */
    private double calculateSingleLocalReward(DatacenterInstance dc) {
        double reward = 0.0;

        LoadBalancingBroker localBroker = dc.getLocalBroker();

        // === Queue size penalty ===
        int waitingCount = dc.getWaitingCloudletCount();
        double queuePenalty = -waitingCount * 0.5;

        // === Utilization reward ===
        double avgUtilization = dc.getAverageHostUtilization();
        // Encourage moderate utilization (too low = waste, too high = risk)
        double utilizationReward = 0.0;
        if (avgUtilization >= 0.5 && avgUtilization <= 0.9) {
            utilizationReward = avgUtilization * 5.0;  // Reward good utilization
        } else if (avgUtilization > 0.9) {
            utilizationReward = (1.0 - avgUtilization) * 2.0;  // Mild penalty for overload
        }

        // === Cloudlet completion reward ===
        // Reward for completed cloudlets in this step
        int completedInStep = 0;  // Could track this with listeners
        double completionReward = completedInStep * 1.0;

        // === Invalid action penalty ===
        // This will be added based on failed assignments (tracked externally)
        double invalidActionPenalty = 0.0;

        // === Combine local rewards ===
        reward = queuePenalty + utilizationReward + completionReward + invalidActionPenalty;

        LOGGER.debug("DC {} Local Reward: total={:.3f} (queue={:.3f}, util={:.3f}, completion={:.3f})",
                dc.getName(), reward, queuePenalty, utilizationReward, completionReward);

        return reward;
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
        return info;
    }

    /**
     * Check if simulation is still running.
     */
    public boolean isRunning() {
        return simulation != null && simulation.isRunning();
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
