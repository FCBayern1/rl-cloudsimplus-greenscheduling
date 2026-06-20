package exe.edu.cspg.multidc;

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

import exe.edu.cspg.common.CloudletDescriptor;
import exe.edu.cspg.common.DatacenterConfig;
import exe.edu.cspg.common.DatacenterSetup;
import exe.edu.cspg.common.SimulationSettings;
import exe.edu.cspg.common.VmAllocationPolicyCustom;
import exe.edu.cspg.energy.GreenEnergyProvider;
import exe.edu.cspg.singledc.LoadBalancingBroker;
import exe.edu.cspg.singledc.ObservationState;
import exe.edu.cspg.tables.CloudletsTableBuilderWithDetails;
import exe.edu.cspg.utils.WorkloadFileReader;
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
    // cloudletId → absolute completion deadline (sim seconds) for the deadline-aware
    // SLA. Populated from the descriptors (deferrable-batch traces). Empty when the
    // trace carries no deadline column → deadline_miss_rate stays 0.
    private final Map<Long, Long> cloudletDeadlineById = new HashMap<>();

    // === Simulation State ===
    private double currentClock = 0.0;
    private double timestepSize;
    private int currentStep = 0;
    private boolean firstReset = true;  // Track if this is the first reset

    // === Running Statistics for Reward Normalisation ===
    // Running max for carbon penalty signal normalisation (persists across episodes).
    // Note: depending on settings.carbon_penalty_mode, this may track:
    // - TOTAL: total step carbon (kg)
    // - PER_MI: carbon-per-MI signal (kg/MI)
    private double runningMaxCarbon = 1e-3;  // Initial small value to avoid division by zero
    private static final double EPSILON = 1e-8;
    private static final double CARBON_RATIO_MAX = 3.0;  // Cap for normalised carbon ratio

    // === Episode-level reward breakdown tracking (for logging/analysis) ===
    // Global reward terms per step: r_global = α·L - β·Ĉ - γ·Rw + per-action sum
    private double epGlobalTermLocalSum = 0.0;   // Σ (α·L)
    private double epGlobalTermCarbonSum = 0.0;  // Σ (-β·Ĉ)
    private double epGlobalTermWasteSum = 0.0;   // Σ (-γ·Rw)
    private double epGlobalTermThroughputSum = 0.0; // Σ (k_T · log1p(finished_mi_this_step))
    private double epGlobalTermCompletionMiSum = 0.0; // Σ (k_C · Δcompletion_rate_mi)

    // 2026-05-16 per-action reward decomposition tracking
    private double epGlobalTermPerActionSum = 0.0;        // Σ over steps of (Σᵢ rᵢ)
    private double epGlobalMarginalCarbonKgSum = 0.0;     // Σ over actions of marginal_kg (proxy "kg")


    // Per-step accumulators populated by accumulatePerActionReward() inside
    // executeGlobalRouting().  Consumed once per step in calculateGlobalReward()
    // (which then resets them).
    private double stepPerActionRewardSum = 0.0;   // Σᵢ rᵢ for this step
    private double stepMarginalCarbonKg = 0.0;     // Σᵢ marginal_kg for this step (logging only)
    private int    stepRoutedCount = 0;            // number of successful routings this step

    // Episode-level MI completion tracking (for completion_rate_mi shaping)
    private double episodeTotalWorkloadMi = 0.0;
    private double episodeFinishedMiCumulative = 0.0;
    private double prevCompletionRateMi = 0.0;

    // === Carbon penalty signal tracking (for debugging/analysis) ===
    // Track the *raw* carbon penalty signal that is normalized to produce Ĉ.
    // - TOTAL mode: signal = step total carbon (kg)
    // - PER_MI mode: signal = step carbon per completed MI (kg/MI) or CARBON_RATIO_MAX if MI==0
    private double epGlobalCarbonSignalSum = 0.0;        // Σ signal, total carbon emission sum at each timestep in kg
    private double epGlobalCarbonPenaltyNormSum = 0.0;   // Σ Ĉ
    private double lastGlobalCarbonSignal = 0.0;         // last step signal
    private double lastGlobalCarbonPenaltyNorm = 0.0;    // last step Ĉ

    // Local reward component sums per DC (Σ over steps within an episode)
    // Keys: dcId -> component sum
    private final Map<Integer, Double> epLocalWaitSum = new HashMap<>();
    private final Map<Integer, Double> epLocalUtilSum = new HashMap<>();
    private final Map<Integer, Double> epLocalQueueSum = new HashMap<>();
    private final Map<Integer, Double> epLocalInvalidSum = new HashMap<>();
    private final Map<Integer, Double> epLocalCompletionSum = new HashMap<>();

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

        LOGGER.info("Initialising MultiDatacenterSimulationCore with {} datacenters",
                datacenterConfigs.size());
    }

    /**
     * Reset and initialize the simulation environment.
     * Creates all datacenters, loads workload, and initializes brokers.
     */
    public void resetSimulation() {
        LOGGER.info("Resetting multi-datacenter simulation environment...");

        // Print cloudlet execution summary from previous episode (skip first reset).
        //
        // BUG-FIX 2026-05-12: This dump was the root cause of the 76-iter
        // training hang.  CloudletsTableBuilderWithDetails.build() writes the
        // entire finished-cloudlet table (~60k rows per episode in 10-DC v2)
        // straight to System.out, which Python captures into the gateway log.
        // After 84 episodes the log was 939MB; the JVM eventually hung mid-
        // dump (Python stdout-pipe back-pressure / disk-IO stall) and Python
        // saw "Answer from Java side is empty" on its next step() call.
        // Default OFF; turn on per-experiment with
        //   print_cloudlet_summary_on_reset: true
        // for one-off debugging.
        if (!firstReset && settings.isPrintCloudletSummaryOnReset()) {
            printCloudletExecutionSummary();
        }
        firstReset = false;

        stopSimulation();

        // Reset VM ID counter to ensure VM IDs start from 0 in each episode
        DatacenterSetup.resetVmIdCounter();

        // Reinitialise CloudSim Plus engine
        simulation = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        currentClock = 0.0;
        currentStep = 0;

        // Reset episode-level reward breakdown trackers
        epGlobalTermLocalSum = 0.0;
        epGlobalTermCarbonSum = 0.0;
        epGlobalTermWasteSum = 0.0;
        epGlobalTermThroughputSum = 0.0;
        epGlobalTermCompletionMiSum = 0.0;
        epGlobalTermPerActionSum = 0.0;
        epGlobalMarginalCarbonKgSum = 0.0;
        stepPerActionRewardSum = 0.0;
        stepMarginalCarbonKg = 0.0;
        stepRoutedCount = 0;
        epGlobalCarbonSignalSum = 0.0;
        epGlobalCarbonPenaltyNormSum = 0.0;
        lastGlobalCarbonSignal = 0.0;
        lastGlobalCarbonPenaltyNorm = 0.0;
        epLocalWaitSum.clear();
        epLocalUtilSum.clear();
        epLocalQueueSum.clear();
        epLocalInvalidSum.clear();
        epLocalCompletionSum.clear();

        // Reset running-max carbon normalisation for EPISODE mode
        if ("EPISODE".equals(settings.getCarbonNormalizationMode())) {
            runningMaxCarbon = 1e-3;
        }

        // === Step 1: Load Cloudlet Workload ===
        allCloudlets = loadAllCloudlets();
        LOGGER.info("Loaded {} cloudlets from workload trace", allCloudlets.size());

        // Pre-compute total workload MI for this episode (used by completion_rate_mi shaping).
        episodeTotalWorkloadMi = 0.0;
        if (allCloudlets != null && !allCloudlets.isEmpty()) {
            for (Cloudlet c : allCloudlets) {
                episodeTotalWorkloadMi += Math.max(0.0, (double) c.getLength());
            }
        }
        episodeFinishedMiCumulative = 0.0;
        prevCompletionRateMi = 0.0;

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

        // === Step 6: Initialise simulation by proceeding clock ===
        proceedClockTo(settings.getMinTimeBetweenEvents());
        LOGGER.info("Simulation clock initialized to {}", currentClock);

        // === Step 7: Preload arrivals for the first decision point ===
        // Keep observation semantics as post-arrival/pre-routing:
        // obs_0 should already include cloudlets that arrive in the first routing window.
        globalBroker.processArrivingCloudlets(currentClock, timestepSize);
        LOGGER.info("Initial arrivals processed. Global waiting queue size: {}",
                globalBroker.getGlobalWaitingCloudletsCount());

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

        // Record completion deadlines (cloudletId → deadline sim-seconds) before the
        // descriptors are converted to Cloudlets (toCloudlet drops the deadline).
        cloudletDeadlineById.clear();
        for (CloudletDescriptor d : descriptors) {
            if (d.hasDeadline()) {
                cloudletDeadlineById.put((long) d.getCloudletId(), d.getDeadlineTime());
            }
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
        if ("SWF".equalsIgnoreCase(settings.getWorkloadMode())) {
            reader.setSwfFilterFailedJobs(settings.isSwfFilterFailedJobs());
            reader.setSwfNormalizeSubmitTimes(settings.isSwfNormalizeSubmitTimes());
            reader.setSwfTimeScale(settings.getSwfTimeScale());
            reader.setSwfMaxPes(settings.getSwfMaxPes());
            reader.setSwfScaleMiByPes(settings.isSwfScaleMiByPes());
        }

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
                        config.getTimeZoneOffsetRows(),       // per-DC tz offset (geo)
                        config.getCompressedPowerDivisor(),   // COMPRESSED-mode divisor
                        config.getSimulationWarmupRows()      // global sim warm-up (eliminates cold start)
                );
                dcInstance.addGreenEnergyProvider(greenEnergyProvider);
            }
            LOGGER.info("Created {} GreenEnergyProvider(s) for DC {} with turbines {} (mode: {}, " +
                       "forecast: short={}rows, long={}rows, tzOffset={}rows, warmup={}rows)",
                    turbineIds.size(), config.getDatacenterId(), turbineIds,
                    config.getTimeScalingMode().getDescription(),
                    config.getShortTermRows(), config.getLongTermRows(),
                    config.getTimeZoneOffsetRows(), config.getSimulationWarmupRows());
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

        // === Phase 3.1: Preload arrivals for the next decision point ===
        // After advancing time, enqueue cloudlets arriving in the next routing window
        // so the returned observation is post-arrival/pre-routing for the next step.
        globalBroker.processArrivingCloudlets(currentClock, timestepSize);
        LOGGER.debug("Processed arrivals for next decision point. Global queue size: {}",
                globalBroker.getGlobalWaitingCloudletsCount());

        // === Phase 3.2: Update Energy Metrics for All Datacenters ===
        updateAllDatacenterEnergyMetrics();
        
        // === Phase 3.3: Update Cloudlet Completion Counts ===
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
     * Execute global routing phase: route a batch to datacenters.
     * 
     * Arrivals are processed outside this method (reset + post-advance phase),
     * so this method only consumes the existing global waiting queue.
     */
    private int executeGlobalRouting(List<Integer> globalActions) {
        // The queue at this point is already post-arrival/pre-routing for this step.
        int queueSize = globalBroker.getGlobalWaitingCloudletsCount();
        LOGGER.debug("Global queue size before routing: {}", queueSize);

        // Get batch for routing (up to the size of global actions provided)
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

        // Route each cloudlet to its target datacenter.
        // 2026-05-16: each successful routing also accumulates a per-action
        // reward (carbon cost + completion probability) into step-scoped
        // fields that calculateGlobalReward() consumes once per step.
        // try/catch around accounting so an exception in proxy formulas
        // can never strand cloudlets (which were removed from the queue
        // by getBatchForRouting and would leak if accounting throws).
        // Architecture B: the defer action index is num_dc (one past the last DC).
        final int deferActionIndex = datacenterInstances.size();
        final boolean deferEnabled = settings.isGlobalDeferEnabled();
        int routedCount = 0;
        int deferredCount = 0;
        for (int i = 0; i < actionCount; i++) {
            Cloudlet cloudlet = cloudletsToRoute.get(i);
            int targetDcIndex = globalActions.get(i);

            // DEFER: hold this cloudlet in the global queue (re-presented next step)
            // instead of routing it now — the temporal lever. The agent should defer
            // during brown and route during green (rewarded by the per-action carbon
            // reward, which uses current green). No per-action reward while held.
            if (deferEnabled && targetDcIndex == deferActionIndex) {
                globalBroker.requeueCloudletToTail(cloudlet);
                deferredCount++;
                continue;
            }

            boolean routed = globalBroker.routeCloudletToDatacenter(cloudlet, targetDcIndex);
            if (routed) {
                routedCount++;
                try {
                    accumulatePerActionReward(cloudlet, targetDcIndex);
                } catch (Throwable t) {
                    LOGGER.error("accumulatePerActionReward failed for cloudlet={} dc={}: {}",
                            cloudlet.getId(), targetDcIndex, t.getMessage(), t);
                }
            } else {
                // IMPORTANT: Do not lose cloudlets on routing failure.
                // Re-queue to tail so training can recover, while still allowing you to
                // penalize invalid/ineffective actions on the Python side if desired.
                globalBroker.requeueCloudletToTail(cloudlet);
            }
        }

        LOGGER.debug("Routed {}/{} cloudlets ({} deferred), {} remain in global queue",
                routedCount, cloudletsToRoute.size(), deferredCount,
                globalBroker.getGlobalWaitingCloudletsCount());
        return routedCount;
    }

    /**
     * Compute and accumulate the per-action reward contribution for one
     * (cloudlet → DC) routing decision.  Called from executeGlobalRouting()
     * immediately after a successful routing.
     *
     * <p>2026-05-16 Stage 1 — DIFFERENCE REWARD redesign.
     *
     * <p>For each cloudlet c_i, two DCs matter: the actual choice d_actual
     * (what the policy picked) and d_baseline (what Round-Robin would have
     * picked at this index).  The per-action reward measures the
     * <em>improvement over RR</em> rather than absolute cost:
     *
     * <pre>
     *   r_i = − w_carbon  · (marginal_kg(c, d_actual) − marginal_kg(c, d_baseline))
     *         + w_compl · (prob_complete(c, d_actual) − prob_complete(c, d_baseline))
     * </pre>
     *
     * <p>Why difference, not absolute (was the old design):
     * <ul>
     *   <li><b>Removes baseline noise.</b>  In the absolute formulation, a slot
     *       that just happened to route to a typical DC got ≈ the same reward
     *       no matter what.  All slots' rewards correlated with each other
     *       (because everyone was getting "the typical cost").  PPO's per-slot
     *       policy gradient drowned in this shared bias.</li>
     *   <li><b>Per-slot signal isolation.</b>  Now r_i = 0 if slot i picked
     *       the RR baseline; r_i ≠ 0 only when slot i actively deviated.
     *       Other slots' actions don't affect slot i's reward.  Inter-slot
     *       gradient noise drops dramatically.</li>
     *   <li><b>Stable initial policy.</b>  Uniform-random policy picks the
     *       RR baseline ~1/N of the time → average r_i ≈ 0.  PPO doesn't
     *       wander; only meaningful deviations create gradient.</li>
     * </ul>
     *
     * <p>Calibrated 2026-05-16 — typical |r_i| ≈ ±0.02 (smaller than absolute
     * version's ±0.05 because the baseline-constant part cancels).  Still
     * comparable to λ·c_step mid-training, so Lagrangian and shaping balance.
     */
    private void accumulatePerActionReward(Cloudlet cloudlet, int dcIndex) {
        if (dcIndex < 0 || dcIndex >= datacenterInstances.size()) {
            return;
        }
        int numDcs = datacenterInstances.size();
        if (numDcs == 0) return;

        // 2026-05-20 reward redesign — switched from "diff vs RR baseline" to
        // ABSOLUTE per-action reward.
        //
        // Why we dropped the RR baseline:
        //   r_i_diff = -w_c·(marg_a − marg_RR) + w_compl·(prob_a − prob_RR)
        // ...looks like a counterfactual baseline, but RR isn't on the Pareto
        // frontier — it's just round-robin and doesn't read the obs.  Once
        // the policy is better than RR on either axis, the diff flips sign
        // (positive when worse, negative when better) and PPO sees a noisy
        // signal that goes negative right when the agent is improving.
        // Empirical evidence: 100-iter run 20260519_185531 had per_action_sum
        // go from +159 (iter 5) → −267 (iter 40, the BEST c/c point) → +133
        // (iter 110, drift point) — i.e., signal points wrong way at the best
        // policy.  See conversation log 2026-05-20.
        //
        // New formula (absolute, normalized to ~[0, 1] so weights are stable):
        //   r_i = -w_c · (marg_actual / marg_normalizer) + w_compl · prob_actual
        //
        // Both terms are in ~[0, 1] range so w_c / w_compl ratio directly
        // controls the carbon/completion trade-off at the per-action level.
        // The Lagrangian outer loop (still active) provides the episode-level
        // SLA pressure on top.
        DcCostFeatures actual = computeDcCostFeatures(cloudlet, dcIndex);
        if (actual == null) {
            // Defensive — shouldn't happen since dcIndex was already validated
            // by routeCloudletToDatacenter.
            return;
        }

        double wCarbon       = settings.getPerActionCarbonWeight();
        double wCompletion   = settings.getPerActionCompletionWeight();
        double margNormalize = Math.max(1e-6, settings.getPerActionMargNormalizer());

        double margNorm = actual.marginalKg / margNormalize;  // ~[0, 1] for typical cloudlets
        double rAbsolute = -wCarbon * margNorm
                         + wCompletion * actual.probComplete;

        stepPerActionRewardSum += rAbsolute;
        stepMarginalCarbonKg   += actual.marginalKg;
        stepRoutedCount++;

        LOGGER.trace(
            "PerActionAbs dc={} marg={} margNorm={} prob={} r_i={}",
            dcIndex, actual.marginalKg, margNorm, actual.probComplete, rAbsolute
        );
    }

    /**
     * Pure-function helper: compute the (marginal_kg, prob_complete) pair for
     * routing a given cloudlet to a given DC index.  Used by
     * {@link #accumulatePerActionReward(Cloudlet, int)} to evaluate both the
     * actual and the baseline routing.  Returns null if dcIndex is invalid.
     */
    private DcCostFeatures computeDcCostFeatures(Cloudlet cloudlet, int dcIndex) {
        if (dcIndex < 0 || dcIndex >= datacenterInstances.size()) {
            return null;
        }
        DatacenterInstance dc = datacenterInstances.get(dcIndex);
        if (dc == null) return null;

        double greenPowerW = dc.isGreenEnergyEnabled()
                ? Math.max(0.0, dc.getCurrentGreenPowerW(currentClock))
                : 0.0;
        double currentPowerW = estimateDcCurrentPowerW(dc);
        double greenRatio = currentPowerW > 1e-9
                ? Math.min(1.0, greenPowerW / currentPowerW)
                : (greenPowerW > 0.0 ? 1.0 : 0.0);

        double brownFactor = dc.getConfig().getBrownCarbonFactor();
        double greenFactor = dc.getConfig().getGreenCarbonFactor();
        int availPes  = Math.max(0, dc.getTotalAvailablePes());
        int queueSize = Math.max(0, dc.getWaitingCloudletCount());
        int dcCapacityPes = Math.max(1, dc.getHostCount() * settings.getSmallVmPes());

        int  cloudletPes = (int)  Math.max(1L, cloudlet.getPesNumber());
        long cloudletMi  = (long) Math.max(0L, cloudlet.getLength());

        double miPerKg = Math.max(1e3, settings.getMiPerKgFactor());
        double effFactor = greenRatio * greenFactor + (1.0 - greenRatio) * brownFactor;
        double marginalKg = ((double) cloudletMi / miPerKg) * effFactor;

        double overflowFrac = Math.max(0.0,
                (double) (queueSize + cloudletPes - availPes) / (double) dcCapacityPes);
        double k = settings.getPerActionOverflowSharpness();
        double probComplete = Math.exp(-k * overflowFrac);

        return new DcCostFeatures(marginalKg, probComplete);
    }

    /** Small pair-struct: (marginalKg, probComplete) for one (cloudlet, dc) candidate. */
    private static final class DcCostFeatures {
        final double marginalKg;
        final double probComplete;
        DcCostFeatures(double mk, double pc) { this.marginalKg = mk; this.probComplete = pc; }
    }

    /**
     * Rough estimate of a DC's current power demand in Watts, used only to
     * compute an instantaneous green_ratio at routing time.  Reads from the
     * latest energy-delta snapshot (populated at the previous step's tail).
     * Returns 0.0 before the first updateAllDatacenterEnergyMetrics call.
     */
    private double estimateDcCurrentPowerW(DatacenterInstance dc) {
        EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
        if (delta == null) {
            return 0.0;
        }
        return Math.max(0.0, delta.getCurrentPowerW());
    }

    /**
     * NEW local agent (dispatch_rate mode): each DC's action value = how many
     * cloudlets to RELEASE from its queue this step (0 = hold all = the temporal
     * lever's "defer"; higher = burst-drain). Each released cloudlet is placed on
     * the least-loaded VM — placement is green-IRRELEVANT (all VMs in a DC share
     * its green; total DC power is invariant to which VM runs a cloudlet when
     * hosts are always-on), so RL owns only the green-relevant temporal decision.
     * Unthrottles the legacy 1-cloudlet/DC/step throughput bottleneck.
     */
    private Map<Integer, Boolean> executeLocalDispatchRate(Map<Integer, Integer> localActions) {
        Map<Integer, Boolean> results = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : localActions.entrySet()) {
            DatacenterInstance dc = resolveDatacenterInstance(entry.getKey());
            if (dc == null) {
                results.put(entry.getKey(), false);
                continue;
            }
            final int dcId = dc.getId();
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker == null) {
                results.put(dcId, false);
                continue;
            }
            int dispatchCount = Math.max(0, entry.getValue());

            // Local free-PE view of the VM fleet. assignCloudletToVm submits each
            // cloudlet via a FUTURE CLOUDLET_SUBMIT event, so vm.getFreePesNumber()
            // does NOT reflect cloudlets dispatched earlier in THIS step. Without our
            // own bookkeeping, "most-free VM" returns the same VM every iteration and
            // the whole step's dispatch piles onto a single VM (fleet starves, util
            // caps ~10-15%). Track the running free-PE count locally and decrement on
            // each assignment so cloudlets spread across the fleet.
            Map<Long, Long> freePes = new HashMap<>();
            for (Vm vm : dc.getVmPool()) {
                if (vm.isCreated() && !vm.isFailed()) {
                    freePes.put(vm.getId(), vm.getFreePesNumber());
                }
            }
            int placed = 0;
            while (placed < dispatchCount && localBroker.hasWaitingCloudlets()) {
                Cloudlet next = localBroker.peekWaitingCloudlet();
                if (next == null) {
                    break;
                }
                final long need = Math.max(1, next.getPesNumber());
                // most-free VM that still fits the next cloudlet (least-loaded, local view)
                long vmId = -1;
                long bestFree = -1;
                for (Map.Entry<Long, Long> e : freePes.entrySet()) {
                    long f = e.getValue();
                    if (f >= need && f > bestFree) {
                        bestFree = f;
                        vmId = e.getKey();
                    }
                }
                if (vmId < 0) {
                    break; // no VM (accounting for in-step fills) can host the next cloudlet
                }
                if (localBroker.assignCloudletToVm(vmId)) {
                    freePes.put(vmId, freePes.get(vmId) - need);
                    placed++;
                } else {
                    break; // refused despite pre-check; avoid spinning
                }
            }
            // Any dispatch count is a valid action (0 = legitimate hold).
            results.put(dcId, true);
        }
        return results;
    }

    /**
     * Execute local scheduling phase: assign cloudlets to VMs within each datacenter.
     */
    private Map<Integer, Boolean> executeLocalScheduling(Map<Integer, Integer> localActions) {
        // NEW dispatch-rate local agent (gated; the legacy vm_placement path
        // below is unchanged when the flag is off). See docs/Deferrable_Jobs_Lever.md.
        if ("dispatch_rate".equals(settings.getLocalDispatchMode())) {
            return executeLocalDispatchRate(localActions);
        }

        Map<Integer, Boolean> results = new HashMap<>();

        for (Map.Entry<Integer, Integer> entry : localActions.entrySet()) {
            int dcKey = entry.getKey();
            int targetVmId = entry.getValue();

            // IMPORTANT: dcKey in localActions is expected to be the Datacenter ID (from config),
            // not necessarily the 0..N-1 index in datacenterInstances.
            // To be robust, try to resolve by DC ID first; fall back to index only if needed.
            DatacenterInstance dc = resolveDatacenterInstance(dcKey);
            if (dc == null) {
                LOGGER.warn("Invalid datacenter key: {} (cannot resolve to any DatacenterInstance)", dcKey);
                results.put(dcKey, false);
                continue;
            }
            final int dcId = dc.getId();
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker == null) {
                LOGGER.error("DC {}: LocalBroker not initialized. Cannot apply local action.", dcId);
                results.put(dcId, false);
                continue;
            }

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

                    // Debug info (use DEBUG level to avoid log spam during training)
                    LOGGER.debug("DC {}: vmPool size: {}, local index: {}, actual VM ID: {}",
                            dcId, vmPool.size(), targetVmId, actualVmId);

                    int brokerVmCount = localBroker.getVmCreatedList().size();
                    LOGGER.debug("DC {}: Broker has {} VMs created", dcId, brokerVmCount);

                    // DEBUG: Print first few VM IDs in vmPool (for first action of each DC)
                    if (targetVmId < 3) {
                        StringBuilder vmPoolIds = new StringBuilder("DC " + dcId + " vmPool IDs: [");
                        for (int i = 0; i < Math.min(5, vmPool.size()); i++) {
                            vmPoolIds.append(vmPool.get(i).getId());
                            if (i < Math.min(5, vmPool.size()) - 1) vmPoolIds.append(", ");
                        }
                        vmPoolIds.append(", ...]");
                        LOGGER.debug(vmPoolIds.toString());

                        // Also print broker's actual VM IDs
                        StringBuilder brokerVmIds = new StringBuilder("DC " + dcId + " broker VM IDs: [");
                        List<Vm> brokerVms = localBroker.getVmCreatedList();
                        for (int i = 0; i < Math.min(5, brokerVms.size()); i++) {
                            brokerVmIds.append(brokerVms.get(i).getId());
                            if (i < Math.min(5, brokerVms.size()) - 1) brokerVmIds.append(", ");
                        }
                        brokerVmIds.append(", ...]");
                        LOGGER.debug(brokerVmIds.toString());
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
        // If a DC is missing from localActions, it means "no local decision provided this step".
        // We still need to mark the action as invalid when cloudlets are waiting, otherwise
        // reward calculation will default it to "valid" and skip the invalid-action penalty.
        
        for (DatacenterInstance dc : datacenterInstances) {
            if (dc == null) {
                continue;
            }
            final int dcId = dc.getId();
            if (results.containsKey(dcId)) {
                continue; // already processed via provided action (or invalid dcKey resolution)
            }

            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker == null) {
                // If broker is missing, treat as invalid/no-op.
                results.put(dcId, false);
                continue;
            }

            // Missing action behaves like an implicit NoAssign decision.
            // Penalize only when there is work waiting.
            if (localBroker.hasWaitingCloudlets()) {
                LOGGER.debug(
                        "DC {}: No local action provided but {} cloudlets are waiting. Invalid action.",
                        dcId, localBroker.getWaitingCloudletCount()
                );
                results.put(dcId, false);
            } else {
                results.put(dcId, true);
            }
        }

        return results;
    }

    /**
     * Resolve a DatacenterInstance from a key that may be either a datacenterId (preferred)
     * or a datacenter index (fallback for backward compatibility).
     */
    private DatacenterInstance resolveDatacenterInstance(int dcKey) {
        // 1) Try resolve by configured datacenterId
        for (DatacenterInstance dc : datacenterInstances) {
            if (dc != null && dc.getId() == dcKey) {
                return dc;
            }
        }
        // 2) Fallback: treat as index
        if (dcKey >= 0 && dcKey < datacenterInstances.size()) {
            return datacenterInstances.get(dcKey);
        }
        return null;
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
        // Per-DC cleanup: finished wait times + cloudletCreatedList.
        // Clearing cloudletCreatedList mirrors singledc/SimulationCore.clearListsIfNeeded()
        // and is gated on settings.isClearCreatedLists() (default true). Without this
        // the list grows monotonically with episode length and CloudSim Plus per-step
        // bookkeeping over it makes per-step latency grow linearly (observed 0.15s -> 1.32s
        // in a 5000-step rollout). Safe because this env does not perform VM upscaling
        // (the only CloudSim path that re-reads the created list).
        boolean clear = settings.isClearCreatedLists();
        for (DatacenterInstance dc : datacenterInstances) {
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker != null) {
                localBroker.clearFinishedWaitTimes();
                if (clear) {
                    localBroker.getCloudletCreatedList().clear();
                }
            }
        }
        if (clear && globalBroker != null) {
            globalBroker.getCloudletCreatedList().clear();
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

        // Get recent completed cloudlets (across all DCs) within the last timestep.
        // Use the broker's per-step finished list rather than the cumulative finished list.
        int recentCompleted = 0;
        for (DatacenterInstance dc : datacenterInstances) {
            LoadBalancingBroker broker = dc.getLocalBroker();
            if (broker == null) continue;
            List<Cloudlet> finishedLastStep = broker.getCloudletsFinishedLastStep(currentClock);
            if (finishedLastStep != null) {
                recentCompleted += finishedLastStep.size();
            }
        }

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
     * NEW DESIGN (v2 - Normalized Hierarchical Reward):
     *
     * r_global = α·L - β·Ĉ - γ·R_w
     *
     * Where:
     * - L = Average local reward (1/N × Σ r_local)
     * - Ĉ = Running-max normalized carbon (C / M_C, capped at CARBON_RATIO_MAX)
     * - R_w = Green waste ratio (G_waste / (G_waste + G_used))
     *
     * Coefficients (from config):
     * - α (alpha): Local performance weight (default: 1.0)
     * - β (beta): Carbon penalty weight (default: 0.5)
     * - γ (gamma): Green waste penalty weight (default: 0.3)
     *
     * @param localRewards Map of DC_ID -> local reward
     */
    private double calculateGlobalReward(Map<Integer, Double> localRewards) {
        int N = datacenterInstances.size();
        if (N == 0) return 0.0;

        // Get coefficients from settings
        double alpha = settings.getGlobalRewardAlpha();  // Local performance weight
        double beta = settings.getGlobalRewardBeta();    // Carbon penalty weight
        double gamma = settings.getGlobalRewardGamma();  // Green waste penalty weight
        double throughputMiCoef = settings.getGlobalThroughputMiCoef();
        double completionRateMiCoef = settings.getGlobalCompletionRateMiCoef();

        // === Component 1: Average Local Reward (L) ===
        double avgLocalReward = localRewards.values().stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);

        // === Component 2: Normalized Carbon Penalty (Ĉ) ===
        double normalizedCarbon = calculateNormalizedCarbonPenalty();

        // === Component 3: Green Waste Ratio (R_w) ===
        double wasteRatio = calculateGreenWasteRatio();

        // === Optional Component 4: Throughput shaping (MI finished in this step) ===
        // Encourages doing more useful work per unit time.
        double completedMiThisStep = getCompletedMiThisStep();
        double throughputTerm = throughputMiCoef * Math.log1p(Math.max(0.0, completedMiThisStep));

        // === Optional Component 5: MI-based completion progress shaping ===
        // completion_rate_mi = finished_mi / total_workload_mi (episode-to-date)
        episodeFinishedMiCumulative += Math.max(0.0, completedMiThisStep);
        double completionRateMiNow = episodeTotalWorkloadMi > 0
                ? (episodeFinishedMiCumulative / (episodeTotalWorkloadMi + EPSILON))
                : 0.0;
        double deltaCompletionMi = completionRateMiNow - prevCompletionRateMi;
        prevCompletionRateMi = completionRateMiNow;
        double completionMiTerm = completionRateMiCoef * deltaCompletionMi;

        // === NEW 2026-05-16 Component 6: Per-action reward sum ===
        // Σᵢ rᵢ accumulated by accumulatePerActionReward() during executeGlobalRouting
        // earlier in this same step.  Each rᵢ depends only on slot i's action, so
        // the resulting per-step contribution is additively decomposable across the
        // N routing slots — gives PPO a real attribution gradient even with shared
        // advantage.  See accumulatePerActionReward javadoc for the math.
        double perActionTerm = stepPerActionRewardSum;

        // === Final Global Reward ===
        double reward = alpha * avgLocalReward
                      - beta * normalizedCarbon
                      - gamma * wasteRatio;
        // Add shaping terms (default coefs are 0.0, so this is backward-compatible)
        reward += throughputTerm + completionMiTerm + perActionTerm;

        // Track episode-level component contributions for analysis (cumulative sums over steps)
        epGlobalTermLocalSum += (alpha * avgLocalReward);
        epGlobalTermCarbonSum += (-beta * normalizedCarbon);
        epGlobalTermWasteSum += (-gamma * wasteRatio);
        epGlobalTermThroughputSum += throughputTerm;
        epGlobalTermCompletionMiSum += completionMiTerm;
        epGlobalTermPerActionSum += perActionTerm;
        epGlobalMarginalCarbonKgSum += stepMarginalCarbonKg;

        // Reset per-step accumulators (consumed for this step's reward).
        stepPerActionRewardSum = 0.0;
        stepMarginalCarbonKg = 0.0;
        stepRoutedCount = 0;

        LOGGER.debug("Global Reward: total={} (α·L={}, β·Ĉ={}, γ·Rw={})",
                String.format("%.4f", reward),
                String.format("%.4f", alpha * avgLocalReward),
                String.format("%.4f", beta * normalizedCarbon),
                String.format("%.4f", gamma * wasteRatio));
        LOGGER.debug("  Components: L={}, Ĉ={}, Rw={} | Coeffs: α={}, β={}, γ={}",
                String.format("%.4f", avgLocalReward),
                String.format("%.4f", normalizedCarbon),
                String.format("%.4f", wasteRatio),
                alpha, beta, gamma);
        LOGGER.debug("  Shaping: throughputMi={} (coef={}, term={}), completionRateMiNow={} (coef={}, Δ={}, term={})",
                String.format("%.2f", completedMiThisStep),
                throughputMiCoef,
                String.format("%.4f", throughputTerm),
                String.format("%.4f", completionRateMiNow),
                completionRateMiCoef,
                String.format("%.6f", deltaCompletionMi),
                String.format("%.4f", completionMiTerm));

        return reward;
    }

    private double computeGlobalCompletionRate() {
        long totalReceived = 0L;
        long totalFinished = 0L;
        for (DatacenterInstance dc : datacenterInstances) {
            if (dc == null) continue;
            // DatacenterInstance tracks per-DC counters via getCloudletsReceived/getCloudletsCompleted
            totalReceived += Math.max(0L, (long) dc.getCloudletsReceived());
            totalFinished += Math.max(0L, (long) dc.getCloudletsCompleted());
        }
        if (totalReceived <= 0L) return 0.0;
        return (double) totalFinished / (double) totalReceived;
    }

    /**
     * Calculate normalized carbon penalty using running maximum.
     *
     * Formula: Ĉ = min(C / (M_C + ε), c_max)
     *
     * Where:
     * - C = current timestep total carbon (kg)
     * - M_C = running maximum carbon (updated each step)
     * - ε = small constant to avoid division by zero
     * - c_max = cap to prevent extreme ratios
     *
     * @return Normalized carbon penalty in range [0, CARBON_RATIO_MAX]
     */
    private double calculateNormalizedCarbonPenalty() {
        // 1) Sum carbon emissions from all datacenters for this timestep
        double totalCarbonKg = 0.0;
        int validDatacenters = 0;
        for (DatacenterInstance dc : datacenterInstances) {
            EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
            if (delta == null) continue;
            validDatacenters++;
            totalCarbonKg += delta.getDeltaCarbonEmissionKg();
        }
        if (validDatacenters == 0) return 0.0;

        // 2) Compute the penalty signal depending on mode
        final String mode = settings.getCarbonPenaltyMode();
        final double signal;

        if ("PER_MI".equals(mode)) {
            // Compute completed MI in this timestep across all datacenters.
            double completedMiThisStep = getCompletedMiThisStep();
            double miFloor = settings.getCarbonMiFloor();
            if (miFloor > 0.0) {
                // Smooth variant: divide by max(completedMI, miFloor).  Produces a bounded but
                // non-saturating penalty on idle steps, yielding usable gradients instead of
                // a constant CARBON_RATIO_MAX clip.
                double denomMi = Math.max(completedMiThisStep, miFloor);
                signal = totalCarbonKg / (denomMi + EPSILON); // kg/MI
            } else if (completedMiThisStep <= 0.0) {
                // Legacy behaviour (miFloor<=0): saturate on idle to discourage "doing nothing".
                lastGlobalCarbonSignal = CARBON_RATIO_MAX;
                lastGlobalCarbonPenaltyNorm = CARBON_RATIO_MAX;
                epGlobalCarbonSignalSum += lastGlobalCarbonSignal;
                epGlobalCarbonPenaltyNormSum += lastGlobalCarbonPenaltyNorm;
                return CARBON_RATIO_MAX;
            } else {
                signal = totalCarbonKg / (completedMiThisStep + EPSILON); // kg/MI
            }
        } else {
            // Default: TOTAL (kg) per timestep
            signal = totalCarbonKg;
        }

        // 3) Normalize the signal according to carbon_normalization_mode
        final String normMode = settings.getCarbonNormalizationMode();
        final double denominator;

        if ("FIXED".equals(normMode)) {
            double fixedMax = settings.getCarbonNormalizationFixedMax();
            if (fixedMax <= 0.0) {
                LOGGER.warn("carbon_normalization_mode=FIXED but carbon_normalization_fixed_max={} (<=0). "
                        + "Falling back to RUNNING_MAX.", fixedMax);
                runningMaxCarbon = Math.max(runningMaxCarbon, signal);
                denominator = runningMaxCarbon;
            } else {
                denominator = fixedMax;
            }
        } else {
            // RUNNING_MAX (default) or EPISODE — both use the same running-max logic,
            // but EPISODE resets runningMaxCarbon in resetSimulation().
            runningMaxCarbon = Math.max(runningMaxCarbon, signal);
            denominator = runningMaxCarbon;
        }

        // NOTE: do NOT add EPSILON to `denominator` here.  EPSILON=1e-8 is sized for
        // dividing-by-MI (lines 1088, 1181, 1190) where the denominator is in the 1e6+
        // range; FIXED mode lets the user supply `carbon_normalization_fixed_max` which
        // is typically 1e-12..1e-9 (kg/MI) and would be silently dominated by EPSILON,
        // producing `norm = signal × 1e8` instead of `signal / fixed_max`.  Use a
        // dedicated tiny floor so we still avoid div-by-zero if `denominator` is 0.
        final double denomFloor = 1e-30;
        double normalizedCarbon = Math.min(
                signal / Math.max(denominator, denomFloor),
                CARBON_RATIO_MAX
        );

        // 4) Track signal + normalized penalty for logging/analysis
        lastGlobalCarbonSignal = signal;
        lastGlobalCarbonPenaltyNorm = normalizedCarbon;
        epGlobalCarbonSignalSum += signal;
        epGlobalCarbonPenaltyNormSum += normalizedCarbon;

        LOGGER.trace("  Carbon(mode={}, norm={}): total={}kg, signal={}, denominator={}, normalized={}",
                mode, normMode,
                String.format("%.6f", totalCarbonKg),
                String.format("%.8f", signal),
                String.format("%.8f", denominator),
                String.format("%.4f", normalizedCarbon));

        return normalizedCarbon;
    }

    /**
     * Computes the total completed work (MI) in the current timestep across all datacenters.
     * Used for carbon penalty normalization in PER_MI mode.
     */
    private double getCompletedMiThisStep() {
        double totalMi = 0.0;
        for (DatacenterInstance dc : datacenterInstances) {
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker == null) continue;
            List<Cloudlet> finished = localBroker.getCloudletsFinishedLastStep(currentClock);
            if (finished == null || finished.isEmpty()) continue;
            for (Cloudlet c : finished) {
                // Cloudlet length is in MI (CloudSim Plus convention)
                totalMi += Math.max(0.0, (double) c.getLength());
            }
        }
        return totalMi;
    }

    /**
     * Calculate green energy waste ratio.
     *
     * Formula: R_w = G_waste / (G_waste + G_used)
     *
     * Range: [0, 1] where 0 = no waste, 1 = all green energy wasted
     *
     * @return Waste ratio in range [0, 1]
     */
    private double calculateGreenWasteRatio() {
        double totalGreenUsed = 0.0;
        double totalGreenWasted = 0.0;

        for (DatacenterInstance dc : datacenterInstances) {
            EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
            if (delta == null) {
                continue;
            }
            totalGreenUsed += delta.getDeltaGreenEnergyUsedWh();
            totalGreenWasted += delta.getDeltaGreenEnergyWastedWh();
        }

        double total = totalGreenUsed + totalGreenWasted;
        if (total <= 0) {
            return 0.0;
        }

        double wasteRatio = totalGreenWasted / total;

        LOGGER.trace("  Green Waste: used={}Wh, wasted={}Wh, ratio={}",
                String.format("%.2f", totalGreenUsed),
                String.format("%.2f", totalGreenWasted),
                String.format("%.4f", wasteRatio));

        return wasteRatio;
    }

    // ============================================================================
    // OLD REWARD FUNCTION (COMMENTED OUT FOR REFERENCE)
    // ============================================================================
    /*
    // OLD: calculateGlobalReward - used sum of local rewards + raw carbon penalty
    private double calculateGlobalReward_OLD(Map<Integer, Double> localRewards) {
        double sumLocalRewards = localRewards.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        double carbonEmissionPenalty = calculateCarbonEmissionPenalty_OLD();
        double reward = sumLocalRewards - carbonEmissionPenalty;

        LOGGER.debug("Global Reward (OLD): total={} (sumLocal={}, carbonPenalty={})",
                reward, sumLocalRewards, carbonEmissionPenalty);
        return reward;
    }

    // OLD: calculateCarbonEmissionPenalty - used raw penalty coefficient × carbon
    private double calculateCarbonEmissionPenalty_OLD() {
        double totalCarbonKg = 0.0;
        int validDatacenters = 0;

        for (DatacenterInstance dc : datacenterInstances) {
            EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
            if (delta == null) continue;
            validDatacenters++;
            totalCarbonKg += delta.getDeltaCarbonEmissionKg();
        }

        if (validDatacenters == 0) return 0.0;

        double penaltyCoef = settings.getCarbonEmissionPenaltyCoef();
        return penaltyCoef * totalCarbonKg;
    }
    */

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

        // === Architecture A: temporal lever at the LOCAL agent (default OFF) ===
        // Only when reward_local_carbon_enabled=true. The local agent then decides
        // hold-vs-release and needs a carbon signal, so its reward becomes a per-DC
        // GREEN-FRACTION reward + completion (drops queue/wait/util that punish
        // holding). In architecture B (global deferral, local = QoS) this stays off
        // and the validated QoS reward below is used unchanged.
        if (settings.isRewardLocalCarbonEnabled()) {
            double localGreenCoef = settings.getRewardLocalGreenCoef();
            double completionCoefD = settings.getRewardCompletionCoef();
            double greenFrac = 0.0;
            EnergyMetricsDelta d = dc.getLatestEnergyDelta();
            if (d != null) {
                double green = d.getDeltaGreenEnergyUsedWh();
                double brown = d.getDeltaBrownEnergyUsedWh();
                double consumed = green + brown;
                if (consumed > 1e-9) {
                    greenFrac = green / consumed;   // ∈ [0,1]; 0 when nothing ran this step
                }
            }
            double greenReward = localGreenCoef * greenFrac;
            int completedNow = (localBroker != null)
                    ? localBroker.getCloudletsFinishedLastStep(currentClock).size() : 0;
            double completionRewardD = completionCoefD * Math.log1p(completedNow);
            double total = greenReward + completionRewardD;
            final int dcIdD = dc.getId();
            epLocalCompletionSum.put(dcIdD, epLocalCompletionSum.getOrDefault(dcIdD, 0.0) + completionRewardD);
            LOGGER.debug("DC {} dispatch-rate local reward: total={} (green={}·{}={}, completion={}[{}])",
                    dc.getName(), String.format("%.3f", total), localGreenCoef,
                    String.format("%.2f", greenFrac), String.format("%.3f", greenReward),
                    String.format("%.3f", completionRewardD), completedNow);
            return total;
        }

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

        // === 2. Utilization & Balance Penalty (Single-DC aligned) ===
        // Only penalize imbalance (variance across active VMs), and only when there is work waiting.
        // Rationale: the local agent controls *distribution* of load, not total workload amount.
        double utilizationPenalty = 0.0;
        int waitingCloudletsForUtilPenalty = dc.getWaitingCloudletCount();
        if (waitingCloudletsForUtilPenalty > 0) {
            List<Vm> runningVms = dc.getVmPool();
            if (runningVms != null && !runningVms.isEmpty()) {
                // Filter for created and active VMs
                List<Vm> activeVms = runningVms.stream()
                        .filter(vm -> vm.isCreated() && !vm.isFailed())
                        .collect(java.util.stream.Collectors.toList());

                if (!activeVms.isEmpty()) {
                    double avgUtilization = activeVms.stream()
                            .mapToDouble(Vm::getCpuPercentUtilization)
                            .average()
                            .orElse(0.0);

                    double variance = activeVms.stream()
                            .mapToDouble(vm -> Math.pow(vm.getCpuPercentUtilization() - avgUtilization, 2))
                            .average()
                            .orElse(0.0);

                    utilizationPenalty = -utilizationCoef * Math.sqrt(variance);
                }
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

        // === 5. Completion Reward (Single-DC aligned) ===
        // Positive shaping reward for completions in this timestep.
        // Use log1p scaling to prevent reward explosion when many tasks finish at once.
        double completionReward = 0.0;
        int completedThisStep = 0;
        if (localBroker != null) {
            completedThisStep = localBroker.getCloudletsFinishedLastStep(currentClock).size();
            completionReward = completionCoef * Math.log1p(completedThisStep);
        }

        // === Total Reward (no clipping, same as LoadBalancerGateway) ===
        double totalReward = waitTimePenalty + utilizationPenalty + queuePenalty + invalidActionPenalty + completionReward;

        // Track per-DC episode component sums for analysis/logging
        final int dcId = dc.getId();
        epLocalWaitSum.put(dcId, epLocalWaitSum.getOrDefault(dcId, 0.0) + waitTimePenalty);
        epLocalUtilSum.put(dcId, epLocalUtilSum.getOrDefault(dcId, 0.0) + utilizationPenalty);
        epLocalQueueSum.put(dcId, epLocalQueueSum.getOrDefault(dcId, 0.0) + queuePenalty);
        epLocalInvalidSum.put(dcId, epLocalInvalidSum.getOrDefault(dcId, 0.0) + invalidActionPenalty);
        epLocalCompletionSum.put(dcId, epLocalCompletionSum.getOrDefault(dcId, 0.0) + completionReward);

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

            // Add local reward breakdown (episode cumulative sums)
            // These are sums of the per-step components accumulated inside calculateSingleLocalReward(...)
            int dcId = dc.getId();
            dcMetrics.put("local_reward_wait_sum", epLocalWaitSum.getOrDefault(dcId, 0.0));
            dcMetrics.put("local_reward_util_sum", epLocalUtilSum.getOrDefault(dcId, 0.0));
            dcMetrics.put("local_reward_queue_sum", epLocalQueueSum.getOrDefault(dcId, 0.0));
            dcMetrics.put("local_reward_invalid_sum", epLocalInvalidSum.getOrDefault(dcId, 0.0));
            dcMetrics.put("local_reward_completion_sum", epLocalCompletionSum.getOrDefault(dcId, 0.0));

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
        double totalWorkloadMi = 0.0;
        double totalFinishedMi = 0.0;

        // Total workload MI (across all cloudlets in the episode workload)
        if (allCloudlets != null && !allCloudlets.isEmpty()) {
            for (Cloudlet c : allCloudlets) {
                totalWorkloadMi += Math.max(0.0, (double) c.getLength());
            }
        }

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

            // Aggregate finished MI from broker's finished list (episode-to-date)
            LoadBalancingBroker localBroker = dc.getLocalBroker();
            if (localBroker != null) {
                List<Cloudlet> finishedCloudlets = localBroker.getCloudletFinishedList();
                if (finishedCloudlets != null && !finishedCloudlets.isEmpty()) {
                    for (Cloudlet c : finishedCloudlets) {
                        totalFinishedMi += Math.max(0.0, (double) c.getLength());
                    }
                }
            }
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

        // Add MI-based workload stats (useful for carbon-per-work metrics)
        stats.put("total_workload_cloudlets", (allCloudlets != null) ? allCloudlets.size() : 0);
        stats.put("total_workload_mi", totalWorkloadMi);
        stats.put("total_finished_mi", totalFinishedMi);
        stats.put("carbon_kg_per_mi", totalFinishedMi > 0 ? (totalCarbonKg / (totalFinishedMi + EPSILON)) : 0.0);
        double completionRateMi = totalWorkloadMi > 0 ? (totalFinishedMi / (totalWorkloadMi + EPSILON)) : 0.0;
        stats.put("completion_rate_mi", completionRateMi);

        // ---------------------------------------------------------------------
        // Deadline-aware SLA signal (deferrable-batch carbon lever, 2026-06-20).
        // A cloudlet with a deadline MISSES if it finished after its deadline, or
        // is still unfinished once the clock has passed its deadline. The
        // Lagrangian uses deadline_miss_rate so the agent may freely defer work
        // into green windows as long as it still completes before the deadline.
        // ---------------------------------------------------------------------
        double deadlineMissRate = 0.0;
        int deadlineTotal = cloudletDeadlineById.size();
        if (deadlineTotal > 0) {
            Map<Long, Double> finishTimeById = new HashMap<>();
            for (DatacenterInstance dc : datacenterInstances) {
                LoadBalancingBroker lb = dc.getLocalBroker();
                if (lb == null) continue;
                List<Cloudlet> fin = lb.getCloudletFinishedList();
                if (fin == null) continue;
                for (Cloudlet c : fin) {
                    finishTimeById.put(c.getId(), c.getFinishTime());
                }
            }
            long misses = 0;
            for (Map.Entry<Long, Long> e : cloudletDeadlineById.entrySet()) {
                double deadline = e.getValue();
                Double finish = finishTimeById.get(e.getKey());
                if (finish != null) {
                    if (finish > deadline) misses++;          // finished late
                } else if (currentClock > deadline) {
                    misses++;                                  // unfinished past deadline
                }
            }
            deadlineMissRate = (double) misses / (double) deadlineTotal;
        }
        stats.put("deadline_miss_rate", deadlineMissRate);
        stats.put("deadline_total", deadlineTotal);

        // ---------------------------------------------------------------------
        // SLA / Lagrangian cost signals
        // ---------------------------------------------------------------------
        // pending_ratio = (received − finished) / received  (episode-to-date)
        // c_step        = max(0, pending_ratio − d)          (dense per-step cost)
        // c_ep          = max(0, c* − completion_rate_mi)    (episode-level cost)
        // Python-side Lagrangian callback reads these keys, updates λ, and the env
        // wrapper applies r_train_global = r_step − λ · c_step.
        double pendingRatio;
        if (totalCloudletsReceived > 0) {
            long pendingCount = (long) totalCloudletsReceived - (long) totalCloudletsCompleted;
            if (pendingCount < 0) pendingCount = 0;
            pendingRatio = (double) pendingCount / (double) totalCloudletsReceived;
        } else {
            pendingRatio = 0.0;
        }
        double slaPendingTarget = settings.getSlaPendingTarget();
        double slaTarget = settings.getSlaTarget();
        double slaCostStep;
        double slaCostEpisode;
        if ("deadline_miss".equals(settings.getSlaMode()) && deadlineTotal > 0) {
            // Deadline-aware: constrain deadline_miss_rate (NOT raw pending/completion),
            // so deferring work into green windows is free as long as deadlines hold.
            double missTarget = settings.getSlaDeadlineMissTarget();
            double missCost = Math.max(0.0, deadlineMissRate - missTarget);
            slaCostStep = missCost;
            slaCostEpisode = missCost;
        } else {
            slaCostStep = Math.max(0.0, pendingRatio - slaPendingTarget);
            slaCostEpisode = Math.max(0.0, slaTarget - completionRateMi);
        }
        stats.put("sla_pending_ratio", pendingRatio);
        stats.put("sla_cost_step", slaCostStep);
        stats.put("sla_cost_episode", slaCostEpisode);
        stats.put("sla_target", slaTarget);
        stats.put("sla_pending_target", slaPendingTarget);

        // Expose the actual carbon penalty signal used by the global reward (pre-normalization)
        stats.put("global_carbon_signal_last", lastGlobalCarbonSignal);
        stats.put("global_carbon_signal_sum", epGlobalCarbonSignalSum);
        stats.put("global_carbon_signal_mean", currentStep > 0 ? epGlobalCarbonSignalSum / currentStep : 0.0);
        // Expose the normalized penalty Ĉ used in r_global = α·L - β·Ĉ - γ·Rw
        stats.put("global_carbon_penalty_norm_last", lastGlobalCarbonPenaltyNorm);
        stats.put("global_carbon_penalty_norm_sum", epGlobalCarbonPenaltyNormSum);
        stats.put("global_carbon_penalty_norm_mean", currentStep > 0 ? epGlobalCarbonPenaltyNormSum / currentStep : 0.0);

        // Add global reward breakdown (episode cumulative term sums)
        // r_global = α·L - β·Ĉ - γ·Rw
        stats.put("global_reward_term_local_sum", epGlobalTermLocalSum);
        stats.put("global_reward_term_carbon_sum", epGlobalTermCarbonSum);
        stats.put("global_reward_term_waste_sum", epGlobalTermWasteSum);
        stats.put("global_reward_term_throughput_sum", epGlobalTermThroughputSum);
        stats.put("global_reward_term_completion_mi_sum", epGlobalTermCompletionMiSum);
        // Optional per-step means (divide by currentStep; 0 if episode not progressed)
        stats.put("global_reward_term_local_mean", currentStep > 0 ? epGlobalTermLocalSum / currentStep : 0.0);
        stats.put("global_reward_term_carbon_mean", currentStep > 0 ? epGlobalTermCarbonSum / currentStep : 0.0);
        stats.put("global_reward_term_waste_mean", currentStep > 0 ? epGlobalTermWasteSum / currentStep : 0.0);
        stats.put("global_reward_term_throughput_mean", currentStep > 0 ? epGlobalTermThroughputSum / currentStep : 0.0);
        stats.put("global_reward_term_completion_mi_mean", currentStep > 0 ? epGlobalTermCompletionMiSum / currentStep : 0.0);

        // 2026-05-16 per-action reward decomposition stats
        stats.put("global_reward_term_per_action_sum", epGlobalTermPerActionSum);
        stats.put("global_reward_term_per_action_mean",
                  currentStep > 0 ? epGlobalTermPerActionSum / currentStep : 0.0);
        // Diagnostic: total "marginal kg" attributed (sum of unweighted marginal_kg values).
        stats.put("global_per_action_marginal_carbon_proxy_sum", epGlobalMarginalCarbonKgSum);

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
        if (globalBroker != null) {
            // Cloudlets not yet injected into the global routing queue
            if (globalBroker.getRemainingCloudletCount() > 0) {
                return true;
            }
            // Cloudlets already arrived but not routed yet
            if (globalBroker.getGlobalWaitingCloudletsCount() > 0) {
                return true;
            }
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
     * <p>Installs a listener that fires after every event processed.  When
     * the future-events queue is about to drain (only the current event left)
     * AND there are still unfinished cloudlets to dispatch, the listener
     * schedules a NONE event one timestep ahead so the simulation does not
     * call {@link org.cloudsimplus.core.CloudSimPlus#finish()} prematurely.
     *
     * <p>BUG-FIX 2026-05-13 (REWRITE of the 2026-05-12 fix):
     *
     * <p>The original implementation used {@code getNumberOfFutureEvents() == 1}
     * which is O(N) over a TreeMap on EVERY event — this caused 1st-step hang
     * on v2 workload (see commit 10d6d87 message).  The 2026-05-12 attempted
     * fix used {@code simulation.noFutureEvents()} (O(1) — checks queue
     * emptiness), but this BROKE cloudlet completion (run
     * {@code 20260512_225043}: 0 cloudlets finished over 7200 steps).
     *
     * <p>The reason: by inspecting the CloudSim Plus 8.5.5 bytecode for
     * {@code processFutureEventsHappeningAtSameTimeOfTheFirstOne}:
     * <pre>
     *   0: processEvent(firstEvent)    ← fires THIS listener
     *   5: future.remove(firstEvent)   ← only NOW is event removed from queue
     * </pre>
     * When the listener runs, the event being processed is STILL in the queue,
     * so {@code noFutureEvents()} (which is {@code future.isEmpty()}) always
     * returns {@code false} — the listener never fires.  Sim then calls
     * {@code finish()} as soon as the last real event drains.
     *
     * <p>The original {@code == 1} semantic was correct: it means "the only
     * event left is the one being processed right now" — i.e. the queue will
     * be empty after the upcoming {@code future.remove()} call.
     *
     * <p>This rewrite keeps the {@code == 1} semantic but evaluates it in
     * O(1) using {@code isThereAnyFutureEvt} with a short-circuiting counter
     * predicate: it returns true on the second event matched, so the predicate
     * runs at most twice regardless of queue size.
     */
    private void ensureAllCloudletsCompleteBeforeSimulationEnds() {
        double interval = settings.getSimulationTimestep();
        simulation.addOnEventProcessingListener(info -> {
            // O(1) equivalent of `getNumberOfFutureEvents() == 1`: the
            // predicate is invoked for each queued event until one returns
            // true.  `++counter[0] > 1` returns true on the SECOND match
            // (counter goes 1→2), so the stream short-circuits after seeing
            // at most 2 events.  isThereAnyFutureEvt then returns:
            //   - true  iff queue has ≥ 2 events  → don't schedule NONE
            //   - false iff queue has ≤ 1 event   → about to drain
            // (We can't have a literal queue size 0 here because the event
            // being processed is still in the queue at this point in the
            // CloudSim Plus event-processing chain.)
            final int[] counter = {0};
            boolean hasMoreThanOne = simulation.isThereAnyFutureEvt(e -> ++counter[0] > 1);
            if (!hasMoreThanOne && hasUnfinishedCloudlets()) {
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
