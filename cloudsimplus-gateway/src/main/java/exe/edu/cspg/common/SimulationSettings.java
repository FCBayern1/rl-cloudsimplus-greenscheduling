package exe.edu.cspg.common;
import exe.edu.cspg.common.SimulationSettings;

import java.util.Map;
import java.util.Objects;
import java.util.ArrayList;
import java.util.List;

import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import exe.edu.cspg.energy.TimeScalingMode;

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
    // Local scheduling mode (2026-06-18, new dispatch-rate local agent):
    //   "vm_placement" = legacy (local action = pick 1 VM, 1 cloudlet/DC/step)
    //   "dispatch_rate"= new (local action = how many to release this step; sim
    //                    best-fit places them). Unthrottles throughput AND makes
    //                    the action the green-relevant temporal lever.
    //                    See docs/Deferrable_Jobs_Lever.md.
    private final String localDispatchMode;
    // Architecture B — temporal lever at the GLOBAL agent (2026-06-20). When true,
    // the global routing action gains a DEFER option (index = num_dc): instead of
    // routing a cloudlet to a DC, the agent HOLDS it in the global queue (re-presented
    // next step) so it can be routed later when green arrives. The per-action carbon
    // reward already rewards routing-during-green (marginalKg uses current green), so
    // deferring-then-routing-on-green is naturally incentivized. Default false.
    private final boolean globalDeferEnabled;
    // Fix A (deadline backstop): when the agent picks DEFER for a cloudlet whose
    // deadline is within deferDeadlineSlackSec of the current clock, override the
    // defer and force-route it to the greenest available DC instead. Prevents the
    // deterministic policy from deferring work indefinitely (starvation collapse).
    private final boolean deferDeadlineForceEnabled;
    private final double deferDeadlineSlackSec;
    // SQT2.2 (2026-08-18): "latest_start" forces iff now + estimated_runtime
    // + slack >= deadline (runtime = MI/(PES*MIPS), the reward-ledger unit).
    // The legacy fixed-lead rule (now + slack >= deadline) fires ~600 s early
    // for tight-slack jobs and would erase the U[200,900] class entirely.
    private final String deferDeadlineForceMode;
    // SQT2.2-Clean physics alignment (Codex ruling, 2026-08-18): per-cloudlet
    // CPU utilization. Default 0.5 preserves byte-level legacy/v3 semantics;
    // SQT2 experiments set 1.0 so execution obeys runtime = MI/(PES*MIPS).
    private final double cloudletCpuUtilization;
    // Fix B (urgency deferral cost): deferring a cloudlet is no longer free. The global
    // per-action reward gets −deferUrgencyWeight·urgency for each deferred cloudlet, where
    // urgency = clamp(1 − (deadline − now)/deferUrgencyWindowSec, 0, 1). Fresh work (far
    // from deadline) ≈ free to defer; near-deadline work is expensive → the agent learns
    // to route urgent work instead of mindlessly deferring (which a forecast otherwise
    // makes look free). Default weight 0.0 = disabled (backward compatible).
    private final double deferUrgencyWeight;
    private final double deferUrgencyWindowSec;
    // Route A (honest defer): a small ALWAYS-ON base cost charged per deferred cloudlet on top
    // of the urgency cost, so even fresh work isn't free to defer. Stops the argmax "always
    // defer" drift. Anchored to a fraction of a green-routing reward. Default 0.0 = disabled.
    private final double deferBaseCost;
    private final String cloudletTraceFile; // Path for trace file
    private final int maxCloudletsToCreateFromWorkloadFile; // Limit for SWF mode
    private final int workloadReaderMips; // MIPS ref for SWF runtime calculation
    /**
     * When true, SWF jobs with status==0 are filtered out (treated as failed).
     * Some traces have status column set to 0 for all rows (meaning "unknown" after cleaning),
     * so this must be configurable per experiment.
     */
    private final boolean swfFilterFailedJobs;
    /** If true, subtract the first seen SWF submitTime so arrivals start at t=0. */
    private final boolean swfNormalizeSubmitTimes;
    /** Divide SWF submit times by this factor to compress long traces into shorter episodes (>= 1.0). */
    private final double swfTimeScale;
    /** Cap SWF requested processors to this value (e.g., 8 to match max VM size). */
    private final int swfMaxPes;
    /**
     * If true, compute Cloudlet MI as runTime * mips * pes to preserve SWF runtime on pes processors.
     * If false, uses runTime * mips (legacy behavior).
     */
    private final boolean swfScaleMiByPes;
    /**
     * Carbon emission factor (kg CO2 per kWh). Used when reporting carbon metrics.
     */
    private final double carbonEmissionFactorKgPerKwh;
    private final boolean splitLargeCloudlets;
    private final int maxCloudletPes;
    /**
     * Whether to dump the full per-episode CloudletsTable to stdout at every
     * resetSimulation() call.  This was on by default historically but is a
     * known cause of pathological log growth (≈60k rows/episode × N episodes,
     * 940MB+ over a 76-iter training run) and stdout-pipe back-pressure that
     * can hang the JVM when the Python side falls behind reading.  Default
     * OFF — turn on only for one-off debugging sessions.
     */
    private final boolean printCloudletSummaryOnReset;

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
    private final double rewardCompletionCoef;   // Coefficient for completion rate reward (positive reward)
    // Local-carbon reward (architecture A — temporal lever at the LOCAL agent).
    // Default OFF. When true (and dispatch_rate), the local reward becomes a per-DC
    // GREEN-FRACTION reward + completion (drops queue/wait/util). Use this ONLY if the
    // lever lives at the local dispatch agent. In architecture B (global deferral,
    // local = pure QoS) leave this false so the validated QoS reward is used.
    private final boolean rewardLocalCarbonEnabled;
    private final double rewardLocalGreenCoef;
    private final double greenWastePenaltyCoef;  // Base coefficient for green waste penalty (scaled by DC count)
    private final double carbonEmissionPenaltyCoef;  // Coefficient for carbon emission penalty (OLD, kept for reference)

    // Global Reward Coefficients (NEW normalized reward scheme)
    // r_global = α·L - β·Ĉ - γ·R_w
    private final double globalRewardAlpha;  // α: Local performance weight (default: 1.0)
    private final double globalRewardBeta;   // β: Carbon penalty weight (default: 0.5)
    private final double globalRewardGamma;  // γ: Green waste penalty weight (default: 0.3)
    // Optional shaping terms to align global reward with throughput + completion objectives:
    // - throughput: encourages finishing more MI per step
    // - completion progress: encourages increasing MI-based completion rate over the episode
    private final double globalThroughputMiCoef;       // k_T: weight for log1p(MI_finished_this_step)
    private final double globalCompletionRateMiCoef;   // k_C: weight for Δcompletion_rate_mi per step

    // ===========================================================================
    // 2026-05-16 per-action reward decomposition (NEW).
    //
    // Each routing decision (cloudlet c_i → DC d_i) gets its own immediate
    // reward r_i = -w_carbon · marginal_kg(c_i, d_i) + w_compl · prob_complete(c_i, d_i),
    // and the step-level global reward is Σ_i r_i.  Because r_i depends only on
    // slot i's action, the per-step reward is additively decomposable across
    // the N batch slots — PPO's policy gradient ∇ log π_i(a_i|s) naturally
    // attributes credit to each slot's choice even though the advantage
    // A(s, a) is shared.
    //
    // Calibrated against smoke 20260515_174514 (uniform policy):
    //   Per typical cloudlet (mi=700K, brown=0.5, green=0.5):
    //     marginal_kg = (700K / 3.5e6) · 0.25 = 0.05  (× w_carbon = 1.0)
    //     prob_complete ≈ 0.9                          (× w_compl = 0.05)
    //   Per-step (10 actions): −0.5 carbon + 0.45 completion ≈ −0.05
    //   Lagrangian λ · c_step ≈ −0.1 at mid-training (comparable magnitude).
    // ===========================================================================
    /** Calibration constant: MI workload that produces 1 unit of marginal kg
     *  at brown_factor=1.0, green_ratio=0.0. Default 3.5e6 — chosen so per-
     *  action reward magnitude sits around ±0.05 (same scale as Lagrangian). */
    private final double miPerKgFactor;
    /** w_carbon: scales the carbon-cost term (−w_carbon · marginal_kg). */
    private final double perActionCarbonWeight;
    /** w_completion: scales the completion-probability bonus (+w_compl · prob). */
    private final double perActionCompletionWeight;
    /** Sigmoid sharpness for the overflow→completion-prob mapping.
     *  prob = exp(−k · overflow_frac); k=3 gives ~0.55 at 20% overflow, ~0.05 at 100%. */
    private final double perActionOverflowSharpness;
    /** 2026-05-20: divisor that maps marginal_kg into a ~[0, 1] range so the
     *  absolute per-action reward (post-RR-baseline-removal) has a stable
     *  magnitude regardless of cloudlet MI distribution.  Default 0.05
     *  corresponds to a "typical" 700K-MI cloudlet at brown=0.5, green=0.5
     *  (marginalKg = 700e3 / 3.5e6 · 0.255 ≈ 0.051).
     *  See MultiDatacenterSimulationCore.accumulatePerActionReward for use. */
    private final double perActionMargNormalizer;
    /** 2026-08-06: when true, the per-action marginal-carbon estimate uses the
     *  MEAN green power over the cloudlet's expected run window (MI / (pes ·
     *  vmPeMips) steps) instead of the instantaneous green power at routing
     *  time. With long jobs the instantaneous ratio actively mis-credits
     *  routing (a DC green now but brown for the run window looks good, and
     *  the window-optimal DC looks bad), teaching the policy greedy behaviour
     *  regardless of forecast observability. Default false: short-job
     *  experiments keep the historical (equivalent) behaviour. */
    private final boolean perActionWindowCarbon;
    /** 2026-08-09: which green series feeds the window-carbon reward.
     *  "actual" (default) = true future green over the run window — an oracle
     *  reward that leaks future information to EVERY arm during training,
     *  including no-forecast ablations (the v2026-gamble lesson: the blind arm
     *  learned oracle-grade timing from reward supervision alone).
     *  "persistence" = current green held flat over the window, so an arm's
     *  learning signal contains no information its observations do not. */
    private final String windowCarbonSource;
    /** 2026-08-13 (V3.1 reward surgery, all default to legacy behaviour):
     *  per_action_completion_mode — "bonus" (legacy +w·p) | "no_offset" (−w·(1−p),
     *  removes the constant +w that routes got but defer did not; ordering
     *  between DCs unchanged). */
    private final String perActionCompletionMode;
    /** defer_cost_mode — "flat" (legacy: −base −w·U charged at EVERY sighting) |
     *  "incremental_urgency" (−base once on the first explicit defer, plus
     *  telescoping −w·[U(now)−U(last)] settled at every later sighting incl.
     *  the final route, so cost is independent of encounter count and a job
     *  cannot escape its last waiting segment; U(s)=clip(1−s/W,0,1)²). */
    private final String deferCostMode;
    /** per_action_carbon_norm — "fixed" (legacy kg/normalizer) | "scale_only"
     *  (kg/σ, physical zero kept) | "centered_zscore" (clip((kg−μ)/σ,±5); the
     *  +w·μ/σ constant on routes is a DELIBERATE carbon-threshold design:
     *  below-mean-carbon routes beat defer). μ/σ come from the offline
     *  calibration artifact (calibrate_reward_norm.py), shared by every arm —
     *  never from online statistics, which would give workers/arms different
     *  reward functions. */
    private final String perActionCarbonNorm;
    private final double perActionCarbonMu;
    private final double perActionCarbonSigma;
    /** 2026-08-14 (V3.2 two-scale split, default legacy-off):
     *  per_action_spatial_center — "none" | "candidate_mean". When on, the route
     *  reward gains −w_s·(C_j − mean over FEASIBLE candidate DCs)/σ_spatial:
     *  a mean-zero DC-ranking term that cannot bias route-vs-defer, separated
     *  from the carbon LEVEL term (centered_zscore) which keeps the
     *  worth-running-now threshold semantics. σ_spatial comes from the offline
     *  calibration artifact (candidate-difference distribution), shared by all arms. */
    private final String perActionSpatialCenter;
    private final double perActionSpatialWeight;
    private final double perActionSpatialSigma;
    /** 2026-08-07: >0 enables the per-episode green-window shift (rows). Each reset
     *  applies offset (1009*episodeIndex mod range) to ALL green providers, so every
     *  episode replays a different slice of the wind year while cross-DC phases stay
     *  fixed. 0 (default) keeps the historical fixed-window behaviour. */
    private final int greenEpisodeOffsetRange;

    /** Explicit episode offsets (rows), cycled by episode index when non-empty. */
    private final java.util.List<Integer> greenEpisodeOffsetAllowlist;

    /** "13016;21088" / "13016,21088" / a List of numbers -> ints; null or blank -> empty. */
    public static java.util.List<Integer> parseIntList(Object raw) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        if (raw == null) return out;
        if (raw instanceof java.util.List<?> list) {
            for (Object o : list) out.add(((Number) o).intValue());
            return out;
        }
        String s = String.valueOf(raw).trim();
        if (s.isEmpty()) return out;
        for (String tok : s.split("[;,\\s]+")) {
            if (!tok.isEmpty()) out.add(Integer.parseInt(tok.trim()));
        }
        return out;
    }

    /** Wind year the turbine resolver prefers; 2021 keeps the historical behaviour. */
    @Getter
    private final int windCsvYear;

    /**
     * Controls how the global carbon penalty signal is computed in multi-DC mode.
     *
     * Options:
     * - TOTAL: penalize total step carbon emission (kg) normalized by running max
     * - PER_MI: penalize carbon per completed MI in the timestep (kg/MI), normalized by running max.
     *          This discourages "doing less work" to reduce total emissions.
     */
    private final String carbonPenaltyMode;

    /**
     * Controls how the carbon penalty signal is normalised.
     *
     * Options:
     * - RUNNING_MAX: (default) divide by cross-episode running maximum of the signal.
     * - FIXED: divide by a fixed constant ({@link #carbonNormalizationFixedMax}).
     *          Keeps reward scale fully stationary across training.
     * - EPISODE: running maximum that resets at the start of each episode.
     */
    private final String carbonNormalizationMode;

    /**
     * Fixed denominator for carbon normalisation when {@link #carbonNormalizationMode} is "FIXED".
     * Must be > 0 when FIXED mode is active.  A good starting value can be estimated from
     * the mean step-carbon of a few random-policy episodes.
     */
    private final double carbonNormalizationFixedMax;

    /**
     * MI floor used in PER_MI carbon penalty mode.  Instead of returning CARBON_RATIO_MAX when
     * no work completes in a step (which produces a constant saturating penalty that adds noise
     * and no gradient), we divide by max(completedMI, carbonMiFloor).  When <=0, PER_MI falls
     * back to the legacy "return CARBON_RATIO_MAX on idle" behaviour.
     */
    private final double carbonMiFloor;

    /**
     * SLA / Lagrangian constraint parameters.  The global reward can be shaped by an outer
     * Lagrangian loop: r_train = r_step − λ · c_step, with λ updated between training
     * iterations based on the episode-level violation c_ep.
     *
     * slaTarget (c*) — minimum acceptable completion rate (MI-based) for an episode.
     * slaPendingTarget (d) — per-step threshold for pending_ratio = (received − finished) /
     *   received.  c_step = max(0, pending_ratio − d) gives dense cost signal.
     * slaLagrangianEnabled — flag purely for Java-side logging; the actual λ lives on the
     *   Python side and multiplies c_step in the env wrapper.  Java only exposes the raw
     *   cost signals in the info dict.
     */
    private final double slaTarget;
    private final double slaPendingTarget;
    private final boolean slaLagrangianEnabled;
    // Deadline-aware SLA (deferrable-batch carbon lever, 2026-06-20). slaMode
    // = "deadline_miss" → Lagrangian constraint = deadline_miss_rate ≤
    // slaDeadlineMissTarget (lets the agent defer freely while meeting deadlines);
    // "completion" (default) keeps the legacy completion-based SLA.
    private final String slaMode;
    private final double slaDeadlineMissTarget;

    // Green Energy Configuration
    private final boolean greenEnergyEnabled;
    private final int turbineId;
    /**
     * Optional multi-turbine support (mainly used when single-DC reuses multi-DC datacenter configs).
     * If not provided, falls back to a singleton list containing turbineId.
     */
    private final List<Integer> turbineIds;
    private final String windDataFile;
    private final TimeScalingMode timeScalingMode;
    private final int timeZoneOffsetRows;
    private final double brownCarbonFactor;
    private final double greenCarbonFactor;
    private final boolean greenPredictionEnabled;
    private final String predictionModelPath;
    private final double predictionCacheDuration;
    private final int predictionHorizon;
    
    // Future energy forecast configuration
    private final int shortTermRows;   // Short-term forecast rows (default: 3 = 30 min)
    private final int longTermRows;    // Long-term forecast rows (default: 144 = 24 hours)

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
        this.localDispatchMode = getStringParam(params, "local_dispatch_mode", "vm_placement").trim();
        this.globalDeferEnabled = getBoolParam(params, "global_defer_enabled", false);
        this.deferDeadlineForceEnabled = getBoolParam(params, "defer_deadline_force_enabled", true);
        this.windCsvYear = getIntParam(params, "wind_csv_year", 2021);
        this.deferDeadlineSlackSec = getDoubleParam(params, "defer_deadline_slack_sec", 600.0);
        this.deferDeadlineForceMode = getStringParam(params, "defer_deadline_force_mode", "legacy").trim();
        this.cloudletCpuUtilization = getDoubleParam(params, "cloudlet_cpu_utilization", 0.5);
        this.deferUrgencyWeight = getDoubleParam(params, "defer_urgency_weight", 0.0);
        this.deferUrgencyWindowSec = getDoubleParam(params, "defer_urgency_window_sec", 3600.0);
        this.deferBaseCost = getDoubleParam(params, "defer_base_cost", 0.0);
        this.cloudletTraceFile = getStringParam(params, "cloudlet_trace_file",
                "traces/LLNL-Atlas-2006-2.1-cln-test.swf");
        this.maxCloudletsToCreateFromWorkloadFile = getIntParam(params, "max_cloudlets_to_create_from_workload_file",
                Integer.MAX_VALUE);
        this.workloadReaderMips = getIntParam(params, "workload_reader_mips", (int) this.hostPeMips);
        this.swfFilterFailedJobs = getBoolParam(params, "swf_filter_failed_jobs", true);
        this.swfNormalizeSubmitTimes = getBoolParam(params, "swf_normalize_submit_times", false);
        this.swfTimeScale = getDoubleParam(params, "swf_time_scale", 1.0);
        this.swfMaxPes = getIntParam(params, "swf_max_pes", Integer.MAX_VALUE);
        this.swfScaleMiByPes = getBoolParam(params, "swf_scale_mi_by_pes", true);
        this.carbonEmissionFactorKgPerKwh = getDoubleParam(params, "carbon_emission_factor_kg_per_kwh", 0.5);

        this.splitLargeCloudlets = getBoolParam(params, "split_large_cloudlets", true);
        // Default maxCloudletPes to the largest VM's PE count if not specified
        int defaultMaxCloudletPes = this.smallVmPes * this.largeVmMultiplier;
        this.maxCloudletPes = getIntParam(params, "max_cloudlet_pes", defaultMaxCloudletPes);
        // Default OFF — dumping ~60k cloudlet rows per episode to stdout cost
        // a 76-iter run a 940MB gateway log and is the prime suspect for the
        // 2026-05-12 JVM hang (stdout pipe back-pressure).  Set to true
        // explicitly via config when you actually want the table.
        this.printCloudletSummaryOnReset = getBoolParam(params, "print_cloudlet_summary_on_reset", false);

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
        this.rewardCompletionCoef = getDoubleParam(params, "reward_completion_coef", 1.0); // Positive reward for completion
        // Default 1.0 (aligned with per_action_carbon_weight): the green-fraction
        // reward (≤1.0) must be comparable to the completion reward (≤~1.2) so the
        // lever's payoff isn't drowned by completion + discounting of delayed work.
        this.rewardLocalCarbonEnabled = getBoolParam(params, "reward_local_carbon_enabled", false);
        this.rewardLocalGreenCoef = getDoubleParam(params, "reward_local_green_coef", 1.0);
        this.greenWastePenaltyCoef = getDoubleParam(params, "green_waste_penalty_coef", 10.0); // Base coef per DC
        this.carbonEmissionPenaltyCoef = getDoubleParam(params, "carbon_emission_penalty_coef", 1.0); // Default 1.0 (OLD, kept for reference)

        // Global Reward Coefficients (NEW normalized reward scheme)
        // r_global = α·L - β·Ĉ - γ·R_w
        this.globalRewardAlpha = getDoubleParam(params, "global_reward_alpha", 1.0);  // α: Local performance weight
        this.globalRewardBeta = getDoubleParam(params, "global_reward_beta", 0.5);    // β: Carbon penalty weight
        this.globalRewardGamma = getDoubleParam(params, "global_reward_gamma", 0.3);  // γ: Green waste penalty weight
        this.globalThroughputMiCoef = getDoubleParam(params, "global_throughput_mi_coef", 0.0);
        // Backward-compat alias: global_completion_rate_coef (older name used in some configs)
        this.globalCompletionRateMiCoef = getDoubleParam(
                params,
                "global_completion_rate_mi_coef",
                getDoubleParam(params, "global_completion_rate_coef", 0.0)
        );

        // 2026-05-16 per-action reward decomposition.  All default to OFF so
        // legacy configs (α/β/γ) continue to behave identically.  Activate by
        // setting per_action_carbon_weight and/or per_action_completion_weight
        // non-zero in the experiment config.
        this.miPerKgFactor              = getDoubleParam(params, "mi_per_kg_factor", 3.5e6);
        this.perActionCarbonWeight      = getDoubleParam(params, "per_action_carbon_weight", 0.0);
        this.perActionCompletionWeight  = getDoubleParam(params, "per_action_completion_weight", 0.0);
        this.perActionOverflowSharpness = getDoubleParam(params, "per_action_overflow_sharpness", 3.0);
        this.perActionMargNormalizer    = getDoubleParam(params, "per_action_marg_normalizer", 0.05);
        this.perActionWindowCarbon      = getBoolParam(params, "per_action_window_carbon", false);
        this.windowCarbonSource         = getStringParam(params, "window_carbon_source", "actual").trim().toLowerCase();
        this.perActionCompletionMode    = getStringParam(params, "per_action_completion_mode", "bonus").trim().toLowerCase();
        this.deferCostMode              = getStringParam(params, "defer_cost_mode", "flat").trim().toLowerCase();
        this.perActionCarbonNorm        = getStringParam(params, "per_action_carbon_norm", "fixed").trim().toLowerCase();
        this.perActionCarbonMu          = getDoubleParam(params, "per_action_carbon_mu", 0.0);
        this.perActionCarbonSigma       = getDoubleParam(params, "per_action_carbon_sigma", 1.0);
        this.perActionSpatialCenter     = getStringParam(params, "per_action_spatial_center", "none").trim().toLowerCase();
        this.perActionSpatialWeight     = getDoubleParam(params, "per_action_spatial_weight", 1.0);
        this.perActionSpatialSigma      = getDoubleParam(params, "per_action_spatial_sigma", 1.0);
        this.greenEpisodeOffsetRange    = getIntParam(params, "green_episode_offset_range", 0);
        // Stage D (2026-09-03): an explicit, frozen list of episode offsets ("a;b;c" or a
        // list) cycled by episode index, replacing the 1009*k schedule so training and
        // judgement windows are exactly the preregistered ones. Empty keeps the schedule.
        this.greenEpisodeOffsetAllowlist = parseIntList(params.get("green_episode_offset_allowlist"));

        this.carbonPenaltyMode = getStringParam(params, "carbon_penalty_mode", "TOTAL").trim().toUpperCase();
        this.carbonNormalizationMode = getStringParam(params, "carbon_normalization_mode", "RUNNING_MAX").trim().toUpperCase();
        this.carbonNormalizationFixedMax = getDoubleParam(params, "carbon_normalization_fixed_max", 0.0);
        this.carbonMiFloor = getDoubleParam(params, "carbon_mi_floor", 0.0);
        this.slaTarget = getDoubleParam(params, "sla_target", 0.85);
        this.slaPendingTarget = getDoubleParam(params, "sla_pending_target", 0.15);
        this.slaMode = getStringParam(params, "sla_mode", "completion").trim();
        this.slaDeadlineMissTarget = getDoubleParam(params, "sla_deadline_miss_target", 0.10);
        this.slaLagrangianEnabled = getBoolParam(params, "sla_lagrangian_enabled", false);
        LOGGER.info("Global Reward Coefficients: α={}, β={}, γ={}",
                this.globalRewardAlpha, this.globalRewardBeta, this.globalRewardGamma);
        LOGGER.info("Global Reward Shaping: throughput_mi_coef={}, completion_rate_mi_coef={}",
                this.globalThroughputMiCoef, this.globalCompletionRateMiCoef);
        LOGGER.info("Per-Action Reward: w_carbon={}, w_completion={}, mi_per_kg_factor={}, overflow_sharpness={}, marg_normalizer={}",
                this.perActionCarbonWeight, this.perActionCompletionWeight,
                this.miPerKgFactor, this.perActionOverflowSharpness, this.perActionMargNormalizer);
        LOGGER.info("Global Carbon Penalty Mode: {}, Normalization: {} (fixedMax={}, miFloor={})",
                this.carbonPenaltyMode, this.carbonNormalizationMode,
                this.carbonNormalizationFixedMax, this.carbonMiFloor);
        LOGGER.info("SLA / Lagrangian: enabled={}, target(c*)={}, pending_target(d)={}",
                this.slaLagrangianEnabled, this.slaTarget, this.slaPendingTarget);

        // Green Energy Configuration
        @SuppressWarnings("unchecked")
        Map<String, Object> greenEnergyConfig = (Map<String, Object>) params.getOrDefault("green_energy", Map.of());
        this.greenEnergyEnabled = getBoolParam(greenEnergyConfig, "enabled", false);
        this.turbineId = getIntParam(greenEnergyConfig, "turbine_id", 1);
        this.turbineIds = parseTurbineIds(greenEnergyConfig, this.turbineId);
        this.windDataFile = getStringParam(greenEnergyConfig, "wind_data_file",
            "windProduction/sdwpf_2001_2112_full.csv");
        this.timeScalingMode = parseTimeScalingMode(getStringParam(greenEnergyConfig, "time_scaling_mode", "REAL_TIME"));
        this.timeZoneOffsetRows = getIntParam(greenEnergyConfig, "time_zone_offset_rows", 0);
        this.brownCarbonFactor = getDoubleParam(greenEnergyConfig, "brown_carbon_factor", 0.5);
        this.greenCarbonFactor = getDoubleParam(greenEnergyConfig, "green_carbon_factor", 0.01);

        // Prediction sub-configuration
        @SuppressWarnings("unchecked")
        Map<String, Object> predictionConfig = (Map<String, Object>) greenEnergyConfig.getOrDefault("prediction", Map.of());
        this.greenPredictionEnabled = getBoolParam(predictionConfig, "enabled", false);
        this.predictionModelPath = getStringParam(predictionConfig, "model_path", "");
        this.predictionCacheDuration = getDoubleParam(predictionConfig, "cache_duration_seconds", 600.0);
        this.predictionHorizon = getIntParam(predictionConfig, "horizon", 8);

        // Future energy forecast configuration (God's Eye features)
        @SuppressWarnings("unchecked")
        Map<String, Object> forecastConfig = (Map<String, Object>) greenEnergyConfig.getOrDefault("future_energy_forecast", Map.of());
        this.shortTermRows = getIntParam(forecastConfig, "short_term_rows", 3);   // Default: 3 rows = 30 min
        this.longTermRows = getIntParam(forecastConfig, "long_term_rows", 144);    // Default: 144 rows = 24 hours

        if (this.greenEnergyEnabled) {
            LOGGER.info("Green Energy: enabled, turbine_ids={}, data_file={}, mode={}, tzOffsetRows={}, carbonFactors(brown={}, green={})",
                this.turbineIds, this.windDataFile, this.timeScalingMode,
                this.timeZoneOffsetRows, this.brownCarbonFactor, this.greenCarbonFactor);
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
                "printCloudletSummaryOnReset=" + printCloudletSummaryOnReset + ",\n" +
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
                "rewardCompletionCoef=" + rewardCompletionCoef + ",\n" +
                "greenEnergyEnabled=" + greenEnergyEnabled + ",\n" +
                "turbineId=" + turbineId + ",\n" +
                "turbineIds=" + turbineIds + ",\n" +
                "windDataFile='" + windDataFile + '\'' + ",\n" +
                "timeScalingMode=" + timeScalingMode + ",\n" +
                "timeZoneOffsetRows=" + timeZoneOffsetRows + ",\n" +
                "brownCarbonFactor=" + brownCarbonFactor + ",\n" +
                "greenCarbonFactor=" + greenCarbonFactor + ",\n" +
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

    /**
     * Parse a list of turbine IDs from config map.
     * Supports:
     * - turbine_ids: [57, 58]
     * - turbine_ids: ["57", "58"]
     * Fallback: singleton list containing fallbackTurbineId.
     */
    private List<Integer> parseTurbineIds(Map<String, Object> map, int fallbackTurbineId) {
        Object idsObj = map.get("turbine_ids");
        if (idsObj instanceof List<?> list) {
            List<Integer> ids = new ArrayList<>();
            for (Object o : list) {
                if (o instanceof Integer i) {
                    ids.add(i);
                } else if (o instanceof Number n) {
                    ids.add(n.intValue());
                } else if (o != null) {
                    try {
                        ids.add(Integer.parseInt(o.toString()));
                    } catch (NumberFormatException ignored) {
                        // Skip invalid entries
                    }
                }
            }
            if (!ids.isEmpty()) {
                return ids;
            }
        }
        return List.of(fallbackTurbineId);
    }

    /**
     * Parse TimeScalingMode from string.
     */
    private TimeScalingMode parseTimeScalingMode(String modeStr) {
        if (modeStr == null || modeStr.isBlank()) {
            return TimeScalingMode.REAL_TIME;
        }
        String upper = modeStr.trim().toUpperCase();
        return switch (upper) {
            case "COMPRESSED" -> TimeScalingMode.COMPRESSED;
            case "REAL_TIME" -> TimeScalingMode.REAL_TIME;
            default -> {
                LOGGER.warn("Unknown time_scaling_mode '{}', defaulting to REAL_TIME", modeStr);
                yield TimeScalingMode.REAL_TIME;
            }
        };
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
     * Gets the carbon emission penalty coefficient (OLD, kept for reference).
     * Penalty = coefficient × total carbon emissions (kg CO2).
     *
     * @return Coefficient for carbon emission penalty
     */
    public double getCarbonEmissionPenaltyCoef() {
        return carbonEmissionPenaltyCoef;
    }

    public double getCarbonEmissionFactorKgPerKwh() {
        return carbonEmissionFactorKgPerKwh;
    }

    // ============================================================================
    // Global Reward Coefficients (NEW normalized reward scheme)
    // r_global = α·L - β·Ĉ - γ·R_w
    // ============================================================================

    /**
     * Get global reward alpha coefficient (local performance weight).
     * @return α coefficient (default: 1.0)
     */
    public double getGlobalRewardAlpha() {
        return globalRewardAlpha;
    }

    /**
     * Get global reward beta coefficient (carbon penalty weight).
     * @return β coefficient (default: 0.5)
     */
    public double getGlobalRewardBeta() {
        return globalRewardBeta;
    }

    /**
     * Get global reward gamma coefficient (green waste penalty weight).
     * @return γ coefficient (default: 0.3)
     */
    public double getGlobalRewardGamma() {
        return globalRewardGamma;
    }

    /**
     * Global throughput shaping coefficient (MI-based).
     * Applied as: r_global += global_throughput_mi_coef * log1p(finished_mi_this_step)
     */
    public double getGlobalThroughputMiCoef() {
        return globalThroughputMiCoef;
    }

    /**
     * Global MI-completion progress shaping coefficient.
     * Applied as: r_global += global_completion_rate_mi_coef * Δcompletion_rate_mi
     */
    public double getGlobalCompletionRateMiCoef() {
        return globalCompletionRateMiCoef;
    }

    /**
     * Returns the global carbon penalty mode ("TOTAL" or "PER_MI").
     */
    public String getCarbonPenaltyMode() {
        return carbonPenaltyMode;
    }

    /**
     * Returns the carbon normalisation mode ("RUNNING_MAX", "FIXED", or "EPISODE").
     */
    public String getCarbonNormalizationMode() {
        return carbonNormalizationMode;
    }

    /**
     * Returns the fixed max value for carbon normalisation (used when mode is "FIXED").
     */
    public double getCarbonNormalizationFixedMax() {
        return carbonNormalizationFixedMax;
    }

    /**
     * Returns the MI floor used in PER_MI carbon penalty mode.
     * When > 0, PER_MI uses denominator = max(completedMI, carbonMiFloor) instead of
     * saturating at CARBON_RATIO_MAX on idle steps, producing a smoother gradient.
     */
    public double getCarbonMiFloor() {
        return carbonMiFloor;
    }

    /** Target completion rate (c*) — episode violation triggers λ increase. */
    public double getSlaTarget() {
        return slaTarget;
    }

    /** SLA mode: "completion" (legacy) or "deadline_miss" (deferrable-batch). */
    public String getSlaMode() {
        return slaMode;
    }

    /** Max acceptable deadline-miss rate when slaMode = "deadline_miss". */
    public double getSlaDeadlineMissTarget() {
        return slaDeadlineMissTarget;
    }

    /** Per-step pending_ratio threshold (d) — excess above this feeds the Lagrangian. */
    public double getSlaPendingTarget() {
        return slaPendingTarget;
    }

    /** Whether the Python Lagrangian loop is active; purely informational on Java side. */
    public boolean isSlaLagrangianEnabled() {
        return slaLagrangianEnabled;
    }
}
