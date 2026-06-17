package exe.edu.cspg.multidc;
import exe.edu.cspg.singledc.ObservationState;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import lombok.Getter;

/**
 * Result object for a hierarchical simulation step in multi-datacenter environment.
 *
 * Contains observations, rewards, and info for both global and local levels:
 * - Global level: Aggregated state of all datacenters (GlobalObservationState)
 * - Local level: Individual state of each datacenter (ObservationState per DC)
 */
@Getter
public class HierarchicalStepResult {
    // === Global Level ===
    private final GlobalObservationState globalObservation;
    private final double globalReward;

    // === Local Level ===
    private final Map<Integer, ObservationState> localObservations;  // Key: DC_ID
    private final Map<Integer, Double> localRewards;                 // Key: DC_ID

    // === Termination Flags ===
    private final boolean terminated;  // Simulation completed naturally
    private final boolean truncated;   // Episode reached max length

    // === Additional Info ===
    private final Map<String, Object> info;

    /**
     * Create a hierarchical step result.
     *
     * @param globalObservation Global observation (all DCs aggregated, GlobalObservationState)
     * @param localObservations Map of local observations (one ObservationState per DC)
     * @param globalReward Global reward
     * @param localRewards Map of local rewards (one per DC)
     * @param terminated Whether simulation terminated naturally
     * @param truncated Whether episode was truncated
     * @param info Additional information dictionary
     */
    public HierarchicalStepResult(
            GlobalObservationState globalObservation,
            Map<Integer, ObservationState> localObservations,
            double globalReward,
            Map<Integer, Double> localRewards,
            boolean terminated,
            boolean truncated,
            Map<String, Object> info) {

        this.globalObservation = globalObservation;
        this.localObservations = localObservations != null ? localObservations : new HashMap<>();
        this.globalReward = globalReward;
        this.localRewards = localRewards != null ? localRewards : new HashMap<>();
        this.terminated = terminated;
        this.truncated = truncated;
        this.info = info != null ? info : new HashMap<>();
    }

    /**
     * Get local observation for a specific datacenter.
     *
     * @param dcId Datacenter ID
     * @return ObservationState or null if not found
     */
    public ObservationState getLocalObservation(int dcId) {
        return localObservations.get(dcId);
    }

    /**
     * Get local reward for a specific datacenter.
     *
     * @param dcId Datacenter ID
     * @return Reward value or 0.0 if not found
     */
    public double getLocalReward(int dcId) {
        return localRewards.getOrDefault(dcId, 0.0);
    }

    /**
     * Get info value by key.
     *
     * @param key Info key
     * @return Info value or null if not found
     */
    public Object getInfo(String key) {
        return info.get(key);
    }

    /**
     * Check if episode is done (terminated or truncated).
     *
     * @return true if episode ended
     */
    public boolean isDone() {
        return terminated || truncated;
    }

    /**
     * Convert info map to a format suitable for Py4J transfer.
     *
     * @return Map with string values
     */
    public Map<String, String> getInfoAsStringMap() {
        Map<String, String> stringMap = new HashMap<>();
        for (Map.Entry<String, Object> entry : info.entrySet()) {
            stringMap.put(entry.getKey(), String.valueOf(entry.getValue()));
        }
        return stringMap;
    }

    /**
     * Flatten the entire step result into a single Map for one bulk Py4J transfer
     * (Level A fast path), replacing ~200 individual getter RPCs per step (5-8x
     * env-step speedup). The Python side parses this with `_parse_*_from_flat`,
     * which a regression test keeps bit-identical to the legacy per-getter path.
     * Restored from commit 782cd34 (dropped by 189b8ba).
     *
     * @return flat map: g.* global obs, l.<dc>.* local obs, r.* rewards, meta.*, info
     */
    public Map<String, Object> getStepAsFlatMap() {
        // Pre-size: 20 global keys + 15 local keys × N DCs + 5 metadata.
        int dcCount = localObservations != null ? localObservations.size() : 0;
        Map<String, Object> out = new LinkedHashMap<>(25 + 15 * Math.max(1, dcCount));

        // === Global observation fields ===
        if (globalObservation != null) {
            out.put("g.dc_current_green_power_w",  globalObservation.getDcCurrentGreenPowerW());
            out.put("g.dc_current_power_w",        globalObservation.getDcCurrentPowerW());
            out.put("g.dc_green_ratio",            globalObservation.getDcGreenRatio());
            out.put("g.dc_cumulative_wasted_green_wh", globalObservation.getDcCumulativeWastedGreenWh());
            out.put("g.dc_future_short_mean",      globalObservation.getDcFutureShortMean());
            out.put("g.dc_future_short_trend",     globalObservation.getDcFutureShortTrend());
            out.put("g.dc_future_long_mean",       globalObservation.getDcFutureLongMean());
            out.put("g.dc_future_long_peak_timing", globalObservation.getDcFutureLongPeakTiming());
            out.put("g.dc_queue_sizes",            globalObservation.getDcQueueSizes());
            out.put("g.dc_utilizations",           globalObservation.getDcUtilizations());
            out.put("g.dc_available_pes",          globalObservation.getDcAvailablePes());
            out.put("g.dc_ram_utilizations",       globalObservation.getDcRamUtilizations());
            out.put("g.upcoming_cloudlets_count",  globalObservation.getUpcomingCloudletsCount());
            out.put("g.batch_cloudlet_pes",        globalObservation.getBatchCloudletPes());
            out.put("g.batch_cloudlet_mi",         globalObservation.getBatchCloudletMi());
            out.put("g.upcoming_cloudlets_pes_distribution",
                    globalObservation.getUpcomingCloudletsPesDistribution());
            out.put("g.load_imbalance",            globalObservation.getLoadImbalance());
            out.put("g.recent_completed_cloudlets", globalObservation.getRecentCompletedCloudlets());
            out.put("g.current_clock",             globalObservation.getCurrentClock());
            out.put("g.num_datacenters",           globalObservation.getNumDatacenters());
        }

        // === Per-DC local observations ===
        if (localObservations != null) {
            for (Map.Entry<Integer, ObservationState> e : localObservations.entrySet()) {
                int dcId = e.getKey();
                ObservationState s = e.getValue();
                if (s == null) continue;
                String p = "l." + dcId + ".";
                out.put(p + "host_loads",          s.getHostLoads());
                out.put(p + "host_ram_usage_ratio", s.getHostRamUsageRatio());
                out.put(p + "vm_loads",            s.getVmLoads());
                out.put(p + "vm_types",            s.getVmTypes());
                out.put(p + "vm_host_map",         s.getVmHostMap());
                out.put(p + "vm_available_pes",    s.getVmAvailablePes());
                out.put(p + "waiting_cloudlets",   s.getWaitingCloudlets());
                out.put(p + "next_cloudlet_pes",   s.getNextCloudletPes());
                out.put(p + "next_cloudlet_mi",    s.getNextCloudletMi());
                out.put(p + "next_cloudlet_wait_time", s.getNextCloudletWaitTime());
                out.put(p + "queue_pes_distribution", s.getQueuePesDistribution());
                out.put(p + "completed_cloudlets_last_10_steps",
                        s.getCompletedCloudletsLast10Steps());
                out.put(p + "actual_vm_count",     s.getActualVmCount());
                out.put(p + "actual_host_count",   s.getActualHostCount());
                out.put(p + "infrastructure_observation", s.getInfrastructureObservation());
            }
        }

        // === Rewards ===
        out.put("r.global", globalReward);
        out.put("r.local",  localRewards);  // Map<Integer, Double>

        // === Termination flags ===
        out.put("meta.terminated", terminated);
        out.put("meta.truncated",  truncated);

        // === Info dict (kept nested; Python merges with CRD overlay separately) ===
        out.put("info", info);

        return out;
    }

    @Override
    public String toString() {
        return String.format(
                "HierarchicalStepResult{globalReward=%.3f, localRewards=%s, terminated=%b, truncated=%b, info=%s}",
                globalReward, localRewards, terminated, truncated, info
        );
    }
}
