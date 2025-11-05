package giu.edu.cspg.multidc;
import giu.edu.cspg.singledc.ObservationState;

import java.util.HashMap;
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

    @Override
    public String toString() {
        return String.format(
                "HierarchicalStepResult{globalReward=%.3f, localRewards=%s, terminated=%b, truncated=%b, info=%s}",
                globalReward, localRewards, terminated, truncated, info
        );
    }
}
