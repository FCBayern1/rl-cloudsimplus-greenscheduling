package giu.edu.cspg.multidc;

import giu.edu.cspg.singledc.ObservationState;
import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

/**
 * Result returned by the reset() method in hierarchical multi-datacenter environments.
 *
 * This class follows the Gymnasium convention where reset() returns only
 * observations and info, without rewards or termination flags.
 */
@Getter
public class HierarchicalResetResult {
    // === Observations ===
    private final GlobalObservationState globalObservation;
    private final Map<Integer, ObservationState> localObservations;  // Key: DC_ID

    // === Additional Info ===
    private final Map<String, Object> info;

    /**
     * Create a hierarchical reset result.
     *
     * @param globalObservation Global observation (all DCs aggregated)
     * @param localObservations Map of local observations (one per DC)
     * @param info Additional information dictionary
     */
    public HierarchicalResetResult(
            GlobalObservationState globalObservation,
            Map<Integer, ObservationState> localObservations,
            Map<String, Object> info) {

        this.globalObservation = globalObservation;
        this.localObservations = localObservations != null ? localObservations : new HashMap<>();
        this.info = info != null ? info : new HashMap<>();
    }

    /**
     * Convenience constructor without info.
     */
    public HierarchicalResetResult(
            GlobalObservationState globalObservation,
            Map<Integer, ObservationState> localObservations) {
        this(globalObservation, localObservations, new HashMap<>());
    }

    // === Getters ===

    public Map<Integer, ObservationState> getLocalObservations() {
        return new HashMap<>(localObservations);
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

    public Map<String, Object> getInfo() {
        return new HashMap<>(info);
    }

    /**
     * Get info value by key.
     *
     * @param key Info key
     * @return Value or null if not found
     */
    public Object getInfoValue(String key) {
        return info.get(key);
    }

    @Override
    public String toString() {
        return String.format(
            "HierarchicalResetResult{globalObs=%s, localObs=%d DCs, info=%d entries}",
            globalObservation != null ? "present" : "null",
            localObservations.size(),
            info.size()
        );
    }
}