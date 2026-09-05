package exe.edu.cspg.multidc;

import java.util.Map;

/**
 * Pure arithmetic of the dispatch-rate VM placement ledger (Codex ruling 2026-09-05 on the
 * fallback's gate-3 forensic). Kept free of CloudSim objects so the unit tests exercise the
 * production formulas: committed PEs of a VM are its scheduler's exec list plus its waiting
 * list plus the cloudlets this broker already submitted to it that are in neither list yet
 * (in flight across a step); the selector takes the most-free VM that fits, lowest id on a
 * tie, and returns -1 when none fits (the caller then leaves the cloudlet queued, which is
 * the unchanged SpaceShared behaviour when no idle VM exists).
 */
final class PlacementLedger {

    private PlacementLedger() { }

    static long freePes(long vmPes, long execPes, long waitingPes, long inflightPes) {
        return Math.max(0L, vmPes - execPes - waitingPes - inflightPes);
    }

    static long selectMostFreeFitting(Map<Long, Long> freePes, long need) {
        long vmId = -1;
        long bestFree = -1;
        for (Map.Entry<Long, Long> e : freePes.entrySet()) {
            long f = e.getValue();
            if (f < need) continue;
            if (f > bestFree || (f == bestFree && e.getKey() < vmId)) {
                bestFree = f;
                vmId = e.getKey();
            }
        }
        return vmId;
    }
}
