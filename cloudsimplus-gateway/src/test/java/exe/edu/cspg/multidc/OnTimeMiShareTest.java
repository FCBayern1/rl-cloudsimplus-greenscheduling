package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * SQT2.3 on-time contract maths (Codex, 2026-08-19). The blind-deferral
 * no-regret loophole survived because the verdict never priced per-job
 * punctuality; ontime_mi_share is the third completion contract. These
 * tests lock the MI weighting and the pending-job convention.
 */
public class OnTimeMiShareTest {

    private Map<Long, Long> map(long... kv) {
        Map<Long, Long> m = new HashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }

    @Test
    public void miWeightingDominatesCounts() {
        // 1 late whale (900 MI) vs 3 punctual minnows (100 MI each):
        // count-based miss rate would say 25% missed; MI share says 75% late.
        Map<Long, Long> ddl = map(1L, 100L, 2L, 100L, 3L, 100L, 4L, 100L);
        Map<Long, Long> mi = map(1L, 100L, 2L, 100L, 3L, 100L, 4L, 900L);
        Map<Long, Double> fin = new HashMap<>();
        fin.put(1L, 50.0); fin.put(2L, 60.0); fin.put(3L, 70.0);
        fin.put(4L, 500.0);                                   // whale late
        assertEquals(300.0 / 1200.0,
                MultiDatacenterSimulationCore.computeOnTimeMiShare(ddl, mi, fin, 600.0),
                1e-12);
    }

    @Test
    public void pendingBeforeDeadlineCountsOnTime() {
        Map<Long, Long> ddl = map(1L, 1000L);
        Map<Long, Long> mi = map(1L, 500L);
        assertEquals(1.0, MultiDatacenterSimulationCore.computeOnTimeMiShare(
                ddl, mi, new HashMap<>(), 900.0), 1e-12);
    }

    @Test
    public void pendingPastDeadlineCountsLate() {
        Map<Long, Long> ddl = map(1L, 1000L);
        Map<Long, Long> mi = map(1L, 500L);
        assertEquals(0.0, MultiDatacenterSimulationCore.computeOnTimeMiShare(
                ddl, mi, new HashMap<>(), 1200.0), 1e-12);
    }

    @Test
    public void exactDeadlineFinishIsOnTime() {
        Map<Long, Long> ddl = map(1L, 1000L);
        Map<Long, Long> mi = map(1L, 500L);
        Map<Long, Double> fin = new HashMap<>();
        fin.put(1L, 1000.0);
        assertEquals(1.0, MultiDatacenterSimulationCore.computeOnTimeMiShare(
                ddl, mi, fin, 5000.0), 1e-12);
    }

    @Test
    public void emptyDeadlineSetIsVacuouslyPerfect() {
        assertEquals(1.0, MultiDatacenterSimulationCore.computeOnTimeMiShare(
                new HashMap<>(), new HashMap<>(), new HashMap<>(), 0.0), 1e-12);
    }
}
