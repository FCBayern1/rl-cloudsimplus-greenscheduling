package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * TB12 reward repair (Codex prereg 2026-08-25): sla_mode "ontime_mi" maths.
 * The RL collapse survived training because sla_mode "completion" priced only
 * eventual completion while the verdict contract is MI-weighted punctuality.
 * These tests lock c_t = max(0, target − ontime_mi_share), the MI weighting,
 * the pending-job conventions, and the 720s-backstop-vs-600s-quantization
 * safety margin.
 */
public class SlaOntimeMiCostTest {

    private Map<Long, Long> map(long... kv) {
        Map<Long, Long> m = new HashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }

    @Test
    public void zeroCostWhenAllOnTime() {
        assertEquals(0.0, MultiDatacenterSimulationCore.ontimeMiSlaCost(0.995, 1.0), 1e-12);
    }

    @Test
    public void collapseFingerprintPaysNearFullTarget() {
        // rl_nofc eval fingerprint: ontime_mi_share == 0.0 → cost == target.
        assertEquals(0.995, MultiDatacenterSimulationCore.ontimeMiSlaCost(0.995, 0.0), 1e-12);
    }

    @Test
    public void miWeightedNotJobCounted() {
        // Big job (900 MI) late, small job (100 MI) punctual: count-based miss
        // rate is 0.5 but the MI contract charges for 90% of the work.
        Map<Long, Long> ddl = map(1L, 100L, 2L, 100L);
        Map<Long, Long> mi = map(1L, 900L, 2L, 100L);
        Map<Long, Double> fin = new HashMap<>();
        fin.put(1L, 500.0);    // whale late
        fin.put(2L, 50.0);     // minnow on time
        double share = MultiDatacenterSimulationCore.computeOnTimeMiShare(ddl, mi, fin, 600.0);
        assertEquals(0.1, share, 1e-12);
        double cost = MultiDatacenterSimulationCore.ontimeMiSlaCost(0.995, share);
        assertEquals(0.895, cost, 1e-12);
        double countBasedCost = MultiDatacenterSimulationCore.ontimeMiSlaCost(0.995, 0.5);
        assertTrue(cost > countBasedCost);
    }

    @Test
    public void unfinishedPastDeadlineIsLate() {
        Map<Long, Long> ddl = map(1L, 100L);
        Map<Long, Long> mi = map(1L, 500L);
        double share = MultiDatacenterSimulationCore.computeOnTimeMiShare(
                ddl, mi, new HashMap<>(), 200.0);       // clock past deadline, unfinished
        assertEquals(0.0, share, 1e-12);
        assertEquals(0.995, MultiDatacenterSimulationCore.ontimeMiSlaCost(0.995, share), 1e-12);
    }

    @Test
    public void unfinishedBeforeDeadlineIsOnTimeOnline() {
        // Online convention: pending work whose deadline is still ahead of the
        // clock is not (yet) charged — deferring into green stays free.
        Map<Long, Long> ddl = map(1L, 1000L);
        Map<Long, Long> mi = map(1L, 500L);
        double share = MultiDatacenterSimulationCore.computeOnTimeMiShare(
                ddl, mi, new HashMap<>(), 200.0);       // clock before deadline
        assertEquals(1.0, share, 1e-12);
        assertEquals(0.0, MultiDatacenterSimulationCore.ontimeMiSlaCost(0.995, share), 1e-12);
    }

    @Test
    public void slack720CoversControlQuantization600() {
        // Decisions only happen at 600s boundaries. With slack 720 the first
        // boundary where the latest-start backstop fires still leaves the job
        // >= 120s before its deadline: at the previous boundary (t-600) it did
        // NOT fire, so t-600+rt+720 < ddl → t+rt <= ddl-120. Sweep deadlines
        // second-by-second to check the worst case mechanically.
        double runtime = 14400.0;                       // 4h TB12 job
        for (double ddl = runtime + 1000.0; ddl < runtime + 4000.0; ddl += 1.0) {
            double fired = -1.0;
            for (double t = 0.0; t <= ddl; t += 600.0) {
                if (PerActionRewardMath.deadlineForceLatestStart(
                        t, ddl, runtime * 50000.0, 1L, 50000.0, 720.0)) {
                    fired = t;
                    break;
                }
            }
            assertTrue(fired >= 0.0, "backstop never fired for ddl=" + ddl);
            assertTrue(fired + runtime <= ddl - 120.0 + 1e-9,
                    "quantized firing too late for ddl=" + ddl);
        }
    }

    @Test
    public void perPeLengthSemanticsRealTb12Job() {
        // Real TB12 job: 5.76e8 MI per-PE, 2 PES, 40000 MIPS → runtime 14400s.
        // The old pes× division estimated 7200s and fired ~1.8h late. Deadline
        // window rt+slack: firing must leave >= 14400s before the deadline.
        double lengthMi = 576_000_000.0, mips = 40_000.0, ddl = 87_820.0;
        double fired = -1.0;
        for (double t = 17_020.0; t <= ddl; t += 600.0) {
            if (PerActionRewardMath.deadlineForceLatestStart(
                    t, ddl, lengthMi, 2L, mips, 720.0)) {
                fired = t;
                break;
            }
        }
        assertTrue(fired >= 0.0);
        assertTrue(fired + 14_400.0 <= ddl - 120.0 + 1e-9,
                "backstop fired too late: t=" + fired);
    }

    @Test
    public void slack600IsRazorThinUnderQuantization() {
        // Documents why 720 (not the legacy 600) is required: with slack 600
        // the guaranteed margin at 600s quantization is exactly zero.
        double runtime = 14400.0;
        double ddl = runtime + 600.0 + 600.0;           // fires exactly at t=600
        double t = 600.0;
        assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                t, ddl, runtime * 50000.0, 1L, 50000.0, 600.0));
        assertEquals(ddl, t + runtime + 600.0, 1e-9);   // zero slack left
    }
}
