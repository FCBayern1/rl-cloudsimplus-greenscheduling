package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Math-only regression tests for the per-action reward formulas.
 *
 * <p>2026-05-20: switched from "diff vs RR baseline" to ABSOLUTE
 * normalized reward.  Old diff formula was:
 *   r_i = -w_c·(marg_a − marg_RR) + w_compl·(prob_a − prob_RR)
 * which empirically flipped sign once the agent beat RR on either axis.
 * New formula:
 *   r_i = -w_c · (marg_a / marg_normalizer) + w_compl · prob_a
 * Both terms live in roughly [0, 1] so weights stay calibrated.
 *
 * <p>These tests do NOT instantiate the simulator — they verify the pure
 * formulas applied inside
 * {@code MultiDatacenterSimulationCore.computeDcCostFeatures()} and
 * {@code MultiDatacenterSimulationCore.accumulatePerActionReward()}.
 *
 * <p>Calibration constants (2026-05-20):
 * <ul>
 *   <li>mi_per_kg_factor = 3.5e6</li>
 *   <li>marg_normalizer  = 0.05 (typical cloudlet marginalKg)</li>
 *   <li>green_factor 0.01, brown_factor 0.5</li>
 *   <li>per_action_overflow_sharpness k = 3</li>
 *   <li>w_carbon = 1.0, w_completion = 0.5</li>
 * </ul>
 */
public class PerActionRewardMathTest {

    // === Pure helpers mirroring the Java SimulationCore math ===

    /** Marginal carbon ("kg proxy") for routing one cloudlet to one DC. */
    static double marginalKg(long mi, double greenRatio,
                             double greenFactor, double brownFactor,
                             double miPerKg) {
        double effFactor = greenRatio * greenFactor + (1.0 - greenRatio) * brownFactor;
        return ((double) mi / Math.max(1e3, miPerKg)) * effFactor;
    }

    /** Completion probability given queue overflow at a DC. */
    static double probComplete(int queueSize, int cloudletPes, int availPes,
                               int capacity, double sharpness) {
        double overflow = Math.max(0.0,
                (double) (queueSize + cloudletPes - availPes) / (double) Math.max(1, capacity));
        return Math.exp(-sharpness * overflow);
    }

    /**
     * 2026-05-20 NEW absolute per-action reward.
     * r_i = -w_c · (marg / marg_normalizer) + w_compl · prob
     */
    static double absoluteReward(double margActual, double probActual,
                                 double wCarbon, double wCompletion,
                                 double margNormalizer) {
        double margNorm = margActual / Math.max(1e-6, margNormalizer);
        return -wCarbon * margNorm + wCompletion * probActual;
    }

    // === Marginal-carbon math (unchanged from Stage 1) ===

    @Test
    public void marginalKg_typicalCloudletAt50_50_GreenBrown() {
        // mi=700K, green_ratio=0.5, brown=0.5, green_factor=0.01, mi_per_kg=3.5e6.
        // eff = 0.5·0.01 + 0.5·0.5 = 0.255
        // marginal = 700_000 / 3.5e6 · 0.255 = 0.0510
        double mk = marginalKg(700_000L, 0.5, 0.01, 0.5, 3.5e6);
        assertEquals(0.051, mk, 1e-4);
    }

    @Test
    public void marginalKg_allGreenIsNearZero() {
        double mk = marginalKg(700_000L, 1.0, 0.01, 0.5, 3.5e6);
        assertEquals(0.002, mk, 1e-6);
    }

    @Test
    public void marginalKg_allBrown_useBrownFactor() {
        double mk = marginalKg(700_000L, 0.0, 0.01, 0.5, 3.5e6);
        assertEquals(0.1, mk, 1e-6);
    }

    // === Probability of completion math (unchanged) ===

    @Test
    public void probComplete_noOverflow_isOne() {
        assertEquals(1.0, probComplete(0, 4, 10, 100, 3.0), 1e-9);
    }

    @Test
    public void probComplete_halfFullCapacityOverflow_isAbout0_55() {
        assertEquals(0.5488, probComplete(10, 10, 0, 100, 3.0), 1e-3);
    }

    @Test
    public void probComplete_fullCapacityOverflow_isNearZero() {
        assertEquals(0.0498, probComplete(100, 100, 100, 100, 3.0), 1e-3);
    }

    // === NEW absolute reward tests (2026-05-20) ===

    @Test
    public void absoluteReward_typicalGreenDcWithoutOverflow_isPositive() {
        // Green DC, no overflow.  marg = 0.002 (very small), prob = 1.0.
        // r = -1.0 · (0.002/0.05) + 0.5 · 1.0 = -0.04 + 0.5 = +0.46
        // Agent should be STRONGLY rewarded for picking such a DC.
        double r = absoluteReward(0.002, 1.0, 1.0, 0.5, 0.05);
        assertEquals(0.46, r, 1e-3);
        assertTrue(r > 0, "green DC with capacity should give positive reward");
    }

    @Test
    public void absoluteReward_dirtyDcWithoutOverflow_isMixed() {
        // Brown DC, no overflow.  marg = 0.1 (dirty), prob = 1.0.
        // r = -1.0 · (0.1/0.05) + 0.5 · 1.0 = -2.0 + 0.5 = -1.5
        // Agent is penalized for choosing brown even when it can complete.
        double r = absoluteReward(0.1, 1.0, 1.0, 0.5, 0.05);
        assertEquals(-1.5, r, 1e-3);
        assertTrue(r < 0, "dirty DC even without overflow should be penalized");
    }

    @Test
    public void absoluteReward_greenDcButOverflowing_dropsPositive() {
        // Green DC but already overflowing.  marg = 0.002, prob = 0.05 (full overflow).
        // r = -1.0 · 0.04 + 0.5 · 0.05 = -0.04 + 0.025 = -0.015
        // Net slightly negative — overflowing greenness no longer attractive.
        double r = absoluteReward(0.002, 0.05, 1.0, 0.5, 0.05);
        assertEquals(-0.015, r, 1e-3);
        assertTrue(r < 0, "overflowing green DC should net negative");
    }

    @Test
    public void absoluteReward_sign_strictlyMonotone_inGreenness() {
        // Holding overflow constant, picking a GREENER DC must yield HIGHER reward.
        // (This is the structural property the diff-vs-RR formula broke.)
        double rBrown = absoluteReward(0.10, 1.0, 1.0, 0.5, 0.05);
        double rMid   = absoluteReward(0.05, 1.0, 1.0, 0.5, 0.05);
        double rGreen = absoluteReward(0.01, 1.0, 1.0, 0.5, 0.05);
        assertTrue(rBrown < rMid, "greener should always be higher: brown < mid");
        assertTrue(rMid   < rGreen, "greener should always be higher: mid < green");
    }

    @Test
    public void absoluteReward_sign_strictlyMonotone_inCapacity() {
        // Holding marg constant, picking a less-overflowing DC must yield HIGHER reward.
        double margMid = 0.05;
        double rNoOverflow   = absoluteReward(margMid, 1.0,  1.0, 0.5, 0.05);
        double rSomeOverflow = absoluteReward(margMid, 0.55, 1.0, 0.5, 0.05);
        double rFullOverflow = absoluteReward(margMid, 0.05, 1.0, 0.5, 0.05);
        assertTrue(rNoOverflow   > rSomeOverflow);
        assertTrue(rSomeOverflow > rFullOverflow);
    }

    @Test
    public void absoluteReward_magnitudeWithinCalibration() {
        // Realistic "average" routing: typical cloudlet to typical DC.
        //   marg = 0.051 (50/50 green/brown), prob = 0.85 (light load).
        // r = -1.0 · (0.051/0.05) + 0.5 · 0.85
        //   = -1.02 + 0.425 = -0.595
        // Per-step (10 actions): ~ -5.95.  Lagrangian penalty ~ -0.3/step;
        // per-action term ~ -0.6/step.  Same order, no domination.
        double r = absoluteReward(0.051, 0.85, 1.0, 0.5, 0.05);
        assertEquals(-0.595, r, 1e-3);
        assertTrue(Math.abs(r) < 3.0,
                "single-action |r| should stay in single digits to keep PPO step sizes sane");
    }

    @Test
    public void absoluteReward_zeroWeights_isZero() {
        // Backward-compat sanity: if both weights are 0, reward is 0
        // regardless of actual DC choice.
        double r = absoluteReward(0.05, 0.5, 0.0, 0.0, 0.05);
        assertEquals(0.0, r, 1e-12);
    }

    // ---- SQT2.2 latest-start backstop (Codex adjudication 2026-08-18) ----

    @Test
    void latestStartForcesExactlyAtBoundary() {
        // runtime = 40e6 MI / (1 pes * 40000 MIPS) = 1000 s; slack 120
        // deadline 3120, now 2000: 2000+1000+120 == 3120 -> force
        assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                2000, 3120, 40_000_000, 1, 40000, 120));
        // one second earlier -> no force (tight job keeps its wait window)
        assertFalse(PerActionRewardMath.deadlineForceLatestStart(
                1999, 3120, 40_000_000, 1, 40000, 120));
    }

    @Test
    void latestStartRuntimeIsPerPeAndIgnoresPesCount() {
        // TB12 audit 2026-08-25: CloudSim length is PER-PE MI — runtime does
        // not shrink with pes. The old assertion (4 pes → 250 s) codified the
        // 2x underestimate that fired the TB12 backstop ~1.8h past latest
        // start. 40e6 MI at 40000 MIPS is 1000 s regardless of pes.
        assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                2000, 2380, 40_000_000, 4, 40000, 120));
        assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                2000, 2380, 40_000_000, 1, 40000, 120));
        // Same job, deadline far enough for 1000+120 s → genuinely waits.
        assertFalse(PerActionRewardMath.deadlineForceLatestStart(
                2000, 3200, 40_000_000, 4, 40000, 120));
    }

    @Test
    void latestStartTightSlackJobNotForcedOnArrival() {
        // the legacy 600 s lead would fire immediately for slack 300; the
        // latest-start rule leaves the genuine 300-120=180 s wait window
        double arrival = 0, runtime = 372 * 40000.0 * 1;   // MI for 372 s job
        double deadline = arrival + 372 + 300;             // slack 300
        assertFalse(PerActionRewardMath.deadlineForceLatestStart(
                arrival, deadline, runtime, 1, 40000, 120));
        assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                arrival + 181, deadline, runtime, 1, 40000, 120));
    }

}
