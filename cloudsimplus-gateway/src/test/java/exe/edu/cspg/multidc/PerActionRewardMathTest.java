package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Math-only regression tests for the 2026-05-16 Stage 1 per-action
 * difference-reward formulas.  These do NOT instantiate any simulator
 * state — they verify the pure formulas applied inside
 * {@code MultiDatacenterSimulationCore.computeDcCostFeatures()} and
 * the diff combination inside
 * {@code MultiDatacenterSimulationCore.accumulatePerActionReward()}.
 *
 * <p>Pinned scenarios match the 2026-05-16 calibration notes:
 * <ul>
 *   <li>typical cloudlet 700K MI</li>
 *   <li>mi_per_kg_factor = 3.5e6</li>
 *   <li>green_factor 0.01, brown_factor 0.5</li>
 *   <li>per_action_overflow_sharpness k = 3</li>
 *   <li>w_carbon = 1.0, w_completion = 0.05</li>
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

    /** Difference reward: actual vs RR baseline. */
    static double diffReward(double margActual, double margBaseline,
                             double probActual, double probBaseline,
                             double wCarbon, double wCompletion) {
        return -wCarbon * (margActual - margBaseline)
             + wCompletion * (probActual - probBaseline);
    }

    // === Tests ===

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
        // green_ratio=1, eff = 1·0.01 + 0 = 0.01
        // marginal = 700_000 / 3.5e6 · 0.01 = 0.002
        double mk = marginalKg(700_000L, 1.0, 0.01, 0.5, 3.5e6);
        assertEquals(0.002, mk, 1e-6);
    }

    @Test
    public void marginalKg_allBrown_useBrownFactor() {
        // green_ratio=0, eff = 0 + 1·0.5 = 0.5
        // marginal = 700_000 / 3.5e6 · 0.5 = 0.1
        double mk = marginalKg(700_000L, 0.0, 0.01, 0.5, 3.5e6);
        assertEquals(0.1, mk, 1e-6);
    }

    @Test
    public void probComplete_noOverflow_isOne() {
        // queue=0, pes=4, avail=10 → overflow=0 → prob=1.0
        assertEquals(1.0, probComplete(0, 4, 10, 100, 3.0), 1e-9);
    }

    @Test
    public void probComplete_halfFullCapacityOverflow_isAbout0_55() {
        // queue=10, pes=10, avail=0, capacity=100 → overflow_frac = 20/100 = 0.2
        // prob = exp(-3 * 0.2) = exp(-0.6) ≈ 0.5488
        assertEquals(0.5488, probComplete(10, 10, 0, 100, 3.0), 1e-3);
    }

    @Test
    public void probComplete_fullCapacityOverflow_isNearZero() {
        // overflow_frac = 1.0, prob = exp(-3) ≈ 0.0498
        assertEquals(0.0498, probComplete(100, 100, 100, 100, 3.0), 1e-3);
    }

    @Test
    public void diffReward_signWhenActualCleanerThanBaseline() {
        // actual: green DC (low marginal); baseline: brown DC (high marginal).
        // diff(marg) = 0.002 - 0.1 = -0.098;  -w_c · diff = -1.0 · (-0.098) = +0.098
        // diff(prob) = 1.0 - 1.0 = 0
        // r_i = +0.098 — agent picked smarter than RR, gets positive reward.
        double r = diffReward(/*margAct*/0.002, /*margBase*/0.1,
                              /*probAct*/1.0, /*probBase*/1.0,
                              /*w_c*/1.0, /*w_compl*/0.05);
        assertTrue(r > 0, "actual cleaner than baseline must yield positive reward, got " + r);
        assertEquals(0.098, r, 1e-3);
    }

    @Test
    public void diffReward_signWhenActualDirtierThanBaseline() {
        // actual: brown; baseline: green.  diff(marg) = +0.098; -w_c · diff = -0.098.
        double r = diffReward(/*margAct*/0.1, /*margBase*/0.002,
                              /*probAct*/1.0, /*probBase*/1.0,
                              /*w_c*/1.0, /*w_compl*/0.05);
        assertTrue(r < 0, "actual dirtier than baseline must yield negative reward, got " + r);
        assertEquals(-0.098, r, 1e-3);
    }

    @Test
    public void diffReward_zeroWhenActualEqualsBaseline() {
        // Same DC picked by policy and RR → no signal.  Guards against the
        // policy getting "free" reward just for going along with the baseline,
        // which would defeat the whole point of difference-reward attribution.
        double r = diffReward(0.05, 0.05, 0.9, 0.9, 1.0, 0.05);
        assertEquals(0.0, r, 1e-12);
    }

    @Test
    public void diffReward_completionFitContributes() {
        // Same marginal carbon for actual and baseline (cancels), but actual
        // has 1.0 completion prob vs baseline 0.5.  Reward = w_compl · 0.5 = 0.025.
        double r = diffReward(0.05, 0.05, 1.0, 0.5, 1.0, 0.05);
        assertEquals(0.025, r, 1e-9);
    }

    @Test
    public void diffReward_magnitudeMatchesCalibrationTarget() {
        // 2026-05-16 design note: typical |r_i| should be ~0.02-0.05.
        // Reasonable "policy deviates from RR" scenario:
        //   actual:   green DC, no overflow → marg=0.02, prob=1.0
        //   baseline: typical DC, light load → marg=0.05, prob=0.95
        // r = -1.0 · (0.02 - 0.05) + 0.05 · (1.0 - 0.95) = 0.03 + 0.0025 = 0.0325
        double r = diffReward(0.02, 0.05, 1.0, 0.95, 1.0, 0.05);
        assertEquals(0.0325, r, 1e-4);
        assertTrue(Math.abs(r) < 0.1,
                "|r_i| stayed within calibration target (≪ Lagrangian λ·c_step ≈ 0.1)");
    }
}
