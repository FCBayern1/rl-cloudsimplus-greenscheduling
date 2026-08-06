package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Math-only regression tests for the window-aware per-action carbon reward
 * (per_action_window_carbon, 2026-08-06).
 *
 * <p>Motivation: with LONG cloudlets (run window spanning green-phase
 * transitions) the instantaneous green ratio at routing time mis-credits
 * routing — a DC that is green now but brown for the run window is rewarded,
 * and the window-optimal DC is punished. The fix replaces the instantaneous
 * green power with the MEAN green power over the cloudlet's expected run
 * window (MI / (pes · vmPeMips) steps, capped).
 *
 * <p>These tests mirror the pure math of
 * {@code GreenEnergyProvider.getMeanFuturePowerW()} (COMPRESSED branch) and
 * the window override inside
 * {@code MultiDatacenterSimulationCore.computeDcCostFeatures()}; they do not
 * instantiate the simulator.
 */
public class WindowCarbonRewardMathTest {

    /** Mirrors getMeanFuturePowerW: mean of kW rows over horizon, wrap, ×1000, ÷divisor. */
    static double meanFuturePowerW(double[] rowsKw, int start, int horizon, double divisor) {
        double sum = 0;
        int n = rowsKw.length;
        for (int i = 0; i < horizon; i++) {
            sum += rowsKw[(start + i) % n];
        }
        return Math.max(0, (sum / horizon) * 1000.0) / divisor;
    }

    /** Mirrors the run-window length derivation in computeDcCostFeatures. */
    static int runSteps(long mi, int pes, double vmMips) {
        return (int) Math.min(3600, Math.max(1, Math.ceil((double) mi / (pes * vmMips))));
    }

    static double effFactor(double greenRatio, double greenFactor, double brownFactor) {
        return greenRatio * greenFactor + (1.0 - greenRatio) * brownFactor;
    }

    @Test
    public void meanIsAverageOfWindowNotInstant() {
        // profile: green for 2 rows then dead for 8 (kW)
        double[] kw = {6.0, 6.0, 0, 0, 0, 0, 0, 0, 0, 0};
        double inst = meanFuturePowerW(kw, 0, 1, 1500.0);
        double win10 = meanFuturePowerW(kw, 0, 10, 1500.0);
        assertEquals(6.0 * 1000 / 1500.0, inst, 1e-9);          // 4 W instant
        assertEquals(6.0 * 2 / 10 * 1000 / 1500.0, win10, 1e-9); // 0.8 W window mean
        assertTrue(win10 < inst);
    }

    @Test
    public void windowWrapsCyclically() {
        double[] kw = {1.0, 2.0, 3.0};
        // start at last row, horizon 3 -> rows {3,1,2} mean = 2
        assertEquals(2.0 * 1000 / 1.0, meanFuturePowerW(kw, 2, 3, 1.0), 1e-9);
    }

    @Test
    public void runStepsMatchesMiOverMips() {
        assertEquals(55, runSteps(55L * 40000, 1, 40000));   // median v2026 job
        assertEquals(1000, runSteps(40_000_000L, 1, 40000)); // capped trace max
        assertEquals(3600, runSteps(1_000_000_000L, 1, 40000)); // safety cap
        assertEquals(1, runSteps(100, 1, 40000));            // floor
    }

    @Test
    public void windowRatioFixesTheMyopicPreference() {
        // Two DCs, demand 4 W each, run window 10 steps, divisor 1.
        // DC A: green NOW (rows kW: 4 then 0...) -> brown for the window.
        // DC B: brown NOW (rows kW: 0 then 4...) -> green for the window.
        double[] aKw = {0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double[] bKw = {0, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004};
        double demandW = 4.0, gF = 0.01, bF = 0.5;

        // instantaneous ratios (old reward): A looks perfect, B looks dead
        double rA_inst = Math.min(1.0, meanFuturePowerW(aKw, 0, 1, 1.0) / demandW);
        double rB_inst = Math.min(1.0, meanFuturePowerW(bKw, 0, 1, 1.0) / demandW);
        assertTrue(effFactor(rA_inst, gF, bF) < effFactor(rB_inst, gF, bF),
                "myopic reward prefers the wrong DC (A)");

        // window ratios (new reward): B is the right choice and now wins
        double rA_win = Math.min(1.0, meanFuturePowerW(aKw, 0, 10, 1.0) / demandW);
        double rB_win = Math.min(1.0, meanFuturePowerW(bKw, 0, 10, 1.0) / demandW);
        assertTrue(effFactor(rB_win, gF, bF) < effFactor(rA_win, gF, bF),
                "window reward must prefer the window-green DC (B)");
    }
}
