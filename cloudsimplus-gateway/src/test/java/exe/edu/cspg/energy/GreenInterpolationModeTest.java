package exe.edu.cspg.energy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * STEP row semantics (Codex 2026-08-23): row i holds on
 * [i*600, (i+1)*600); the ledger integrates rows exactly; default SPLINE
 * stays bit-identical to the legacy constructor. Uses the shipped
 * Turbine_100_2021.csv (real SDWPF, jagged) so the spline-vs-step gap is
 * exercised on the data class that exposed the defect.
 */
class GreenInterpolationModeTest {

    private static final String CSV = "windProduction/simplified/Turbine_100_2021.csv";

    private GreenEnergyProvider make(GreenInterpolationMode mode) {
        return new GreenEnergyProvider(100, CSV, TimeScalingMode.REAL_TIME,
                3, 144, 0, 60.0, 0, 1.0, mode);
    }

    @Test
    void defaultConstructorIsSplineBitIdentical() {
        GreenEnergyProvider legacy = new GreenEnergyProvider(100, CSV,
                TimeScalingMode.REAL_TIME, 3, 144, 0, 60.0, 0, 1.0);
        GreenEnergyProvider spline = make(GreenInterpolationMode.SPLINE);
        for (double t : new double[] { 0, 601, 7200, 86_399, 1_234_567 }) {
            assertEquals(legacy.getCurrentPowerW(t), spline.getCurrentPowerW(t), 0.0,
                    "legacy path must stay bit-identical at t=" + t);
        }
    }

    @Test
    void stepHoldsTheRowValueAcrossItsWholeUnit() {
        GreenEnergyProvider step = make(GreenInterpolationMode.STEP);
        for (int row : new int[] { 0, 7, 8, 1000, 44_000 }) {
            double atKnot = step.getCurrentPowerW(row * 600.0);
            assertEquals(atKnot, step.getCurrentPowerW(row * 600.0 + 1), 1e-9,
                    "row " + row + " must hold at +1s");
            assertEquals(atKnot, step.getCurrentPowerW(row * 600.0 + 599), 1e-9,
                    "row " + row + " must hold at +599s");
        }
    }

    @Test
    void stepNeverInventsPowerBetweenSamples() {
        // The defect: cubic oscillation glides over deep lulls. Under STEP a
        // zero row must read exactly zero anywhere inside its unit.
        GreenEnergyProvider step = make(GreenInterpolationMode.STEP);
        boolean sawZeroRow = false;
        for (int row = 0; row < 5000; row++) {
            double v = step.getCurrentPowerW(row * 600.0);
            if (v == 0.0) {
                sawZeroRow = true;
                assertEquals(0.0, step.getCurrentPowerW(row * 600.0 + 300), 0.0,
                        "zero row " + row + " must stay zero mid-unit");
                break;
            }
        }
        assertTrue(sawZeroRow, "real SDWPF must contain calm rows in the first 5000");
    }

    @Test
    void intervalEnergyIsTheExactRowIntegral() {
        GreenEnergyProvider step = make(GreenInterpolationMode.STEP);
        // One aligned step: energy == row power * 600s
        double p0 = step.getCurrentPowerW(1200.0);
        assertEquals(p0 * 600.0 / 3600.0, step.getIntervalEnergyWh(1200.0, 1800.0), 1e-9);
        // Straddling a boundary: exact split between the two rows
        double p1 = step.getCurrentPowerW(1800.0);
        double expected = (p0 * 300.0 + p1 * 300.0) / 3600.0;
        assertEquals(expected, step.getIntervalEnergyWh(1500.0, 2100.0), 1e-9);
        // Additivity over a long span
        double whole = step.getIntervalEnergyWh(0.0, 6000.0);
        double parts = step.getIntervalEnergyWh(0.0, 2500.0)
                + step.getIntervalEnergyWh(2500.0, 6000.0);
        assertEquals(whole, parts, 1e-9);
    }

    @Test
    void tzOffsetShiftsRowsSynchronously() {
        GreenEnergyProvider a = new GreenEnergyProvider(100, CSV,
                TimeScalingMode.REAL_TIME, 3, 144, 0, 60.0, 0, 1.0,
                GreenInterpolationMode.STEP);
        GreenEnergyProvider b = new GreenEnergyProvider(100, CSV,
                TimeScalingMode.REAL_TIME, 3, 144, 500, 60.0, 0, 1.0,
                GreenInterpolationMode.STEP);
        for (double t : new double[] { 0, 600, 12_345, 600_000 }) {
            assertEquals(a.getCurrentPowerW(t + 500 * 600.0), b.getCurrentPowerW(t), 1e-9,
                    "offset provider must read the shifted row at t=" + t);
        }
    }

    @Test
    void legacyLedgerArithmeticPreservedOutsideStep() {
        GreenEnergyProvider spline = make(GreenInterpolationMode.SPLINE);
        double t0 = 3600.0, t1 = 4200.0;
        assertEquals(spline.getCurrentPowerW(t1) * (t1 - t0) / 3600.0,
                spline.getIntervalEnergyWh(t0, t1), 0.0,
                "non-STEP interval energy must be the legacy end-sample x delta");
    }

    // ---- P0.5-1(Codex):时间戳全部不可解析,power rows 仍有效 ----

    private GreenEnergyProvider badTs(GreenInterpolationMode mode) {
        return new GreenEnergyProvider(9999,
                "windProduction/simplified/Turbine_9999_2021.csv",
                TimeScalingMode.REAL_TIME, 3, 12, 0, 60.0, 0, 1.0, mode);
    }

    @Test
    void unparsableTimestamps_currentPowerStillWorks() {
        GreenEnergyProvider step = badTs(GreenInterpolationMode.STEP);
        assertEquals(0.0, step.getCurrentPowerW(0.0), 1e-9);          // row 0 = 0 kW
        assertEquals(100_000.0, step.getCurrentPowerW(600.0), 1e-6);  // row 1 = 100 kW
        assertEquals(300_000.0, step.getCurrentPowerW(1500.0), 1e-6); // row 2 held mid-unit
    }

    @Test
    void unparsableTimestamps_futureBinsStillWork() {
        GreenEnergyProvider step = badTs(GreenInterpolationMode.STEP);
        double[] bins = step.getFuturePowerPredictions(0.0, new int[] { 600, 1200 });
        assertEquals(2, bins.length, "future bins must not collapse to empty on spline failure");
        assertEquals(100_000.0, bins[0], 1e-6);
        assertEquals(300_000.0, bins[1], 1e-6);
    }

    @Test
    void unparsableTimestamps_trendFeaturesStillWork() {
        GreenEnergyProvider step = badTs(GreenInterpolationMode.STEP);
        double[] f = step.computeFutureTrendFeatures(0.0);
        assertEquals(4, f.length);
        // short window rows {0,100,300}kW, max=300 -> mean/max = 400/3/300
        assertEquals((0.0 + 100.0 + 300.0) / 3.0 / 300.0, f[0], 1e-9,
                "trend must read raw rows, normalised by raw max");
    }

    @Test
    void unparsableTimestamps_meanPathStillWorks() {
        GreenEnergyProvider step = badTs(GreenInterpolationMode.STEP);
        // REAL_TIME -> mean path falls back to current power; must not NPE
        assertEquals(step.getCurrentPowerW(600.0),
                step.getMeanFuturePowerW(600.0, 3), 1e-9);
    }
}
