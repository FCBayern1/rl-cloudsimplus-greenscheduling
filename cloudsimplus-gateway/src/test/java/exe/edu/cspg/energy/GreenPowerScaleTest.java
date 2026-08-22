package exe.edu.cspg.energy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * green_power_scale (testbed 12, Codex 2026-08-22): a mode-independent
 * multiplier applied AFTER time-scaling-mode handling. The default 1.0 must
 * keep every existing experiment bit-identical; the frozen testbed-12 value
 * (9.05562658195e-5, calibrated on T100+101/2020 only) must scale every
 * power read path uniformly.
 */
class GreenPowerScaleTest {

    private static final double TB12_SCALE = 9.05562658195e-5;

    private GreenEnergyProvider provider(double scale) {
        // Turbine 100 / 2021 ships in src/main/resources and is on the test
        // classpath; REAL_TIME so the COMPRESSED divisor stays inert.
        return new GreenEnergyProvider(100,
                "windProduction/simplified/Turbine_100_2021.csv",
                TimeScalingMode.REAL_TIME, 3, 144, 0, 60.0, 0, scale);
    }

    @Test
    void defaultScaleIsBitIdenticalToLegacyConstructor() {
        GreenEnergyProvider legacy = new GreenEnergyProvider(100,
                "windProduction/simplified/Turbine_100_2021.csv",
                TimeScalingMode.REAL_TIME, 3, 144, 0, 60.0, 0);
        GreenEnergyProvider scaled1 = provider(1.0);
        for (double t : new double[] { 0, 3600, 86400, 1_000_000 }) {
            assertEquals(legacy.getCurrentPowerW(t), scaled1.getCurrentPowerW(t), 0.0,
                    "scale=1.0 must be bit-identical to the legacy constructor at t=" + t);
        }
    }

    @Test
    void scaleMultipliesCurrentPowerUniformly() {
        GreenEnergyProvider base = provider(1.0);
        GreenEnergyProvider tb12 = provider(TB12_SCALE);
        boolean sawNonZero = false;
        for (double t : new double[] { 0, 7200, 86400, 400_000, 2_000_000 }) {
            double raw = base.getCurrentPowerW(t);
            assertEquals(raw * TB12_SCALE, tb12.getCurrentPowerW(t), 1e-12,
                    "scaled reading must equal raw * scale at t=" + t);
            if (raw > 0) sawNonZero = true;
        }
        assertTrue(sawNonZero, "test would be vacuous if every sampled reading were zero");
    }

    @Test
    void nonPositiveScaleFallsBackToLegacyOne() {
        GreenEnergyProvider bad = provider(0.0);
        GreenEnergyProvider one = provider(1.0);
        assertEquals(one.getCurrentPowerW(86400), bad.getCurrentPowerW(86400), 0.0,
                "scale<=0 must fall back to 1.0, not zero out the farm");
    }

    @Test
    void tb12ScaleLandsNearRegisteredMeanGreen() {
        // rho=0.5 calibration: mean scaled T100+T101 green should be ~47.9 W.
        // One turbine alone (T100) should land in the same order of magnitude:
        // this is a sanity anchor, not a tight bound (T100 != half of the sum
        // exactly, and 2021 drifts from the 2020 calibration year).
        GreenEnergyProvider tb12 = provider(TB12_SCALE);
        double sum = 0;
        int n = 0;
        for (double t = 0; t < 31_000_000; t += 60_000) { // ~1 year, 517 samples
            sum += tb12.getCurrentPowerW(t);
            n++;
        }
        double mean = sum / n;
        assertTrue(mean > 5 && mean < 100,
                "mean scaled single-turbine green " + mean + " W out of plausible range");
    }
}
