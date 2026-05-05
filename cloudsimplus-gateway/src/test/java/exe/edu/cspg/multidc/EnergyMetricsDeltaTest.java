package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Verifies that the pure analytical helpers added to EnergyMetricsDelta
 * (used by the CRD framework's forecast counterfactual) produce numerically
 * identical results to the inline formula in
 * {@link DatacenterInstance#updateEnergyMetrics(double)} (lines 320-355).
 */
public class EnergyMetricsDeltaTest {

    private static final double EPS = 1e-9;

    /** Re-derives the expected carbon kg using the formula from DatacenterInstance. */
    private static double expectedCarbonKg(
            double availableGreenW, double demandW, double durationHours,
            double greenFactor, double brownFactor) {
        double demandWh = Math.max(0.0, demandW) * durationHours;
        double greenAvailableWh = Math.max(0.0, availableGreenW) * durationHours;
        double greenUsedWh = Math.min(demandWh, greenAvailableWh);
        double brownUsedWh = demandWh - greenUsedWh;
        double greenKWh = greenUsedWh / 1000.0;
        double brownKWh = brownUsedWh / 1000.0;
        return greenKWh * greenFactor + brownKWh * brownFactor;
    }

    @Test
    public void carbonMatchesInlineFormula_typicalCase() {
        double green = 800_000.0;     // 800 kW
        double demand = 1_500_000.0;  // 1500 kW
        double dt = 60.0 / 3600.0;    // 60s = 0.01667 h
        double gf = 0.0;              // green is carbon-free
        double bf = 0.5;              // 0.5 kgCO2/kWh for grid

        double got = EnergyMetricsDelta.computeCarbonKg(green, demand, dt, gf, bf);
        double exp = expectedCarbonKg(green, demand, dt, gf, bf);
        assertEquals(exp, got, EPS, "carbon kg must match the inline formula");
    }

    @Test
    public void carbonMatchesInlineFormula_greenExceedsDemand() {
        double green = 2_000_000.0;
        double demand = 500_000.0;
        double dt = 0.5;
        double gf = 0.04;
        double bf = 0.5;

        double got = EnergyMetricsDelta.computeCarbonKg(green, demand, dt, gf, bf);
        double exp = expectedCarbonKg(green, demand, dt, gf, bf);
        assertEquals(exp, got, EPS);
        // No brown used → all green. demand*dt = 500_000W * 0.5h = 250 kWh consumed,
        // all from green → carbon = 250 kWh * gf = 250 * 0.04 = 10.0 kg.
        assertEquals(10.0, got, EPS, "no brown → carbon = greenKWh * greenFactor");
    }

    @Test
    public void carbonMatchesInlineFormula_zeroGreen() {
        double green = 0.0;
        double demand = 1_000_000.0;
        double dt = 1.0;
        double gf = 0.0;
        double bf = 0.5;

        double got = EnergyMetricsDelta.computeCarbonKg(green, demand, dt, gf, bf);
        // 1000 kWh * 0.5 kg/kWh = 500 kg
        assertEquals(500.0, got, EPS);
    }

    @Test
    public void carbonMatchesInlineFormula_zeroDemand() {
        double got = EnergyMetricsDelta.computeCarbonKg(1_000_000.0, 0.0, 1.0, 0.0, 0.5);
        assertEquals(0.0, got, EPS, "zero demand → zero carbon");
    }

    @Test
    public void greenUsedWastedSplit_typicalCase() {
        double green = 800_000.0;
        double demand = 1_500_000.0;
        double dt = 1.0;  // 1 hour for easy mental math

        double[] uw = EnergyMetricsDelta.computeGreenUsedWastedWh(green, demand, dt);
        // greenAvailableWh = 800000, demandWh = 1500000, used = min = 800000, wasted = 0
        assertArrayEquals(new double[]{800_000.0, 0.0}, uw, EPS);
    }

    @Test
    public void greenUsedWastedSplit_greenExceedsDemand() {
        double green = 2_000_000.0;
        double demand = 500_000.0;
        double dt = 1.0;

        double[] uw = EnergyMetricsDelta.computeGreenUsedWastedWh(green, demand, dt);
        // used = 500000, wasted = 1500000
        assertArrayEquals(new double[]{500_000.0, 1_500_000.0}, uw, EPS);
    }

    @Test
    public void wasteRatio_zeroWhenNoGreen() {
        double r = EnergyMetricsDelta.computeWasteRatio(0.0, 1_000.0, 1.0);
        assertEquals(0.0, r, EPS);
    }

    @Test
    public void wasteRatio_typical() {
        // green=2000W, demand=500W, dt=1h
        // used=500, wasted=1500, ratio = 1500 / 2000 = 0.75
        double r = EnergyMetricsDelta.computeWasteRatio(2000.0, 500.0, 1.0);
        assertEquals(0.75, r, EPS);
    }

    @Test
    public void negativeInputs_clampedToZero() {
        double got = EnergyMetricsDelta.computeCarbonKg(-100.0, -200.0, 1.0, 0.0, 0.5);
        assertEquals(0.0, got, EPS, "negative inputs must be treated as 0");
    }
}
