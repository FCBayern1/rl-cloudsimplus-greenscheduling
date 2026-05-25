package exe.edu.cspg.common;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression test for the 2026-05-16 per-action reward decomposition.
 *
 * <p>The per-step global reward gets an additive Σᵢ rᵢ term computed at
 * routing time, where rᵢ depends only on slot i's action.  This makes the
 * reward decomposable across the N routing slots so PPO's per-slot policy
 * gradient gets a useful attribution signal even with shared advantage.
 *
 * <p>This test pins down:
 *   (1) defaults are all OFF (zero weights) so any legacy config behaves identically;
 *   (2) every new parameter round-trips correctly through the YAML param map;
 *   (3) numeric YAML coercion works (integers passed where doubles expected).
 */
public class SimulationSettingsRewardDecompositionTest {

    private Map<String, Object> minimalParams() {
        Map<String, Object> p = new HashMap<>();
        p.put("simulation_timestep", 1.0);
        p.put("min_time_between_events", 0.1);
        return p;
    }

    @Test
    public void defaultsAreSafeNoOpForLegacyConfigs() {
        SimulationSettings s = new SimulationSettings(minimalParams());

        // Weights MUST default to 0 so reward decomposition is opt-in.
        // Otherwise any legacy config without these keys would silently
        // change reward shape.
        assertEquals(0.0, s.getPerActionCarbonWeight(),     1e-12,
                "w_carbon MUST default OFF");
        assertEquals(0.0, s.getPerActionCompletionWeight(), 1e-12,
                "w_completion MUST default OFF");

        // Calibration constants have sensible non-zero defaults so the
        // formulas don't divide by zero when someone first opts in:
        //   mi_per_kg_factor — calibrated so per-action |reward| ≈ 0.05
        //     for a typical 700K-MI cloudlet at brown=0.5, green=0.5
        //     (smoke 20260515_174514).
        //   overflow_sharpness — k=3 in exp(−k · overflow_frac) gives
        //     prob_complete ≈ 0.55 at 20% overflow, ≈ 0.05 at 100%.
        assertEquals(3.5e6, s.getMiPerKgFactor(),             1.0,
                "mi_per_kg_factor default 3.5e6 (calibrated 2026-05-16)");
        assertEquals(3.0,   s.getPerActionOverflowSharpness(), 1e-12,
                "overflow_sharpness default k=3");
        // 2026-05-20: marg_normalizer maps actual marg into [0, 1] for the
        // absolute reward formula; default 0.05 = typical 700K-MI cloudlet at
        // 50/50 green/brown ratio.
        assertEquals(0.05, s.getPerActionMargNormalizer(), 1e-12,
                "marg_normalizer default 0.05 (2026-05-20 absolute reward calibration)");
    }

    @Test
    public void weightsRoundTripFromYamlParamMap() {
        Map<String, Object> p = minimalParams();
        p.put("per_action_carbon_weight",     1.0);
        p.put("per_action_completion_weight", 0.05);
        p.put("mi_per_kg_factor",             5.0e6);
        p.put("per_action_overflow_sharpness", 5.0);
        p.put("per_action_marg_normalizer",   0.08);

        SimulationSettings s = new SimulationSettings(p);

        assertEquals(1.0,   s.getPerActionCarbonWeight(),       1e-12);
        assertEquals(0.05,  s.getPerActionCompletionWeight(),   1e-12);
        assertEquals(5.0e6, s.getMiPerKgFactor(),               1.0);
        assertEquals(5.0,   s.getPerActionOverflowSharpness(),  1e-12);
        assertEquals(0.08,  s.getPerActionMargNormalizer(),     1e-12);
    }

    @Test
    public void integerCoefsAreAcceptedAndCoerced() {
        // YAML parsers often hand integers (1) where doubles (1.0) are
        // expected.  getDoubleParam handles this — verify the new keys do too.
        Map<String, Object> p = minimalParams();
        p.put("per_action_carbon_weight", 2);          // int, not double
        p.put("per_action_completion_weight", 0);
        p.put("mi_per_kg_factor", 7000000);            // int

        SimulationSettings s = new SimulationSettings(p);
        assertEquals(2.0,     s.getPerActionCarbonWeight(),     1e-12);
        assertEquals(0.0,     s.getPerActionCompletionWeight(), 1e-12);
        assertEquals(7.0e6,   s.getMiPerKgFactor(),             1.0);
    }
}
