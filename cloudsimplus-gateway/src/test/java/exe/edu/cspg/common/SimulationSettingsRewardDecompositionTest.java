package exe.edu.cspg.common;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Regression test for the 2026-05-12 per-action reward decomposition fields.
 *
 * Background: the previous global reward (`α·L − β·Ĉ − γ·R_w + completion_mi`)
 * shared one scalar across all N routing decisions per step, leaving PPO with
 * no per-action gradient signal.  We added three decomposable terms:
 *   fit-score shaping (λ_shape · mean_i fit_i),
 *   marginal carbon penalty (−β_marg · marginal_kg),
 *   per-step completion shaping (k · (finished − baseline)).
 *
 * These tests pin down:
 *   (1) defaults are all OFF (zero coefs) — no behaviour change for legacy configs.
 *   (2) every new parameter round-trips through the YAML param map.
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

        // The four fit-score weights have non-zero defaults (1.0/1.0/1.0/2.0)
        // so users only need to flip λ_shape to enable shaping.
        assertEquals(1.0, s.getGlobalFitWeightGreen(),    1e-12);
        assertEquals(1.0, s.getGlobalFitWeightBrown(),    1e-12);
        assertEquals(1.0, s.getGlobalFitWeightQueue(),    1e-12);
        assertEquals(2.0, s.getGlobalFitWeightOverflow(), 1e-12);

        // The three top-level scalars MUST default to 0.0 — otherwise any
        // config that doesn't mention them silently changes reward shape.
        assertEquals(0.0, s.getGlobalFitLambda(),             1e-12,
                "λ_shape MUST default OFF — pre-fix configs would silently change behaviour otherwise");
        assertEquals(0.0, s.getGlobalRewardBetaMarginal(),    1e-12,
                "β_marg MUST default OFF");
        assertEquals(0.0, s.getGlobalCompletionPerStepCoef(), 1e-12,
                "completion-per-step coef MUST default OFF");
    }

    @Test
    public void allFieldsRoundTripFromYamlParamMap() {
        Map<String, Object> p = minimalParams();
        p.put("global_fit_weight_green",    0.7);
        p.put("global_fit_weight_brown",    1.3);
        p.put("global_fit_weight_queue",    0.5);
        p.put("global_fit_weight_overflow", 3.0);
        p.put("global_fit_lambda",          0.42);
        p.put("global_reward_beta_marginal", 0.077);
        p.put("global_completion_per_step_coef", 0.013);

        SimulationSettings s = new SimulationSettings(p);

        assertEquals(0.7,   s.getGlobalFitWeightGreen(),       1e-12);
        assertEquals(1.3,   s.getGlobalFitWeightBrown(),       1e-12);
        assertEquals(0.5,   s.getGlobalFitWeightQueue(),       1e-12);
        assertEquals(3.0,   s.getGlobalFitWeightOverflow(),    1e-12);
        assertEquals(0.42,  s.getGlobalFitLambda(),            1e-12);
        assertEquals(0.077, s.getGlobalRewardBetaMarginal(),   1e-12);
        assertEquals(0.013, s.getGlobalCompletionPerStepCoef(), 1e-12);
    }

    @Test
    public void integerCoefsAreAcceptedAndCoerced() {
        // YAML parsers often hand integers (1) where doubles (1.0) are expected.
        // getDoubleParam handles this — verify the new keys do too.
        Map<String, Object> p = minimalParams();
        p.put("global_fit_lambda", 1);  // int, not double
        p.put("global_reward_beta_marginal", 2);
        SimulationSettings s = new SimulationSettings(p);
        assertEquals(1.0, s.getGlobalFitLambda(),          1e-12);
        assertEquals(2.0, s.getGlobalRewardBetaMarginal(), 1e-12);
    }
}
