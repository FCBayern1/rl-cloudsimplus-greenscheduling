package exe.edu.cspg.multidc;

/**
 * Pure math for the V3.1 per-action reward surgery (2026-08-13).
 *
 * <p>Every function here is called by BOTH {@link MultiDatacenterSimulationCore}
 * and the unit tests, so the tests exercise the real production formulas rather
 * than a hand-copied mirror (mirror drift bit us before). All functions are
 * stateless and side-effect free.
 *
 * <p>Design notes (docs/V31_WORK_ORDERS.md, fourth-review revision):
 * <ul>
 *   <li>{@code completionTerm} "no_offset" mode: algebraically
 *       {@code −w(1−p) = wp − w}, i.e. it preserves the completion ORDERING
 *       between DCs and only removes the constant +w that every route action
 *       received but defer did not. It is a route-offset removal, not a new
 *       learning signal.</li>
 *   <li>{@code urgency} uses {@code clip(1 − s/W, 0, 1)²}: zero for all
 *       s ≥ W (the first-draft {@code min(1,(1−s/W)²)} wrongly ROSE again for
 *       s &gt; W), quadratic ramp inside the window, and 1 for negative slack
 *       (overdue work is maximally urgent).</li>
 *   <li>{@code urgencySettlement} implements the incremental (telescoping)
 *       charge {@code −w·[U(now) − U(last)]}: summed over every sighting of a
 *       cloudlet — including the final one where it is routed — the total is
 *       {@code −w·[U(final) − U(first)]}, independent of how many times or at
 *       what intervals the job re-entered the batch. This replaces the flat
 *       per-encounter tax whose total grew with encounter count.</li>
 *   <li>{@code normalizeCarbon} "centered_zscore" deliberately adds a
 *       {@code +w·μ/σ} advantage to below-mean-carbon routes relative to
 *       defer. That constant is NOT an accident of normalisation: it defines
 *       the carbon threshold below which routing now beats waiting, and it is
 *       the term that lets truth-table row 1 (green route &gt; defer) hold.
 *       "scale_only" keeps the physical zero point and therefore cannot pass
 *       row 1 on its own (green routes stay slightly negative vs defer ≈ 0).</li>
 * </ul>
 */
final class PerActionRewardMath {

    /** Symmetric clip bound for the centered z-score mode. */
    static final double ZSCORE_CLIP = 5.0;

    private PerActionRewardMath() { }

    /**
     * Completion term of the per-action route reward.
     *
     * @param mode "bonus" (legacy, {@code +w·p}) or "no_offset" ({@code −w·(1−p)})
     */
    static double completionTerm(String mode, double wCompletion, double probComplete) {
        if ("no_offset".equals(mode)) {
            return -wCompletion * (1.0 - probComplete);
        }
        // legacy default — kept byte-identical to the pre-surgery formula
        return wCompletion * probComplete;
    }

    /**
     * Urgency U(s) = clip(1 − s/W, 0, 1)² for slack s seconds and window W.
     * U(s ≥ W) = 0, U(0) = 1, U(s &lt; 0) = 1 (overdue = maximally urgent).
     */
    static double urgency(double slackSec, double windowSec) {
        double w = Math.max(1.0, windowSec);
        double x = Math.max(0.0, Math.min(1.0, 1.0 - slackSec / w));
        return x * x;
    }

    /** Incremental urgency settlement {@code −w·[uNow − uLast]}. */
    static double urgencySettlement(double wUrgency, double uNow, double uLast) {
        return -wUrgency * (uNow - uLast);
    }

    /**
     * Carbon-term normalisation.
     *
     * @param mode            "fixed" (legacy {@code kg/normalizer}),
     *                        "scale_only" ({@code kg/σ}, physical zero kept), or
     *                        "centered_zscore" ({@code clip((kg−μ)/σ, ±5)})
     * @param flooredNormalizer the legacy normalizer, already floored by the
     *                        caller ({@code max(1e-6, per_action_marg_normalizer)})
     *                        so the "fixed" branch reproduces the old arithmetic
     *                        bit-for-bit
     */
    static double normalizeCarbon(String mode, double marginalKg, double flooredNormalizer,
                                  double mu, double sigma) {
        switch (mode) {
            case "scale_only":
                return marginalKg / Math.max(1e-12, sigma);
            case "centered_zscore":
                double z = (marginalKg - mu) / Math.max(1e-12, sigma);
                return Math.max(-ZSCORE_CLIP, Math.min(ZSCORE_CLIP, z));
            default: // "fixed" — legacy
                return marginalKg / flooredNormalizer;
        }
    }

    /** True when the centered z-score of {@code marginalKg} falls outside ±{@link #ZSCORE_CLIP}. */
    static boolean zscoreWouldClip(double marginalKg, double mu, double sigma) {
        double z = (marginalKg - mu) / Math.max(1e-12, sigma);
        return z > ZSCORE_CLIP || z < -ZSCORE_CLIP;
    }

    /**
     * V3.2 candidate-centered SPATIAL term (docs/V32_FORECAST_REVIVAL_PLAN.md
     * §4.3 as amended by §11 Q4): {@code −w_s·(C_j − mean_feasible)/σ_spatial}.
     *
     * <p>Mean-zero across the feasible candidate set by construction, so it
     * ranks DCs against each other WITHOUT adding any route-vs-defer bias in
     * expectation — the temporal threshold semantics stay with the level term
     * (centered_zscore). This restores the DC-vs-DC gradient that the single
     * global σ compressed ~70x in V3.1 (control-channel TV 0.003 vs 0.82).
     */
    static double spatialCenteredTerm(double wSpatial, double marginalKg,
                                      double candidateMean, double sigmaSpatial) {
        return -wSpatial * (marginalKg - candidateMean) / Math.max(1e-12, sigmaSpatial);
    }

    /**
     * Full per-action ROUTE reward under the given modes:
     * {@code −wC·normalize(kg) + completionTerm(p)}. Used directly by the
     * truth-table tests; {@code accumulatePerActionReward} composes the same
     * two calls inline.
     */
    static double routeReward(String completionMode, String carbonNormMode,
                              double wCarbon, double wCompletion,
                              double marginalKg, double probComplete,
                              double flooredNormalizer, double mu, double sigma) {
        return -wCarbon * normalizeCarbon(carbonNormMode, marginalKg, flooredNormalizer, mu, sigma)
                + completionTerm(completionMode, wCompletion, probComplete);
    }
}
