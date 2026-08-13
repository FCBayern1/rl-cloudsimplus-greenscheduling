package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * V3.1 reward-surgery tests (2026-08-13). Unlike the older mirror-style math
 * tests, these call the PRODUCTION formulas in {@link PerActionRewardMath}
 * directly — the same functions {@code MultiDatacenterSimulationCore} uses.
 *
 * <p>Four groups:
 * <ol>
 *   <li>Regression lock: default modes reproduce the legacy arithmetic exactly.</li>
 *   <li>Unit properties: no_offset identity/ordering, urgency shape,
 *       telescoping settlement (incl. the anti-escape final-route charge),
 *       z-score clip, scale_only zero point.</li>
 *   <li>The four-row REWARD TRUTH TABLE under the recommended switch combo —
 *       encoded as short discounted PATH comparisons, because route-vs-defer
 *       is an intertemporal choice, not a single-slot one. A switch combo that
 *       fails these rows must not enter training (docs/V31_WORK_ORDERS.md).</li>
 *   <li>The documented STRUCTURAL negative: scale_only + no_offset cannot pass
 *       row 1 (green route stays below defer's 0), which is why
 *       centered_zscore's +wμ/σ carbon threshold is load-bearing.</li>
 * </ol>
 */
public class PerActionRewardSurgeryTest {

    // Recommended-combo parameters used by the truth table. Carbon numbers are
    // a plausible v3-scale two-point distribution: green route ~0.3 kg-units,
    // brown route 55x higher (effFactor 0.55 vs 0.01), mu/sigma placed so the
    // two sit at roughly ∓1 sigma. The REAL certification values come from the
    // calibration artifact; preflight re-runs these rows with those values.
    private static final double W_CARBON = 0.5;
    private static final double W_COMPLETION = 2.0;
    private static final double W_URGENCY = 2.0;
    private static final double WINDOW = 3600.0;
    private static final double GAMMA = 0.999;
    // Overridable via -Dv31.* so the SAME truth table re-runs with the real
    // calibration-artifact values before the cert combo is locked:
    //   ./gradlew test --tests "*.PerActionRewardSurgeryTest" \
    //       -Dv31.mu=3.524 -Dv31.sigma=2.512 -Dv31.margGreen=... -Dv31.margBrown=...
    // Defaults keep the synthetic structural gate unchanged.
    private static final double MARG_GREEN =
            Double.parseDouble(System.getProperty("v31.margGreen", "0.3"));
    private static final double MARG_BROWN =
            Double.parseDouble(System.getProperty("v31.margBrown", "16.5"));
    private static final double MU =
            Double.parseDouble(System.getProperty("v31.mu", "8.4"));
    private static final double SIGMA =
            Double.parseDouble(System.getProperty("v31.sigma", "8.1"));
    private static final double NORMALIZER = 0.05; // legacy fixed normalizer

    private static double u(double slack) {
        return PerActionRewardMath.urgency(slack, WINDOW);
    }

    private static double route(String completionMode, String carbonMode,
                                double marginalKg, double p) {
        return PerActionRewardMath.routeReward(completionMode, carbonMode,
                W_CARBON, W_COMPLETION, marginalKg, p, NORMALIZER, MU, SIGMA);
    }

    /** Discounted defer path: settle U over the wait, then route k steps later. */
    private static double deferPath(double slackNow, double slackAtRoute, int steps,
                                    double routeRewardAtEnd) {
        double settlement = PerActionRewardMath.urgencySettlement(
                W_URGENCY, u(slackAtRoute), u(slackNow));
        return settlement + Math.pow(GAMMA, steps) * routeRewardAtEnd;
    }

    // ------------------------------------------------------------------
    // 1. Regression lock
    // ------------------------------------------------------------------

    @Test
    public void defaultModesReproduceLegacyArithmeticExactly() {
        double[] kgs = {0.0, 0.004, 0.05, 0.3, 1.7, 16.5};
        double[] ps = {0.0, 0.05, 0.5, 0.97, 1.0};
        for (double kg : kgs) {
            for (double p : ps) {
                double legacy = -W_CARBON * (kg / NORMALIZER) + W_COMPLETION * p;
                double now = route("bonus", "fixed", kg, p);
                assertEquals(legacy, now, 0.0,   // bit-exact, no tolerance
                        "legacy mismatch at kg=" + kg + " p=" + p);
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Unit properties
    // ------------------------------------------------------------------

    @Test
    public void noOffsetIsPureShiftPreservingDcOrdering() {
        // identity: −w(1−p) = w·p − w
        for (double p : new double[]{0.0, 0.3, 0.5, 1.0}) {
            assertEquals(W_COMPLETION * p - W_COMPLETION,
                    PerActionRewardMath.completionTerm("no_offset", W_COMPLETION, p), 1e-12);
        }
        // p=1 -> 0, p=0.5 -> −w/2
        assertEquals(0.0, PerActionRewardMath.completionTerm("no_offset", W_COMPLETION, 1.0), 0.0);
        assertEquals(-W_COMPLETION / 2,
                PerActionRewardMath.completionTerm("no_offset", W_COMPLETION, 0.5), 1e-12);
        // ordering between two DCs is the SAME difference in both modes
        double dBonus = PerActionRewardMath.completionTerm("bonus", W_COMPLETION, 0.9)
                - PerActionRewardMath.completionTerm("bonus", W_COMPLETION, 0.4);
        double dNoOff = PerActionRewardMath.completionTerm("no_offset", W_COMPLETION, 0.9)
                - PerActionRewardMath.completionTerm("no_offset", W_COMPLETION, 0.4);
        assertEquals(dBonus, dNoOff, 1e-12);
    }

    @Test
    public void urgencyShapeIsClippedQuadratic() {
        assertEquals(1.0, u(0.0), 1e-12);
        assertEquals(0.25, u(WINDOW / 2), 1e-12);
        assertEquals(0.0, u(WINDOW), 1e-12);
        assertEquals(0.0, u(2 * WINDOW), 1e-12);   // the first-draft formula returned 1 here
        assertEquals(0.0, u(10 * WINDOW), 1e-12);
        assertEquals(1.0, u(-100.0), 1e-12);       // overdue = maximally urgent
        // monotone non-increasing in slack
        double prev = Double.MAX_VALUE;
        for (double s = -200; s <= 2 * WINDOW; s += 100) {
            double v = u(s);
            assertTrue(v <= prev + 1e-12, "urgency rose at slack=" + s);
            prev = v;
        }
    }

    @Test
    public void settlementTelescopesRegardlessOfEncounterPattern() {
        // A job waits from slack 3000 down to 600 and is then routed. Whatever
        // the intermediate sighting pattern, total charge must be
        // −w·[U(600) − U(3000)] — including the FINAL route settlement, so the
        // last waiting segment cannot escape.
        double expected = -W_URGENCY * (u(600) - u(3000));
        double[][] patterns = {
                {3000, 600},                          // one hop, settled at route
                {3000, 2400, 1800, 1200, 600},        // even re-encounters
                {3000, 2990, 700, 600},               // wildly uneven gaps
                {3000, 1000, 900, 800, 700, 650, 600} // many late sightings
        };
        for (double[] slacks : patterns) {
            double total = 0.0;
            Double last = null;                        // ledger: first sighting = baseline
            for (double s : slacks) {
                if (last != null) {
                    total += PerActionRewardMath.urgencySettlement(W_URGENCY, u(s), u(last));
                }
                last = s;
            }
            assertEquals(expected, total, 1e-12,
                    "telescoping broken for pattern of length " + slacks.length);
        }
    }

    @Test
    public void zscoreClipsAndCentersCorrectly() {
        assertEquals(0.0,
                PerActionRewardMath.normalizeCarbon("centered_zscore", MU, NORMALIZER, MU, SIGMA), 1e-12);
        double sixSigma = MU + 6 * SIGMA;
        assertEquals(PerActionRewardMath.ZSCORE_CLIP,
                PerActionRewardMath.normalizeCarbon("centered_zscore", sixSigma, NORMALIZER, MU, SIGMA), 1e-12);
        assertTrue(PerActionRewardMath.zscoreWouldClip(sixSigma, MU, SIGMA));
        assertFalse(PerActionRewardMath.zscoreWouldClip(MU + 4 * SIGMA, MU, SIGMA));
    }

    @Test
    public void scaleOnlyKeepsPhysicalZero() {
        assertEquals(0.0,
                PerActionRewardMath.normalizeCarbon("scale_only", 0.0, NORMALIZER, MU, SIGMA), 0.0);
        assertEquals(MARG_GREEN / SIGMA,
                PerActionRewardMath.normalizeCarbon("scale_only", MARG_GREEN, NORMALIZER, MU, SIGMA), 1e-12);
    }

    // ------------------------------------------------------------------
    // 3. Truth table — recommended combo: no_offset + centered_zscore
    //    + incremental_urgency. Rows are discounted path comparisons.
    // ------------------------------------------------------------------

    @Test
    public void row1_greenNow_routeBeatsDefer() {
        double routeNow = route("no_offset", "centered_zscore", MARG_GREEN, 1.0);
        assertTrue(routeNow > 0, "green route must be positive under centered_zscore");
        // defer 60 steps and route green later: pays urgency drift + discount
        double defer = deferPath(3000, 2940, 60,
                route("no_offset", "centered_zscore", MARG_GREEN, 1.0));
        assertTrue(routeNow > defer, "row 1: route_green_now must beat defer");
    }

    @Test
    public void row2_brownNow_greenComing_ampleSlack_deferWins() {
        double routeBrownNow = route("no_offset", "centered_zscore", MARG_BROWN, 1.0);
        // wait 300 steps for green, ample slack (3000 -> 2700)
        double defer = deferPath(3000, 2700, 300,
                route("no_offset", "centered_zscore", MARG_GREEN, 1.0));
        assertTrue(defer > routeBrownNow,
                "row 2: waiting for green must beat routing brown now");
    }

    @Test
    public void row3_deadlineTight_noGreenInTime_routeWins() {
        // Slack 600s, green will NOT arrive before the backstop force-routes at
        // slack~120 — the defer path pays steep urgency drift and still routes
        // brown. (The green-arrives-in-time variant is an environment-dynamics
        // question — probComplete/backstop — not a per-action reward property.)
        double routeBrownNow = route("no_offset", "centered_zscore", MARG_BROWN, 1.0);
        double defer = deferPath(600, 120, 480,
                route("no_offset", "centered_zscore", MARG_BROWN, 1.0));
        assertTrue(routeBrownNow > defer,
                "row 3: with a tight deadline and no green in time, route now must win");
    }

    @Test
    public void row4_congestedDc_losesToAlternativeAndToDefer() {
        double congested = route("no_offset", "centered_zscore", MARG_GREEN, 0.3);
        double otherDc = route("no_offset", "centered_zscore", MARG_GREEN, 1.0);
        double defer = deferPath(3000, 2940, 60,
                route("no_offset", "centered_zscore", MARG_GREEN, 1.0));
        assertTrue(otherDc > congested, "row 4: uncongested DC must beat congested");
        assertTrue(defer > congested, "row 4: defer must beat congested route");
    }

    // ------------------------------------------------------------------
    // 4. Structural negative (documents WHY centered_zscore is load-bearing)
    // ------------------------------------------------------------------

    @Test
    public void scaleOnlyPlusNoOffsetStructurallyFailsRow1() {
        // Under scale_only the green route reward is a small NEGATIVE number
        // while a fresh defer charges ~0 -> defer weakly dominates every route
        // and row 1 cannot hold. This is the all-defer failure mode; the test
        // pins the fact so nobody ships this combo by accident.
        double routeGreen = route("no_offset", "scale_only", MARG_GREEN, 1.0);
        assertTrue(routeGreen < 0, "scale_only green route is negative by construction");
        double freshDefer = 0.0; // first sighting = baseline, no charge
        assertTrue(freshDefer > routeGreen,
                "structural fact: scale_only + no_offset -> defer dominates green route");
    }
}
