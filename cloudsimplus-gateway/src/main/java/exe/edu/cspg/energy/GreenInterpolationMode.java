package exe.edu.cspg.energy;

/**
 * How CSV wind rows are turned into a continuous power signal (testbed 12,
 * Codex ruling 2026-08-23).
 *
 * SDWPF rows are 10-minute SCADA samples; the dataset defines no physical
 * curve between them. The legacy unconstrained cubic spline oscillates
 * between jagged samples and undershoots negative near deep lulls, and the
 * subsequent max(0,·) clamps only the negative side - an asymmetric bias
 * that rewrites the wind the planner, the observation and the carbon
 * ledger all consume.
 *
 * STEP:   row i holds over its registered 600-second unit
 *         [i*rowSeconds, (i+1)*rowSeconds). The registered semantics for
 *         formal TB12 cells; makes offline planning and simulator
 *         execution bit-consistent.
 * LINEAR: piecewise-linear between samples (bounded by endpoints);
 *         sensitivity cell only, not a formal main cell.
 * SPLINE: legacy cubic; kept ONLY to reproduce historical results.
 */
public enum GreenInterpolationMode {
    SPLINE,
    LINEAR,
    STEP;

    public static GreenInterpolationMode parse(String s) {
        if (s == null) return SPLINE;
        switch (s.trim().toUpperCase()) {
            case "STEP":   return STEP;
            case "LINEAR": return LINEAR;
            case "SPLINE": return SPLINE;
            default:       return SPLINE;
        }
    }
}
