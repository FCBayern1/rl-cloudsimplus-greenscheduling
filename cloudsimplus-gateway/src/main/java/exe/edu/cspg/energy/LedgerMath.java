package exe.edu.cspg.energy;

import java.util.List;

/**
 * Pure carbon-ledger arithmetic (P0.5-3, Codex 2026-08-23), extracted so the
 * per-segment min() can be unit-tested without a live simulation.
 *
 * Integrating green ENERGY over a whole interval and then taking
 * min(demandWh, greenWh) lets surplus green in one row cover a deficit in
 * another row of the same control step - implicit intra-step storage. The
 * correct settlement is per overlap segment:
 *
 *     greenUsedWh = sum_j min(P_demand, P_green_j) * dt_j / 3600
 */
public final class LedgerMath {

    private LedgerMath() { }

    /** One constant-power segment of the interval. */
    public static final class Segment {
        public final double greenPowerW;
        public final double durationSec;
        public Segment(double greenPowerW, double durationSec) {
            this.greenPowerW = greenPowerW;
            this.durationSec = durationSec;
        }
    }

    /** Per-segment settlement: {greenUsedWh, brownUsedWh, greenWastedWh}. */
    public static double[] settlePerSegment(double demandPowerW, List<Segment> segments) {
        double used = 0.0, wasted = 0.0, demandWs = 0.0;
        for (Segment seg : segments) {
            double g = Math.max(0.0, seg.greenPowerW);
            double u = Math.min(demandPowerW, g);
            used += u * seg.durationSec;
            wasted += (g - u) * seg.durationSec;
            demandWs += demandPowerW * seg.durationSec;
        }
        double brown = Math.max(0.0, demandWs - used);
        return new double[] { used / 3600.0, brown / 3600.0, wasted / 3600.0 };
    }

    /** Legacy settlement (whole-interval energies) - kept for the non-STEP
     *  path and as the contrast case in tests. */
    public static double[] settleWholeInterval(double demandWh, double greenWh) {
        double used = Math.min(demandWh, greenWh);
        return new double[] { used, demandWh - used, greenWh - used };
    }
}
