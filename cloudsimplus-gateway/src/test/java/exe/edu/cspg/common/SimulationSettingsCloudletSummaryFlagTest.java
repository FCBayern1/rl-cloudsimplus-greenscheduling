package exe.edu.cspg.common;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression test for the 2026-05-12 JVM-hang fix.
 *
 * Background: the previous resetSimulation() unconditionally dumped the
 * entire finished-cloudlet table to stdout, producing a 939MB log over
 * 76 training iterations and (probably) hanging the JVM on stdout-pipe
 * back-pressure.  The fix gates that dump behind
 * {@code print_cloudlet_summary_on_reset}, defaulting to {@code false}.
 *
 * These tests guard against:
 *   (1) someone flipping the default back to {@code true} by accident, and
 *   (2) the flag being misspelled / unparseable from the YAML config map.
 */
public class SimulationSettingsCloudletSummaryFlagTest {

    /** Minimal param map containing every key SimulationSettings requires. */
    private Map<String, Object> minimalParams() {
        Map<String, Object> p = new HashMap<>();
        // The constructor reads with getXxxParam(..., default), so an empty
        // map is technically legal — but we set a few fields explicitly to
        // keep the smoke tests realistic.
        p.put("simulation_timestep", 1.0);
        p.put("min_time_between_events", 0.1);
        return p;
    }

    @Test
    public void defaultIsFalse_neverFloodStdoutWithoutOptIn() {
        SimulationSettings s = new SimulationSettings(minimalParams());
        assertFalse(s.isPrintCloudletSummaryOnReset(),
                "Default MUST be false to prevent the 940MB log regression");
    }

    @Test
    public void explicitTrue_isHonoured() {
        Map<String, Object> p = minimalParams();
        p.put("print_cloudlet_summary_on_reset", true);
        SimulationSettings s = new SimulationSettings(p);
        assertTrue(s.isPrintCloudletSummaryOnReset(),
                "When explicitly enabled the flag must round-trip");
    }

    @Test
    public void explicitFalse_isHonoured() {
        Map<String, Object> p = minimalParams();
        p.put("print_cloudlet_summary_on_reset", false);
        SimulationSettings s = new SimulationSettings(p);
        assertFalse(s.isPrintCloudletSummaryOnReset());
    }
}
