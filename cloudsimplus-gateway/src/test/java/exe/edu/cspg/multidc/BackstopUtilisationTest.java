package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The latest-start backstop has to use the runtime the simulator will actually deliver.
 *
 * A cloudlet executes at cloudlet_cpu_utilization of a VM PE, so it occupies its site for
 * length / (mips * u). Measured 2026-08-30 against the simulator's own finish events:
 * twelve of twelve cloudlets ran 2.0166x the nominal length/mips at u = 0.5, sd 0.0156,
 * and 4.0201x at u = 0.25, with no dependence on PES. Before this, the backstop believed
 * every job was half as long as it is and fired about half a runtime too late.
 */
public class BackstopUtilisationTest {

    private static final double MIPS = 40000.0;
    private static final double LENGTH = 800000.0;    // 20s nominal, 40s at u = 0.5

    @Test
    public void halfUtilisationDoublesTheRuntimeTheBackstopAssumes() {
        // Deadline 30s away: safe at full speed, already too late at half speed.
        assertFalse(PerActionRewardMath.deadlineForceLatestStart(
                0.0, 30.0, LENGTH, 2, MIPS, 0.0, 1.0));
        assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                0.0, 30.0, LENGTH, 2, MIPS, 0.0, 0.5));
    }

    @Test
    public void runtimeDoesNotShrinkWithPes() {
        for (long pes : new long[]{1, 2, 4, 8}) {
            assertTrue(PerActionRewardMath.deadlineForceLatestStart(
                    0.0, 39.9, LENGTH, pes, MIPS, 0.0, 0.5),
                    "pes " + pes + " changed the assumed runtime");
        }
    }

    @Test
    public void theLegacyOverloadKeepsFullSpeedSoOldCallersAreUnchanged() {
        assertFalse(PerActionRewardMath.deadlineForceLatestStart(
                0.0, 30.0, LENGTH, 2, MIPS, 0.0));
    }

    @Test
    public void anUnusableUtilisationFallsBackToFullSpeedRatherThanDividingByZero() {
        assertEquals(1.0, PerActionRewardMath.clampUtilization(0.0));
        assertEquals(1.0, PerActionRewardMath.clampUtilization(-1.0));
        assertEquals(1.0, PerActionRewardMath.clampUtilization(2.0));
        assertEquals(0.5, PerActionRewardMath.clampUtilization(0.5));
    }
}
