package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the per-episode green-window offset schedule (2026-08-07).
 *
 * <p>Motivation: with a fixed window every episode replays the same wind
 * trajectory, a trained policy memorises the future, and forecast
 * observations become redundant by construction (godeye == noforecast).
 * The schedule (1009*k mod range) re-bases each episode onto a different
 * slice of the wind year. It must be deterministic in the episode index
 * (arm parity: both arms and eval see the same window sequence), keep
 * episode 0 on the historical window, and cover the range densely.
 */
public class EpisodeOffsetScheduleTest {

    @Test
    public void episodeZeroKeepsHistoricalWindow() {
        assertEquals(0, MultiDatacenterSimulationCore.episodeOffsetFor(0, 4800));
    }

    @Test
    public void disabledRangeAlwaysZero() {
        assertEquals(0, MultiDatacenterSimulationCore.episodeOffsetFor(7, 0));
        assertEquals(0, MultiDatacenterSimulationCore.episodeOffsetFor(7, -5));
        assertEquals(0, MultiDatacenterSimulationCore.episodeOffsetFor(-1, 4800));
    }

    @Test
    public void deterministicInEpisodeIndex() {
        for (int k = 0; k < 50; k++) {
            assertEquals(MultiDatacenterSimulationCore.episodeOffsetFor(k, 4800),
                         MultiDatacenterSimulationCore.episodeOffsetFor(k, 4800));
        }
    }

    @Test
    public void staysInRangeAndSpreadsWidely() {
        int range = 4800;
        Set<Integer> seen = new HashSet<>();
        for (int k = 0; k < 200; k++) {
            int off = MultiDatacenterSimulationCore.episodeOffsetFor(k, range);
            assertTrue(off >= 0 && off < range, "offset out of range: " + off);
            seen.add(off);
        }
        // 1009 is coprime with 4800 -> first 200 episodes give 200 distinct windows
        assertEquals(200, seen.size());
    }

    @Test
    public void noOverflowAtLargeEpisodeIndex() {
        int off = MultiDatacenterSimulationCore.episodeOffsetFor(Integer.MAX_VALUE, 4800);
        assertTrue(off >= 0 && off < 4800);
    }
}
