package exe.edu.cspg.multidc;

import exe.edu.cspg.common.SimulationSettings;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Stage D window allowlist: explicit offsets cycled by episode index, schedule otherwise. */
public class EpisodeOffsetAllowlistTest {

    @Test
    public void allowlistIsCycledByEpisodeIndex() {
        List<Integer> allow = List.of(13016, 21088, 29160);
        assertEquals(13016, MultiDatacenterSimulationCore.episodeOffsetFor(0, 44950, allow));
        assertEquals(21088, MultiDatacenterSimulationCore.episodeOffsetFor(1, 44950, allow));
        assertEquals(29160, MultiDatacenterSimulationCore.episodeOffsetFor(2, 44950, allow));
        assertEquals(13016, MultiDatacenterSimulationCore.episodeOffsetFor(3, 44950, allow));
        // the allowlist wins even when the schedule range is 0
        assertEquals(21088, MultiDatacenterSimulationCore.episodeOffsetFor(4, 0, allow));
    }

    @Test
    public void emptyAllowlistKeepsTheSchedule() {
        assertEquals(2018, MultiDatacenterSimulationCore.episodeOffsetFor(2, 44950, List.of()));
        assertEquals(2018, MultiDatacenterSimulationCore.episodeOffsetFor(2, 44950, null));
    }

    @Test
    public void parsesStringsAndLists() {
        assertEquals(List.of(13016, 21088), SimulationSettings.parseIntList("13016;21088"));
        assertEquals(List.of(1, 2, 3), SimulationSettings.parseIntList("1, 2 3"));
        assertEquals(List.of(7, 8), SimulationSettings.parseIntList(List.of(7, 8L)));
        assertTrue(SimulationSettings.parseIntList(null).isEmpty());
        assertTrue(SimulationSettings.parseIntList("  ").isEmpty());
    }
}
