package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Encoding contract for the evaluator-only execution trace (Codex, 2026-08-30).
 *
 * The info map bridges to Python as strings, so an array put there verbatim arrives as
 * "[J@1b6d". Records are joined with ';' and fields with ':', and the per-datacentre PE
 * vectors with ','. These tests pin the separators, because a planner that parses this
 * trace wrongly fails in exactly the silent way the dc_available_pes misreading did.
 */
public class ExecutionTraceCsvTest {

    @Test
    public void perDatacentreVectorsAreCommaJoinedInDatacentreOrder() {
        assertEquals("480,384,296,240,144",
                MultiDatacenterSimulationCore.joinLongsForTest(
                        new long[]{480, 384, 296, 240, 144}));
    }

    @Test
    public void anEmptyVectorEncodesAsTheEmptyString() {
        assertEquals("", MultiDatacenterSimulationCore.joinLongsForTest(new long[]{}));
    }

    @Test
    public void aSingleSiteStillHasNoTrailingSeparator() {
        assertEquals("7", MultiDatacenterSimulationCore.joinLongsForTest(new long[]{7}));
    }
}
