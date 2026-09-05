package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Placement ledger repair (Codex ruling 2026-09-05): the four required properties.
 */
class PlacementLedgerTest {

    private static Map<Long, Long> free(long... idFree) {
        Map<Long, Long> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < idFree.length; i += 2) m.put(idFree[i], idFree[i + 1]);
        return m;
    }

    @Test
    void inFlightSubmissionCountsAsCommitted() {
        // the k5 forensic: VM 4 had exec 0, waiting 0, one 32-PE cloudlet in flight
        assertEquals(0L, PlacementLedger.freePes(32, 0, 0, 32));
        assertEquals(32L, PlacementLedger.freePes(32, 0, 0, 0));
        assertEquals(0L, PlacementLedger.freePes(32, 32, 0, 0));
        assertEquals(0L, PlacementLedger.freePes(32, 16, 16, 16));      // never negative
    }

    @Test
    void idleVmWinsOverBusyOrInFlightVm() {
        // VM 4 looks free to the old count but is in flight; VM 5 is idle
        Map<Long, Long> m = free(0, 0, 4, PlacementLedger.freePes(32, 0, 0, 32), 5, 32);
        assertEquals(5L, PlacementLedger.selectMostFreeFitting(m, 32));
    }

    @Test
    void sameStepDispatchesDoNotStackOnOneVm() {
        // the caller decrements the map after each assignment; the next pick moves on
        Map<Long, Long> m = free(0, 32, 1, 32, 2, 32);
        long first = PlacementLedger.selectMostFreeFitting(m, 32);
        m.put(first, m.get(first) - 32);
        long second = PlacementLedger.selectMostFreeFitting(m, 32);
        m.put(second, m.get(second) - 32);
        long third = PlacementLedger.selectMostFreeFitting(m, 32);
        assertEquals(0L, first);
        assertEquals(1L, second);
        assertEquals(2L, third);
    }

    @Test
    void noFittingVmLeavesTheCloudletQueued() {
        assertEquals(-1L, PlacementLedger.selectMostFreeFitting(free(0, 8, 1, 16), 32));
        assertEquals(-1L, PlacementLedger.selectMostFreeFitting(free(), 1));
    }

    @Test
    void tiesGoToTheLowestVmIdDeterministically() {
        assertEquals(3L, PlacementLedger.selectMostFreeFitting(free(7, 32, 3, 32, 9, 32), 8));
        assertEquals(9L, PlacementLedger.selectMostFreeFitting(free(7, 16, 3, 24, 9, 32), 8));
    }
}
