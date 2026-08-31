package exe.edu.cspg.multidc;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Evaluator-only stable cloudlet identity (Codex, 2026-08-30). A curve planner has to
 * key its reservation ledger on something that survives a cloudlet being deferred out
 * of the first batch and coming back many steps later. Shape alone collides, so the
 * batch carries the simulation id. An empty slot reports -1 rather than 0, because 0
 * is a legitimate cloudlet id and padding must never be planned as work.
 */
public class BatchCloudletIdChannelTest {

    private GlobalObservationState withIds(int batchSize, long[] ids) {
        return new GlobalObservationState(
                new double[2], new double[2], new double[2], new double[2],
                new double[2], new double[2], new double[2], new double[2],
                new int[2], new double[2], new int[2], new double[2],
                0,
                new int[batchSize], new long[batchSize], new double[batchSize],
                new double[batchSize], new int[batchSize], new int[batchSize],
                new int[batchSize], ids,
                0, 0L, new int[3], 0.0, 0, 2, 0.0);
    }

    @Test
    public void emptySlotsReportMinusOneNotZero() {
        GlobalObservationState obs = GlobalObservationState.createEmpty(2, 4);
        assertArrayEquals(new long[]{-1L, -1L, -1L, -1L}, obs.getBatchCloudletIds());
    }

    @Test
    public void idsSurviveTheRoundTripAndAreDefensivelyCopied() {
        long[] ids = {7L, 0L, 41L, -1L};
        GlobalObservationState obs = withIds(4, ids);
        assertArrayEquals(ids, obs.getBatchCloudletIds());

        long[] handedOut = obs.getBatchCloudletIds();
        assertNotSame(ids, handedOut);
        handedOut[0] = 999L;
        assertEquals(7L, obs.getBatchCloudletIds()[0]);

        ids[2] = 123L;
        assertEquals(41L, obs.getBatchCloudletIds()[2]);
    }

    @Test
    public void lengthMustMatchTheRestOfTheBatch() {
        assertThrows(IllegalArgumentException.class, () -> withIds(4, new long[]{1L, 2L}));
    }
}
