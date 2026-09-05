package exe.edu.cspg.multidc;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Option hold ledger (reports/OPTION_ACTION_DESIGN.md §2.1, Addendum B): a held cloudlet
 * leaves the routing queue for good, counts in the deferred aggregates, remembers its
 * committed datacenter, and comes back only through takeHeld().
 */
class GlobalBrokerHoldLedgerTest {

    private static Cloudlet cloudlet(long id, long mi) {
        return new CloudletSimple(id, mi, 2);
    }

    private static GlobalBroker broker(List<Cloudlet> cloudlets) {
        return new GlobalBroker(new CloudSimPlus(), cloudlets, List.of());
    }

    @Test
    void heldCloudletIsNeverBatchedAgainAndCountsAsDeferredBacklog() {
        Cloudlet a = cloudlet(1, 10_000);
        Cloudlet b = cloudlet(2, 20_000);
        GlobalBroker broker = broker(List.of(a, b));
        broker.requeueCloudletToTail(a);          // both in the routing queue
        broker.requeueCloudletToTail(b);
        List<Cloudlet> batch = broker.getBatchForRouting(2);
        assertEquals(2, batch.size());

        broker.holdCloudlet(a, 3);                 // a leaves for the ledger, b is deferred to tail
        broker.deferCloudletToTail(b);

        assertEquals(List.of(b), broker.getBatchForRouting(4));   // a is not re-presented
        assertEquals(1, broker.getHeldCount());
        assertEquals(3, broker.getHeldDc(1));
        assertEquals(2, broker.getGlobalDeferredCount());          // held + deferred
        assertEquals(30_000L, broker.getGlobalDeferredMi());
        assertTrue(broker.isCloudletDeferred(a));                  // first-defer charge semantics
        assertEquals(1, broker.getCloudletDeferCount(a));
    }

    @Test
    void takeHeldReturnsTheCloudletOnceAndClearsTheCommitment() {
        Cloudlet a = cloudlet(5, 1_000);
        GlobalBroker broker = broker(List.of(a));
        broker.holdCloudlet(a, 0);

        assertSame(a, broker.takeHeld(5));
        assertNull(broker.takeHeld(5));            // a second take finds nothing
        assertEquals(0, broker.getHeldCount());
        assertEquals(-1, broker.getHeldDc(5));
        assertTrue(broker.getHeldIds().isEmpty());
        // the deferred aggregates are cleared by the route path's lifecycle hook, as for DEFER
        broker.clearDeferredLifecycle(a);
        assertFalse(broker.isCloudletDeferred(a));
        assertEquals(0, broker.getGlobalDeferredCount());
    }

    @Test
    void unknownIdIsNotHeld() {
        GlobalBroker broker = broker(List.of());
        assertNull(broker.takeHeld(42));
        assertEquals(-1, broker.getHeldDc(42));
    }
}
