package exe.edu.cspg.multidc;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GlobalBrokerDeferredLifecycleTest {

    private static Cloudlet cloudlet(long id, long mi) {
        return new CloudletSimple(id, mi, 2);
    }

    private static GlobalBroker broker(List<Cloudlet> cloudlets) {
        return new GlobalBroker(new CloudSimPlus(), cloudlets, List.of());
    }

    @Test
    void repeatedExplicitDeferCountsOneCloudletAndEveryAction() {
        Cloudlet job = cloudlet(7, 12_345);
        GlobalBroker broker = broker(List.of(job));

        broker.deferCloudletToTail(job);
        broker.getBatchForRouting(1); // mirror the next decision's dequeue
        broker.deferCloudletToTail(job);

        assertEquals(1, broker.getGlobalDeferredCount());
        assertEquals(12_345L, broker.getGlobalDeferredMi());
        assertEquals(2, broker.getCloudletDeferCount(job));
        assertTrue(broker.isCloudletDeferred(job));
    }

    @Test
    void successfulRouteLifecycleHookRemovesDeferredAggregates() {
        Cloudlet job = cloudlet(8, 4_000);
        GlobalBroker broker = broker(List.of(job));
        broker.deferCloudletToTail(job);

        broker.clearDeferredLifecycle(job);

        assertEquals(0, broker.getGlobalDeferredCount());
        assertEquals(0L, broker.getGlobalDeferredMi());
        assertEquals(0, broker.getCloudletDeferCount(job));
        assertFalse(broker.isCloudletDeferred(job));
    }

    @Test
    void genericRoutingFailureRequeueDoesNotMarkDeferred() {
        Cloudlet job = cloudlet(9, 5_000);
        GlobalBroker broker = broker(List.of(job));

        broker.requeueCloudletToTail(job);

        assertEquals(0, broker.getGlobalDeferredCount());
        assertEquals(0, broker.getCloudletDeferCount(job));
        assertFalse(broker.isCloudletDeferred(job));
    }

    @Test
    void resetClearsLedgerAndMiAccountingSumsDistinctJobs() {
        Cloudlet first = cloudlet(10, 1_500);
        Cloudlet second = cloudlet(11, 2_500);
        GlobalBroker broker = broker(List.of(first, second));
        broker.deferCloudletToTail(first);
        broker.deferCloudletToTail(second);

        assertEquals(2, broker.getGlobalDeferredCount());
        assertEquals(4_000L, broker.getGlobalDeferredMi());

        broker.resetStatistics();

        assertEquals(0, broker.getGlobalDeferredCount());
        assertEquals(0L, broker.getGlobalDeferredMi());
        assertEquals(0, broker.getGlobalWaitingCloudletsCount());
    }
}
