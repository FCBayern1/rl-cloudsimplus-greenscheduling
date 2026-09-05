package exe.edu.cspg.multidc;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Stage D' mask-margin probe: route -> exec-start delay statistics (export-only). */
class RouteToStartDelayTest {

    @Test
    void maxAndNearestRankP95OverMatchedCloudlets() {
        Map<Long, Double> routed = new HashMap<>();
        Map<Long, Double> start = new HashMap<>();
        for (long id = 1; id <= 20; id++) {
            routed.put(id, 100.0);
            start.put(id, 100.0 + id);          // delays 1..20
        }
        routed.put(99L, 5.0);                    // routed but never started: ignored
        start.put(77L, 1.0);                     // started but never routed: ignored
        double[] d = PerActionRewardMath.routeToStartDelays(routed, start);
        assertEquals(20.0, d[0], 1e-9);
        assertEquals(19.0, d[1], 1e-9);          // nearest rank: ceil(0.95*20)=19 -> 19th smallest
        assertEquals(20.0, d[2], 1e-9);
    }

    @Test
    void emptyAndNegativeDelaysAreSafe() {
        double[] none = PerActionRewardMath.routeToStartDelays(new HashMap<>(), new HashMap<>());
        assertEquals(0.0, none[0], 1e-9);
        assertEquals(0.0, none[2], 1e-9);
        Map<Long, Double> routed = new HashMap<>();
        Map<Long, Double> start = new HashMap<>();
        routed.put(1L, 50.0);
        start.put(1L, 40.0);                     // impossible ordering clamps to 0
        double[] d = PerActionRewardMath.routeToStartDelays(routed, start);
        assertEquals(0.0, d[0], 1e-9);
        assertEquals(1.0, d[2], 1e-9);
    }
}
