package exe.edu.cspg.common;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * SQT2.2-Clean physics alignment (Codex ruling, 2026-08-18).
 *
 * Every SQT2 certification quantity (deadline slack, B_eff, latest-start
 * backstop, capacity audit, oracle labels) is derived from
 * runtime = MI / (PES * vmPeMips), but cloudlets historically executed at
 * 50% CPU utilization - a measured ~2.5x runtime stretch that silently
 * invalidated the whole chain. The fix is CONFIGURATION-SCOPED, never a
 * global default change:
 *
 *   cloudlet_cpu_utilization  default 0.5  (legacy/v3 byte-level semantics)
 *   SQT2 / SQT2HO experiments set 1.0      (physics obeys registered maths)
 *
 * These tests lock both halves of that contract.
 */
public class CloudletCpuUtilizationTest {

    private Map<String, Object> minimalParams() {
        Map<String, Object> p = new HashMap<>();
        p.put("simulation_timestep", 1.0);
        p.put("min_time_between_events", 0.1);
        return p;
    }

    @Test
    public void settingsDefaultIsLegacyHalfUtilization() {
        SimulationSettings s = new SimulationSettings(minimalParams());
        assertEquals(0.5, s.getCloudletCpuUtilization(), 1e-12,
                "Default MUST stay 0.5: legacy/v3 reruns keep their physics");
    }

    @Test
    public void settingsExplicitFullUtilizationIsHonoured() {
        Map<String, Object> p = minimalParams();
        p.put("cloudlet_cpu_utilization", 1.0);
        SimulationSettings s = new SimulationSettings(p);
        assertEquals(1.0, s.getCloudletCpuUtilization(), 1e-12);
    }

    @Test
    public void legacyToCloudletStillBuildsHalfUtilization() {
        CloudletDescriptor d = new CloudletDescriptor(1, 0, 4_000_000, 1);
        Cloudlet c = d.toCloudlet();
        assertEquals(0.5, c.getUtilizationModelCpu().getUtilization(0.0), 1e-12,
                "No-arg toCloudlet() MUST keep the 0.5 legacy semantics");
    }

    @Test
    public void parameterisedToCloudletBuildsRequestedUtilization() {
        CloudletDescriptor d = new CloudletDescriptor(1, 0, 4_000_000, 1);
        Cloudlet c = d.toCloudlet(1.0);
        assertEquals(1.0, c.getUtilizationModelCpu().getUtilization(0.0), 1e-12,
                "SQT2 full-utilization cloudlets must request 100% CPU");
    }
}
