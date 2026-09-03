package exe.edu.cspg.common;

import org.cloudsimplus.hosts.Host;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Zero-floor host twins: the dynamic span of the SPEC curve with no static floor, so the
 * power a host draws is exactly the sum of its jobs' dynamic power. This is the physics
 * the simulator-free lever model (g1/compressed_timecap_s2/toy_lever.py) assumes, and the
 * alignment between the two is what the Level-1 spiral checks.
 */
public class ZeroFloorProfileTest {

    private static final double EPS = 1e-9;

    private static Host build(String profile) {
        DatacenterConfig cfg = DatacenterConfig.builder()
                .datacenterId(0).datacenterName("zero-floor")
                .hostProfiles(Map.of(profile, 1))
                .build();
        List<Host> hosts = DatacenterSetup.createHostsForDatacenter(cfg);
        assertEquals(1, hosts.size());
        return hosts.get(0);
    }

    @Test
    public void rs500aDynIsTheDynamicSpanWithNoFloor() {
        HostProfile p = HostProfile.fromName("SPEC_ASUS_RS500A_DYN");
        assertEquals(64, p.getPes());
        assertEquals(1.0, p.getIdlePowerW(), EPS, "1 W technical floor (CloudSim Plus minimum)");
        assertEquals(162.6, p.getMaxPowerW(), EPS, "214 - 51.4 W at full load");

        Host h = build("SPEC_ASUS_RS500A_DYN");
        assertEquals(1.0, h.getPowerModel().getPower(0.0), EPS, "idle host draws the 1 W floor");
        assertEquals(1.0 + 0.5 * 161.6, h.getPowerModel().getPower(0.5), EPS, "32 of 64 PEs: 81.8 W");
        assertEquals(162.6, h.getPowerModel().getPower(1.0), EPS, "64 PEs draw 162.6 W");
    }

    @Test
    public void rs700aDynMatchesThePerCoreDrawOfRs500aDyn() {
        HostProfile p = HostProfile.fromName("SPEC_ASUS_RS700A_DYN");
        assertEquals(128, p.getPes());
        assertEquals(1.0, p.getIdlePowerW(), EPS);
        assertEquals(324.0, p.getMaxPowerW(), EPS, "430 - 106 W at full load");
        Host h = build("SPEC_ASUS_RS700A_DYN");
        // 32 of 128 PEs is a quarter of the span above the floor: 1 + 323/4 = 81.75 W
        assertEquals(1.0 + 0.25 * 323.0, h.getPowerModel().getPower(0.25), EPS);
    }

    @Test
    public void legacyProfilesAreUntouched() {
        HostProfile p = HostProfile.fromName("SPEC_ASUS_RS500A");
        assertEquals(51.4, p.getIdlePowerW(), EPS);
        assertEquals(214.0, p.getMaxPowerW(), EPS);
    }
}
