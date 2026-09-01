package exe.edu.cspg.common;

import org.cloudsimplus.hosts.Host;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The host power curve a profile actually gets, measured through the same construction
 * path the simulator uses.
 *
 * CloudSim Plus 8.5.5 takes the second argument of PowerModelHostSimple as static power in
 * WATTS: the constructor rejects a value above maxPower with "maxPower has to be higher
 * than staticPower", and getStaticPower returns it unchanged. HostProfile stores idle as a
 * PERCENTAGE and offers getIdlePowerW() for the watt value, so passing the percentage
 * straight through gives an RS500A a 24 W idle floor instead of its SPEC 51.4 W.
 *
 * TB13-v4 registers the RS500A curve as 51.4 / 132.7 / 214.0 W at 0 / 32 / 64 busy PEs of
 * 64. The profile stores idle as 24.0% of 214 W, which is 51.36 W, so the wired curve is
 * 51.36 / 132.68 / 214.00. These tests assert the profile-derived curve; whether the
 * registered values should be met exactly is a question for the sentinel, not for here.
 */
public class HostPowerWiringTest {

    private static final double EPS = 1e-6;
    private static final double IDLE_W = 51.4;
    private static final double PEAK_W = 214.0;
    private static final int PES = 64;

    private static Host rs500a() {
        DatacenterConfig cfg = DatacenterConfig.builder()
                .datacenterId(0)
                .datacenterName("sentinel")
                .hostProfiles(Map.of("SPEC_ASUS_RS500A", 1))
                .build();
        List<Host> hosts = DatacenterSetup.createHostsForDatacenter(cfg);
        assertEquals(1, hosts.size(), "the profile must yield exactly one host");
        return hosts.get(0);
    }

    @Test
    public void profileHostExposesSixtyFourPes() {
        assertEquals(PES, rs500a().getPeList().size(), "RS500A is a 64-core server");
    }

    @Test
    public void idlePowerIsTheProfileIdleNotThePercentage() {
        Host h = rs500a();
        double idle = HostProfile.SPEC_ASUS_RS500A().getIdlePowerW();
        assertEquals(51.36, idle, EPS, "24.0% of 214 W is 51.36 W");
        assertEquals(idle, h.getPowerModel().getPower(0.0), EPS,
                "an idle RS500A draws its idle watts, not its idle percentage");
        // The wiring defect this test was written for: the percentage went through as
        // watts, so the floor was 24 W.
        assertTrue(h.getPowerModel().getPower(0.0) > 50.0,
                "a 24 W floor means the percentage reached the model as watts");
    }

    @Test
    public void halfTheCoresBusyDrawsTheMidpointOfTheProfileCurve() {
        Host h = rs500a();
        double idle = HostProfile.SPEC_ASUS_RS500A().getIdlePowerW();
        double expected = idle + 32 * (PEAK_W - idle) / PES;
        assertEquals(132.68, expected, EPS, "the midpoint follows from the profile");
        assertEquals(expected, h.getPowerModel().getPower(32.0 / PES), EPS,
                "32 of 64 busy PEs draw the midpoint of the profile curve");
    }

    @Test
    public void allCoresBusyDrawsPeak() {
        assertEquals(PEAK_W, rs500a().getPowerModel().getPower(1.0), EPS,
                "64 of 64 busy PEs draw the SPEC peak");
    }

    @Test
    public void everyProfileIdlesAtItsOwnSpecIdle() {
        for (String name : List.of("SPEC_ASUS_RS500A", "SPEC_ASUS_RS700A")) {
            HostProfile p = HostProfile.fromName(name);
            DatacenterConfig cfg = DatacenterConfig.builder()
                    .datacenterId(0).datacenterName("sentinel")
                    .hostProfiles(Map.of(name, 1)).build();
            Host h = DatacenterSetup.createHostsForDatacenter(cfg).get(0);
            assertTrue(p.getIdlePowerW() > 0, "the profile must state an idle power");
            assertEquals(p.getIdlePowerW(), h.getPowerModel().getPower(0.0), EPS,
                    name + " must idle at its own SPEC idle power");
        }
    }
}
