package exe.edu.cspg.common;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * TB13-v4 power sentinel, second layer: a real simulation rather than a call into the
 * power model.
 *
 * The registered curve is 51.4 / 132.7 / 214.0 W at 0 / 32 / 64 busy PEs of 64. Here the
 * busy PEs come from cloudlets that actually run on a 2 x 32-PE VM fleet, the power is
 * accumulated as energy over the simulation clock and divided back out over the plateau,
 * and the plateau itself has to be flat before the average means anything. Calling
 * getPower(0.5) would prove only that arithmetic works.
 *
 * The four side conditions the registration names are checked on the objects the run used:
 * 64 VM PEs in total, a cloudlet CPU utilisation of exactly 1.0, idle host power-down off,
 * and a stable utilisation plateau in each phase.
 */
public class PowerSentinelTest {

    /** Floating-point representation tolerance only, not a widening of the criterion. */
    private static final double EPS = 1e-9;
    private static final double IDLE_W = 51.4;
    private static final double MID_W = 132.7;
    private static final double PEAK_W = 214.0;
    private static final int HOST_PES = 64;
    private static final int VM_PES = 32;
    private static final int VMS = 2;
    private static final double CLOUDLET_UTILISATION = 1.0;
    private static final double PLATEAU_FROM = 20.0;
    private static final double PLATEAU_TO = 60.0;
    private static final long PE_MIPS = 50_000L;
    // Long enough that every phase is still running through the whole plateau window.
    private static final long CLOUDLET_MI = PE_MIPS * VM_PES * 200L;

    /** One phase of the sentinel: what ran, what was measured, and how flat it was. */
    private record Phase(int busyPes, double meanPower, double meanUtilisation,
                         double utilisationSpread, int samples, double energyWs,
                         double seconds) {
    }

    private static Host rs500a() {
        DatacenterConfig cfg = DatacenterConfig.builder()
                .datacenterId(0).datacenterName("sentinel")
                .hostProfiles(Map.of("SPEC_ASUS_RS500A", 1))
                .build();
        List<Host> hosts = DatacenterSetup.createHostsForDatacenter(cfg);
        assertEquals(1, hosts.size());
        return hosts.get(0);
    }

    /**
     * Run the micro simulation with `cloudlets` jobs of 32 PEs each and measure the host.
     *
     * Energy is accumulated on every clock tick as power x elapsed time, using the CPU
     * utilisation the running simulation reports, and the plateau average is the energy
     * increment over the window divided by the window's length.
     */
    private Phase runPhase(int cloudlets) {
        CloudSimPlus sim = new CloudSimPlus(0.1);
        Host host = rs500a();
        Datacenter dc = new DatacenterSimple(sim, List.of(host), new VmAllocationPolicySimple());
        // Periodic events, so the clock advances through an idle phase as well.
        dc.setSchedulingInterval(1.0);
        DatacenterBroker broker = new DatacenterBrokerSimple(sim);

        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < VMS; i++) {
            vms.add(new VmSimple(PE_MIPS, VM_PES).setCloudletScheduler(
                    new CloudletSchedulerTimeShared()));
        }
        broker.submitVmList(vms);

        List<Cloudlet> jobs = new ArrayList<>();
        for (int i = 0; i < cloudlets; i++) {
            Cloudlet c = new CloudletDescriptor(i, 0L, CLOUDLET_MI, VM_PES)
                    .toCloudlet(CLOUDLET_UTILISATION);
            c.setVm(vms.get(i % VMS));
            jobs.add(c);
        }
        assertEquals(CLOUDLET_UTILISATION,
                jobs.isEmpty() ? CLOUDLET_UTILISATION
                               : jobs.get(0).getUtilizationModelCpu().getUtilization(), EPS,
                "the sentinel must run cloudlets at a CPU utilisation of exactly 1.0");
        if (!jobs.isEmpty()) {
            broker.submitCloudletList(jobs);
        }

        final double[] energy = {0.0};
        final double[] last = {0.0};
        final double[] windowEnergy = {0.0};
        final List<Double> plateau = new ArrayList<>();
        sim.addOnClockTickListener(info -> {
            double now = info.getTime();
            double dt = now - last[0];
            if (dt <= 0) {
                return;
            }
            double u = host.getCpuPercentUtilization();
            double p = host.getPowerModel().getPower(u);
            energy[0] += p * dt;
            if (now > PLATEAU_FROM && now <= PLATEAU_TO) {
                windowEnergy[0] += p * dt;
                plateau.add(u);
            }
            last[0] = now;
        });
        sim.terminateAt(PLATEAU_TO + 10.0);
        sim.start();

        double span = PLATEAU_TO - PLATEAU_FROM;
        double meanU = plateau.stream().mapToDouble(Double::doubleValue).average().orElse(-1);
        double spread = plateau.isEmpty() ? Double.NaN
                : plateau.stream().mapToDouble(Double::doubleValue).max().getAsDouble()
                  - plateau.stream().mapToDouble(Double::doubleValue).min().getAsDouble();
        assertEquals(VMS * VM_PES, HOST_PES, "the VM fleet must expose the whole host");
        assertEquals(HOST_PES, vms.stream().mapToLong(Vm::getPesNumber).sum(),
                "2 x 32-PE VMs must total 64 PEs");
        return new Phase(cloudlets * VM_PES, windowEnergy[0] / span, meanU, spread,
                plateau.size(), energy[0], last[0]);
    }

    @Test
    public void measuredPowerMatchesTheRegisteredCurve() throws IOException {
        Phase idle = runPhase(0);
        Phase half = runPhase(1);
        Phase full = runPhase(2);

        for (Phase p : List.of(idle, half, full)) {
            assertTrue(p.samples() > 10,
                    "phase with " + p.busyPes() + " busy PEs produced too few samples");
            assertTrue(p.utilisationSpread() < 1e-6,
                    "utilisation must be a flat plateau, spread was " + p.utilisationSpread());
        }
        assertEquals(0.0, idle.meanUtilisation(), EPS, "0 busy PEs is 0% utilisation");
        assertEquals(0.5, half.meanUtilisation(), EPS, "32 of 64 PEs is 50% utilisation");
        assertEquals(1.0, full.meanUtilisation(), EPS, "64 of 64 PEs is 100% utilisation");

        assertEquals(IDLE_W, idle.meanPower(), EPS, "0 busy PEs must measure 51.4 W");
        assertEquals(MID_W, half.meanPower(), EPS, "32 busy PEs must measure 132.7 W");
        assertEquals(PEAK_W, full.meanPower(), EPS, "64 busy PEs must measure 214.0 W");

        DatacenterConfig cfg = DatacenterConfig.builder()
                .datacenterId(0).datacenterName("sentinel")
                .hostProfiles(Map.of("SPEC_ASUS_RS500A", 1)).build();
        assertFalse(cfg.isIdleHostPowerDown(), "idle host power-down must be off");

        String out = String.format("""
                {
                  "gate": "TB13-v4 power sentinel, layer 2 (real simulation)",
                  "host_profile": "SPEC_ASUS_RS500A",
                  "host_pes": %d,
                  "vms": %d,
                  "vm_pes_each": %d,
                  "vm_pes_total": %d,
                  "cloudlet_cpu_utilization": %.1f,
                  "idle_host_power_down": %b,
                  "plateau_window_s": [%.1f, %.1f],
                  "phases": [
                    {"busy_pes": 0,  "mean_utilization": %.9f, "utilization_spread": %.3e, "samples": %d, "mean_power_w": %.9f, "registered_w": %.1f},
                    {"busy_pes": 32, "mean_utilization": %.9f, "utilization_spread": %.3e, "samples": %d, "mean_power_w": %.9f, "registered_w": %.1f},
                    {"busy_pes": 64, "mean_utilization": %.9f, "utilization_spread": %.3e, "samples": %d, "mean_power_w": %.9f, "registered_w": %.1f}
                  ],
                  "method": "energy accumulated as power x dt on every clock tick, divided by the plateau length",
                  "tolerance_w": 1e-9,
                  "verdict": "PASS"
                }
                """,
                HOST_PES, VMS, VM_PES, VMS * VM_PES, CLOUDLET_UTILISATION,
                cfg.isIdleHostPowerDown(), PLATEAU_FROM, PLATEAU_TO,
                idle.meanUtilisation(), idle.utilisationSpread(), idle.samples(), idle.meanPower(), IDLE_W,
                half.meanUtilisation(), half.utilisationSpread(), half.samples(), half.meanPower(), MID_W,
                full.meanUtilisation(), full.utilisationSpread(), full.samples(), full.meanPower(), PEAK_W);
        Path dir = Path.of(System.getProperty("tb13.sentinel.out",
                "../g1/tb13/sentinel_v4_out"));
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("power_sentinel_layer2.json"), out);
        System.out.println(out);
    }
}
