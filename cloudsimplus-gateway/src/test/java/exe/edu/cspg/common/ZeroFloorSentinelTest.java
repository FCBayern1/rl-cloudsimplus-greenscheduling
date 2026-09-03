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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Scheme 2-HZ power semantics, pinned on a real simulation (Codex 2026-09-03, item 6).
 *
 * The HZ fleet exposes a SPEC_ASUS_RS500A_DYN host as 32-PE VMs whose PEs run at the
 * experiment's vm_pe_mips of 40000 against 50000-MIPS host cores. One 32-PE job at CPU
 * utilisation 1.0 therefore drives the host to 32*40000 / (64*50000) = 0.4 utilisation and
 * the registered curve gives 1.0 + 0.4 * (162.6 - 1.0) = 65.64 W. Two jobs on the two VMs
 * give 0.8 and 130.28 W. Idle is the 1 W technical floor. These three numbers are the only
 * "job power" the HZ prereg, the toy model (P_DYN_W) and the reports may quote.
 */
public class ZeroFloorSentinelTest {

    private static final double EPS = 1e-9;
    private static final int HOST_PES = 64;
    private static final int VM_PES = 32;
    private static final int VMS = 2;
    private static final long VM_PE_MIPS = 40_000L;   // experiment vm_pe_mips
    private static final double FLOOR_W = 1.0;
    private static final double SPAN_W = 214.0 - 51.4 - FLOOR_W;   // 161.6
    private static final double ONE_JOB_W = FLOOR_W + 0.4 * SPAN_W;  // 65.64
    private static final double TWO_JOBS_W = FLOOR_W + 0.8 * SPAN_W; // 130.28
    private static final double PLATEAU_FROM = 20.0;
    private static final double PLATEAU_TO = 60.0;
    private static final long CLOUDLET_MI = VM_PE_MIPS * 200L;

    private record Phase(int jobs, double meanPower, double meanUtilisation, double spread, int samples) {
    }

    private static Host dynHost() {
        DatacenterConfig cfg = DatacenterConfig.builder()
                .datacenterId(0).datacenterName("hz-sentinel")
                .hostProfiles(Map.of("SPEC_ASUS_RS500A_DYN", 1))
                .build();
        List<Host> hosts = DatacenterSetup.createHostsForDatacenter(cfg);
        assertEquals(1, hosts.size());
        return hosts.get(0);
    }

    private Phase runPhase(int jobs) {
        CloudSimPlus sim = new CloudSimPlus(0.1);
        Host host = dynHost();
        Datacenter dc = new DatacenterSimple(sim, List.of(host), new VmAllocationPolicySimple());
        dc.setSchedulingInterval(1.0);
        DatacenterBroker broker = new DatacenterBrokerSimple(sim);
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < VMS; i++) {
            vms.add(new VmSimple(VM_PE_MIPS, VM_PES).setCloudletScheduler(new CloudletSchedulerTimeShared()));
        }
        broker.submitVmList(vms);
        List<Cloudlet> list = new ArrayList<>();
        for (int i = 0; i < jobs; i++) {
            Cloudlet c = new CloudletDescriptor(i, 0L, CLOUDLET_MI, VM_PES).toCloudlet(1.0);
            c.setVm(vms.get(i % VMS));
            list.add(c);
        }
        if (!list.isEmpty()) {
            broker.submitCloudletList(list);
        }
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
            if (now > PLATEAU_FROM && now <= PLATEAU_TO) {
                windowEnergy[0] += host.getPowerModel().getPower(u) * dt;
                plateau.add(u);
            }
            last[0] = now;
        });
        sim.terminateAt(PLATEAU_TO + 10.0);
        sim.start();
        double meanU = plateau.stream().mapToDouble(Double::doubleValue).average().orElse(-1);
        double spread = plateau.isEmpty() ? Double.NaN
                : plateau.stream().mapToDouble(Double::doubleValue).max().getAsDouble()
                  - plateau.stream().mapToDouble(Double::doubleValue).min().getAsDouble();
        assertEquals(HOST_PES, vms.stream().mapToLong(Vm::getPesNumber).sum());
        return new Phase(jobs, windowEnergy[0] / (PLATEAU_TO - PLATEAU_FROM), meanU, spread, plateau.size());
    }

    @Test
    public void oneThirtyTwoPeJobAtVmMips40000Draws65_64W() {
        Phase idle = runPhase(0);
        Phase one = runPhase(1);
        Phase two = runPhase(2);
        for (Phase p : List.of(idle, one, two)) {
            assertTrue(p.samples() > 10, "too few plateau samples for " + p.jobs() + " jobs");
            assertTrue(p.spread() < 1e-6, "utilisation must be flat, spread " + p.spread());
        }
        assertEquals(0.0, idle.meanUtilisation(), EPS);
        assertEquals(0.4, one.meanUtilisation(), EPS, "32 PEs at 40000 of 64 x 50000 MIPS is 0.4");
        assertEquals(0.8, two.meanUtilisation(), EPS);
        assertEquals(FLOOR_W, idle.meanPower(), EPS, "idle zero-floor host draws the 1 W technical floor");
        assertEquals(ONE_JOB_W, one.meanPower(), EPS, "one 32-PE job draws 65.64 W");
        assertEquals(TWO_JOBS_W, two.meanPower(), EPS, "two 32-PE jobs draw 130.28 W");
    }
}
