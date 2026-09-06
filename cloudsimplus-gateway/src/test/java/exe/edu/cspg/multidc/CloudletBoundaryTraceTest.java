package exe.edu.cspg.multidc;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Settlement diagnostic B (Codex ruling 2026-09-06): source-level trace of the one-second
 * end-boundary effect. Pure CloudSim Plus 8.5.5, no gateway code, stepped exactly like
 * {@code MultiDatacenterSimulationCore.proceedClockTo} (runFor(1 s) loops), one 64-PE host with
 * two 32-PE VMs (space-shared), 1,920,000 MI cloudlets at 40,000 MIPS per PE (48 s).
 *
 * L1: one cloudlet submitted at 20 s.  L3a: a second one submitted at 40 s while the first runs.
 * For every whole second the test records the host CPU utilisation the energy sampler would
 * read, and at the end the cloudlets' exec start, finish and actual CPU time.
 */
class CloudletBoundaryTraceTest {

    private static final long MI = 1_920_000L;
    private static final int PES = 32;
    private static final double VM_MIPS = 40_000.0;
    private static final double HOST_MIPS = 50_000.0;

    private static final class Trace {
        final List<Double> utilAtSecond = new ArrayList<>();   // index = clock second
        final List<Cloudlet> cloudlets = new ArrayList<>();
    }

    private static Trace run(double... submitTimes) {
        return runWith(0.001, submitTimes);
    }

    private static Trace runWith(double minTimeBetweenEvents, double... submitTimes) {
        return runCase(minTimeBetweenEvents, 1.0, submitTimes);
    }

    private static Trace runCase(double minTimeBetweenEvents, double schedulingInterval, double... submitTimes) {
        CloudSimPlus sim = new CloudSimPlus(minTimeBetweenEvents);
        List<Pe> pes = new ArrayList<>();
        for (int i = 0; i < 64; i++) pes.add(new PeSimple(HOST_MIPS));
        Host host = new HostSimple(262_144, 25_000, 1_000_000, pes);
        List<Host> hosts = List.of(host);
        DatacenterSimple dc = new DatacenterSimple(sim, hosts);
        dc.setSchedulingInterval(schedulingInterval);
        DatacenterBrokerSimple broker = new DatacenterBrokerSimple(sim);
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            Vm vm = new VmSimple(VM_MIPS, PES).setRam(8192).setBw(1000).setSize(4000);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
            vms.add(vm);
        }
        broker.submitVmList(vms);
        Trace tr = new Trace();
        List<Cloudlet> cls = new ArrayList<>();
        for (int i = 0; i < submitTimes.length; i++) {
            Cloudlet c = new CloudletSimple(MI, PES).setUtilizationModel(new UtilizationModelFull());
            c.setSubmissionDelay(submitTimes[i]);
            c.setVm(vms.get(i));
            cls.add(c);
        }
        broker.submitCloudletList(cls);
        tr.cloudlets.addAll(cls);

        sim.startSync();
        double clock = 0.0;
        for (int second = 0; second < 200 && sim.isRunning(); second++) {
            // proceedClockTo(clock + 1) as the gateway does it
            double target = clock + 1.0;
            double interval = target - clock;
            int guard = 0;
            while (sim.runFor(interval) < target) {
                clock = sim.clock();
                interval = target - clock;
                if (interval <= 0) interval = 0.001;
                if (++guard > 1000) break;
            }
            clock = sim.clock();
            // the energy sampler reads this right after the clock advance
            tr.utilAtSecond.add(host.getCpuPercentUtilization());
        }
        return tr;
    }

    private static int busySeconds(Trace tr, double from, double to) {
        int n = 0;
        for (int s = 0; s < tr.utilAtSecond.size(); s++) {
            if (s + 1 >= from && s + 1 <= to && tr.utilAtSecond.get(s) > 1e-9) n++;
        }
        return n;
    }

    private static String describe(Trace tr) {
        StringBuilder sb = new StringBuilder();
        for (Cloudlet c : tr.cloudlets) {
            sb.append(String.format("cloudlet %d: execStart=%.6f finish=%.6f actualCpu=%.6f status=%s%n",
                    c.getId(), c.getStartTime(), c.getFinishTime(), c.getTotalExecutionTime(), c.getStatus()));
        }
        for (int s = 0; s < tr.utilAtSecond.size(); s++) {
            double u = tr.utilAtSecond.get(s);
            if (u > 1e-9) sb.append(String.format("t=%d util=%.4f%n", s + 1, u));
        }
        return sb.toString();
    }

    /**
     * Recorded facts (CloudSim Plus 8.5.5, this configuration): a cloudlet submitted at T starts
     * at T + minTimeBetweenEvents (0.001) and its finish time is 48.01 s after its exec start,
     * not 48.0: the span carries a +0.01 s tail from the scheduler's processing updates. A finish
     * at 68.011 is therefore processed AFTER the whole-second sample at 68, so the host still
     * reports utilisation at that sample: 48 busy samples (21..68) for a start at 20.001.
     */
    @Test
    void singleCloudletSpanIs48PlusATailAndSampledAt48WholeSeconds() {
        Trace tr = run(20.0);
        System.out.println("== L1\n" + describe(tr));
        Cloudlet c = tr.cloudlets.get(0);
        assertEquals(20.001, c.getStartTime(), 1e-9, "exec start = submission + minTimeBetweenEvents");
        double span = c.getFinishTime() - c.getStartTime();
        assertEquals(48.01, span, 1e-3, "execution span carries a +0.01 s tail");
        assertEquals(48, busySeconds(tr, 0, 200), "seconds with non-zero host utilisation");
        assertEquals(0.0, tr.utilAtSecond.get(19), 1e-12, "sample at 20 (before the start) is idle");
        assertEquals(0.4, tr.utilAtSecond.get(67), 1e-12, "sample at 68 still sees the cloudlet (finish 68.011)");
        assertEquals(0.0, tr.utilAtSecond.get(68), 1e-12, "sample at 69 is idle");
    }

    /**
     * The experiment cell sets min_time_between_events = 1.0. Records what CloudSim does then:
     * a lone cloudlet vs one submitted while another is being processed on the same datacenter.
     */
    @Test
    void withOneSecondMinTimeBetweenEventsTraceLoneAndMidRunCloudlets() {
        Trace lone = runWith(1.0, 20.0);
        Trace pair = runWith(1.0, 20.0, 40.0);
        System.out.println("== L1 (minTimeBetweenEvents=1.0)\n" + describe(lone));
        System.out.println("== L3a (minTimeBetweenEvents=1.0)\n" + describe(pair));
        Cloudlet a = pair.cloudlets.get(0), b = pair.cloudlets.get(1), c = lone.cloudlets.get(0);
        System.out.printf("lone span=%.6f  pair spanA=%.6f spanB=%.6f  busy lone=%d pair=%d%n",
                c.getFinishTime() - c.getStartTime(), a.getFinishTime() - a.getStartTime(),
                b.getFinishTime() - b.getStartTime(), busySeconds(lone, 0, 200), busySeconds(pair, 0, 200));
    }

    /**
     * The gateway's finding reproduced at the CloudSim level: with min_time_between_events 1.0
     * and NO scheduling interval (CloudSim's default 0: updates only at estimated finishes) a
     * cloudlet submitted while another is processed on the same datacenter is finished one
     * second late (49 s), the lone one exactly (48 s); with the scheduling interval set to the
     * 1 s step both are exact. This is the certification twin's fix (diagnostic B).
     */
    @Test
    void schedulingIntervalAlignedToTheStepRemovesTheExtraSecond() {
        Trace noInterval = runCase(1.0, 0.0, 20.0, 40.0);
        Trace aligned = runCase(1.0, 1.0, 20.0, 40.0);
        System.out.println("== L3a (mte 1.0, interval 0)\n" + describe(noInterval));
        System.out.println("== L3a (mte 1.0, interval 1)\n" + describe(aligned));
        Cloudlet a0 = noInterval.cloudlets.get(0), b0 = noInterval.cloudlets.get(1);
        Cloudlet a1 = aligned.cloudlets.get(0), b1 = aligned.cloudlets.get(1);
        System.out.printf("interval 0: spanA=%.3f spanB=%.3f | interval 1: spanA=%.3f spanB=%.3f%n",
                a0.getFinishTime() - a0.getStartTime(), b0.getFinishTime() - b0.getStartTime(),
                a1.getFinishTime() - a1.getStartTime(), b1.getFinishTime() - b1.getStartTime());
        assertEquals(48.0, a1.getFinishTime() - a1.getStartTime(), 1e-9, "aligned: lone span");
        assertEquals(48.0, b1.getFinishTime() - b1.getStartTime(), 1e-9, "aligned: mid-run span");
        assertEquals(68, busySeconds(aligned, 0, 200));
    }

    @Test
    void secondCloudletSubmittedMidRunHasTheSameSpanAndTail() {
        Trace tr = run(20.0, 40.0);
        System.out.println("== L3a\n" + describe(tr));
        Cloudlet a = tr.cloudlets.get(0), b = tr.cloudlets.get(1);
        double spanA = a.getFinishTime() - a.getStartTime();
        double spanB = b.getFinishTime() - b.getStartTime();
        System.out.printf("spanA=%.6f spanB=%.6f%n", spanA, spanB);
        assertEquals(48.01, spanA, 2e-3);
        assertEquals(48.01, spanB, 2e-3);
        // host-busy whole seconds: 21..88 = 68 (both cloudlets, overlapping 41..68)
        assertEquals(68, busySeconds(tr, 0, 200));
        assertEquals(0.8, tr.utilAtSecond.get(40), 1e-12, "sample at 41 sees both");
        assertEquals(0.4, tr.utilAtSecond.get(87), 1e-12, "sample at 88 still sees cloudlet b (finish 88.012)");
    }
}
