package exe.edu.cspg.common;

import java.util.Objects;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.util.DataCloudTags;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

public class CloudletDescriptor {
    private final int cloudletId;
    private final long submissionDelay;
    private final long mi;
    private final int numberOfCores;
    // Deferrable-jobs temporal lever (2026-06-18): how many global steps this
    // cloudlet may be held (deferred) by the router before it MUST be routed.
    // 0 = latency-sensitive / not deferrable (route immediately). >0 = deferrable,
    // may wait up to this many steps so the agent can release it onto a DC when a
    // forecast-predicted green peak arrives. See docs/Deferrable_Jobs_Lever.md.
    private final long deferDeadlineSteps;
    // Absolute COMPLETION deadline in simulation seconds (deferrable-batch carbon
    // lever, 2026-06-20): the job should FINISH by this sim-time. <=0 means no
    // deadline. Used by the deadline-aware Lagrangian SLA: a job whose finish time
    // exceeds this (or that never finishes) is a deadline MISS. Distinct from
    // deferDeadlineSteps (a routing-hold budget). Read from the CSV 'deadline' col.
    private final long deadlineTime;

    /** Full constructor with both the routing-defer budget and the completion deadline. */
    public CloudletDescriptor(int cloudletId, long submissionDelay, long mi, int numberOfCores,
                              long deferDeadlineSteps, long deadlineTime) {
        this.cloudletId = cloudletId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.numberOfCores = numberOfCores;
        this.deferDeadlineSteps = Math.max(0, deferDeadlineSteps);
        this.deadlineTime = deadlineTime;
    }

    /** Constructor with the deferral deadline only (no completion deadline). */
    public CloudletDescriptor(int cloudletId, long submissionDelay, long mi, int numberOfCores,
                              long deferDeadlineSteps) {
        this(cloudletId, submissionDelay, mi, numberOfCores, deferDeadlineSteps, 0);
    }

    /**
     * Backward-compatible constructor: defaults to NON-deferrable
     * (deferDeadlineSteps = 0). Keeps every existing caller working unchanged.
     */
    public CloudletDescriptor(int cloudletId, long submissionDelay, long mi, int numberOfCores) {
        this(cloudletId, submissionDelay, mi, numberOfCores, 0);
    }

    public int getCloudletId() {
        return cloudletId;
    }

    public long getSubmissionDelay() {
        return submissionDelay;
    }

    public long getMi() {
        return mi;
    }

    public int getNumberOfCores() {
        return numberOfCores;
    }

    /** Max global steps this cloudlet may be deferred (0 = not deferrable). */
    public long getDeferDeadlineSteps() {
        return deferDeadlineSteps;
    }

    /** True if the router is allowed to defer (hold) this cloudlet. */
    public boolean isDeferrable() {
        return deferDeadlineSteps > 0;
    }

    /** Absolute completion deadline in sim seconds (&lt;=0 = none). */
    public long getDeadlineTime() {
        return deadlineTime;
    }

    /** True if this cloudlet has a finite completion deadline. */
    public boolean hasDeadline() {
        return deadlineTime > 0;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;
        CloudletDescriptor that = (CloudletDescriptor) o;
        return getCloudletId() == that.getCloudletId() &&
                getSubmissionDelay() == that.getSubmissionDelay() &&
                getMi() == that.getMi() &&
                getNumberOfCores() == that.getNumberOfCores() &&
                getDeferDeadlineSteps() == that.getDeferDeadlineSteps() &&
                getDeadlineTime() == that.getDeadlineTime();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getCloudletId(), getSubmissionDelay(), getMi(), getNumberOfCores(),
                getDeferDeadlineSteps(), getDeadlineTime());
    }

    @Override
    public String toString() {
        return "CloudletDescriptor{" +
                "cloudletId=" + cloudletId +
                ", submissionDelay=" + submissionDelay +
                ", mi=" + mi +
                ", numberOfCores=" + numberOfCores +
                ", deferDeadlineSteps=" + deferDeadlineSteps +
                ", deadlineTime=" + deadlineTime +
                '}';
    }

    public Cloudlet toCloudlet() {
        // Legacy default: 0.5 CPU utilization, byte-level semantics preserved
        // for every pre-SQT2 scenario (Codex ruling, 2026-08-18).
        return toCloudlet(0.5);
    }

    public Cloudlet toCloudlet(double cpuUtilization) {
        // Use UtilizationModelDynamic for realistic resource usage
        // Lower RAM/BW requests to reduce resource contention on VMs
        Cloudlet cloudlet = new CloudletSimple(cloudletId, mi, numberOfCores)
                .setFileSize(DataCloudTags.DEFAULT_MTU)
                .setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new org.cloudsimplus.utilizationmodels.UtilizationModelDynamic(cpuUtilization))
                .setUtilizationModelRam(new org.cloudsimplus.utilizationmodels.UtilizationModelDynamic(0.15))
                .setUtilizationModelBw(new org.cloudsimplus.utilizationmodels.UtilizationModelDynamic(0.1));
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}
