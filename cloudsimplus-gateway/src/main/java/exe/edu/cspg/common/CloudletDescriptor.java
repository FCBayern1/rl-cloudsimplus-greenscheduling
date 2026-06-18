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

    /** Full constructor with the deferral deadline. */
    public CloudletDescriptor(int cloudletId, long submissionDelay, long mi, int numberOfCores,
                              long deferDeadlineSteps) {
        this.cloudletId = cloudletId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.numberOfCores = numberOfCores;
        this.deferDeadlineSteps = Math.max(0, deferDeadlineSteps);
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
                getDeferDeadlineSteps() == that.getDeferDeadlineSteps();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getCloudletId(), getSubmissionDelay(), getMi(), getNumberOfCores(),
                getDeferDeadlineSteps());
    }

    @Override
    public String toString() {
        return "CloudletDescriptor{" +
                "cloudletId=" + cloudletId +
                ", submissionDelay=" + submissionDelay +
                ", mi=" + mi +
                ", numberOfCores=" + numberOfCores +
                ", deferDeadlineSteps=" + deferDeadlineSteps +
                '}';
    }

    public Cloudlet toCloudlet() {
        // Use UtilizationModelDynamic for realistic resource usage
        // Lower RAM/BW requests to reduce resource contention on VMs
        Cloudlet cloudlet = new CloudletSimple(cloudletId, mi, numberOfCores)
                .setFileSize(DataCloudTags.DEFAULT_MTU)
                .setOutputSize(DataCloudTags.DEFAULT_MTU)
                .setUtilizationModelCpu(new org.cloudsimplus.utilizationmodels.UtilizationModelDynamic(0.5))
                .setUtilizationModelRam(new org.cloudsimplus.utilizationmodels.UtilizationModelDynamic(0.15))
                .setUtilizationModelBw(new org.cloudsimplus.utilizationmodels.UtilizationModelDynamic(0.1));
        cloudlet.setSubmissionDelay(submissionDelay);
        return cloudlet;
    }
}
