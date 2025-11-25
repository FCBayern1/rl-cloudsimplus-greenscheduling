package giu.edu.cspg.common;

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

    public CloudletDescriptor(int cloudletId, long submissionDelay, long mi, int numberOfCores) {
        this.cloudletId = cloudletId;
        this.submissionDelay = submissionDelay;
        this.mi = mi;
        this.numberOfCores = numberOfCores;
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
                getNumberOfCores() == that.getNumberOfCores();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getCloudletId(), getSubmissionDelay(), getMi(), getNumberOfCores());
    }

    @Override
    public String toString() {
        return "CloudletDescriptor{" +
                "cloudletId=" + cloudletId +
                ", submissionDelay=" + submissionDelay +
                ", mi=" + mi +
                ", numberOfCores=" + numberOfCores +
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
