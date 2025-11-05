package giu.edu.cspg.singledc;

import java.util.Arrays;

/**
 * Represents the observable state passed to the RL agent.
 * Uses padding to handle dynamic numbers of VMs and hosts up to configured
 * maximums.
 */
public class ObservationState {

    // Padded arrays - size determined by settings (maxHosts, maxPotentialVms)
    private final double[] hostLoads; // CPU load [0, 1] for each host (padded with 0)
    private final double[] hostRamUsageRatio; // RAM usage ratio [0, 1] for each host (padded with 0)
    private final double[] vmLoads; // CPU load [0, 1] for each potential VM slot (padded with 0)
    private final int[] vmTypes; // Type of VM in slot (0=Off/None, 1=S, 2=M, 3=L) (padded with 0)
    private final int[] vmHostMap; // Host ID the VM runs on (-1 if slot empty) (padded with -1)
    private final int[] infrastructureObservation; // Infrastructure observation (padded with 0)

    // Scalar values
    private final int waitingCloudlets; // Number of cloudlets in broker queue
    private final int nextCloudletPes; // PEs required by the next waiting cloudlet (0 if none)
    private final int[] vmAvailablePes; // Available PEs on each VM (size num_vms, 0 if VM off/non-existent)
    private final int actualVmCount; // Actual number of currently running VMs (for potential use in Python)
    private final int actualHostCount; // Actual number of hosts (usually fixed, but good practice)

    // Enhanced cloudlet information
    private final long nextCloudletMi; // MI (Million Instructions) of the next waiting cloudlet (0 if none)
    private final double nextCloudletWaitTime; // Time the next cloudlet has been waiting (seconds, 0 if none)
    private final int[] queuePesDistribution; // Distribution of PEs in queue: [small (1-2 PEs), medium (3-4 PEs), large (5+ PEs)]

    // Historical statistics (for temporal awareness)
    private final int completedCloudletsLast10Steps; // Number of cloudlets completed in last 10 steps

    /**
     * Constructor for ObservationState.
     *
     * @param hostLoads                      Padded array of host CPU loads.
     * @param hostRamUsageRatio              Padded array of host RAM usage ratios.
     * @param vmLoads                        Padded array of VM CPU loads.
     * @param vmTypes                        Padded array of VM types (0=Off, 1=S, 2=M, 3=L).
     * @param vmHostMap                      Padded array mapping VM slot index to host ID (-1 if empty).
     * @param infrastructureObservation      Infrastructure observation array.
     * @param waitingCloudlets               Number of cloudlets in the broker's queue.
     * @param nextCloudletPes                PEs required by the next cloudlet in the queue.
     * @param vmAvailablePes                 Available PEs on each VM.
     * @param actualVmCount                  The actual number of running VMs represented in the padded arrays.
     * @param actualHostCount                The actual number of hosts represented.
     * @param nextCloudletMi                 MI of the next waiting cloudlet (0 if none).
     * @param nextCloudletWaitTime           Wait time of the next cloudlet (0 if none).
     * @param queuePesDistribution           Distribution of PEs in queue [small, medium, large].
     * @param completedCloudletsLast10Steps  Number of cloudlets completed in last 10 steps.
     */
    public ObservationState(double[] hostLoads,
                            double[] hostRamUsageRatio,
                            double[] vmLoads,
                            int[] vmTypes,
                            int[] vmHostMap,
                            int[] infrastructureObservation,
                            int waitingCloudlets,
                            int nextCloudletPes,
                            int[] vmAvailablePes,
                            int actualVmCount,
                            int actualHostCount,
                            long nextCloudletMi,
                            double nextCloudletWaitTime,
                            int[] queuePesDistribution,
                            int completedCloudletsLast10Steps) {
        // Use defensive copies for arrays
        this.hostLoads = Arrays.copyOf(hostLoads, hostLoads.length);
        this.hostRamUsageRatio = Arrays.copyOf(hostRamUsageRatio, hostRamUsageRatio.length);
        this.vmLoads = Arrays.copyOf(vmLoads, vmLoads.length);
        this.vmTypes = Arrays.copyOf(vmTypes, vmTypes.length);
        this.vmHostMap = Arrays.copyOf(vmHostMap, vmHostMap.length);
        this.infrastructureObservation = Arrays.copyOf(infrastructureObservation, infrastructureObservation.length);

        this.waitingCloudlets = waitingCloudlets;
        this.nextCloudletPes = nextCloudletPes;
        this.vmAvailablePes = Arrays.copyOf(vmAvailablePes, vmAvailablePes.length);
        this.actualVmCount = actualVmCount;
        this.actualHostCount = actualHostCount;

        // Enhanced cloudlet information
        this.nextCloudletMi = nextCloudletMi;
        this.nextCloudletWaitTime = nextCloudletWaitTime;
        this.queuePesDistribution = Arrays.copyOf(queuePesDistribution, queuePesDistribution.length);

        // Historical statistics
        this.completedCloudletsLast10Steps = completedCloudletsLast10Steps;
    }

    // --- Getters ---
    // Return copies of arrays to maintain immutability
    public double[] getHostLoads() {
        return Arrays.copyOf(hostLoads, hostLoads.length);
    }

    public double[] getHostRamUsageRatio() {
        return Arrays.copyOf(hostRamUsageRatio, hostRamUsageRatio.length);
    }

    public double[] getVmLoads() {
        return Arrays.copyOf(vmLoads, vmLoads.length);
    }

    public int[] getVmTypes() {
        return Arrays.copyOf(vmTypes, vmTypes.length);
    }

    public int[] getVmHostMap() {
        return Arrays.copyOf(vmHostMap, vmHostMap.length);
    }

    public int getWaitingCloudlets() {
        return waitingCloudlets;
    }

    public int getNextCloudletPes() {
        return nextCloudletPes;
    }

    public int[] getVmAvailablePes() {
        return Arrays.copyOf(vmAvailablePes, vmAvailablePes.length);
    }

    public int getActualVmCount() {
        return actualVmCount;
    }

    public int getActualHostCount() {
        return actualHostCount;
    }

    public int[] getInfrastructureObservation() {
        return Arrays.copyOf(infrastructureObservation, infrastructureObservation.length);
    }

    // Enhanced cloudlet information getters
    public long getNextCloudletMi() {
        return nextCloudletMi;
    }

    public double getNextCloudletWaitTime() {
        return nextCloudletWaitTime;
    }

    public int[] getQueuePesDistribution() {
        return Arrays.copyOf(queuePesDistribution, queuePesDistribution.length);
    }

    // Historical statistics getters
    public int getCompletedCloudletsLast10Steps() {
        return completedCloudletsLast10Steps;
    }

    @Override
    public String toString() {
        return "ObservationState{" +
                "actualHosts=" + actualHostCount +
                ", actualVms=" + actualVmCount +
                ", waitingCloudlets=" + waitingCloudlets +
                ", nextCloudletPes=" + nextCloudletPes +
                ", nextCloudletMi=" + nextCloudletMi +
                ", nextCloudletWaitTime=" + String.format("%.2f", nextCloudletWaitTime) +
                ", queuePesDistribution=" + Arrays.toString(queuePesDistribution) +
                ", completedCloudletsLast10Steps=" + completedCloudletsLast10Steps +
                ", hostLoads=" + Arrays.toString(Arrays.copyOf(hostLoads, actualHostCount)) +
                ", vmLoads=" + Arrays.toString(Arrays.copyOf(vmLoads, actualVmCount)) +
                ", vmAvailablePes=" + Arrays.toString(vmAvailablePes) +
                ", vmTypes=" + Arrays.toString(Arrays.copyOf(vmTypes, actualVmCount)) +
                ", vmHostMap=" + Arrays.toString(Arrays.copyOf(vmHostMap, actualVmCount)) +
                ", hostRamUsageRatio=" + Arrays.toString(Arrays.copyOf(hostRamUsageRatio, actualHostCount)) +
                ", infrastructureObservation=" + Arrays.toString(Arrays.copyOf(infrastructureObservation, actualHostCount)) +
                '}';
    }
}
