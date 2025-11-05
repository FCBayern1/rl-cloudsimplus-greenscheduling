package giu.edu.cspg.multidc;
import giu.edu.cspg.singledc.ObservationState;

import java.util.Arrays;

/**
 * Global observation state for hierarchical multi-datacenter MARL.
 *
 * This class represents the aggregated, datacenter-level view that the Global Agent
 * uses for routing decisions. It contains metrics about each datacenter's state,
 * not individual VM or Host details.
 *
 * Design Philosophy:
 * - Global Agent sees datacenter-level aggregated metrics
 * - Local Agents see VM/Host-level detailed metrics (ObservationState)
 * - Clear separation of concerns between routing and scheduling
 */
public class GlobalObservationState {

    // === Datacenter-level Resource Metrics ===

    /**
     * Green energy power available at each datacenter (kW).
     * Index corresponds to datacenter ID.
     */
    private final double[] dcGreenPower;

    /**
     * Number of cloudlets waiting in each datacenter's local queue.
     * Index corresponds to datacenter ID.
     */
    private final int[] dcQueueSizes;

    /**
     * Average CPU utilization across all hosts in each datacenter [0.0, 1.0].
     * Index corresponds to datacenter ID.
     */
    private final double[] dcUtilizations;

    /**
     * Total available PEs (Processing Elements/Cores) across all VMs in each datacenter.
     * Index corresponds to datacenter ID.
     */
    private final int[] dcAvailablePes;

    /**
     * Average RAM utilization across all hosts in each datacenter [0.0, 1.0].
     * Index corresponds to datacenter ID.
     */
    private final double[] dcRamUtilizations;

    // === Workload Information ===

    /**
     * Number of cloudlets arriving in the upcoming time window.
     * These cloudlets need to be routed to datacenters.
     */
    private final int upcomingCloudletsCount;

    /**
     * PEs (cores) required by the next arriving cloudlet.
     * 0 if no cloudlets are arriving.
     */
    private final int nextCloudletPes;

    /**
     * Million Instructions (MI) of the next arriving cloudlet.
     * 0 if no cloudlets are arriving.
     */
    private final long nextCloudletMi;

    /**
     * Distribution of PEs requirements in upcoming cloudlets.
     * Format: [small (1-2 PEs), medium (3-4 PEs), large (5+ PEs)]
     */
    private final int[] upcomingCloudletsPesDistribution;

    // === Cross-Datacenter Metrics ===

    /**
     * Load imbalance metric across datacenters.
     * Higher values indicate more uneven load distribution.
     */
    private final double loadImbalance;

    /**
     * Total number of cloudlets completed across all datacenters
     * in the last time window.
     */
    private final int recentCompletedCloudlets;

    // === Metadata ===

    /**
     * Number of datacenters in the system.
     */
    private final int numDatacenters;

    /**
     * Current simulation clock (seconds).
     */
    private final double currentClock;

    /**
     * Constructor for GlobalObservationState.
     *
     * @param dcGreenPower Green power available at each DC (kW)
     * @param dcQueueSizes Queue sizes at each DC
     * @param dcUtilizations CPU utilization at each DC [0, 1]
     * @param dcAvailablePes Available PEs at each DC
     * @param dcRamUtilizations RAM utilization at each DC [0, 1]
     * @param upcomingCloudletsCount Number of arriving cloudlets
     * @param nextCloudletPes PEs required by next cloudlet
     * @param nextCloudletMi MI of next cloudlet
     * @param upcomingCloudletsPesDistribution Distribution of PEs in upcoming cloudlets [small, medium, large]
     * @param loadImbalance Load imbalance metric across DCs
     * @param recentCompletedCloudlets Cloudlets completed recently
     * @param numDatacenters Number of datacenters
     * @param currentClock Current simulation time
     */
    public GlobalObservationState(
            double[] dcGreenPower,
            int[] dcQueueSizes,
            double[] dcUtilizations,
            int[] dcAvailablePes,
            double[] dcRamUtilizations,
            int upcomingCloudletsCount,
            int nextCloudletPes,
            long nextCloudletMi,
            int[] upcomingCloudletsPesDistribution,
            double loadImbalance,
            int recentCompletedCloudlets,
            int numDatacenters,
            double currentClock) {

        // Defensive copies for arrays
        this.dcGreenPower = Arrays.copyOf(dcGreenPower, dcGreenPower.length);
        this.dcQueueSizes = Arrays.copyOf(dcQueueSizes, dcQueueSizes.length);
        this.dcUtilizations = Arrays.copyOf(dcUtilizations, dcUtilizations.length);
        this.dcAvailablePes = Arrays.copyOf(dcAvailablePes, dcAvailablePes.length);
        this.dcRamUtilizations = Arrays.copyOf(dcRamUtilizations, dcRamUtilizations.length);

        // Scalar values
        this.upcomingCloudletsCount = upcomingCloudletsCount;
        this.nextCloudletPes = nextCloudletPes;
        this.nextCloudletMi = nextCloudletMi;
        this.upcomingCloudletsPesDistribution = Arrays.copyOf(
                upcomingCloudletsPesDistribution,
                upcomingCloudletsPesDistribution.length
        );

        this.loadImbalance = loadImbalance;
        this.recentCompletedCloudlets = recentCompletedCloudlets;
        this.numDatacenters = numDatacenters;
        this.currentClock = currentClock;
    }

    // === Getters (Return copies for immutability) ===

    public double[] getDcGreenPower() {
        return Arrays.copyOf(dcGreenPower, dcGreenPower.length);
    }

    public int[] getDcQueueSizes() {
        return Arrays.copyOf(dcQueueSizes, dcQueueSizes.length);
    }

    public double[] getDcUtilizations() {
        return Arrays.copyOf(dcUtilizations, dcUtilizations.length);
    }

    public int[] getDcAvailablePes() {
        return Arrays.copyOf(dcAvailablePes, dcAvailablePes.length);
    }

    public double[] getDcRamUtilizations() {
        return Arrays.copyOf(dcRamUtilizations, dcRamUtilizations.length);
    }

    public int getUpcomingCloudletsCount() {
        return upcomingCloudletsCount;
    }

    public int getNextCloudletPes() {
        return nextCloudletPes;
    }

    public long getNextCloudletMi() {
        return nextCloudletMi;
    }

    public int[] getUpcomingCloudletsPesDistribution() {
        return Arrays.copyOf(upcomingCloudletsPesDistribution, upcomingCloudletsPesDistribution.length);
    }

    public double getLoadImbalance() {
        return loadImbalance;
    }

    public int getRecentCompletedCloudlets() {
        return recentCompletedCloudlets;
    }

    public int getNumDatacenters() {
        return numDatacenters;
    }

    public double getCurrentClock() {
        return currentClock;
    }

    // === Helper Methods ===

    /**
     * Get the datacenter ID with the most available green power.
     *
     * @return Datacenter ID with maximum green power
     */
    public int getDatacenterWithMaxGreenPower() {
        int maxIndex = 0;
        double maxPower = dcGreenPower[0];
        for (int i = 1; i < dcGreenPower.length; i++) {
            if (dcGreenPower[i] > maxPower) {
                maxPower = dcGreenPower[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * Get the datacenter ID with the smallest queue size.
     *
     * @return Datacenter ID with minimum queue
     */
    public int getDatacenterWithMinQueue() {
        int minIndex = 0;
        int minQueue = dcQueueSizes[0];
        for (int i = 1; i < dcQueueSizes.length; i++) {
            if (dcQueueSizes[i] < minQueue) {
                minQueue = dcQueueSizes[i];
                minIndex = i;
            }
        }
        return minIndex;
    }

    /**
     * Get the datacenter ID with the lowest utilization.
     *
     * @return Datacenter ID with minimum utilization
     */
    public int getDatacenterWithMinUtilization() {
        int minIndex = 0;
        double minUtil = dcUtilizations[0];
        for (int i = 1; i < dcUtilizations.length; i++) {
            if (dcUtilizations[i] < minUtil) {
                minUtil = dcUtilizations[i];
                minIndex = i;
            }
        }
        return minIndex;
    }

    /**
     * Check if a specific datacenter has sufficient resources for the next cloudlet.
     *
     * @param dcIndex Datacenter index
     * @return true if datacenter has enough available PEs
     */
    public boolean datacenterCanAcceptNextCloudlet(int dcIndex) {
        if (dcIndex < 0 || dcIndex >= numDatacenters) {
            return false;
        }
        return dcAvailablePes[dcIndex] >= nextCloudletPes;
    }

    /**
     * Get total green power across all datacenters.
     *
     * @return Total green power (kW)
     */
    public double getTotalGreenPower() {
        return Arrays.stream(dcGreenPower).sum();
    }

    /**
     * Get total waiting cloudlets across all datacenters.
     *
     * @return Total cloudlets in all DC queues
     */
    public int getTotalWaitingCloudlets() {
        return Arrays.stream(dcQueueSizes).sum();
    }

    @Override
    public String toString() {
        return String.format(
                "GlobalObservationState{" +
                "numDCs=%d, " +
                "clock=%.2fs, " +
                "dcGreenPower=%s kW, " +
                "dcQueueSizes=%s, " +
                "dcUtilizations=%s, " +
                "dcAvailablePes=%s, " +
                "upcomingCloudlets=%d, " +
                "nextCloudletPes=%d, " +
                "loadImbalance=%.3f, " +
                "recentCompleted=%d" +
                "}",
                numDatacenters,
                currentClock,
                Arrays.toString(dcGreenPower),
                Arrays.toString(dcQueueSizes),
                Arrays.toString(dcUtilizations),
                Arrays.toString(dcAvailablePes),
                upcomingCloudletsCount,
                nextCloudletPes,
                loadImbalance,
                recentCompletedCloudlets
        );
    }

    /**
     * Create a simple global observation with minimal information.
     * Useful for testing and initialization.
     *
     * @param numDatacenters Number of datacenters
     * @return Empty GlobalObservationState
     */
    public static GlobalObservationState createEmpty(int numDatacenters) {
        return new GlobalObservationState(
                new double[numDatacenters],  // dcGreenPower
                new int[numDatacenters],     // dcQueueSizes
                new double[numDatacenters],  // dcUtilizations
                new int[numDatacenters],     // dcAvailablePes
                new double[numDatacenters],  // dcRamUtilizations
                0,                           // upcomingCloudletsCount
                0,                           // nextCloudletPes
                0L,                          // nextCloudletMi
                new int[3],                  // upcomingCloudletsPesDistribution
                0.0,                         // loadImbalance
                0,                           // recentCompletedCloudlets
                numDatacenters,
                0.0                          // currentClock
        );
    }
}
