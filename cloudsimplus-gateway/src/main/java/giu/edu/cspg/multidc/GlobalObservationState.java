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
     * Current green power available at each datacenter (W).
     * This is the instantaneous green power supply at the current timestep.
     * Index corresponds to datacenter ID.
     */
    private final double[] dcCurrentGreenPowerW;

    /**
     * Current total power consumption at each datacenter (W).
     * This is the instantaneous power draw from all hosts at the current timestep.
     * Index corresponds to datacenter ID.
     */
    private final double[] dcCurrentPowerW;

    /**
     * Green energy usage ratio at each datacenter [0.0, 1.0].
     * Ratio of green energy consumed to total energy consumed.
     * Index corresponds to datacenter ID.
     */
    private final double[] dcGreenRatio;

    /**
     * Cumulative wasted green energy at each datacenter (Wh).
     * Green energy that was available but not used (e.g., low load with high green supply).
     * Index corresponds to datacenter ID.
     */
    private final double[] dcCumulativeWastedGreenWh;

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
     * PEs (cores) required by each cloudlet in the batch to be routed.
     * Array length = batch size. Values are 0 if no cloudlet at that position.
     */
    private final int[] batchCloudletPes;

    /**
     * Million Instructions (MI) of each cloudlet in the batch to be routed.
     * Array length = batch size. Values are 0 if no cloudlet at that position.
     */
    private final long[] batchCloudletMi;

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
     * @param dcCurrentGreenPowerW Current green power at each DC (W)
     * @param dcCurrentPowerW Current total power consumption at each DC (W)
     * @param dcGreenRatio Green energy usage ratio at each DC [0, 1]
     * @param dcCumulativeWastedGreenWh Cumulative wasted green energy at each DC (Wh)
     * @param dcQueueSizes Queue sizes at each DC
     * @param dcUtilizations CPU utilization at each DC [0, 1]
     * @param dcAvailablePes Available PEs at each DC
     * @param dcRamUtilizations RAM utilization at each DC [0, 1]
     * @param upcomingCloudletsCount Number of arriving cloudlets
     * @param batchCloudletPes Array of PEs required by each cloudlet in the batch
     * @param batchCloudletMi Array of MI for each cloudlet in the batch
     * @param upcomingCloudletsPesDistribution Distribution of PEs in upcoming cloudlets [small, medium, large]
     * @param loadImbalance Load imbalance metric across DCs
     * @param recentCompletedCloudlets Cloudlets completed recently
     * @param numDatacenters Number of datacenters
     * @param currentClock Current simulation time
     */
    public GlobalObservationState(
            double[] dcCurrentGreenPowerW,
            double[] dcCurrentPowerW,
            double[] dcGreenRatio,
            double[] dcCumulativeWastedGreenWh,
            int[] dcQueueSizes,
            double[] dcUtilizations,
            int[] dcAvailablePes,
            double[] dcRamUtilizations,
            int upcomingCloudletsCount,
            int[] batchCloudletPes,
            long[] batchCloudletMi,
            int[] upcomingCloudletsPesDistribution,
            double loadImbalance,
            int recentCompletedCloudlets,
            int numDatacenters,
            double currentClock) {

        // Defensive copies for arrays
        this.dcCurrentGreenPowerW = Arrays.copyOf(dcCurrentGreenPowerW, dcCurrentGreenPowerW.length);
        this.dcCurrentPowerW = Arrays.copyOf(dcCurrentPowerW, dcCurrentPowerW.length);
        this.dcGreenRatio = Arrays.copyOf(dcGreenRatio, dcGreenRatio.length);
        this.dcCumulativeWastedGreenWh = Arrays.copyOf(dcCumulativeWastedGreenWh, dcCumulativeWastedGreenWh.length);
        this.dcQueueSizes = Arrays.copyOf(dcQueueSizes, dcQueueSizes.length);
        this.dcUtilizations = Arrays.copyOf(dcUtilizations, dcUtilizations.length);
        this.dcAvailablePes = Arrays.copyOf(dcAvailablePes, dcAvailablePes.length);
        this.dcRamUtilizations = Arrays.copyOf(dcRamUtilizations, dcRamUtilizations.length);

        // Scalar values
        this.upcomingCloudletsCount = upcomingCloudletsCount;
        this.batchCloudletPes = Arrays.copyOf(batchCloudletPes, batchCloudletPes.length);
        this.batchCloudletMi = Arrays.copyOf(batchCloudletMi, batchCloudletMi.length);
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

    public double[] getDcCurrentGreenPowerW() {
        return Arrays.copyOf(dcCurrentGreenPowerW, dcCurrentGreenPowerW.length);
    }

    public double[] getDcCurrentPowerW() {
        return Arrays.copyOf(dcCurrentPowerW, dcCurrentPowerW.length);
    }

    public double[] getDcGreenRatio() {
        return Arrays.copyOf(dcGreenRatio, dcGreenRatio.length);
    }

    public double[] getDcCumulativeWastedGreenWh() {
        return Arrays.copyOf(dcCumulativeWastedGreenWh, dcCumulativeWastedGreenWh.length);
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

    public int[] getBatchCloudletPes() {
        return Arrays.copyOf(batchCloudletPes, batchCloudletPes.length);
    }

    public long[] getBatchCloudletMi() {
        return Arrays.copyOf(batchCloudletMi, batchCloudletMi.length);
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
        double maxPower = dcCurrentGreenPowerW[0];
        for (int i = 1; i < dcCurrentGreenPowerW.length; i++) {
            if (dcCurrentGreenPowerW[i] > maxPower) {
                maxPower = dcCurrentGreenPowerW[i];
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
     * Check if a specific datacenter has sufficient resources for the first cloudlet in batch.
     *
     * @param dcIndex Datacenter index
     * @return true if datacenter has enough available PEs for first cloudlet
     */
    public boolean datacenterCanAcceptNextCloudlet(int dcIndex) {
        if (dcIndex < 0 || dcIndex >= numDatacenters) {
            return false;
        }
        // Check against first cloudlet in batch (if any)
        int firstCloudletPes = (batchCloudletPes != null && batchCloudletPes.length > 0) ? batchCloudletPes[0] : 0;
        return dcAvailablePes[dcIndex] >= firstCloudletPes;
    }

    /**
     * Get total green power across all datacenters.
     *
     * @return Total green power (W)
     */
    public double getTotalGreenPower() {
        return Arrays.stream(dcCurrentGreenPowerW).sum();
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
                "batchCloudletPes=%s, " +
                "loadImbalance=%.3f, " +
                "recentCompleted=%d" +
                "}",
                numDatacenters,
                currentClock,
                Arrays.toString(dcCurrentGreenPowerW),
                Arrays.toString(dcQueueSizes),
                Arrays.toString(dcUtilizations),
                Arrays.toString(dcAvailablePes),
                upcomingCloudletsCount,
                Arrays.toString(batchCloudletPes),
                loadImbalance,
                recentCompletedCloudlets
        );
    }

    /**
     * Create a simple global observation with minimal information.
     * Useful for testing and initialization.
     *
     * @param numDatacenters Number of datacenters
     * @param batchSize Batch size for global routing
     * @return Empty GlobalObservationState
     */
    public static GlobalObservationState createEmpty(int numDatacenters, int batchSize) {
        return new GlobalObservationState(
                new double[numDatacenters],  // dcCurrentGreenPowerW
                new double[numDatacenters],  // dcCurrentPowerW
                new double[numDatacenters],  // dcGreenRatio
                new double[numDatacenters],  // dcCumulativeWastedGreenWh
                new int[numDatacenters],     // dcQueueSizes
                new double[numDatacenters],  // dcUtilizations
                new int[numDatacenters],     // dcAvailablePes
                new double[numDatacenters],  // dcRamUtilizations
                0,                           // upcomingCloudletsCount
                new int[batchSize],          // batchCloudletPes
                new long[batchSize],         // batchCloudletMi
                new int[3],                  // upcomingCloudletsPesDistribution
                0.0,                         // loadImbalance
                0,                           // recentCompletedCloudlets
                numDatacenters,
                0.0                          // currentClock
        );
    }

    /**
     * Create empty observation with default batch size of 5.
     * @deprecated Use createEmpty(int numDatacenters, int batchSize) instead
     */
    @Deprecated
    public static GlobalObservationState createEmpty(int numDatacenters) {
        return createEmpty(numDatacenters, 5);
    }
}
