package giu.edu.cspg.multidc;
import giu.edu.cspg.singledc.LoadBalancingBroker;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import lombok.Getter;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *   Global Broker for multi-datacenter environment (Smart Router Mode).
 *
 *   Instead of maintaining a global queue, this broker acts as an intelligent router:
 * - Stores all cloudlets with their arrival times (from workload trace)
 * - Provides cloudlets that arrive in the current time window
 * - Routes cloudlets to target datacenters based on Global Agent's decisions
 *
 * Design: No queuing delay - cloudlets are routed immediately upon arrival
 */
public class GlobalBroker extends DatacenterBrokerSimple {
    private static final Logger logger = LoggerFactory.getLogger(GlobalBroker.class.getSimpleName());

    // === All Cloudlets (sorted by arrival time) ===
    private final List<Cloudlet> allCloudlets;

    // === Datacenter Instances ===
    private final List<DatacenterInstance> datacenterInstances;

    // === Routing State ===
    private int nextCloudletIndex = 0;  // Index of next cloudlet to process

    /**
     * -- GETTER --
     *  Get the number of cloudlets that have been routed to datacenters.
     *
     * @return Total cloudlets routed
     */
    // === Statistics ===
    @Getter
    private int totalCloudletsRouted = 0;
    private int totalCloudletsCompleted = 0;

    /**
     * Create a GlobalBroker with all cloudlets and datacenter instances.
     *
     * @param simulation The CloudSimPlus simulation instance
     * @param allCloudlets All cloudlets from workload trace (sorted by arrival time)
     * @param datacenterInstances List of all datacenter instances
     */
    public GlobalBroker(CloudSimPlus simulation, List<Cloudlet> allCloudlets,
                       List<DatacenterInstance> datacenterInstances) {
        super(simulation);
        this.setName("GlobalBroker");

        if (allCloudlets == null || datacenterInstances == null) {
            throw new IllegalArgumentException("Cloudlet list and datacenter instances cannot be null");
        }

        // Store all cloudlets (should be sorted by submission delay / arrival time)
        this.allCloudlets = new ArrayList<>(allCloudlets);
        this.datacenterInstances = new ArrayList<>(datacenterInstances);

        logger.info("GlobalBroker initialized with {} cloudlets and {} datacenters",
                allCloudlets.size(), datacenterInstances.size());

        // Disable auto-submission
        setShutdownWhenIdle(false);
    }

    /**
     * Get cloudlets that arrive in the time window [currentTime, currentTime + timestep).
     *
     * @param currentTime Current simulation time (seconds)
     * @param timestep Time window size (seconds)
     * @return List of cloudlets arriving in this time window
     */
    public List<Cloudlet> getArrivingCloudlets(double currentTime, double timestep) {
        List<Cloudlet> arriving = new ArrayList<>();
        double windowEnd = currentTime + timestep;

        // Scan from nextCloudletIndex onwards
        while (nextCloudletIndex < allCloudlets.size()) {
            Cloudlet cloudlet = allCloudlets.get(nextCloudletIndex);
            double arrivalTime = cloudlet.getSubmissionDelay();

            // Check if cloudlet arrives in current window
            if (arrivalTime >= currentTime && arrivalTime < windowEnd) {
                arriving.add(cloudlet);
                nextCloudletIndex++;
            } else if (arrivalTime >= windowEnd) {
                // Future cloudlet, stop scanning
                break;
            } else {
                // Cloudlet arrived in the past (should not happen if sorted)
                logger.warn("Cloudlet {} has arrival time {} < current time {}. Skipping.",
                        cloudlet.getId(), arrivalTime, currentTime);
                nextCloudletIndex++;
            }
        }

        if (!arriving.isEmpty()) {
            logger.debug("{}: {} cloudlets arriving in window [{}, {})",
                    getSimulation().clockStr(), arriving.size(), currentTime, windowEnd);
        }

        return arriving;
    }

    /**
     * Route a cloudlet to a specific datacenter.
     * Called by Global Agent's routing decision.
     *
     * @param cloudlet The cloudlet to route
     * @param datacenterIndex The target datacenter index (0 to num_dcs-1)
     * @return true if routing was successful, false otherwise
     */
    public boolean routeCloudletToDatacenter(Cloudlet cloudlet, int datacenterIndex) {
        // Validate datacenter index
        if (datacenterIndex < 0 || datacenterIndex >= datacenterInstances.size()) {
            logger.warn("Invalid datacenter index: {}. Valid range: [0, {})",
                    datacenterIndex, datacenterInstances.size());
            return false;
        }

        // Get target datacenter
        DatacenterInstance targetDC = datacenterInstances.get(datacenterIndex);
        LoadBalancingBroker localBroker = targetDC.getLocalBroker();

        if (localBroker == null) {
            logger.error("LocalBroker not initialized for datacenter {}", targetDC.getName());
            return false;
        }

        // Route cloudlet to local broker's queue
        boolean success = localBroker.receiveCloudlet(cloudlet);

        if (success) {
            totalCloudletsRouted++;
            targetDC.incrementCloudletsReceived();
            logger.debug("{}: Cloudlet {} routed to {} (Local queue: {})",
                    getSimulation().clockStr(), cloudlet.getId(), targetDC.getName(),
                    localBroker.getWaitingCloudletCount());
        } else {
            logger.warn("Failed to route Cloudlet {} to {}", cloudlet.getId(), targetDC.getName());
        }

        return success;
    }

    /**
     * Batch route multiple cloudlets to their target datacenters.
     *
     * @param cloudlets List of cloudlets to route
     * @param datacenterIndices Corresponding target DC indices
     * @return Number of successfully routed cloudlets
     */
    public int batchRouteCloudlets(List<Cloudlet> cloudlets, List<Integer> datacenterIndices) {
        if (cloudlets.size() != datacenterIndices.size()) {
            logger.error("Cloudlet count ({}) does not match datacenter indices count ({})",
                    cloudlets.size(), datacenterIndices.size());
            return 0;
        }

        int successCount = 0;
        for (int i = 0; i < cloudlets.size(); i++) {
            boolean success = routeCloudletToDatacenter(cloudlets.get(i), datacenterIndices.get(i));
            if (success) {
                successCount++;
            }
        }

        return successCount;
    }

    /**
     * Check if there are more cloudlets to be routed.
     *
     * @return true if there are unprocessed cloudlets
     */
    public boolean hasMoreCloudlets() {
        return nextCloudletIndex < allCloudlets.size();
    }

    /**
     * Get the number of cloudlets remaining to be routed.
     *
     * @return Number of unprocessed cloudlets
     */
    public int getRemainingCloudletCount() {
        return allCloudlets.size() - nextCloudletIndex;
    }

    /**
     * Get the total number of cloudlets in the workload.
     *
     * @return Total cloudlets
     */
    public int getTotalCloudletCount() {
        return allCloudlets.size();
    }

    /**
     * Get the total number of cloudlets completed across all datacenters.
     *
     * @return Total cloudlets completed
     */
    public int getTotalCloudletsCompleted() {
        return datacenterInstances.stream()
                .mapToInt(DatacenterInstance::getCloudletsCompleted)
                .sum();
    }

    /**
     * Get the total number of cloudlets waiting across all datacenters.
     *
     * @return Total cloudlets in all local queues
     */
    public int getTotalCloudletsWaitingInAllDCs() {
        return datacenterInstances.stream()
                .mapToInt(DatacenterInstance::getWaitingCloudletCount)
                .sum();
    }

    /**
     * Get a specific datacenter instance.
     *
     * @param datacenterIndex The datacenter index
     * @return DatacenterInstance or null if invalid index
     */
    public DatacenterInstance getDatacenterInstance(int datacenterIndex) {
        if (datacenterIndex < 0 || datacenterIndex >= datacenterInstances.size()) {
            return null;
        }
        return datacenterInstances.get(datacenterIndex);
    }

    /**
     * Get the number of datacenters.
     *
     * @return Number of datacenters
     */
    public int getDatacenterCount() {
        return datacenterInstances.size();
    }

    /**
     * Reset all statistics.
     */
    public void resetStatistics() {
        totalCloudletsRouted = 0;
        totalCloudletsCompleted = 0;
        nextCloudletIndex = 0;
        datacenterInstances.forEach(DatacenterInstance::resetStatistics);
        logger.info("GlobalBroker statistics reset");
    }

    /**
     * Get a summary of current global state.
     */
    public String getStateSummary() {
        return String.format("GlobalBroker[Total: %d, Routed: %d, Remaining: %d, Completed: %d, DCs: %d]",
                getTotalCloudletCount(), totalCloudletsRouted, getRemainingCloudletCount(),
                getTotalCloudletsCompleted(), datacenterInstances.size());
    }

    @Override
    public String toString() {
        return getStateSummary();
    }
}
