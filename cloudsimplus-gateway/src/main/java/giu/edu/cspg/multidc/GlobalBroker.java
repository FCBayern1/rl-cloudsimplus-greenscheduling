package giu.edu.cspg.multidc;
import giu.edu.cspg.singledc.LoadBalancingBroker;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import lombok.Getter;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *   Global Broker for multi-datacenter environment (Batch Routing Mode).
 *
 *   This broker implements a batch routing strategy:
 * - Stores all cloudlets with their arrival times (from workload trace)
 * - Maintains a waiting queue for cloudlets to be routed
 * - Each timestep, provides a fixed-size batch of cloudlets for routing
 * - Newly arriving cloudlets are added to the waiting queue
 *
 * Design: Fixed batch size per step for stable action space
 */
public class GlobalBroker extends DatacenterBrokerSimple {
    private static final Logger logger = LoggerFactory.getLogger(GlobalBroker.class.getSimpleName());

    // === All Cloudlets (sorted by arrival time) ===
    private final List<Cloudlet> allCloudlets;

    // === Global Waiting Queue ===
    // Use LinkedList for efficient O(1) removal from head
    private final LinkedList<Cloudlet> globalWaitingQueue = new LinkedList<>();

    // === Datacenter Instances ===
    private final List<DatacenterInstance> datacenterInstances;

    // === Routing State ===
    private int nextCloudletIndex = 0;  // Index of next cloudlet to process from allCloudlets

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
     * Process newly arriving cloudlets for the current time window.
     * Adds arriving cloudlets to the global waiting queue.
     *
     * @param currentTime Current simulation time (seconds)
     * @param timestep Time window size (seconds)
     */
    public void processArrivingCloudlets(double currentTime, double timestep) {
        double windowEnd = currentTime + timestep;
        
        // For the first step, CloudSim clock starts at minTimeBetweenEvents (e.g., 0.1),
        // but cloudlets may have arrival time starting from 0.0.
        // Use 0.0 as window start if this appears to be the first step.
        double windowStart = (currentTime < timestep) ? 0.0 : currentTime;

        int arrivedCount = 0;
        // Scan from nextCloudletIndex onwards
        while (nextCloudletIndex < allCloudlets.size()) {
            Cloudlet cloudlet = allCloudlets.get(nextCloudletIndex);
            double arrivalTime = cloudlet.getSubmissionDelay();

            // Check if cloudlet arrives in current window
            if (arrivalTime >= windowStart && arrivalTime < windowEnd) {
                globalWaitingQueue.add(cloudlet);
                nextCloudletIndex++;
                arrivedCount++;
            } else if (arrivalTime >= windowEnd) {
                // Future cloudlet, stop scanning
                break;
            } else {
                // Cloudlet arrived in the past (should not happen if sorted)
                logger.warn("Cloudlet {} has arrival time {} < window start {}. Skipping.",
                        cloudlet.getId(), arrivalTime, windowStart);
                nextCloudletIndex++;
            }
        }

        if (arrivedCount > 0) {
            logger.debug("{}: {} cloudlets arrived, global queue size now: {}",
                    getSimulation().clockStr(), arrivedCount, globalWaitingQueue.size());
        }
    }

    /**
     * Get cloudlets that arrive in the time window [currentTime, currentTime + timestep).
     * (Legacy method for backward compatibility - use processArrivingCloudlets instead)
     *
     * @param currentTime Current simulation time (seconds)
     * @param timestep Time window size (seconds)
     * @return List of cloudlets arriving in this time window
     */
    public List<Cloudlet> getArrivingCloudlets(double currentTime, double timestep) {
        List<Cloudlet> arriving = new ArrayList<>();
        double windowEnd = currentTime + timestep;
        
        double windowStart = (currentTime < timestep) ? 0.0 : currentTime;

        // Peek at arriving cloudlets without removing them from allCloudlets
        for (int i = nextCloudletIndex; i < allCloudlets.size(); i++) {
            Cloudlet cloudlet = allCloudlets.get(i);
            double arrivalTime = cloudlet.getSubmissionDelay();

            if (arrivalTime >= windowStart && arrivalTime < windowEnd) {
                arriving.add(cloudlet);
            } else if (arrivalTime >= windowEnd) {
                break;
            }
        }

        return arriving;
    }

    /**
     * Get a batch of cloudlets from the waiting queue for routing.
     * Returns up to batchSize cloudlets (or fewer if queue has less).
     *
     * @param batchSize Maximum number of cloudlets to return
     * @return List of cloudlets to be routed (removed from waiting queue)
     */
    public List<Cloudlet> getBatchForRouting(int batchSize) {
        if (batchSize <= 0 || globalWaitingQueue.isEmpty()) {
            return new ArrayList<>();
        }

        int actualSize = Math.min(batchSize, globalWaitingQueue.size());
        List<Cloudlet> batch = new ArrayList<>(actualSize);
        
        // Remove cloudlets from head of queue (O(1) per removal with LinkedList)
        for (int i = 0; i < actualSize; i++) {
            batch.add(globalWaitingQueue.removeFirst());
        }
        
        logger.debug("Provided batch of {} cloudlets for routing, {} remaining in queue",
                actualSize, globalWaitingQueue.size());
        
        return batch;
    }

    /**
     * Get the number of cloudlets currently in the global waiting queue.
     *
     * @return Number of cloudlets waiting to be routed
     */
    public int getGlobalWaitingCloudletsCount() {
        return globalWaitingQueue.size();
    }

    /**
     * Peek at the next cloudlet in the waiting queue without removing it.
     *
     * @return Next cloudlet or null if queue is empty
     */
    public Cloudlet peekNextCloudlet() {
        return globalWaitingQueue.peekFirst();  // Returns null if empty
    }

    /**
     * Peek at the next batch of cloudlets without removing them from the queue.
     * Returns up to batchSize cloudlets from the head of the queue.
     *
     * @param batchSize Maximum number of cloudlets to peek
     * @return List of cloudlets (may be smaller than batchSize if queue has fewer cloudlets)
     */
    public List<Cloudlet> peekBatch(int batchSize) {
        List<Cloudlet> batch = new ArrayList<>();
        int count = Math.min(batchSize, globalWaitingQueue.size());

        // Use iterator to peek without modifying the queue
        int i = 0;
        for (Cloudlet cloudlet : globalWaitingQueue) {
            if (i >= count) break;
            batch.add(cloudlet);
            i++;
        }

        return batch;
    }

    /**
     * Calculate the REAL distribution of cloudlets in the waiting queue by PE requirements.
     * Categories: Small (1-2 PEs), Medium (3-4 PEs), Large (5+ PEs)
     *
     * @return Array of [smallCount, mediumCount, largeCount]
     */
    public int[] calculatePesDistribution() {
        int[] distribution = new int[3];  // [small, medium, large]
        
        if (globalWaitingQueue.isEmpty()) {
            return distribution;  // All zeros
        }
        
        // Iterate through the queue and classify each cloudlet
        for (Cloudlet cloudlet : globalWaitingQueue) {
            long pesCount = cloudlet.getPesNumber();
            
            if (pesCount <= 2) {
                distribution[0]++;  // Small (1-2 PEs)
            } else if (pesCount <= 4) {
                distribution[1]++;  // Medium (3-4 PEs)
            } else {
                distribution[2]++;  // Large (5+ PEs)
            }
        }
        
        logger.trace("Queue PE distribution: Small={}, Medium={}, Large={}", 
                distribution[0], distribution[1], distribution[2]);
        
        return distribution;
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
     * Reset all statistics and queues.
     */
    public void resetStatistics() {
        totalCloudletsRouted = 0;
        totalCloudletsCompleted = 0;
        nextCloudletIndex = 0;
        globalWaitingQueue.clear();
        datacenterInstances.forEach(DatacenterInstance::resetStatistics);
        logger.info("GlobalBroker statistics and waiting queue reset");
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
