package giu.edu.cspg.singledc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.core.events.SimEvent;
import org.cloudsimplus.listeners.CloudletVmEventInfo;
import org.cloudsimplus.listeners.EventListener;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LoadBalancingBroker extends DatacenterBrokerSimple {
    private static final Logger logger = LoggerFactory.getLogger(LoadBalancingBroker.class.getSimpleName());

    // Our internal queue to hold cloudlets until assigned by the agent
    private final Queue<Cloudlet> cloudletWaitingQueue = new LinkedList<>();
    private final List<Cloudlet> inputCloudlets; // List of cloudlets to be submitted
    private final PriorityQueue<Cloudlet> cloudletsQueue; // cloudlets to be submitted - sorted by arrival time
    private final Map<Long, Double> cloudletArrivalTimeMap; // map to keep track of arrival times
    private final List<Double> cloudletsFinishedWaitTimeLastTimestep = new ArrayList<>();
    private final List<Cloudlet> cloudletsFinishedLastTimestep = new ArrayList<>();

    public LoadBalancingBroker(CloudSimPlus simulation, List<Cloudlet> inputCloudlets) {
        super(simulation);
        this.inputCloudlets = new ArrayList<>(inputCloudlets);
        initializeCloudletListeners();
        // PriorityQueue requires capacity >= 1. Local brokers in multi-DC mode
        // start empty (cloudlets arrive later), so guard against zero.
        final int initialCapacity = Math.max(1, inputCloudlets.size());
        this.cloudletsQueue = new PriorityQueue<>(initialCapacity,
                (c1, c2) -> Double.compare(c1.getSubmissionDelay(), c2.getSubmissionDelay()));
        this.cloudletsQueue.addAll(inputCloudlets);
        this.cloudletArrivalTimeMap = cloudletsQueue.stream()
                .collect(Collectors.toMap(Cloudlet::getId, Cloudlet::getSubmissionDelay));

        // Keep VMs alive until explicitly destroyed by the agent/gateway
        // If a VM becomes idle, it just sits there.
        this.setVmDestructionDelayFunction(vm -> Double.MAX_VALUE);
    }

    /**
     * Initializes cloudlet listeners for each cloudlet in the cloudletWaitingQueue.
     * This method adds both start and finish listeners to each cloudlet.
     */
    private void initializeCloudletListeners() {
        inputCloudlets.forEach(this::addOnStartListener);
        inputCloudlets.forEach(this::addOnFinishListener);
    }

    /**
     * Adds an event listener to the specified Cloudlet that triggers when the
     * Cloudlet starts      running. The listener logs a debug message with the Cloudlet ID, VM ID, and
     * the current
     * simulation time.
     *
     * @param cloudlet the Cloudlet to which the start listener will be added
     */
    @SuppressWarnings("Convert2Lambda")
    private void addOnStartListener(Cloudlet cloudlet) {
        cloudlet.addOnStartListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.info("Cloudlet: {} started running on VM {} at {} ", cloudlet.getId(),
                        cloudlet.getVm().getId(), getSimulation().clockStr());
            }
        });
    }

    /**
     * Adds an on-finish listener to the specified Cloudlet. The listener logs
     * detailed information about the Cloudlet's
     * execution and calculates the wait time for the Cloudlet
     * upon its completion.
     *
     * @param cloudlet the Cloudlet to which the on-finish listener will be added
     */
    @SuppressWarnings("Convert2Lambda")
    private void addOnFinishListener(Cloudlet cloudlet) {
        cloudlet.addOnFinishListener(new EventListener<CloudletVmEventInfo>() {
            @Override
            public void update(CloudletVmEventInfo info) {
                LOGGER.info(
                        "{}: Cloudlet: {} that was running on Vm {} (runs {} cloudlets) on host {} (runs {} Vms) finished at {} with total execution time {}",
                        getSimulation().clockStr(), cloudlet.getId(), cloudlet.getVm().getId(),
                        cloudlet.getVm().getCloudletScheduler().getCloudletExecList().size(),
                        cloudlet.getVm().getHost(), cloudlet.getVm().getHost().getVmList().size(),
                        getSimulation().clockStr(), String.format("%.2f", cloudlet.getTotalExecutionTime()));
                cloudletsFinishedLastTimestep.add(cloudlet);
                
                // Safe null-check for arrival time
                final Double arrivalTime = cloudletArrivalTimeMap.get(cloudlet.getId());
                final double waitTime = (arrivalTime != null) 
                    ? Math.ceil(cloudlet.getStartTime() - arrivalTime)
                    : 0.0;
                cloudletsFinishedWaitTimeLastTimestep.add(waitTime);
                LOGGER.info("{}: CloudletWaitTime: {}", getSimulation().clockStr(), waitTime);
            }
        });
    }

    /**
     * Adds a list of cloudlets to the internal waiting queue, bypassing
     * the default broker submission logic.
     * 
     * @param list The list of cloudlets to queue.
     */
    public void queueCloudletList(List<Cloudlet> list) {
        if (list == null || list.isEmpty()) {
            logger.debug("{}: {} Received empty list for queueing.", getSimulation().clockStr(), getName());
            return;
        }
        logger.info("{}: {} Queueing {} cloudlets.", getSimulation().clockStr(), getName(), list.size());
        list.stream().filter(c -> c != null).forEach(this.cloudletWaitingQueue::offer);
        logger.info("{}: {} Cloudlet queue size now: {}.", getSimulation().clockStr(), getName(),
                cloudletWaitingQueue.size());
    }

    /**
     * Gets the cloudlets that are ready to be submitted at this timestep.
     * This method checks the submission delay of each cloudlet and submits those
     * that are ready.
     *
     * @param targetTime   The target time for submission.
     * @param startTime    The start time of the simulation.
     * @param currentClock The current simulation clock time.
     */
    public List<Cloudlet> getCloudletsToQueueAtThisTimestep(double targetTime, double startTime, double currentClock) {
        final List<Cloudlet> cloudletsToQueue = cloudletsQueue.stream()
                .takeWhile(cloudlet -> cloudlet.getSubmissionDelay() <= targetTime)
                .collect(Collectors.toList());
        return cloudletsToQueue;
    }

    public void queueCloudletsAtThisTimestep(List<Cloudlet> cloudletsToQueue, double startTime, double currentClock) {
        if (!cloudletsToQueue.isEmpty()) {
            cloudletsQueue.removeAll(cloudletsToQueue); // Remove from the submission queue
            LOGGER.info("[{} - {}]: Queueing {} cloudlets", startTime, currentClock, cloudletsToQueue.size());
            queueCloudletList(cloudletsToQueue); // Add to the internal queue
        } else {
            LOGGER.info("[{} - {}]: No cloudlets to queue", startTime, currentClock);
        }
    }

    /**
     * Receive a cloudlet from GlobalBroker and add it to the local waiting queue.
     * This method is called when the Global Agent routes a cloudlet to this datacenter.
     *
     * @param cloudlet The cloudlet to receive
     * @return true if cloudlet was successfully added to queue, false otherwise
     */
    public boolean receiveCloudlet(Cloudlet cloudlet) {
        if (cloudlet == null) {
            logger.warn("{}: {} Attempted to receive null cloudlet",
                    getSimulation().clockStr(), getName());
            return false;
        }

        // Record arrival time for wait time calculation
        final double arrivalTime = getSimulation().clock();
        cloudletArrivalTimeMap.put(cloudlet.getId(), arrivalTime);

        // Add cloudlet to waiting queue
        cloudletWaitingQueue.offer(cloudlet);

        // IMPORTANT: Also track in inputCloudlets for hasUnfinishedCloudlets() check
        // This ensures multi-DC mode correctly tracks all received cloudlets
        inputCloudlets.add(cloudlet);

        // Add listeners if not already added
        addOnStartListener(cloudlet);
        addOnFinishListener(cloudlet);

        logger.debug("{}: {} Received Cloudlet {} from GlobalBroker (Queue size: {})",
                getSimulation().clockStr(), getName(), cloudlet.getId(),
                cloudletWaitingQueue.size());

        return true;
    }

    /**
     * Assigns the next waiting cloudlet from the internal queue to the specified
     * VM.
     * Called by the LoadBalancerGateway based on the RL agent's action
     * (action_type=1).
     *
     * @param vmId The ID of the VM to assign the cloudlet to.
     * @return true if a cloudlet was successfully assigned and sent for submission,
     *         false otherwise.
     */
    public boolean assignCloudletToVm(long vmId) {
        if (cloudletWaitingQueue.isEmpty()) {
            logger.warn("{}: {} No cloudlets in queue to assign.", getSimulation().clockStr(), getName());
            return false;
        }

        // Find the target VM within the broker's currently known execution list
        // Use stream().filter() to find by long ID instead of getVmFromCreatedList(int)
        Vm vm = getVmCreatedList().stream()
                .filter(v -> v.getId() == vmId)
                .findFirst()
                .orElse(Vm.NULL);

        if (vm == Vm.NULL) {
            logger.warn("{}: {} Cannot assign cloudlet: Target VM {} not found in broker's list.",
                    getSimulation().clockStr(), getName(), vmId);
            return false; // Invalid action or VM was destroyed
        }

        if (!vm.isCreated() || vm.isFailed()) {
            logger.warn("{}: {} Cannot assign cloudlet: Target VM {} found but not running/failed (State: {}).",
                    getSimulation().clockStr(), getName(), vmId, vm.getStateHistory());
            return false; // VM exists but is not operational
        }

        Cloudlet cloudlet = cloudletWaitingQueue.poll(); // Remove from our queue
        if (cloudlet == null)
            return false; // Should not happen if queue wasn't empty, but safeguard.

        // Check if the VM is capable of executing the cloudlet
        if (vm.isSuitableForCloudlet(cloudlet)) {
            logger.info("{}: {} Cloudlet {} is suitable for VM {}.", getSimulation().clockStr(), getName(),
                    cloudlet.getId(), vmId);
        } else {
            logger.debug("Cloudlet assignment ignored, VM not suitable");
            logger.warn("{}: {} Cloudlet {} is NOT suitable for VM {}. Queue size: {}.",
                    getSimulation().clockStr(), getName(), cloudlet.getId(), vmId, cloudletWaitingQueue.size());
            // Re-add the cloudlet to the queue if not suitable
            cloudletWaitingQueue.offer(cloudlet);
            return false;
        }

        logger.info("{}: {} Assigning Cloudlet {} (req PEs: {}) to VM {} (avail PEs: {}).",
                getSimulation().clockStr(), getName(), cloudlet.getId(), cloudlet.getPesNumber(),
                vmId, vm.getFreePesNumber());

        // Calculate delay relative to current time - typically 0 for immediate
        // assignment by agent
        double calculatedDelay = Math.max(0, cloudlet.getSubmissionDelay() - getSimulation().clock());
        logger.info("{}: Calculated submission delay for Cloudlet {}: {}", getSimulation().clockStr(),
                cloudlet.getId(), calculatedDelay);

        // Set the submission delay for the cloudlet
        cloudlet.setSubmissionDelay(calculatedDelay);
        // Associate the cloudlet with the target VM
        cloudlet.setVm(vm);
        // Set the broker for the cloudlet
        cloudlet.setBroker(this);

        // Add to the broker's internal tracking list (for CloudSim Plus state
        // consistency)
        getCloudletSubmittedList().add(cloudlet);

        // Submit the Cloudlet to the Datacenter associated with the VM
        send(getDatacenter(vm), calculatedDelay, CloudSimTag.CLOUDLET_SUBMIT, cloudlet);

        logger.info("{}: {} Cloudlet {} submitted to DC for VM {}. Queue size: {}.",
                getSimulation().clockStr(), getName(), cloudlet.getId(), vmId, cloudletWaitingQueue.size());
        return true;
    }

    /**
     * Resets cloudlets that were executing or waiting on a destroyed VM
     * and adds them back to the primary waiting queue.
     *
     * @param vm The VM that has been (or is about to be) destroyed.
     */
    public void rescheduleFailedCloudlets(Vm vm) {
        if (vm == null || vm == Vm.NULL) {
            logger.warn("{}: {} Attempted to reschedule cloudlets for a null VM.", getSimulation().clockStr(),
                    getName());
            return;
        }
        logger.info("{}: {} Rescheduling cloudlets from destroyed VM {}.", getSimulation().clockStr(), getName(),
                vm.getId());

        // Terminate returns the list of CloudletExecution objects that were terminated.
        List<CloudletExecution> terminatedExecList = vm.getCloudletScheduler().getCloudletExecList();
        List<CloudletExecution> terminatedWaitingList = vm.getCloudletScheduler().getCloudletWaitingList();

        // Combine, reset, and re-queue
        List<Cloudlet> cloudletsToReschedule = Stream.concat(
                terminatedExecList.parallelStream(),
                terminatedWaitingList.parallelStream())
                .map(CloudletExecution::getCloudlet) // Get the Cloudlet from CloudletExecution
                .map(cl -> {
                    long originalLength = cl.getLength();
                    long finishedLength = cl.getFinishedLengthSoFar();
                    // Calculate remaining length, ensure it's not negative
                    long remainingLength = Math.max(0, originalLength - finishedLength);

                    logger.info("Cloudlet {} from VM {}: Original Length={}, Finished={}, Remaining={}",
                            cl.getId(), vm.getId(), originalLength, finishedLength, remainingLength);

                    cl.setVm(Vm.NULL); // Remove VM association
                    cl.reset(); // Reset status, execution times etc.
                    cl.setLength(remainingLength); // Set the length to the remaining work
                    cl.setSubmissionDelay(0); // Reset submission delay to 0
                    return cl; // Return the modified cloudlet
                })
                .peek(cl -> logger.debug("Re-queueing Cloudlet {} (remaining length {}) from destroyed VM {}",
                        cl.getId(), cl.getLength(), vm.getId()))
                .collect(Collectors.toList());

        if (!cloudletsToReschedule.isEmpty()) {
            cloudletsQueue.addAll(cloudletsToReschedule);

            logger.info("{}: {} Re-queued {} cloudlets from destroyed VM {}",
                    getSimulation().clockStr(), getName(), cloudletsToReschedule.size(), vm.getId());
        } else {
            logger.info("{}: {} No cloudlets needed rescheduling from destroyed VM {}.", getSimulation().clockStr(),
                    getName(), vm.getId());
        }

        // The broker's internal lists (vmExecList etc.) should be updated automatically
        // when the Datacenter processes the VM_DESTROY ack, but ensure no references
        // linger.
        // The VM itself is destroyed elsewhere (Datacenter/SimulationCore/Gateway).
    }

    @Override
    public void processEvent(final SimEvent evt) {
        super.processEvent(evt);
        /*
         * This is important! CLOUDLET_RETURN is sent whenever a cloudlet finishes
         * executing. The
         * default behaviour in CloudSim Plus is to destroy a Vm when it has no more
         * cloudlets to
         * execute. Here we override the default behaviour and we trigger the creation
         * of waiting
         * cloudlets so they can be possibly allocated inside a Vm. This is because in
         * our case, we
         * may reschedule some cloudlets so we want the VMs to trigger the creation of
         * those waiting
         * cloudlets
         */

        if (evt.getTag() == CloudSimTag.CLOUDLET_RETURN) {
            final Cloudlet cloudlet = (Cloudlet) evt.getData();
            final Vm vm = cloudlet.getVm();
            LOGGER.debug("Cloudlet {} in VM {} returned. Scheduling more cloudlets...",
                    cloudlet.getId(), vm.getId());
            requestDatacentersToCreateWaitingCloudlets();
            // if (vm.getCloudletScheduler().isEmpty()) {
            // LOGGER.info("VM {} is empty, destroying...", vm.getId());
            // getDatacenter(vm).getVmAllocationPolicy().deallocateHostForVm(vm);
            // // schedule(getDatacenter(vm), 0, CloudSimTag.VM_DESTROY, vm);
            // // send(getDatacenter(vm), 0, CloudSimTag.VM_DESTROY, vm);
            // }
        }

        // Clean the vm created list because over an episode we may create/destroy
        // many vms so we do not want to cause OOM.
        // if (evt.getTag() == CloudSimTag.VM_CREATE_ACK) {
        // LOGGER.debug("Cleaning the vmCreatedList");
        // getVmCreatedList().clear();
        // }
    }

    /**
     * Overridden to prevent automatic VM mapping by the broker itself
     * if any cloudlets accidentally end up in the superclass's waiting list.
     * Should ideally not be called in our flow.
     */
    @Override
    protected Vm defaultVmMapper(Cloudlet cloudlet) {
        logger.warn("{}: {} DefaultVmMapper called for Cloudlet {} - this should not happen for queued cloudlets.",
                getSimulation().clockStr(), getName(), cloudlet.getId());
        // Return NULL to prevent accidental assignment if this is somehow called.
        return Vm.NULL;
    }

    public List<Cloudlet> getInputCloudlets() {
        return inputCloudlets;
    }

    /**
     * Gets the waiting times of cloudlets that finished execution since the last
     * time this method was called.
     *
     * @return A list containing the waiting times (time from submission to start of
     *         execution)
     *         for cloudlets finished in the last simulation interval.
     */
    public List<Double> getFinishedWaitTimesLastStep(double currentClock) {
        return cloudletsFinishedWaitTimeLastTimestep;
    }

    /**
     * Gets the list of cloudlets that finished execution since the last time this
     * method was called.
     *
     * @return A list containing the Cloudlet objects that finished in the last
     *         simulation interval.
     */
    public List<Cloudlet> getCloudletsFinishedLastStep(double currentClock) {
        return cloudletsFinishedLastTimestep;
    }

    /**
     * Gets the arrival time map for cloudlets. This map contains the submission
     * delay
     * for each cloudlet, which is used to determine when the cloudlet should be
     * executed.
     */
    public Map<Long, Double> getCloudletArrivalTimeMap() {
        return cloudletArrivalTimeMap;
    }

    /** Gets the number of cloudlets currently waiting in the internal queue. */
    public int getWaitingCloudletCount() {
        return cloudletWaitingQueue.size();
    }

    /** Checks if there are any cloudlets waiting in the internal queue. */
    public boolean hasWaitingCloudlets() {
        return !cloudletWaitingQueue.isEmpty();
    }

    /** Checks if there are any cloudlets that have not finished execution. */
    public boolean hasUnfinishedCloudlets() {
        return this.getCloudletFinishedList().size() < inputCloudlets.size();
    }

    /** Gets (but does not remove) the next cloudlet waiting in the queue. */
    public Cloudlet peekWaitingCloudlet() {
        return cloudletWaitingQueue.peek();
    }

    /** Clears the internal waiting queue. */
    public void clearWaitingQueue() {
        cloudletWaitingQueue.clear();
    }

    /** Clears the list of finished wait times. Called during reset. */
    public void clearFinishedWaitTimes() {
        cloudletsFinishedLastTimestep.clear();
        cloudletsFinishedWaitTimeLastTimestep.clear();
    }
}
