package giu.edu.cspg.singledc;
import giu.edu.cspg.common.VmAllocationPolicyCustom;
import giu.edu.cspg.common.DatacenterSetup;
import giu.edu.cspg.common.CloudletDescriptor;
import giu.edu.cspg.common.SimulationSettings;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import lombok.Getter;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.utils.WorkloadFileReader;


@Getter
public final class SimulationCore {
    private static final Logger LOGGER = LoggerFactory.getLogger(SimulationCore.class.getSimpleName());

    private final SimulationSettings settings;
    private CloudSimPlus simulation;
    private Datacenter datacenter;
    private LoadBalancingBroker broker;
    private List<Host> hostList;
    private List<Vm> vmPool; // This will hold the initial fleet + dynamically created ones
    private List<Cloudlet> cloudletList; // Full list of cloudlets from the trace
    private boolean firstStep; // Flag to indicate if it's the first step of the simulation

    /**
     * Initializes the simulation core with specific settings.
     * 
     * @param settings The simulation settings to use.
     */
    public SimulationCore(SimulationSettings settings) {
        if (settings == null) {
            throw new IllegalArgumentException("SimulationSettings cannot be null.");
        }
        this.settings = settings;
        this.resetSimulation(); // Internal reset to initialize the simulation environment
    }

    /**
     * Resets and initializes the simulation environment with a fixed initial fleet.
     * Loads SWF trace and queues cloudlets.
     */
    public void resetSimulation() {
        LOGGER.info("Resetting CloudSim simulation environment (Initial Fixed Fleet)...");
        stopSimulation(); // Ensure previous run is stopped cleanly

        // Initialize core CloudSimPlus engine
        simulation = new CloudSimPlus(settings.getMinTimeBetweenEvents());
        firstStep = true;

        // Initialize lists
        hostList = new ArrayList<>();
        vmPool = new ArrayList<>(); // Start with empty pool for initial fleet creation
        cloudletList = new ArrayList<>();

        // --- Step 1: Load Cloudlet Descriptors using WorkloadFileReader ---
        List<CloudletDescriptor> descriptors = loadCloudletDescriptors();
        LOGGER.info("Loaded and processed {} cloudlet descriptors from trace.", descriptors.size());

        // --- Step 2: Optionally Split Descriptors ---
        if (settings.isSplitLargeCloudlets()) {
            LOGGER.info("Splitting large cloudlets (maxCloudletPes = {})...", settings.getMaxCloudletPes());
            descriptors = splitLargeCloudletDescriptors(descriptors, settings.getMaxCloudletPes());
            LOGGER.info("Total descriptors after splitting: {}", descriptors.size());
        } else {
            LOGGER.info("Cloudlet splitting disabled.");
        }

        // --- Step 3: Convert Descriptors to Cloudlets ---
        // Convert descriptors to actual Cloudlet objects, associating with the broker
        this.cloudletList = descriptors.stream()
                .map(CloudletDescriptor::toCloudlet)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());
        LOGGER.info("Converted {} descriptors to Cloudlet objects.", this.cloudletList.size());

        // --- Step 4: Create Broker ---
        broker = new LoadBalancingBroker(simulation, this.cloudletList);
        broker.setShutdownWhenIdle(false); // important to keep the broker running

        // --- Step 5: Create Hosts and Datacenter ---
        // DatacenterSetup.createDatacenter creates the fixed number of hosts
        // Use VmAllocationPolicyCustom for future dynamic VM creation by agent
        datacenter = DatacenterSetup.createDatacenter(simulation, settings, this.hostList,
                new VmAllocationPolicyCustom());
        LOGGER.info("Datacenter created with {} hosts.", hostList.size());

        // --- Step 6: Create Initial VM Fleet ---
        DatacenterSetup.createInitialVmFleet(settings, this.vmPool); // Populates vmPool
        LOGGER.info("Initial VM fleet created with {} VMs.", vmPool.size());

        // --- Step 7: Submit the Initial VM Fleet to the broker ---
        // Broker needs to know about these VMs for placement and tracking
        broker.submitVmList(new ArrayList<>(vmPool)); // Submit a copy

        // --- Step 8: Associate cloudlets with the broker ---
        this.cloudletList.forEach(cloudlet -> {
            cloudlet.setBroker(broker);
        });

        // This is important to ensure the broker knows about the cloudlets
        ensureAllCloudletsCompleteBeforeSimulationEnds();

        // --- Step 9: Start the simulation engine ---
        simulation.startSync();
        // initialize the simulation to allow the datacenter to be created
        proceedClockTo(settings.getMinTimeBetweenEvents());

        LOGGER.info("Simulation reset complete. {} Hosts, {} Initial VMs created. {} Cloudlets queued.",
                hostList.size(), vmPool.size(), broker.getWaitingCloudletCount());
    }

    /**
     * Loads cloudlet descriptors using the appropriate reader based on settings.
     */
    private List<CloudletDescriptor> loadCloudletDescriptors() {
        String filePath = settings.getCloudletTraceFile();
        String mode = settings.getWorkloadMode();
        int mips = settings.getWorkloadReaderMips(); // Needed for SWF
        int maxLines = settings.getMaxCloudletsToCreateFromWorkloadFile();

        LOGGER.info("Loading cloudlet descriptors trace from file: {} using MIPS reference: {}", filePath, mips);
        try {
            WorkloadFileReader reader = new WorkloadFileReader(filePath, mode, mips);
            reader.setMaxLinesToRead(maxLines);
            return reader.generateDescriptors();
        } catch (RuntimeException e) {
            LOGGER.error("Fatal error reading workload file {}: {}", filePath, e.getMessage(), e);
            // Rethrow or handle as appropriate (e.g., return empty list if non-fatal)
            throw new RuntimeException("Failed to load workload from " + filePath, e);
        }
    }

    /**
     * Splits CloudletDescriptors requiring more PEs than maxCloudletPes into
     * multiple smaller descriptors.
     *
     * @param originalDescriptors The initial list of descriptors.
     * @param maxCloudletPes      The maximum number of PEs allowed per resulting
     *                            descriptor.
     * @return A new list containing the original descriptors (if PEs <=
     *         maxCloudletPes) and potentially
     *         multiple split descriptors for larger cloudlets.
     */
    private List<CloudletDescriptor> splitLargeCloudletDescriptors(final List<CloudletDescriptor> originalDescriptors,
            final int maxCloudletPes) {
        List<CloudletDescriptor> splitList = new ArrayList<>();
        // Use a high starting point for split IDs to avoid collision with original IDs
        int nextSplitId = originalDescriptors.stream().mapToInt(CloudletDescriptor::getCloudletId).max().orElse(0)
                + 1000000;

        for (CloudletDescriptor originalDesc : originalDescriptors) {
            int originalPes = originalDesc.getNumberOfCores();

            if (originalPes <= maxCloudletPes) {
                // No need to split, add the original directly
                splitList.add(originalDesc);
            } else {
                // Need to split
                long totalMi = originalDesc.getMi();
                int remainingPes = originalPes;
                long miPerOriginalPe = (totalMi > 0 && originalPes > 0) ? totalMi / originalPes : 1;

                LOGGER.debug("Splitting Cloudlet ID {}: PEs={}, MI={}, maxCloudletPes={}",
                        originalDesc.getCloudletId(), originalPes, totalMi, maxCloudletPes);

                while (remainingPes > 0) {
                    int pesForThisSplit = Math.min(remainingPes, maxCloudletPes);
                    // Distribute MI proportionally to the PEs allocated to this split part
                    long miForThisSplit = Math.max(1, miPerOriginalPe * pesForThisSplit);

                    CloudletDescriptor splitDesc = new CloudletDescriptor(
                            nextSplitId++,
                            originalDesc.getSubmissionDelay(),
                            miForThisSplit,
                            pesForThisSplit);

                    splitList.add(splitDesc);
                    remainingPes -= pesForThisSplit;

                    LOGGER.trace("  -> Created split part: ID={}, PEs={}, MI={}",
                            splitDesc.getCloudletId(), pesForThisSplit, miForThisSplit);
                }
            }
        }
        return splitList;
    }

    /**
     * Ensures that all cloudlets are completed before the simulation ends.
     * This method sets up an event listener that checks if there are unfinished
     * cloudlets when there is only one future event left.
     * If there are unfinished cloudlets, it sends an empty event
     * to keep the simulation running.
     */
    private void ensureAllCloudletsCompleteBeforeSimulationEnds() {
        double interval = settings.getSimulationTimestep();
        simulation.addOnEventProcessingListener(info -> {
            if (getNumberOfFutureEvents() == 1 && broker.hasUnfinishedCloudlets()) {
                LOGGER.trace("Cloudlets not finished. Sending empty event to keep simulation running.");
                simulation.send(datacenter, datacenter, interval, CloudSimTag.NONE, null);
            }
        });
    }

    /**
     * Advances the simulation clock to the specified target time. This method runs
     * the simulation
     * in increments until the target time is reached or the maximum number of
     * iterations is
     * exceeded to prevent an infinite loop.
     *
     * @param targetTime The target time to advance the simulation clock to.
     */
    private void proceedClockTo(final double targetTime) {
        if (simulation == null) {
            throw new IllegalStateException("Simulation not initialized. Call resetSimulation first.");
        }
        if (!simulation.isRunning()) {
            LOGGER.warn("Attempting to run a simulation that is not running.");
        }

        double adjustedInterval = targetTime - getClock();
        int maxIterations = 1000; // Safety check to prevent infinite loop
        int iterations = 0;

        // Run the simulation until the target time is reached
        while (simulation.runFor(adjustedInterval) < targetTime) {
            // Calculate the remaining time to the target
            adjustedInterval = targetTime - getClock();
            // Use the minimum time between events if the remaining time is non-positive
            adjustedInterval = adjustedInterval <= 0 ? settings.getMinTimeBetweenEvents() : adjustedInterval;

            // Increment the iteration counter and break if it exceeds the maximum allowed
            // iterations
            if (++iterations >= maxIterations) {
                LOGGER.warn(
                        "Exceeded maximum iterations in runForInternal. Breaking the loop to prevent infinite loop.");
                break;
            }
        }
    }

    double calculateTargetTime() {
        // if first step we have already done 0.1 and we need to finish the first step
        // at 1 else, just add 1 to the current time
        final double targetTime;
        if (firstStep) {
            targetTime = settings.getSimulationTimestep();
        } else {
            targetTime = getClock() + settings.getSimulationTimestep();
        }
        return targetTime;
    }

    public void runOneTimestep() {
        final double oldClock = getClock();
        final double startTime = Math.max(getClock() - settings.getSimulationTimestep(), 0);
        final double targetTime = calculateTargetTime();
        ensureSimulationIsRunning();
        broker.clearFinishedWaitTimes(); // Clear finished wait times for the next timestep
        clearListsIfNeeded();
        List<Cloudlet> cloudletsToQueueList = broker.getCloudletsToQueueAtThisTimestep(targetTime, startTime,
                this.getClock());
        proceedClockTo(targetTime);
        final double startTimeNew = getClock() - settings.getSimulationTimestep();
        broker.queueCloudletsAtThisTimestep(cloudletsToQueueList, startTimeNew, getClock());
        if (shouldPrintStats()) {
            printStats();
        }
        if (firstStep) {
            firstStep = false;
        }
        LOGGER.info("VMs running: {}", broker.getVmExecList().size());
        LOGGER.info("{}: Proceeding clock to {}", oldClock, targetTime);
    }

    /**
     * Ensures that the simulation is currently running. If the simulation is not
     * running, it throws an IllegalStateException.
     *
     * @throws IllegalStateException if the simulation is not running.
     */
    private void ensureSimulationIsRunning() {
        if (!isRunning()) {
            throw new IllegalStateException(
                    "Simulation is not running. Please reset or create a new one!");
        }
    }

    /**
     * Clears the lists of created cloudlets and VMs if needed to prevent
     * OutOfMemoryError (OOM).
     * The lists can grow to large sizes as cloudlets are re-scheduled when VMs are
     * terminated. This
     * method checks a setting to determine if the lists should be cleared. It is
     * safe to clear
     * these lists in the current environment because they are only used in
     * CloudSimPlus when a VM
     * is being upscaled, which is not performed in this environment.
     */
    private void clearListsIfNeeded() {
        if (settings.isClearCreatedLists()) {
            broker.getCloudletCreatedList().clear();
            // broker.getVmCreatedList().clear();
        }
    }

    private boolean shouldPrintStats() {
        return ((int) Math.round(getClock()) % 1 == 0) || !isRunning();
    }

    /**
     * Logs various statistics about the cloudlets and VMs in the simulation.
     * <ul>
     * <li>Total number of VMs created.</li>
     * <li>Number of VMs currently running.</li>
     * <li>Total number of cloudlets.</li>
     * <li>Count of cloudlets by their status.</li>
     * </ul>
     */
    public void printStats() {
        // Contradictory to the previous functions which are called before the clock is
        // procceded,
        // this function is called after the clock is procceded. So, we need to
        // calculate the start
        // time of the timestep (instead of the target time) to get the correct
        // statistics.
        final double startTime = getClock() - settings.getSimulationTimestep();

        LOGGER.info("[{} - {}]: All Cloudlets: {} ", startTime, getClock(), cloudletList.size());
        Map<Cloudlet.Status, Integer> countByStatus = new HashMap<>();
        for (Cloudlet c : cloudletList) {
            final Cloudlet.Status status = c.getStatus();
            int count = countByStatus.getOrDefault(status, 0);
            countByStatus.put(status, count + 1);
        }

        for (Map.Entry<Cloudlet.Status, Integer> e : countByStatus.entrySet()) {
            LOGGER.info("[{} - {}]: {}: {}", startTime,
                    getClock(), e.getKey().toString(),
                    e.getValue());
        }

        LOGGER.info("[{} - {}]: Arrived Cloudlets: {}", startTime, getClock(), getArrivedCloudletsCount());
    }

    /** Stops the simulation. */
    public void stopSimulation() {
        if (simulation != null && this.isRunning()) {
            simulation.terminate();
            LOGGER.info("CloudSim simulation terminated.");
        }
    }

    /** Checks if the simulation is considered active. */
    public boolean isRunning() {
        if (simulation == null || !simulation.isRunning()) {
            return false;
        }
        boolean waiting = broker != null && broker.hasUnfinishedCloudlets();
        return waiting && simulation.isRunning();
    }

    // This now returns the potentially dynamic list of VMs known to the broker
    public List<Vm> getVmPool() {
        return broker != null ? broker.getVmExecList() : new ArrayList<>();
    }

    public double getClock() {
        return simulation != null ? simulation.clock() : 0;
    }

    public int getCloudletTotalCount() {
        return this.cloudletList != null ? cloudletList.size() : 0;
    }

    public long getNumberOfFutureEvents() {
        return simulation.getNumberOfFutureEvents(simEvent -> true);
    }

    public long getAllocatedCores() {
        final long reduce = broker.getVmExecList().parallelStream().map(Vm::getPesNumber)
                .reduce(Long::sum).orElse(0L);
        return reduce;
    }

    public long getTotalHostCores() {
        return this.getDatacenter().getHostList().stream()
                .mapToLong(Host::getPesNumber).sum();
    }

    public long getArrivedCloudletsCount() {
        return broker.getCloudletArrivalTimeMap().entrySet().parallelStream()
                .filter(entry -> entry.getValue() <= getClock()).count();
    }

    public long getNotYetRunningCloudletsCount() {
        return broker.getInputCloudlets().parallelStream()
                .filter(cloudlet -> broker.getCloudletArrivalTimeMap().get(cloudlet.getId()) <= getClock())
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.INEXEC))
                .filter(cloudlet -> !cloudlet.getStatus().equals(Cloudlet.Status.SUCCESS)).count();
    }

    // --- Methods for dynamic VM management (to be called by Gateway) ---

    /** Requests the creation of a VM on a specific host. */
    public boolean createVmOnHost(String type, long hostId) {
        if (simulation == null || broker == null) {
            LOGGER.error("Cannot create VM, simulation or broker not initialized.");
            return false;
        }
        LOGGER.debug("{}: Agent Requesting creation of {} VM on Host {}", getSimulation().clockStr(), type, hostId);

        final Host host = datacenter.getHostById(hostId);

        if (host == Host.NULL) {
            LOGGER.debug("Vm creating ignored, no host with given id found");
            return false;
        }

        Vm newVm = DatacenterSetup.createVm(settings, type); // Uses the static method now
        newVm.setDescription(type + "-" + hostId); // Temporary description for placement policy
        newVm.setBroker(broker); // Associate VM with broker

        if (!host.isSuitableForVm(newVm)) {
            LOGGER.debug("Vm creating ignored, host not suitable");
            LOGGER.debug("Host Vm List Size: {}", host.getVmList().size());
            LOGGER.debug("Total Vm Exec List Size: {}", broker.getVmExecList().size());
            return false;
        }

        broker.submitVm(newVm); // Submit to broker
        return true; // Request submitted
    }

    /** Requests the destruction of a VM by its ID. */
    public boolean destroyVmById(int vmId) {
        if (simulation == null || broker == null) {
            LOGGER.error("Cannot destroy VM, simulation or broker not initialized.");
            return false;
        }
        LOGGER.debug("{}: Agent Requesting destruction of VM {}", getSimulation().clockStr(), vmId);

        if (vmId >= broker.getVmExecList().size()) {
            LOGGER.warn("Can't kill VM with id {}: no such id found", vmId);
            return false;
        }

        Vm vmToDestroy = broker.getVmExecList().get(vmId);
        if (vmToDestroy == Vm.NULL) {
            LOGGER.warn("Cannot destroy VM {}: Not found in broker's list.", vmId);
            return false;
        }
        if (!vmToDestroy.isCreated() || vmToDestroy.isFailed()) {
            LOGGER.warn("Cannot destroy VM {}: Not running or already failed (State: {}).", vmId,
                    vmToDestroy.getStateHistory());
            return false;
        }

        // Reschedule cloudlets first
        broker.rescheduleFailedCloudlets(vmToDestroy);

        // Send destruction request (happens after delay)
        LOGGER.info("Destroying VM {} with delay {}", vmId, settings.getVmShutdownDelay());
        // Deallocate VM from host (this internally calls Host.destroyVm())
        datacenter.getVmAllocationPolicy().deallocateHostForVm(vmToDestroy);

        return true; // Destruction initiated
    }

    public int getVmPoolSize() {
        return vmPool.size();
    }
}
