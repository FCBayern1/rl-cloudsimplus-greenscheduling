package giu.edu.cspg.loadbalancers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

import org.cloudsimplus.autoscaling.HorizontalVmScalingSimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterCharacteristicsSimple;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.common.CloudletDescriptor;
import giu.edu.cspg.common.OptimizedCloudletScheduler;
import giu.edu.cspg.common.SimulationSettings;
import giu.edu.cspg.common.VmAllocationPolicyCustom;
import giu.edu.cspg.utils.ConfigLoader;
import giu.edu.cspg.utils.SimulationResultUtils;
import giu.edu.cspg.utils.WorkloadFileReader;

/**
 * An example simulating a load balancing scenario using CloudSim Plus's
 * built-in Horizontal VM Scaling mechanism.
 * It loads configuration from config.yml and uses common result reporting.
 */
public final class HorizontalVmScalingLoadBalancer {
    private static final Logger LOGGER = LoggerFactory.getLogger(HorizontalVmScalingLoadBalancer.class.getSimpleName());
    private static final String CONFIG_FILE = "config.yml"; // Or load from args/env var
    private static final String EXPERIMENT_ID = "experiment_5"; // Or load from args/env var

    private final CloudSimPlus simulation;
    private final SimulationSettings settings;
    private final Datacenter datacenter;
    private final DatacenterBrokerSimple broker;
    private final List<Vm> initialVmList;
    private List<Cloudlet> cloudletList;
    private final Map<Long, Double> cloudletArrivalTimeMap;
    private final PriorityQueue<Cloudlet> cloudletsQueue;
    private int vmIdCounter = 0;

    public static void main(String[] args) {
        new HorizontalVmScalingLoadBalancer().run();
    }

    public HorizontalVmScalingLoadBalancer() {
        LOGGER.info("Starting HorizontalVmScalingLoadBalancer Experiment...");

        // 1. Load Configuration
        Map<String, Object> params = ConfigLoader.loadConfig(CONFIG_FILE, EXPERIMENT_ID);
        if (params.isEmpty()) {
            LOGGER.error("Failed to load configuration. Exiting.");
            throw new RuntimeException("Configuration could not be loaded.");
        }
        this.settings = new SimulationSettings(params);
        LOGGER.info("Simulation settings dump\n{}", settings.printSettings());

        // 2. Setup Simulation Core Components
        this.simulation = new CloudSimPlus(0.1);
        this.simulation.terminateAt(this.settings.getMaxEpisodeLength());

        this.datacenter = createDatacenter();
        this.broker = new DatacenterBrokerSimple(simulation);
        this.broker.setVmDestructionDelay(10000);

        // 3. Create Initial VMs (that are scalable)
        this.initialVmList = createInitialVmFleet(settings);
        this.broker.submitVmList(initialVmList);

        // 4. Load Workload
        loadCloudletsFromTrace();
        this.cloudletsQueue = new PriorityQueue<>(cloudletList.size(),
                (c1, c2) -> Double.compare(c1.getSubmissionDelay(), c2.getSubmissionDelay()));
        this.cloudletsQueue.addAll(cloudletList);
        this.cloudletArrivalTimeMap = cloudletsQueue.stream()
                .collect(Collectors.toMap(Cloudlet::getId, Cloudlet::getSubmissionDelay));
        this.broker.submitCloudletList(this.cloudletList); // Submit all cloudlets at once

        LOGGER.info("Setup complete. {} initial VMs submitted. {} Cloudlets submitted.", initialVmList.size(),
                cloudletList.size());
    }

    public void run() {
        // 5. Start Simulation
        LOGGER.info("Starting simulation...");
        simulation.start();
        LOGGER.info("Simulation finished at clock: {}", String.format("%.2f", simulation.clock()));

        // 6. Print and Save Results
        // Pass the broker instance to the utility function
        SimulationResultUtils.printAndSaveResults(this.broker,
                cloudletArrivalTimeMap, this.datacenter,
                simulation.clock(),
                settings.getSimulationName() + "_HorizontalScale");

        LOGGER.info("HorizontalVmScalingLoadBalancer Experiment Finished.");
    }

    /** Creates the Datacenter based on settings. */
    private Datacenter createDatacenter() {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < settings.getHostsCount(); i++) {
            hostList.add(createHost(i));
        }
        // Use standard VmAllocationPolicyCustom - scaling decisions are made by
        // VMs/Broker
        VmAllocationPolicyCustom allocationPolicy = new VmAllocationPolicyCustom();
        Datacenter dc = new DatacenterSimple(simulation, hostList, allocationPolicy);
        dc.setCharacteristics(new DatacenterCharacteristicsSimple(0.75, 0.02, 0.001, 0.005));
        dc.setSchedulingInterval(5);
        return dc;
    }

    /** Creates a single Host based on settings. */
    private Host createHost(int index) {
        final List<Pe> peList = new ArrayList<>();
        long hostPeMips = settings.getHostPeMips();
        // Using potentially heterogeneous hosts from settings, adjusted logic if needed
        int pes = settings.getHostPes() + (index * 2);
        for (int i = 0; i < pes; i++) {
            peList.add(new PeSimple(hostPeMips, new PeProvisionerSimple()));
        }
        Host host = new HostSimple(settings.getHostRam(), settings.getHostBw(), settings.getHostStorage(), peList)
                .setRamProvisioner(new ResourceProvisionerSimple())
                .setBwProvisioner(new ResourceProvisionerSimple())
                .setVmScheduler(new VmSchedulerTimeShared());

        host.setStateHistoryEnabled(true);

        return host;
    }

    /**
     * Creates the initial fixed fleet of VMs based on counts in SimulationSettings.
     * Resets the global VM ID counter.
     *
     * @param settings Simulation settings containing initial VM counts and specs.
     * @param vmPool   The list to populate with created initial VMs.
     */
    public List<Vm> createInitialVmFleet(SimulationSettings settings) {
        List<Vm> newList = new ArrayList<>();
        vmIdCounter = 0; // Reset VM ID counter for this simulation instance

        int[] initialCounts = settings.getInitialVmCounts(); // [S, M, L] counts
        LOGGER.info("Creating initial VM fleet: S={}, M={}, L={}",
                initialCounts[0], initialCounts[1], initialCounts[2]);

        // Create Small VMs
        for (int i = 0; i < initialCounts[0]; i++) {
            Vm vm = createVmInstance(SimulationSettings.SMALL);
            createHorizontalVmScaling(vm); // Attach scaling mechanism
            newList.add(vm);
        }
        // Create Medium VMs
        for (int i = 0; i < initialCounts[1]; i++) {
            Vm vm = createVmInstance(SimulationSettings.MEDIUM);
            createHorizontalVmScaling(vm); // Attach scaling mechanism
            newList.add(vm);
        }
        // Create Large VMs
        for (int i = 0; i < initialCounts[2]; i++) {
            Vm vm = createVmInstance(SimulationSettings.LARGE);
            createHorizontalVmScaling(vm); // Attach scaling mechanism
            newList.add(vm);
        }

        return newList;
    }

    /** Creates a single VM instance of a specific type. */
    private Vm createVmInstance(String type) {
        int multiplier = settings.getSizeMultiplier(type);
        int vmPes = settings.getSmallVmPes() * multiplier;
        long vmRam = settings.getSmallVmRam() * multiplier;
        long vmBw = settings.getSmallVmBw();
        long vmStorage = settings.getSmallVmStorage();

        Vm vm = new VmSimple(vmIdCounter++, settings.getHostPeMips(), vmPes)
                .setRam(vmRam).setBw(vmBw).setSize(vmStorage)
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                .setDescription(type);

        vm.setSubmissionDelay(settings.getVmStartupDelay());
        vm.setShutDownDelay(settings.getVmShutdownDelay());
        vm.enableUtilizationStats();

        LOGGER.trace("Created VM instance: ID={}, Type={}, PEs={}", vm.getId(), type, vmPes);
        return vm;
    }

    /** Attaches the Horizontal Scaling mechanism to a VM. */
    private void createHorizontalVmScaling(final Vm vm) {
        HorizontalVmScalingSimple horizontalScaling = new HorizontalVmScalingSimple();

        // Supplier provides *new* VMs when scaling up
        horizontalScaling.setVmSupplier(() -> createVmInstance(vm.getDescription()));

        // Predicate checks if the VM is overloaded (e.g., >70% CPU) OR if there are
        // queued cloudlets due to PE limits
        final double overloadThreshold = 0.7;
        horizontalScaling.setOverloadPredicate(v -> v.getCpuPercentUtilization() > overloadThreshold ||
                !v.getCloudletScheduler().getCloudletWaitingList().isEmpty());

        vm.setHorizontalScaling(horizontalScaling);
        LOGGER.trace("Attached Horizontal Scaling to VM {}", vm.getId());
    }

    /** Loads cloudlets using the appropriate reader based on settings. */
    private void loadCloudletsFromTrace() {
        String filePath = settings.getCloudletTraceFile();
        String mode = settings.getWorkloadMode();
        int mips = settings.getWorkloadReaderMips();
        int maxLines = settings.getMaxCloudletsToCreateFromWorkloadFile();

        WorkloadFileReader reader = new WorkloadFileReader(filePath, mode, mips);
        reader.setMaxLinesToRead(maxLines);

        LOGGER.info("Loading workload descriptors (Mode: {})...", mode);
        List<CloudletDescriptor> descriptors = reader.generateDescriptors();

        descriptors = splitLargeCloudletDescriptors(descriptors, settings.getMaxCloudletPes());

        // Convert descriptors to Cloudlets
        this.cloudletList = descriptors.stream()
                .map(CloudletDescriptor::toCloudlet)
                .filter(Objects::nonNull)
                .collect(Collectors.toList());

        LOGGER.info("Loaded and created {} cloudlets from trace.", this.cloudletList.size());
    }

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

}
