package giu.edu.cspg;

import java.util.ArrayList;
import java.util.List;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterCharacteristicsSimple;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DatacenterSetup {
    private static final Logger LOGGER = LoggerFactory.getLogger(DatacenterSetup.class.getSimpleName());
    private static int vmIdCounter = 0; // Counter to ensure unique VM IDs across resets if needed

    /**
     * Creates a Datacenter with a fixed number of Hosts based on settings.
     *
     * @param simulation         The CloudSimPlus instance.
     * @param settings           Simulation settings containing Host count and
     *                           specs.
     * @param hostList           An empty list to be populated with the created
     *                           Hosts.
     * @param vmAllocationPolicy The policy to use for placing VMs onto hosts
     *                           (initial and dynamic).
     * @return The created Datacenter.
     */
    public static Datacenter createDatacenter(CloudSimPlus simulation, SimulationSettings settings, List<Host> hostList,
            VmAllocationPolicy vmAllocationPolicy) {
        hostList.clear(); // Ensure list is empty
        int numHosts = settings.getHostsCount();
        LOGGER.info("Creating {} hosts...", numHosts);
        for (int i = 0; i < numHosts; i++) {
            hostList.add(createHost(settings));
        }

        LOGGER.info("Creating Datacenter with {} hosts and {} allocation policy.",
                hostList.size(), vmAllocationPolicy.getClass().getSimpleName());
        // Use the provided allocation policy (which should be VmAllocationPolicyCustom)
        return new DatacenterSimple(simulation, hostList, vmAllocationPolicy)
                .setCharacteristics(new DatacenterCharacteristicsSimple(0.75, 0.02, 0.001, 0.005));
    }

    /**
     * Creates a single Host instance based on settings.
     */
    private static Host createHost(SimulationSettings settings) {
        final List<Pe> peList = new ArrayList<>();
        long hostPeMips = settings.getHostPeMips();
        for (int i = 0; i < settings.getHostPes(); i++) {
            peList.add(new PeSimple(hostPeMips, new PeProvisionerSimple()));
        }

        // Use HostWithoutCreatedList for potential memory optimization
        // If HostWithoutCreatedList is not defined, use HostSimple
        Host host = new HostWithoutCreatedList(settings.getHostRam(),
                settings.getHostBw(), settings.getHostStorage(), peList)
                .setRamProvisioner(new ResourceProvisionerSimple())
                .setBwProvisioner(new ResourceProvisionerSimple())
                .setVmScheduler(new VmSchedulerTimeShared()) // Simple time-shared scheduler for Host PEs
                .setStateHistoryEnabled(true);

        // Set power model for energy consumption tracking
        // PowerModelHostSimple: linear power model
        // maxPower: 250W (typical server), staticPowerPercent: 70% (idle power is 70% of max)
        host.setPowerModel(new PowerModelHostSimple(250, 70));
        host.setStateHistoryEnabled(true); // Enable for energy consumption tracking

        return host;
    }

    /**
     * Creates the initial fixed fleet of VMs based on counts in SimulationSettings.
     * Resets the global VM ID counter.
     *
     * @param settings Simulation settings containing initial VM counts and specs.
     * @param vmPool   The list to populate with created initial VMs.
     */
    public static void createInitialVmFleet(SimulationSettings settings, List<Vm> vmPool) {
        vmPool.clear(); // Ensure the list is empty
        vmIdCounter = 0; // Reset VM ID counter for this simulation instance

        int[] initialCounts = settings.getInitialVmCounts(); // [S, M, L] counts
        LOGGER.info("Creating initial VM fleet: S={}, M={}, L={}",
                initialCounts[0], initialCounts[1], initialCounts[2]);

        // Create Small VMs
        for (int i = 0; i < initialCounts[0]; i++) {
            vmPool.add(createVm(settings, SimulationSettings.SMALL));
        }
        // Create Medium VMs
        for (int i = 0; i < initialCounts[1]; i++) {
            vmPool.add(createVm(settings, SimulationSettings.MEDIUM));
        }
        // Create Large VMs
        for (int i = 0; i < initialCounts[2]; i++) {
            vmPool.add(createVm(settings, SimulationSettings.LARGE));
        }

        LOGGER.info("Created initial VM pool with {} VMs.", vmPool.size());
    }

    /**
     * Creates a single VM instance of a specific type (S, M, L).
     * Uses a shared counter for unique IDs.
     *
     * @param settings Simulation settings containing base VM specs and multipliers.
     * @param type     The type of VM ("S", "M", or "L").
     * @return The created Vm object.
     */
    public static Vm createVm(SimulationSettings settings, String type) {
        int multiplier = settings.getSizeMultiplier(type);
        int vmPes = settings.getSmallVmPes() * multiplier; // decides it is a S/M/L VM
        long vmRam = settings.getSmallVmRam() * multiplier;
        long vmBw = settings.getSmallVmBw(); // Keep BW same for simplicity, or scale: * multiplier;
        long vmStorage = settings.getSmallVmStorage(); // Keep Storage same, or scale: * multiplier;

        // Create VM with unique ID and scaled resources
        Vm vm = new VmSimple(vmIdCounter++, settings.getHostPeMips(), vmPes)
                .setRam(vmRam)
                .setBw(vmBw)
                .setSize(vmStorage)
                // Use OptimizedCloudletScheduler if available & desired
                .setCloudletScheduler(new OptimizedCloudletScheduler())
                // .setCloudletScheduler(new CloudletSchedulerTimeShared())
                .setDescription(type); // Set description to S, M, or L initially

        // Set startup/shutdown delays specified in settings
        vm.setSubmissionDelay(settings.getVmStartupDelay());
        vm.setShutDownDelay(settings.getVmShutdownDelay());

        vm.enableUtilizationStats();

        LOGGER.trace("Created VM ID {} Type {} (PEs={}, RAM={}, BW={}, Storage={})",
                vm.getId(), type, vmPes, vmRam, vmBw, vmStorage);
        return vm;
    }

    public int getLastCreatedVmId() {
        return vmIdCounter - 1;
    }
}
