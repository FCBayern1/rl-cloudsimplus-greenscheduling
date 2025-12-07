package giu.edu.cspg.common;
import giu.edu.cspg.common.HostProfile;
import giu.edu.cspg.common.SimulationSettings;
import giu.edu.cspg.common.DatacenterConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerSpaceShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DatacenterSetup {
    private static final Logger LOGGER = LoggerFactory.getLogger(DatacenterSetup.class.getSimpleName());
    private static int vmIdCounter = 0; // Counter to ensure unique VM IDs across resets if needed

    /**
     * Reset the VM ID counter to 0.
     * Should be called at the beginning of each simulation reset.
     */
    public static void resetVmIdCounter() {
        vmIdCounter = 0;
        LOGGER.debug("VM ID counter reset to 0");
    }

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

        if (settings.isEnableHeterogeneousHosts()) {
            // Create heterogeneous hosts based on real server profiles (SPEC power_ssj2008)
            LOGGER.info("Creating heterogeneous datacenter with mixed server types...");

            // Create Low-Power hosts (HP ProLiant DL360 G7: 58W idle, 208W peak)
            for (int i = 0; i < settings.getHostCountLowPower(); i++) {
                hostList.add(createHost(HostProfile.LOW_POWER()));
            }

            // Create Medium hosts (Dell PowerEdge R720: 98W idle, 345W peak)
            for (int i = 0; i < settings.getHostCountMedium(); i++) {
                hostList.add(createHost(HostProfile.MEDIUM()));
            }

            // Create High-Performance hosts (Cisco UCS C240 M4: 142W idle, 476W peak)
            for (int i = 0; i < settings.getHostCountHighPerf(); i++) {
                hostList.add(createHost(HostProfile.HIGH_PERFORMANCE()));
            }

            // Create Ultra hosts (HPE ProLiant DL380 Gen10: 194W idle, 634W peak)
            for (int i = 0; i < settings.getHostCountUltra(); i++) {
                hostList.add(createHost(HostProfile.ULTRA_HIGH()));
            }

            // SPEC power_ssj2008 Real Server Profiles
            // Create Acer Altos R520 hosts (Legacy: 155W idle, 269W peak - 57.6% idle)
            for (int i = 0; i < settings.getHostCountSpecAcerR520(); i++) {
                hostList.add(createHost(HostProfile.SPEC_ACER_R520()));
            }

            // Create Acer AR360 F2 hosts (Medium: 69.4W idle, 315W peak - 22.0% idle)
            for (int i = 0; i < settings.getHostCountSpecAcerAR360(); i++) {
                hostList.add(createHost(HostProfile.SPEC_ACER_AR360()));
            }

            // Create ASUSTeK RS720-E9/RS8 hosts (Modern efficient: 48.2W idle, 385W peak - 12.5% idle)
            for (int i = 0; i < settings.getHostCountSpecAsusRS720E9(); i++) {
                hostList.add(createHost(HostProfile.SPEC_ASUS_RS720_E9()));
            }

            // Create ASUSTeK RS500A-E10-PS4 hosts (Large AMD EPYC: 51.4W idle, 214W peak - 24.0% idle)
            for (int i = 0; i < settings.getHostCountSpecAsusRS500A(); i++) {
                hostList.add(createHost(HostProfile.SPEC_ASUS_RS500A()));
            }

            // Create ASUSTeK RS700A-E9-RS4V2 hosts (Ultra dual-socket: 106W idle, 430W peak - 24.7% idle)
            for (int i = 0; i < settings.getHostCountSpecAsusRS700A(); i++) {
                hostList.add(createHost(HostProfile.SPEC_ASUS_RS700A()));
            }

            LOGGER.info("Created heterogeneous datacenter: {} LowPower, {} Medium, {} HighPerf, {} Ultra = {} total hosts",
                settings.getHostCountLowPower(), settings.getHostCountMedium(),
                settings.getHostCountHighPerf(), settings.getHostCountUltra(), hostList.size());

            // Log SPEC servers if any are configured
            int specServerCount = settings.getHostCountSpecAcerR520() + settings.getHostCountSpecAcerAR360() +
                                  settings.getHostCountSpecAsusRS720E9() + settings.getHostCountSpecAsusRS500A() +
                                  settings.getHostCountSpecAsusRS700A();
            if (specServerCount > 0) {
                LOGGER.info("SPEC Servers: {} AcerR520, {} AcerAR360, {} AsusRS720, {} AsusRS500A, {} AsusRS700A",
                    settings.getHostCountSpecAcerR520(), settings.getHostCountSpecAcerAR360(),
                    settings.getHostCountSpecAsusRS720E9(), settings.getHostCountSpecAsusRS500A(),
                    settings.getHostCountSpecAsusRS700A());
            }
        } else {
            // Create homogeneous hosts (legacy mode)
            int numHosts = settings.getHostsCount();
            LOGGER.info("Creating {} homogeneous hosts...", numHosts);
            for (int i = 0; i < numHosts; i++) {
                hostList.add(createHost(settings));
            }
        }

        LOGGER.info("Creating Datacenter with {} hosts and {} allocation policy.",
                hostList.size(), vmAllocationPolicy.getClass().getSimpleName());
        // Use the provided allocation policy (which should be VmAllocationPolicyCustom)
        return new DatacenterSimple(simulation, hostList, vmAllocationPolicy)
                .setCharacteristics(new DatacenterCharacteristicsSimple(0.75, 0.02, 0.001, 0.005));
    }

    /**
     * Creates a single Host instance based on settings (homogeneous mode).
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
                .setVmScheduler(new VmSchedulerSpaceShared())
                .setStateHistoryEnabled(true);

        // Set power model for energy consumption tracking
        // PowerModelHostSimple: linear power model
        // maxPower: 250W (typical server), staticPowerPercent: 70% (idle power is 70% of max)
        host.setPowerModel(new PowerModelHostSimple(250, 70));
        host.setStateHistoryEnabled(true); // Enable for energy consumption tracking

        return host;
    }

    /**
     * Creates a single Host instance based on a specific server profile (heterogeneous mode).
     * Power consumption characteristics based on SPEC power_ssj2008 real server data.
     *
     * @param profile The server hardware profile to use
     * @return The created Host with profile-specific power model
     */
    private static Host createHost(HostProfile profile) {
        final List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < profile.getPes(); i++) {
            peList.add(new PeSimple(profile.getPeMips(), new PeProvisionerSimple()));
        }

        Host host = new HostWithoutCreatedList(profile.getRam(),
                profile.getBw(), profile.getStorage(), peList)
                .setRamProvisioner(new ResourceProvisionerSimple())
                .setBwProvisioner(new ResourceProvisionerSimple())
                .setVmScheduler(new VmSchedulerSpaceShared())
                .setStateHistoryEnabled(true);

        // Set power model based on real server characteristics
        // PowerModelHostSimple(maxPowerW, staticPowerPercent)
        host.setPowerModel(new PowerModelHostSimple(profile.getMaxPowerW(), profile.getStaticPowerPercent()));
        host.setStateHistoryEnabled(true);

        LOGGER.debug("Created host with profile: {}", profile);

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
                // Use OptimizedCloudletScheduler for better fault tolerance and accurate time estimation
                .setCloudletScheduler(new OptimizedCloudletScheduler())
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

    // ====================================================================
    // Multi-Datacenter Support: Methods for DatacenterConfig
    // ====================================================================

    /**
     * Create hosts for a specific datacenter based on DatacenterConfig.
     * Supports heterogeneous host composition via Map<String, Integer>.
     *
     * @param config Datacenter configuration
     * @return List of created hosts
     */
    public static List<Host> createHostsForDatacenter(DatacenterConfig config) {
        List<Host> hostList = new ArrayList<>();

        // NEW: Support heterogeneous hosts via Map
        if (config.getHostProfiles() != null && !config.getHostProfiles().isEmpty()) {
            LOGGER.info("Creating heterogeneous hosts for datacenter {} ({})",
                    config.getDatacenterId(), config.getDatacenterName());

            // Iterate through the map: profileName -> count
            for (Map.Entry<String, Integer> entry : config.getHostProfiles().entrySet()) {
                String profileName = entry.getKey();
                int count = entry.getValue();

                // Skip if count is 0
                if (count <= 0) {
                    continue;
                }

                // Get the HostProfile
                HostProfile profile = HostProfile.fromName(profileName);

                LOGGER.info("  - Creating {} hosts with profile: {} [{}]",
                        count, profileName, profile);

                // Create hosts with this profile
                for (int i = 0; i < count; i++) {
                    hostList.add(createHost(profile));
                }
            }

            LOGGER.info("Created {} total hosts for datacenter {} across {} profile types",
                    hostList.size(), config.getDatacenterId(),
                    config.getHostProfiles().size());

        } else {
            // LEGACY: Homogeneous hosts mode
            LOGGER.info("Creating homogeneous hosts for datacenter {} (legacy mode)",
                    config.getDatacenterId());

            int count = config.getHostsCount() != null ? config.getHostsCount() : 0;

            for (int i = 0; i < count; i++) {
                // Create PEs list for this host
                List<Pe> peList = new ArrayList<>();
                for (int j = 0; j < config.getHostPes(); j++) {
                    peList.add(new PeSimple(config.getHostPeMips(), new PeProvisionerSimple()));
                }

                // Create host using HostWithoutCreatedList (memory optimized)
                Host host = new HostWithoutCreatedList(
                        config.getHostRam(),
                        config.getHostBw(),
                        config.getHostStorage(),
                        peList
                )
                .setRamProvisioner(new ResourceProvisionerSimple())
                .setBwProvisioner(new ResourceProvisionerSimple())
                .setVmScheduler(new VmSchedulerSpaceShared())
                .setStateHistoryEnabled(true);

                // Add default power model
                double maxPower = 250.0; // Watts
                double staticPowerPercent = 70;  // 70% static power
                host.setPowerModel(new PowerModelHostSimple(maxPower, staticPowerPercent));

                hostList.add(host);
            }

            LOGGER.info("Created {} hosts for datacenter {}",
                    hostList.size(), config.getDatacenterId());
        }

        return hostList;
    }

    /**
     * Create a Datacenter from DatacenterConfig.
     *
     * @param simulation CloudSimPlus instance
     * @param config Datacenter configuration
     * @param hostList List of hosts for this datacenter
     * @param vmAllocationPolicy VM allocation policy
     * @return Created Datacenter
     */
    public static Datacenter createDatacenterFromConfig(
            CloudSimPlus simulation,
            DatacenterConfig config,
            List<Host> hostList,
            VmAllocationPolicy vmAllocationPolicy) {

        // Create datacenter using CloudSim Plus API
        Datacenter datacenter = new DatacenterSimple(simulation, hostList, vmAllocationPolicy)
                .setCharacteristics(new DatacenterCharacteristicsSimple(0.75, 0.02, 0.001, 0.005));

        datacenter.setName(config.getDatacenterName());

        LOGGER.info("Datacenter created: {} (ID: {}) with {} hosts",
                config.getDatacenterName(), config.getDatacenterId(), hostList.size());

        return datacenter;
    }

    /**
     * Create VM fleet for a specific datacenter based on DatacenterConfig.
     *
     * @param config Datacenter configuration
     * @param vmPool List to populate with created VMs
     */
    public static void createVmFleetForDatacenter(DatacenterConfig config, List<Vm> vmPool) {
        vmPool.clear();

        // Create Small VMs
        for (int i = 0; i < config.getInitialSmallVmCount(); i++) {
            Vm vm = createVm(
                    vmIdCounter++,
                    config.getSmallVmPes(),
                    config.getSmallVmRam(),
                    config.getSmallVmBw(),
                    config.getSmallVmStorage()
            );
            vmPool.add(vm);
        }

        // Create Medium VMs
        int mediumPes = config.getMediumVmPes();
        long mediumRam = config.getMediumVmRam();
        for (int i = 0; i < config.getInitialMediumVmCount(); i++) {
            Vm vm = createVm(
                    vmIdCounter++,
                    mediumPes,
                    mediumRam,
                    config.getSmallVmBw() * config.getMediumVmMultiplier(),
                    config.getSmallVmStorage() * config.getMediumVmMultiplier()
            );
            vmPool.add(vm);
        }

        // Create Large VMs
        int largePes = config.getLargeVmPes();
        long largeRam = config.getLargeVmRam();
        for (int i = 0; i < config.getInitialLargeVmCount(); i++) {
            Vm vm = createVm(
                    vmIdCounter++,
                    largePes,
                    largeRam,
                    config.getSmallVmBw() * config.getLargeVmMultiplier(),
                    config.getSmallVmStorage() * config.getLargeVmMultiplier()
            );
            vmPool.add(vm);
        }

        LOGGER.info("Created {} VMs for datacenter {} (S:{}, M:{}, L:{})",
                vmPool.size(), config.getDatacenterId(),
                config.getInitialSmallVmCount(),
                config.getInitialMediumVmCount(),
                config.getInitialLargeVmCount());
    }

    /**
     * Helper method to create a single VM.
     */
    private static Vm createVm(int id, int pes, long ram, long bw, long storage) {
        Vm vm = new VmSimple(id, 1000, pes);  // 1000 MIPS per PE
        vm.setRam(ram)
                .setBw(bw)
                .setSize(storage)
                .setCloudletScheduler(new OptimizedCloudletScheduler());
        return vm;
    }
}
