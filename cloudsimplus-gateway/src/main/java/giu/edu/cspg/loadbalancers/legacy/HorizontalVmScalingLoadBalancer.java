package giu.edu.cspg.loadbalancers.legacy;

import java.util.ArrayList;
import java.util.Comparator;
import static java.util.Comparator.comparingLong;
import java.util.List;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.autoscaling.HorizontalVmScalingSimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.builders.tables.CsvTable;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.distributions.ContinuousDistribution;
import org.cloudsimplus.distributions.UniformDistr;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.listeners.EventInfo;
import org.cloudsimplus.provisioners.PeProvisionerSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmCost;
import org.cloudsimplus.vms.VmSimple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

public class HorizontalVmScalingLoadBalancer {

    private static final Logger logger = LoggerFactory.getLogger(HorizontalVmScalingLoadBalancer.class.getSimpleName());
    private static final Config defaultConfig = ConfigFactory.parseResources("legacy/defaults.conf");

    private static final double COST_PER_BW = defaultConfig.getDouble("datacenter.cost_per_bw");
    private static final double COST_PER_SEC = defaultConfig.getDouble("datacenter.cost_per_sec");

    private static final int NUMBER_OF_VMS = defaultConfig.getInt("vm.number");
    private static final int VM_PES = defaultConfig.getInt("vm.pes");
    private static final int VM_RAM = defaultConfig.getInt("vm.ram");
    private static final int VM_BW = defaultConfig.getInt("vm.bw");
    private static final int VM_STORAGE = defaultConfig.getInt("vm.storage");
    private static final int VM_MIPS = defaultConfig.getInt("vm.mips");

    private static final int NUMBER_OF_HOSTS = defaultConfig.getInt("hosts.number");
    private static final int HOST_MIPS = defaultConfig.getInt("hosts.mips");
    private static final int HOST_PES = defaultConfig.getInt("hosts.pes");
    private static final int HOST_RAM = defaultConfig.getInt("hosts.ram");
    private static final int HOST_STORAGE = defaultConfig.getInt("hosts.storage");
    private static final int HOST_BW = defaultConfig.getInt("hosts.bw");

    private static final int NUMBER_OF_CLOUDLETS = defaultConfig.getInt("cloudlet.number");
    private static final int CLOUDLET_PES = defaultConfig.getInt("cloudlet.pes");
    private static final long CLOUDLET_FILE_SIZE = defaultConfig.getLong("cloudlet.file_size");
    private static final long CLOUDLET_OUTPUT_SIZE = defaultConfig.getLong("cloudlet.op_size");
    private static final int[] CLOUDLET_LENGTHS = new int[] { 2000, 10000, 30000, 16000, 4000, 2000, 20000 };

    private static final int DYNAMIC_CLOUDLETS_AT_A_TIME = defaultConfig.getInt("cloudlet.number_of_dynamic_cloudlets");
    private static final int INITIAL_CLOUDLETS = defaultConfig.getInt("cloudlet.initial_cloudlets");

    private static final int SCHEDULING_INTERVAL = defaultConfig.getInt("simulation.scheduling_interval");
    private static final int CLOUDLETS_CREATION_INTERVAL = SCHEDULING_INTERVAL * 2;
    private static final double TIME_TO_TERMINATE_SIMULATION = defaultConfig.getDouble("simulation.time_to_terminate");

    // Variables for generating unique IDs
    private int createVms = 0;
    private int cloudletId = 0;

    // Simulation objects
    private final CloudSimPlus simulation;
    @SuppressWarnings("unused")
    private final Datacenter datacenter;
    private final DatacenterBrokerSimple broker;
    private final List<Vm> vmList;
    private final List<Cloudlet> cloudletList;
    private final ContinuousDistribution rand;

    public HorizontalVmScalingLoadBalancer() {
        /*
         * You can remove the seed parameter to get a dynamic one, based on current
         * computer time.
         * With a dynamic seed you will get different results at each simulation run.
         */
        final long seed = 1;
        rand = new UniformDistr(0, CLOUDLET_LENGTHS.length, seed);

        logger.info("Starting " + this.getClass().getSimpleName());
        simulation = new CloudSimPlus();
        simulation.terminateAt(TIME_TO_TERMINATE_SIMULATION);

        // Add dynamic cloudlet creation listener.
        simulation.addOnClockTickListener(this::createCloudlets);

        // Initialize simulation components.
        datacenter = createDatacenter();
        broker = new DatacenterBrokerSimple(simulation);
        vmList = new ArrayList<>(NUMBER_OF_VMS);
        cloudletList = new ArrayList<>();

        // Set VM destruction delay (using a lambda expression).
        broker.setVmDestructionDelayFunction(vm -> defaultConfig.getDouble("simulation.vm_destruction_time"));

        // Create scalable VMs.
        vmList.addAll(createListOfScalableVms(NUMBER_OF_VMS));
        // Create initial cloudlets.
        createInitialCloudlets();

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        simulation.start();
        showSimulationResults();
        showTotalCost();
    }

    @SuppressWarnings("unused")
    public static void main(String[] args) {
        HorizontalVmScalingLoadBalancer horizontalVmScalingLoadBalancer = new HorizontalVmScalingLoadBalancer();
    }

    private void showTotalCost() {
        logger.info("Calculating total cost");
        double totalCost = 0.0;
        double totalVmCost = 0.0;
        double totalCompletionTime = 0.0;
        double totalVmCpuUtilization = 0.0;

        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        int count = finishedCloudlets.size();
        int vmCount = 0;

        for (Cloudlet cloudlet : finishedCloudlets) {
            CloudletCost cloudletCost = new CloudletCost(cloudlet);
            totalCost += cloudletCost.getTotalCost();
            totalCompletionTime += (cloudlet.getTotalExecutionTime() + cloudlet.getStartWaitTime());
        }

        for (Vm vm : broker.getVmCreatedList()) {
            if (vm.hasStarted() || vm.isFinished()) {
                if (vm.getCpuUtilizationStats().getMean() > 0) {
                    // Only consider VMs with CPU utilization > 0
                    totalVmCpuUtilization += vm.getCpuUtilizationStats().getMean() * 100.0;
                    totalVmCost += new VmCost(vm).getTotalCost();
                    vmCount++;
                }
            }
        }

        System.out
                .println("Total cost of executing " + count + " Cloudlets = $" + Math.round(totalCost * 100.0) / 100.0);
        System.out.println("Total cost of executing " + vmCount + " VMs = $" + String.format("%.2f", totalVmCost));
        if (count > 0) {
            System.out.println("Mean cost of executing " + count + " Cloudlets = $"
                    + Math.round((totalCost / count) * 100.0) / 100.0);
            System.out.println("Mean cost of executing " + vmCount + " VMs = $"
                    + String.format("%.2f", totalVmCost / vmCount));
            System.out.println("Mean actual completion time of " + count + " Cloudlets = "
                    + Math.round((totalCompletionTime / count) * 100.0) / 100.0);
            System.out.println("Mean CPU utilization of " + vmCount + " VMs = "
                    + String.format("%.2f", totalVmCpuUtilization / vmCount) + "%");
        }

        System.out.println(this.getClass().getSimpleName() + " finished!");
    }

    private void showSimulationResults() {
        logger.info("Print Simulation results");
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();

        // Sort the list before printing
        // Sort by VM ID first, then by Cloudlet Start Time, then by Cloudlet ID
        final Comparator<Cloudlet> sortByVmId = comparingLong(c -> c.getVm().getId());
        List<Cloudlet> sortedList = new ArrayList<>(finishedCloudlets); // Create a copy to sort
        sortedList.sort(sortByVmId.thenComparing(Cloudlet::getStartTime).thenComparing(Cloudlet::getId));

        new CloudletsTableBuilderWithCost(sortedList, new CsvTable()).build();
        System.out.println("Number of Actual cloudlets = " + INITIAL_CLOUDLETS + "; dynamic cloudlets = "
                + (cloudletList.size() - INITIAL_CLOUDLETS));

        // new VmsTableBuilderWithDetails(vmList, new CsvTable())
        // .build();
        // System.out.println("Number of VMs = " + vmList.size());
    }

    private Datacenter createDatacenter() {
        logger.info("Creating a datacenter");
        int numberOfHosts = NUMBER_OF_HOSTS;
        List<Host> hostList = new ArrayList<>(numberOfHosts);
        for (int i = 0; i < numberOfHosts; i++) {
            hostList.add(createHost());
        }
        Datacenter dc = new DatacenterSimple(simulation, hostList, new VmAllocationPolicySimple());
        dc.setSchedulingInterval(SCHEDULING_INTERVAL);
        dc.getCharacteristics().setCostPerBw(COST_PER_BW);
        dc.getCharacteristics().setCostPerSecond(COST_PER_SEC);

        return dc;
    }

    private Host createHost() {
        logger.info("Creating a new Host");
        List<Pe> peList = createPEs(HOST_PES, HOST_MIPS);
        Host host = new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, peList);
        host.setRamProvisioner(new ResourceProvisionerSimple())
                .setBwProvisioner(new ResourceProvisionerSimple())
                .setVmScheduler(new VmSchedulerTimeShared())
                .enableUtilizationStats();
        return host;
    }

    private List<Pe> createPEs(int numberOfPEs, int mips) {
        logger.info("Creating " + numberOfPEs + " PEs with mips " + mips);
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < numberOfPEs; i++) {
            peList.add(new PeSimple(mips, new PeProvisionerSimple()));
        }
        return peList;
    }

    private List<Vm> createListOfScalableVms(int numberOfVms) {
        logger.info("Creating " + numberOfVms + " scalable VMs");
        List<Vm> newList = new ArrayList<>(numberOfVms);
        for (int i = 0; i < numberOfVms; i++) {
            Vm vm = createVm();
            createHorizontalVmScaling(vm);
            newList.add(vm);
        }
        return newList;
    }

    private Vm createVm() {
        int id = createVms;
        logger.info("Creating VM with id " + id);
        createVms++;
        Vm vm = new VmSimple(id, VM_MIPS, VM_PES);
        vm.setRam(VM_RAM).setBw(VM_BW).setSize(VM_STORAGE).setCloudletScheduler(new CloudletSchedulerSpaceShared());
        vm.enableUtilizationStats();
        return vm;
    }

    private void createHorizontalVmScaling(Vm vm) {
        logger.info("Creating a horizontal VM scaling object for VM " + vm.getId());
        HorizontalVmScalingSimple horizontalScaling = new HorizontalVmScalingSimple();
        horizontalScaling.setVmSupplier(() -> createVm());
        horizontalScaling.setOverloadPredicate(v -> isVmOverloaded(v));
        vm.setHorizontalScaling(horizontalScaling);
    }

    private boolean isVmOverloaded(Vm vm) {
        return vm.getCpuPercentUtilization() > 0.7;
    }

    private void createInitialCloudlets() {
        logger.info("Creating initial cloudlets");
        for (int i = 0; i < INITIAL_CLOUDLETS; i++) {
            cloudletList.add(createCloudlet());
        }
    }

    /**
     * This method is used as a clock-tick listener for dynamic cloudlet creation.
     *
     * @param eventInfo the event information.
     */
    public void createCloudlets(EventInfo eventInfo) {
        logger.info("Creating new cloudlet dynamically at runtime");
        long time = (long) eventInfo.getTime();
        if (time % CLOUDLETS_CREATION_INTERVAL == 0 && time <= TIME_TO_TERMINATE_SIMULATION) {
            int numberOfCloudlets = DYNAMIC_CLOUDLETS_AT_A_TIME;
            ArrayList<Cloudlet> newCloudletList = new ArrayList<>(numberOfCloudlets);
            for (int i = 0; i < numberOfCloudlets; i++) {
                if (cloudletList.size() < NUMBER_OF_CLOUDLETS) {
                    System.out.printf("\t#Creating %d Cloudlets at time %d.\n", numberOfCloudlets, time);
                    Cloudlet cloudlet = createCloudlet();
                    cloudletList.add(cloudlet);
                    newCloudletList.add(cloudlet);
                }
            }
            broker.submitCloudletList(newCloudletList);
        }
    }

    /**
     * Creates a Cloudlet with a unique ID and assigns parameters.
     *
     * @return the newly created Cloudlet.
     */
    public Cloudlet createCloudlet() {
        logger.info("Creating cloudlet");
        int id = cloudletId;
        cloudletId++;
        int length = CLOUDLET_LENGTHS[(int) rand.sample()];
        Cloudlet cloudlet = new CloudletSimple(id, length, CLOUDLET_PES);
        cloudlet
                .setFileSize(CLOUDLET_FILE_SIZE)
                .setOutputSize(CLOUDLET_OUTPUT_SIZE)
                .setUtilizationModel(new UtilizationModelFull());
        return cloudlet;
    }
}
