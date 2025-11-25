package giu.edu.cspg.loadbalancers.legacy;

import java.util.ArrayList;
import java.util.Comparator;
import static java.util.Comparator.comparingLong;
import java.util.List;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.builders.tables.CsvTable;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.distributions.UniformDistr;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.network.NetworkHost;
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

/**
 * Abstract class that defines the architecture.
 * <p>
 * This class replicates the functionality from the Scala version by
 * merging the simulation bootstrapping, datacenter creation, VM and cloudlet
 * submission,
 * dynamic arrival of cloudlets (via clock ticks), and result reporting with
 * cost calculations.
 * The default parameters are loaded via a Typesafe configuration file
 * ("defaults.conf").
 * </p>
 */
public abstract class BaseAbstract {

    // --- Static configuration and constants loaded from defaults.conf ---
    private static final Logger logger = LoggerFactory.getLogger(BaseAbstract.class.getSimpleName());
    private static final Config defaultConfig = ConfigFactory.parseResources("legacy/defaults.conf");

    public static final int NUMBER_OF_VMS = defaultConfig.getInt("vm.number");
    public static final int VM_PES = defaultConfig.getInt("vm.pes");
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

    private static final long CLOUDLET_FILE_SIZE = defaultConfig.getLong("cloudlet.file_size");
    private static final long CLOUDLET_OUTPUT_SIZE = defaultConfig.getLong("cloudlet.op_size");
    private static final int CLOUDLET_PES = defaultConfig.getInt("cloudlet.pes");

    public static final int DYNAMIC_CLOUDLETS_AT_A_TIME = defaultConfig.getInt("cloudlet.number_of_dynamic_cloudlets");
    public static final int INITIAL_CLOUDLETS = defaultConfig.getInt("cloudlet.initial_cloudlets");
    public static final int NUMBER_OF_CLOUDLETS = defaultConfig.getInt("cloudlet.number");

    private static final int[] CLOUDLET_LENGTHS = new int[] { 2000, 10000, 30000, 16000, 4000, 2000, 20000 };
    private static final double COST_PER_BW = defaultConfig.getDouble("datacenter.cost_per_bw");
    private static final double COST_PER_SECOND = defaultConfig.getDouble("datacenter.cost_per_sec");

    private static final int SCHEDULING_INTERVAL = defaultConfig.getInt("simulation.scheduling_interval");
    private static final double RANDOM_SAMPLE = defaultConfig.getDouble("simulation.random_sample");
    private static final double TIME_TO_TERMINATE_SIMULATION = defaultConfig.getDouble("simulation.time_to_terminate");

    // --- Instance fields ---
    protected final CloudSimPlus simulation;
    protected final Datacenter datacenter;
    protected final DatacenterBroker broker;
    protected final List<Vm> vmList;
    protected List<Cloudlet> cloudletList;
    protected final UniformDistr random;
    protected static int cloudletlistsize = 0;

    /**
     * Constructor that bootstraps the simulation.
     */
    public BaseAbstract() {
        logger.info("Starting " + this.getClass().getSimpleName());

        // Create simulation and set termination time.
        simulation = new CloudSimPlus();
        simulation.terminateAt(TIME_TO_TERMINATE_SIMULATION);

        // Register a clock tick listener for dynamic cloudlet arrival.
        simulation.addOnClockTickListener(this::createRandomCloudlets);

        // Create Uniform distribution for dynamic cloudlet generation.
        /*
         * You can remove the seed parameter to get a dynamic one, based on current
         * computer time.
         * With a dynamic seed you will get different results at each simulation run.
         */
        final long seed = 1;
        random = new UniformDistr(seed);

        // Create datacenter, broker, VMs and cloudlets.
        datacenter = createDatacenter();
        broker = new DatacenterBrokerSimple(simulation);
        vmList = createAndSubmitVMs(broker);

        // Initialize cloudletList (empty for now)
        cloudletList = new ArrayList<>(NUMBER_OF_CLOUDLETS);
    }

    /**
     * Initializes the initial cloudlets, starts the simulation, and shows results.
     * This method should be called after the constructor to avoid calling the
     * overridable method createCloudlets() during construction.
     */
    public void run() {
        // Create initial set of cloudlets via abstract method.
        this.cloudletList = createCloudlets();
        broker.submitCloudletList(this.cloudletList);

        // Start the simulation.
        logger.info("Starting simulation.");
        simulation.start();

        // Show simulation results.
        showSimulationResults();
    }

    // --- Datacenter and Host creation methods ---

    /**
     * Creates the datacenter.
     *
     * @return the created Datacenter.
     */
    private Datacenter createDatacenter() {
        logger.info("Creating a datacenter");
        // Total number of hosts: NUMBER_OF_HOSTS multiplied by switch fan-out.
        int hostsNumber = NUMBER_OF_HOSTS;
        List<Host> hostList = new ArrayList<>(hostsNumber);
        for (int i = 0; i < hostsNumber; i++) {
            hostList.add(createHost(i));
        }
        // Create a datacenter with a simple VM allocation policy.
        Datacenter dc = new DatacenterSimple(simulation, hostList, new VmAllocationPolicySimple());
        dc.setSchedulingInterval(SCHEDULING_INTERVAL);
        dc.getCharacteristics().setCostPerBw(COST_PER_BW);
        dc.getCharacteristics().setCostPerSecond(COST_PER_SECOND);

        return dc;
    }

    /**
     * Creates a new Host.
     *
     * @return the created host.
     */
    private Host createHost(int hostNumber) {
        logger.info("Creating a new NetworkHost with id " + hostNumber);
        List<Pe> peList = createPEs(HOST_PES, HOST_MIPS);
        // Create host with specified RAM, BW, and storage.
        NetworkHost host = new NetworkHost(HOST_RAM, HOST_BW, HOST_STORAGE, peList);
        host.setRamProvisioner(new ResourceProvisionerSimple())
                .setBwProvisioner(new ResourceProvisionerSimple())
                .setVmScheduler(new VmSchedulerTimeShared());
        return host;
    }

    /**
     * Creates a list of Processing Elements (PEs).
     *
     * @param numberOfPEs number of PEs.
     * @param mips        MIPS for each PE.
     * @return List of created PEs.
     */
    private List<Pe> createPEs(int numberOfPEs, int mips) {
        logger.info("Creating " + numberOfPEs + " PEs with " + mips + " mips each");
        List<Pe> peList = new ArrayList<>(numberOfPEs);
        for (int i = 0; i < numberOfPEs; i++) {
            peList.add(new PeSimple(mips, new PeProvisionerSimple()));
        }
        return peList;
    }

    // --- VM creation methods ---

    /**
     * Creates and submits a list of VMs for the given broker.
     *
     * @param broker the datacenter broker.
     * @return the list of created VMs.
     */
    private List<Vm> createAndSubmitVMs(DatacenterBroker broker) {
        logger.info("Creating and submitting VMs to broker " + broker);
        List<Vm> vmListNew = new ArrayList<>();
        for (int id = 0; id < NUMBER_OF_VMS; id++) {
            vmListNew.add(createVm(id));
        }
        broker.submitVmList(vmListNew);
        return vmListNew;
    }

    /**
     * Creates a single Vm with the given id.
     *
     * @param id the VM id.
     * @return the created Vm.
     */
    private Vm createVm(int id) {
        logger.info("Creating Vm with id " + id);
        Vm vm = new VmSimple(id, VM_MIPS, VM_PES);
        vm.setRam(VM_RAM)
                .setBw(VM_BW)
                .setSize(VM_STORAGE)
                .setCloudletScheduler(new CloudletSchedulerSpaceShared());

        vm.enableUtilizationStats();

        return vm;
    }

    // --- Task creation methods for Cloudlets ---

    /**
     * Creates a Cloudlet.
     *
     * @param vm the VM that will run the created Cloudlet.
     * @return the newly created Cloudlet.
     */
    protected Cloudlet createCloudlet(Vm vm) {
        // Determine a unique id based on the current cloudletList size.
        int id = cloudletList.size();
        logger.info("Creating cloudlet with id " + id);

        // Choose a length value from the CLOUDLET_LENGTHS array in a cyclic manner.
        int rand = cloudletList.size() % CLOUDLET_LENGTHS.length;
        int length = CLOUDLET_LENGTHS[rand];

        // Create and configure the cloudlet.
        Cloudlet cloudlet = new CloudletSimple(id, length, CLOUDLET_PES);
        cloudlet
                .setFileSize(CLOUDLET_FILE_SIZE)
                .setOutputSize(CLOUDLET_OUTPUT_SIZE)
                .setUtilizationModel(new UtilizationModelFull());

        // Assign the provided VM to the cloudlet.
        cloudlet.setVm(vm);

        return cloudlet;
    }

    /**
     * Creates a list of {@link Cloudlet} that together represents the
     * distributed processes of a given fictitious application.
     * Must be implemented my the implementing class
     * 
     * @return the list of create Cloudlets
     */
    protected abstract List<Cloudlet> createCloudlets();

    // --- Dynamic Cloudlet Arrival via Clock Tick Listener ---

    /**
     * This method is registered as a clock tick listener.
     * At each simulation tick, it may create additional cloudlets based on a
     * probability.
     *
     * @param evt the event information.
     */
    protected void createRandomCloudlets(EventInfo evt) {
        if (random.sample() <= RANDOM_SAMPLE && cloudletList.size() != NUMBER_OF_CLOUDLETS) {
            logger.info("\n# Randomly creating more cloudlets at time " + evt.getTime() + "\n");
            // Submit additional dynamically created cloudlets.
            List<Cloudlet> newCloudlets = createCloudlets();
            broker.submitCloudletList(newCloudlets);
        }
    }

    // --- Simulation Results Reporting ---

    /**
     * Displays simulation results including a detailed table, cost calculations,
     * and host data transfers.
     */
    private void showSimulationResults() {
        logger.info("Printing simulation results");
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();

        // Sort the list before printing
        // Sort by VM ID first, then by Cloudlet Start Time, then by Cloudlet ID
        final Comparator<Cloudlet> sortByVmId = comparingLong(c -> c.getVm().getId());
        List<Cloudlet> sortedList = new ArrayList<>(finishedCloudlets); // Create a copy to sort
        sortedList.sort(sortByVmId.thenComparing(Cloudlet::getStartTime).thenComparing(Cloudlet::getId));

        new CloudletsTableBuilderWithCost(finishedCloudlets, new CsvTable()).build();
        System.out.println("Number of Actual cloudlets = " + INITIAL_CLOUDLETS
                + "; dynamic cloudlets = " + (cloudletList.size() - INITIAL_CLOUDLETS));

        // new VmsTableBuilderWithDetails(vmList, new CsvTable())
        // .build();
        // System.out.println("Number of VMs = " + vmList.size());

        showTotalCost(finishedCloudlets);
        System.out.println(this.getClass().getSimpleName() + " finished!");
    }

    /**
     * Calculates and prints the total and mean cost as well as the mean CPU time
     * for the finished cloudlets.
     *
     * @param finishedCloudlets list of finished cloudlets.
     */
    private void showTotalCost(List<Cloudlet> finishedCloudlets) {
        logger.info("Calculating total cost");
        double totalCost = 0.0;
        double totalVmCost = 0.0;
        double totalCompletionTime = 0.0;
        double totalVmCpuUtilization = 0.0;
        int count;
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

        count = finishedCloudlets.size();

        System.out
                .println("Total cost of executing " + count + " Cloudlets = $" + String.format("%.2f", totalCost));
        System.out.println("Total cost of executing " + vmCount + " VMs = $" + String.format("%.2f", totalVmCost));
        if (count > 0) {
            System.out.println("Mean cost of executing " + count + " Cloudlets = $"
                    + String.format("%.2f", totalCost / count));
            System.out.println("Mean cost of executing " + vmCount + " VMs = $"
                    + String.format("%.2f", totalVmCost / vmCount));
            System.out.println("Mean actual Completion time of " + count + " Cloudlets = "
                    + String.format("%.2f", totalCompletionTime / count));
            System.out.println("Mean CPU utilization of " + vmCount + " VMs = "
                    + String.format("%.2f", totalVmCpuUtilization / vmCount) + "%");
        } else {
            System.out.println("No finished Cloudlets to calculate mean costs/time.");
        }
    }
}
