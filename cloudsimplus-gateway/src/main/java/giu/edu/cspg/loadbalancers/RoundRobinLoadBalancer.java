package giu.edu.cspg.loadbalancers;

import java.util.List;
import java.util.Map;

import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.singledc.LoadBalancingBroker;
import giu.edu.cspg.singledc.SimulationCore;
import giu.edu.cspg.common.SimulationSettings;
import giu.edu.cspg.utils.ConfigLoader;
import giu.edu.cspg.utils.SimulationResultUtils;

public class RoundRobinLoadBalancer {
    private static final Logger LOGGER = LoggerFactory.getLogger(RoundRobinLoadBalancer.class.getSimpleName());
    private static final String CONFIG_FILE = "config.yml"; // Or load from args/env var
    private static final String EXPERIMENT_ID = "experiment_3"; // Or load from args/env var

    private int roundRobinVmIndex = -1; // Start before the first VM

    public static void main(String[] args) {
        new RoundRobinLoadBalancer().run(); // Create instance to manage index state
    }

    public void run() {
        LOGGER.info("Starting RoundRobinLoadBalancer Experiment...");

        // 1. Load Configuration
        Map<String, Object> params = ConfigLoader.loadConfig(CONFIG_FILE, EXPERIMENT_ID);
        if (params.isEmpty()) {
            LOGGER.error("Failed to load configuration. Exiting.");
            return;
        }
        SimulationSettings settings = new SimulationSettings(params);
        LOGGER.info("Simulation settings dump\n{}", settings.printSettings());

        // 2. Create Simulation Core
        SimulationCore simulationCore = new SimulationCore(settings);
        LoadBalancingBroker broker = simulationCore.getBroker();

        // 3. Main Simulation Loop
        LOGGER.info("Starting simulation loop...");
        int step = 0;
        while (step < settings.getMaxEpisodeLength() && simulationCore.isRunning()) {
            step++;
            LOGGER.debug("--- Step {} (Clock: {}) ---", step, String.format("%.2f", simulationCore.getClock()));

            // Assign waiting cloudlets using Round Robin logic
            while (broker.hasWaitingCloudlets()) {
                List<Vm> runningVms = broker.getVmExecList();
                if (!runningVms.isEmpty()) {
                    // Increment index and wrap around
                    roundRobinVmIndex = (roundRobinVmIndex + 1) % runningVms.size();
                    Vm targetVm = runningVms.get(roundRobinVmIndex);
                    LOGGER.debug("RoundRobin: Attempting to assign next cloudlet to VM {} (Index {})", targetVm.getId(),
                            roundRobinVmIndex);
                    boolean assigned = broker.assignCloudletToVm((int) targetVm.getId());
                    if (!assigned) {
                        LOGGER.warn(
                                "RoundRobin assignment failed for VM {}. May be unsuitable. Trying next VM if available.",
                                targetVm.getId());
                        // If assignment fails, maybe try the *next* RR index immediately?
                        // Or just break and let the next step handle it? Breaking is simpler.
                        break;
                    }
                } else {
                    LOGGER.warn("RoundRobin LB: No running VMs available to assign cloudlet. Queue size: {}",
                            broker.getWaitingCloudletCount());
                    break; // Exit inner loop if no VMs
                }
            }

            // Advance simulation time by one step
            simulationCore.runOneTimestep();
        }
        LOGGER.info("Simulation loop finished at clock: {}", String.format("%.2f", simulationCore.getClock()));

        // 4. Print and Save Results
        SimulationResultUtils.printAndSaveResults(simulationCore, settings.getSimulationName() + "_RoundRobin");

        LOGGER.info("RoundRobinLoadBalancer Experiment Finished.");
    }
}
