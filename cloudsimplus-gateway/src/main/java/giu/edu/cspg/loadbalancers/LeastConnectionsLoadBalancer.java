package giu.edu.cspg.loadbalancers;

import java.util.List;
import java.util.Map;

import org.cloudsimplus.distributions.ContinuousDistribution;
import org.cloudsimplus.distributions.UniformDistr;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.singledc.LoadBalancingBroker;
import giu.edu.cspg.singledc.SimulationCore;
import giu.edu.cspg.common.SimulationSettings;
import giu.edu.cspg.utils.ConfigLoader;
import giu.edu.cspg.utils.SimulationResultUtils;

public class LeastConnectionsLoadBalancer {
    private static final Logger LOGGER = LoggerFactory.getLogger(LeastConnectionsLoadBalancer.class.getSimpleName());
    private static final String CONFIG_FILE = "config.yml"; // Or load from args/env var
    private static final String EXPERIMENT_ID = "experiment_4"; // Or load from args/env var
    private static int minConnections = Integer.MAX_VALUE;

    public static void main(String[] args) {
        new LeastConnectionsLoadBalancer().run();
    }

    public void run() {
        LOGGER.info("Starting LeastConnectionsLoadBalancer Experiment...");

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

        long seed = params.containsKey("seed") ? ((Number) params.get("seed")).longValue() : 1;
        ContinuousDistribution rand = new UniformDistr(0, simulationCore.getVmPoolSize(), seed);

        // 3. Main Simulation Loop
        LOGGER.info("Starting simulation loop...");
        int step = 0;
        while (step < settings.getMaxEpisodeLength() && simulationCore.isRunning()) {
            step++;
            LOGGER.debug("--- Step {} (Clock: {}) ---", step, String.format("%.2f", simulationCore.getClock()));

            // Assign waiting cloudlets using Least Connections logic
            while (broker.hasWaitingCloudlets()) {
                List<Vm> runningVms = broker.getVmExecList();
                if (!runningVms.isEmpty()) {
                    Vm leastLoadedVm = null;

                    for (Vm vm : runningVms) {
                        int activeConnections = vm.getCloudletScheduler().getCloudletList().size();
                        if (activeConnections < minConnections) {
                            minConnections = activeConnections;
                            leastLoadedVm = vm;
                        }
                    }

                    if (leastLoadedVm != null) {
                        LOGGER.debug("LeastConnections: Assigning cloudlet to VM {} with {} active cloudlets",
                                leastLoadedVm.getId(), minConnections);
                        boolean assigned = broker.assignCloudletToVm((int) leastLoadedVm.getId());
                        if (!assigned) {
                            LOGGER.warn("Assignment failed for VM {}. Skipping to next step.",
                                    leastLoadedVm.getId());
                            break; // Avoid infinite loop
                        }
                    } else {
                        // Randomly assign to any VM if all are busy and no least loaded VM found
                        int randomIndex = (int) rand.sample();
                        Vm randomVm = runningVms.get(randomIndex);
                        LOGGER.debug("LeastConnections: No least loaded VM found. Assigning to random VM {}",
                                randomVm.getId());
                        boolean assigned = broker.assignCloudletToVm((int) randomVm.getId());
                        if (!assigned) {
                            LOGGER.warn("Random assignment failed for VM {}. Skipping to next step.",
                                    randomVm.getId());
                            break; // Avoid infinite loop
                        }
                    }
                } else {
                    LOGGER.warn("LeastConnections LB: No running VMs available to assign cloudlet. Queue size: {}",
                            broker.getWaitingCloudletCount());
                    break;
                }
            }

            // Advance simulation time by one step
            simulationCore.runOneTimestep();
        }

        LOGGER.info("Simulation loop finished at clock: {}", String.format("%.2f", simulationCore.getClock()));

        // 4. Print and Save Results
        SimulationResultUtils.printAndSaveResults(simulationCore, settings.getSimulationName() + "_LeastConnections");

        LOGGER.info("LeastConnectionsLoadBalancer Experiment Finished.");
    }
}
