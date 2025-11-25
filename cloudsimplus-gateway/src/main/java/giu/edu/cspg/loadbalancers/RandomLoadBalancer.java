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

public class RandomLoadBalancer {
    private static final Logger LOGGER = LoggerFactory.getLogger(RandomLoadBalancer.class.getSimpleName());
    private static final String CONFIG_FILE = "config.yml"; // Or load from args/env var
    private static final String EXPERIMENT_ID = "experiment_2"; // Or load from args/env var

    public static void main(String[] args) {
        LOGGER.info("Starting RandomLoadBalancer Experiment...");

        // 1. Load Configuration using ConfigLoader
        Map<String, Object> params = ConfigLoader.loadConfig(CONFIG_FILE, EXPERIMENT_ID);
        if (params.isEmpty()) {
            LOGGER.error("Failed to load configuration. Exiting.");
            return;
        }
        SimulationSettings settings = new SimulationSettings(params);
        LOGGER.info("Simulation settings dump\n{}", settings.printSettings());

        // 2. Create Simulation Core (Handles setup based on settings)
        SimulationCore simulationCore = new SimulationCore(settings);
        LoadBalancingBroker broker = simulationCore.getBroker();

        long seed = params.containsKey("seed") ? ((Number) params.get("seed")).longValue() : 1;
        ContinuousDistribution rand = new UniformDistr(0, simulationCore.getVmPoolSize(), seed);

        // 3. Main Simulation Loop (Mimicking Agent Interaction)
        LOGGER.info("Starting simulation loop...");
        int step = 0;
        while (step < settings.getMaxEpisodeLength() && simulationCore.isRunning()) {
            step++;
            LOGGER.debug("--- Step {} (Clock: {}) ---", step, String.format("%.2f", simulationCore.getClock()));

            // Assign waiting cloudlets using Random logic
            while (broker.hasWaitingCloudlets()) {
                List<Vm> runningVms = broker.getVmExecList();
                if (!runningVms.isEmpty()) {
                    int randomIndex = (int) rand.sample();
                    Vm targetVm = runningVms.get(randomIndex);
                    LOGGER.debug("Random: Attempting to assign next cloudlet to VM {}", targetVm.getId());
                    boolean assigned = broker.assignCloudletToVm((int) targetVm.getId());
                    if (!assigned) {
                        LOGGER.warn("Random assignment failed for VM {}. May be unsuitable.", targetVm.getId());
                        // Basic handling: if assignment fails, break the inner loop for this step
                        // to avoid potentially getting stuck trying to assign to the same bad VM.
                        // A more robust approach might try another random VM or wait.
                        break;
                    }
                } else {
                    LOGGER.warn("Random LB: No running VMs available to assign cloudlet. Queue size: {}",
                            broker.getWaitingCloudletCount());
                    break; // No point looping if no VMs are available
                }
            }

            // Advance simulation time by one step
            simulationCore.runOneTimestep();
        }
        LOGGER.info("Simulation loop finished at clock: {}", String.format("%.2f", simulationCore.getClock()));

        // 4. Print and Save Results using common utility
        SimulationResultUtils.printAndSaveResults(simulationCore, settings.getSimulationName() + "_Random");

        LOGGER.info("RandomLoadBalancer Experiment Finished.");
    }
}
