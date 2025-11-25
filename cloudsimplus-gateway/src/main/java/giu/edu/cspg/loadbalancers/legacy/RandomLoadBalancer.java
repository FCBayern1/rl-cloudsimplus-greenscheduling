package giu.edu.cspg.loadbalancers.legacy;

import java.util.ArrayList;
import java.util.List;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.distributions.ContinuousDistribution;
import org.cloudsimplus.distributions.UniformDistr;
import org.cloudsimplus.vms.Vm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class creates a list of Cloudlet objects and assigns each
 * one to a random virtual machine.
 */
public class RandomLoadBalancer extends BaseAbstract {

    private static final Logger logger = LoggerFactory.getLogger(RandomLoadBalancer.class.getSimpleName());

    public static void main(String[] args) {
        RandomLoadBalancer randomLoadBalancer = new RandomLoadBalancer();
        randomLoadBalancer.run();
    }

    /**
     * Creates a list of Cloudlet objects that together represent the
     * distributed processes of a given fictitious application.
     * The cloudlets are assigned to a random VM.
     *
     * @return the list of created Cloudlets.
     */
    @Override
    protected List<Cloudlet> createCloudlets() {
        logger.info("Create list of network cloudlets and assign VM randomly");

        int numberOfCloudlets;
        if (cloudletList.isEmpty()) {
            numberOfCloudlets = INITIAL_CLOUDLETS;
        } else {
            numberOfCloudlets = DYNAMIC_CLOUDLETS_AT_A_TIME;
        }

        ArrayList<Cloudlet> newCloudletList = new ArrayList<>(numberOfCloudlets);
        // Using the current size of the global cloudletList to control creation.
        cloudletlistsize = cloudletList.size();

        /*
         * You can remove the seed parameter to get a dynamic one, based on current
         * computer time.
         * With a dynamic seed you will get different results at each simulation run.
         */
        final long seed = 1;
        ContinuousDistribution rand = new UniformDistr(0, NUMBER_OF_VMS - 1, seed);

        // Loop for the required number of new cloudlets.
        for (int i = 0; i < numberOfCloudlets; i++) {
            if (cloudletlistsize < NUMBER_OF_CLOUDLETS) {
                // Randomly select a VM.
                int vmIndex = (int) rand.sample();
                final Vm vm = vmList.get(vmIndex);
                // Create cloudlet assigned to that VM.
                Cloudlet cloudlet = createCloudlet(vm);
                // Add cloudlet to both the global list and the local list.
                cloudletList.add(cloudlet);
                newCloudletList.add(cloudlet);
                cloudletlistsize = cloudletList.size();
            }
        }

        return newCloudletList;
    }
}
