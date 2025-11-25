package giu.edu.cspg.loadbalancers.legacy;

import java.util.ArrayList;
import java.util.List;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class creates a list of Cloudlet objects and assigns each
 * one to a VM using a round-robin strategy.
 */
public class RoundRobinLoadBalancer extends BaseAbstract {

    private static final Logger logger = LoggerFactory.getLogger(RoundRobinLoadBalancer.class.getSimpleName());

    public static void main(String[] args) {
        RoundRobinLoadBalancer roundRobinLoadBalancer = new RoundRobinLoadBalancer();
        roundRobinLoadBalancer.run();
    }

    /**
     * Creates a list of Cloudlet objects that together represent the
     * distributed processes of a given fictitious application.
     * Each cloudlet is assigned to a VM in a round-robin manner.
     *
     * @return the list of created Cloudlets.
     */
    @Override
    protected List<Cloudlet> createCloudlets() {
        logger.info("Create list of network cloudlets and assign VM in round robin manner");

        int numberOfCloudlets;
        if (cloudletList.isEmpty()) {
            numberOfCloudlets = INITIAL_CLOUDLETS;
        } else {
            numberOfCloudlets = DYNAMIC_CLOUDLETS_AT_A_TIME;
        }

        ArrayList<Cloudlet> newCloudletList = new ArrayList<>(numberOfCloudlets);
        cloudletlistsize = cloudletList.size();

        // Loop for the required number of new cloudlets.
        for (int i = 0; i < numberOfCloudlets; i++) {
            if (cloudletlistsize < NUMBER_OF_CLOUDLETS) {
                // Determine the VM index using round-robin assignment.
                int vmIndex = cloudletlistsize % NUMBER_OF_VMS;
                // Create a cloudlet assigned to the determined VM.
                Cloudlet cloudlet = createCloudlet(vmList.get(vmIndex));
                // Add cloudlet to both the global cloudlet list and the local list.
                cloudletList.add(cloudlet);
                newCloudletList.add(cloudlet);
                cloudletlistsize = cloudletList.size();
            }
        }

        return newCloudletList;
    }
}
