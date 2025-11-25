package giu.edu.cspg.common;

import java.lang.reflect.Field;
import java.util.List;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.schedulers.MipsShare;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerAbstract;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;

public class OptimizedCloudletScheduler extends CloudletSchedulerSpaceShared {

    @Override
    protected double cloudletSubmitInternal(final CloudletExecution cle, final double fileTransferTime) {
        if (!getVm().isCreated()) {
            // It is possible, that we schedule a cloudlet, an event with processing
            // update is issued (tag: 16), but the VM gets killed before the event
            // is processed. In such a case the cloudlet does not get rescheduled,
            // because we don't know yet that this cloudlet should be!
            final Cloudlet cloudlet = cle.getCloudlet();
            final DatacenterBroker broker = cloudlet.getBroker();
            broker.submitCloudlet((cloudlet.reset()));
            return -1.0;
        }
        return super.cloudletSubmitInternal(cle, fileTransferTime);
    }

    @Override
    public double updateProcessing(final double currentTime, final MipsShare mipsShare) {
        final int sizeBefore = getCloudletWaitingList().size();
        final double nextSimulationTime = super.updateProcessing(currentTime, mipsShare);
        final int sizeAfter = this.getCloudletWaitingList().size();

        // if we have a new cloudlet being processed,
        // schedule another recalculation, which should trigger a proper
        // estimation of end time
        if (sizeAfter != sizeBefore && Double.MAX_VALUE == nextSimulationTime) {
            return getVm().getSimulation().getMinTimeBetweenEvents();
        }

        return nextSimulationTime;
    }

    private List<?> getModifiableCloudletReturnedList()
            throws IllegalAccessException, NoSuchFieldException {

        final Field field = CloudletSchedulerAbstract.class.getDeclaredField("cloudletReturnedList");
        field.setAccessible(true);
        Object fieldValue = field.get(this);

        if (!(fieldValue instanceof List<?>)) {
            return null;
        }

        return (List<?>) fieldValue;
    }

    // Here we modify the private field cloudletReturnedList.
    // The lists cloudletWaitingList cloudletExecList are cleaned
    // inside the parent's class.
    // It is safe to override this function:
    // it is used only in one place - DatacenterBrokerAbstract:827
    @Override
    @SuppressWarnings("CallToPrintStackTrace")
    public void clear() {
        super.clear();
        try {
            List<?> cloudletReturnedList = getModifiableCloudletReturnedList();
            if (cloudletReturnedList != null) {
                cloudletReturnedList.clear();
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }
}
