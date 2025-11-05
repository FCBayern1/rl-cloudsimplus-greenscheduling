package giu.edu.cspg.common;

import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.provisioners.ResourceProvisioner;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.vms.Vm;

import java.util.List;

public class HostWithoutCreatedList extends HostSimple {
    public HostWithoutCreatedList(final List<Pe> peList) {
        super(peList);
    }

    public HostWithoutCreatedList(final List<Pe> peList, final boolean activate) {
        super(peList, activate);
    }

    public HostWithoutCreatedList(final ResourceProvisioner ramProvisioner,
            final ResourceProvisioner bwProvisioner, final long storage, final List<Pe> peList) {
        super(ramProvisioner, bwProvisioner, storage, peList);
    }

    public HostWithoutCreatedList(final long ram, final long bw, final long storage,
            final List<Pe> peList) {
        super(ram, bw, storage, peList);
    }

    public HostWithoutCreatedList(final long ram, final long bw, final long storage,
            final List<Pe> peList, final boolean activate) {
        super(ram, bw, storage, peList, activate);
    }

    @Override
    protected void addVmToCreatedList(final Vm vm) {
        // Do nothing to avoid accumulating vm data. You cannot do a
        // Host.getVmCreatedList().clear()
        // because it is unmodifiable.
    }
}
