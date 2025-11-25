package giu.edu.cspg.loadbalancers.legacy;

import java.util.Objects;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterCharacteristics;
import org.cloudsimplus.vms.Vm;

/**
 * Computes the monetary ($) cost to run a given Cloudlet.
 * This class follows the pattern of the {@link org.cloudsimplus.vms.VmCost}
 * class, externalizing the cost calculation logic from the Cloudlet entity
 * itself.
 *
 * <p>
 * In the current CloudSim Plus architecture (post cost-field removal from
 * Cloudlet),
 * this class calculates costs based on the Cloudlet's total work (length *
 * PEs),
 * total data transferred (file size + output size), and the monetary rates
 * of the Datacenter where the Cloudlet finished execution.
 * </p>
 *
 * <p>
 * This approach simplifies cost tracking compared to the older version that
 * tracked costs across different Datacenter execution segments, which is no
 * longer directly supported by the Cloudlet's internal state.
 * </p>
 */
public class CloudletCost {

    /**
     * The Cloudlet for which the total monetary cost will be computed.
     */
    private final Cloudlet cloudlet;

    /**
     * Creates an instance to compute the monetary cost ($) to run a given Cloudlet.
     * 
     * @param cloudlet the Cloudlet to compute its monetary cost.
     */
    public CloudletCost(final Cloudlet cloudlet) {
        this.cloudlet = Objects.requireNonNull(cloudlet);
    }

    /**
     * Gets the Datacenter where the Cloudlet was executed.
     * {@return the Datacenter or {@link Datacenter#NULL} if not assigned to a
     * VM/Host/Datacenter}
     */
    private Datacenter getExecutionDatacenter() {
        final Vm vm = cloudlet.getVm();
        if (vm == null || vm == Vm.NULL || vm.getHost() == null) {
            return Datacenter.NULL;
        }
        return vm.getHost().getDatacenter();
    }

    /**
     * Gets the characteristics of the Datacenter where the Cloudlet was executed.
     * {@return the Datacenter characteristics or
     * {@link DatacenterCharacteristics#NULL}
     * if not assigned to a VM/Host/Datacenter}
     */
    private DatacenterCharacteristics getExecutionDatacenterCharacteristics() {
        final Datacenter dc = getExecutionDatacenter();
        if (dc == Datacenter.NULL) {
            return DatacenterCharacteristics.NULL;
        }
        return dc.getCharacteristics();
    }

    /**
     * {@return the processing monetary cost ($)} for executing the Cloudlet.
     *
     * <p>
     * This cost is calculated based on the Cloudlet's total length (MIs)
     * and the cost per MI of the Datacenter where it was executed (derived from
     * the Datacenter's cost per second and the Host's MIPS).
     * </p>
     *
     * <p>
     * Returns 0 if the Cloudlet was not successfully executed on a Datacenter.
     * </p>
     */
    public double getProcessingCost() {
        final DatacenterCharacteristics dcCharacteristics = getExecutionDatacenterCharacteristics();
        final Vm vm = cloudlet.getVm();

        // Cannot calculate cost if not executed on a valid VM/Host/DC
        if (dcCharacteristics == DatacenterCharacteristics.NULL || vm == Vm.NULL || vm.getHost() == null) {
            return 0.0;
        }

        // Cost per Million Instructions (MI)
        final double hostMips = vm.getHost().getMips();
        final double costPerMI = hostMips <= 0 ? 0.0 : dcCharacteristics.getCostPerSecond() / hostMips;
        final double cloudletTotalTime = cloudlet.getTotalExecutionTime() + cloudlet.getStartWaitTime();

        return costPerMI * cloudlet.getTotalLength() * cloudletTotalTime;
    }

    /**
     * {@return the bandwidth monetary cost ($)} related to the Cloudlet's input and
     * output file transfers.
     *
     * <p>
     * This cost is calculated based on the total size of input and output
     * files and the cost per BW of the Datacenter where the Cloudlet was
     * executed.
     * </p>
     *
     * <p>
     * Returns 0 if the Cloudlet was not successfully executed on a Datacenter.
     * </p>
     */
    public double getBwCost() {
        final DatacenterCharacteristics dcCharacteristics = getExecutionDatacenterCharacteristics();

        // Cannot calculate cost if not executed on a valid VM/Host/DC
        if (dcCharacteristics == DatacenterCharacteristics.NULL) {
            return 0.0;
        }

        final double totalDataSize = cloudlet.getFileSize() + cloudlet.getOutputSize(); // fileSize and outputSize are
                                                                                        // long, casting to double
        return totalDataSize * dcCharacteristics.getCostPerBw();
    }

    /**
     * {@return the total monetary cost ($)} of the Cloudlet execution,
     * including processing and bandwidth costs.
     */
    public double getTotalCost() {
        return getProcessingCost() + getBwCost();
    }

    @Override
    public String toString() {
        return "Cloudlet %d costs ($) for %8.2f execution seconds - CPU: %8.2f$ BW: %8.2f$ Total: %8.2f$"
                .formatted(cloudlet.getId(), cloudlet.getTotalExecutionTime(), getProcessingCost(), getBwCost(),
                        getTotalCost());
    }
}
