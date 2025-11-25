package giu.edu.cspg.tables;

import java.util.List;

import org.cloudsimplus.builders.tables.Table;
import org.cloudsimplus.builders.tables.TableBuilderAbstract;
import org.cloudsimplus.builders.tables.TableColumn;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmCost;

/**
 * Extends TableBuilderAbstract to add columns for the average CPU Utilization
 * and estimated Total Cost of the VM that executed during the simulation.
 */
public class VmsTableBuilderWithDetails extends TableBuilderAbstract<Vm> {
    private static final String ID = "ID"; // Column name for IDs
    private static final String CPU_CORES = "CPU Cores";
    private static final String PERCENT = "%";
    private static final String DOLLARS = "$";
    private static final String COST_FORMAT = "%.2f"; // Show more precision for cost
    private static final String UTIL_FORMAT = "%.1f"; // Format for utilization percentage

    /**
     * Creates a new VmsTableBuilderWithDetails instance.
     * 
     * @param list The list of finished Vms to display.
     */
    public VmsTableBuilderWithDetails(List<? extends Vm> list) {
        super(list);
    }

    public VmsTableBuilderWithDetails(List<? extends Vm> list, Table table) {
        super(list, table);
    }

    /**
     * Overrides the method to create table columns, adding VM Utilization and Cost.
     */
    @Override
    protected void createTableColumns() {
        addColumn(getTable().newColumn(ID), vm -> vm.getId());
        addColumn(getTable().newColumn("Type"), vm -> vm.getDescription());
        addColumn(getTable().newColumn("Status"), vm -> this.getStatus(vm));
        addColumn(getTable().newColumn("PEs " + CPU_CORES),
                vm -> vm.getPesNumber());
        // 1. VM CPU Utilization (Average during Cloudlet execution - approximation)
        TableColumn cpuUtilCol = getTable().newColumn("CPU Util " + PERCENT).setFormat(UTIL_FORMAT);
        addColumn(cpuUtilCol, vm ->
        // Use the utilization mean provided by the VM's UtilizationHistory
        vm.getCpuUtilizationStats().getMean() * 100 // Get mean utilization (0-1) and convert to %
        );
        // 2. Estimated Total Cost of the VM
        TableColumn costCol = getTable().newColumn("Cost " + DOLLARS).setFormat(COST_FORMAT);
        addColumn(costCol, vm -> new VmCost(vm).getTotalCost());
    }

    /**
     * Returns the status of the VM as a string.
     * 
     * @param vm The VM to get the status from.
     * @return The status of the VM as a string.
     */
    private String getStatus(Vm vm) {
        return switch (vm) {
            case Vm v when v.isFailed() -> "FAILED";
            case Vm v when v.isFinished() -> "FINISHED";
            case Vm v when v.isShuttingDown() -> "SHUTTING_DOWN";
            case Vm v when v.isInMigration() -> "MIGRATING";
            case Vm v when v.isIdle() -> "IDLE";
            case Vm v when v.hasStarted() -> "EXECUTING";
            case Vm v when v.isStartingUp() -> "STARTING";
            case Vm v when v.isWorking() -> "WORKING";
            default -> "UNKNOWN";
        };
    }
}
