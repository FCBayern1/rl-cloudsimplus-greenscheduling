package giu.edu.cspg.loadbalancers.legacy;

import java.util.List;

import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.builders.tables.Table;
import org.cloudsimplus.builders.tables.TableColumn;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.Identifiable;

public class CloudletsTableBuilderWithCost extends CloudletsTableBuilder {

    private static final String TIME_FORMAT = "%.0f";
    private static final String SECONDS = "Seconds";
    private static final String CPU_CORES = "CPU cores";
    private static final String DOLLARS = "Dollars";
    private static final String COST_FORMAT = "%.2f";

    public CloudletsTableBuilderWithCost(List<? extends Cloudlet> list) {
        super(list);
    }

    public CloudletsTableBuilderWithCost(List<? extends Cloudlet> list, Table table) {
        super(list, table);
    }

    @Override
    protected void createTableColumns() {
        final String ID = "ID";
        addColumn(getTable().newColumn("Cloudlet", ID), Identifiable::getId);
        addColumn(getTable().newColumn("Status "), cloudlet -> cloudlet.getStatus().name());
        addColumn(getTable().newColumn("DC", ID),
                cloudlet -> cloudlet.getVm().getHost().getDatacenter().getId());
        addColumn(getTable().newColumn("Host", ID), cloudlet -> cloudlet.getVm().getHost().getId());
        addColumn(getTable().newColumn("Host PEs ", CPU_CORES),
                cloudlet -> cloudlet.getVm().getHost().getWorkingPesNumber());
        addColumn(getTable().newColumn("VM", ID), cloudlet -> cloudlet.getVm().getId());
        addColumn(getTable().newColumn("VM PEs   ", CPU_CORES),
                cloudlet -> cloudlet.getVm().getPesNumber());
        addColumn(getTable().newColumn("CloudletLen", "MI"), Cloudlet::getLength);
        addColumn(getTable().newColumn("CloudletPEs", CPU_CORES), Cloudlet::getPesNumber);

        TableColumn col = getTable().newColumn("StartTime", SECONDS).setFormat(TIME_FORMAT);
        addColumn(col, cloudlet -> cloudlet.getStartTime());

        col = getTable().newColumn("FinishTime", SECONDS).setFormat(TIME_FORMAT);
        addColumn(col, cloudlet -> cloudlet.getFinishTime());

        col = getTable().newColumn("ExecTime", SECONDS).setFormat(TIME_FORMAT);
        addColumn(col, cloudlet -> Math.ceil(cloudlet.getTotalExecutionTime()));

        col = getTable().newColumn("WaitTime", SECONDS).setFormat(TIME_FORMAT);
        addColumn(col, cloudlet -> Math.ceil(cloudlet.getStartWaitTime()));

        col = getTable().newColumn("CompletionTime", SECONDS).setFormat(TIME_FORMAT);
        addColumn(col, cloudlet -> Math.ceil(cloudlet.getTotalExecutionTime() + cloudlet.getStartWaitTime()));

        col = getTable().newColumn("Cost", DOLLARS).setFormat(COST_FORMAT);
        addColumn(col, cloudlet -> new CloudletCost(cloudlet).getTotalCost());
    }
}
