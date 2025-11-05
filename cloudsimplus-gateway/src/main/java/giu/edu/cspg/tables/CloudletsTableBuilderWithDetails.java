package giu.edu.cspg.tables;

import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.builders.tables.Table;
import org.cloudsimplus.builders.tables.TableColumn;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.Identifiable;

import giu.edu.cspg.common.CloudletCost;

/**
 * Extends CloudletsTableBuilder to add columns for the average CPU Utilization
 * and estimated Total Cost of the VM that executed each Cloudlet.
 */
public class CloudletsTableBuilderWithDetails extends CloudletsTableBuilder {
	private static final String ID = "ID"; // Column name for IDs
	private static final String TIME_FORMAT = "%.0f"; // Use 0 decimal places for time
	private static final String SECONDS = "Seconds";
	private static final String CPU_CORES = "CPU Cores";
	private static final String MI = "MI"; // Million Instructions
	private static final String DOLLARS = "$";
	private static final String COST_FORMAT = "%.2f"; // Show more precision for cost
	private final Map<Long, Double> cloudletArrivalTimeMap;

	/**
	 * Creates a new CloudletsTableBuilderWithDetails instance.
	 * 
	 * @param list The list of finished Cloudlets to display.
	 */
	public CloudletsTableBuilderWithDetails(List<? extends Cloudlet> list, Map<Long, Double> cloudletArrivalTimeMap) {
		super(list);
		this.cloudletArrivalTimeMap = Objects.requireNonNull(cloudletArrivalTimeMap);
	}

	public CloudletsTableBuilderWithDetails(List<? extends Cloudlet> list, Table table,
			Map<Long, Double> cloudletArrivalTimeMap) {
		super(list, table);
		this.cloudletArrivalTimeMap = Objects.requireNonNull(cloudletArrivalTimeMap);
	}

	/**
	 * Overrides the method to create table columns, adding VM Utilization and Cost.
	 */
	@Override
	protected void createTableColumns() {
		addColumn(getTable().newColumn(ID), Identifiable::getId);
		addColumn(getTable().newColumn("Status"), cloudlet -> cloudlet.getStatus().name());
		addColumn(getTable().newColumn("DC " + ID),
				cloudlet -> cloudlet.getVm().getHost().getDatacenter().getId());
		addColumn(getTable().newColumn("Host " + ID), cloudlet -> cloudlet.getVm().getHost().getId());
		addColumn(getTable().newColumn("Host PEs " + CPU_CORES),
				cloudlet -> cloudlet.getVm().getHost().getWorkingPesNumber());
		addColumn(getTable().newColumn("VM " + ID), cloudlet -> cloudlet.getVm().getId());
		addColumn(getTable().newColumn("VM PEs " + CPU_CORES),
				cloudlet -> cloudlet.getVm().getPesNumber());
		addColumn(getTable().newColumn("CloudletLen " + MI), Cloudlet::getLength);
		addColumn(getTable().newColumn("FinishedLen " + MI), Cloudlet::getFinishedLengthSoFar);
		addColumn(getTable().newColumn("CloudletPEs " + CPU_CORES), Cloudlet::getPesNumber);

		TableColumn col = getTable().newColumn("ArrivalTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> cloudletArrivalTimeMap.get(cloudlet.getId()));

		col = getTable().newColumn("SubmssionTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> cloudlet.getCreationTime());

		col = getTable().newColumn("StartWaitTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> cloudlet.getStartWaitTime());

		col = getTable().newColumn("StartTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> cloudlet.getStartTime());

		col = getTable().newColumn("FinishTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> cloudlet.getFinishTime());

		col = getTable().newColumn("ExecTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> Math.ceil(cloudlet.getTotalExecutionTime()));

		col = getTable().newColumn("TotalWaitTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> Math.ceil(cloudlet.getStartTime() - cloudletArrivalTimeMap.get(cloudlet.getId())));

		col = getTable().newColumn("CompletionTime " + SECONDS).setFormat(TIME_FORMAT);
		addColumn(col, cloudlet -> Math.ceil(cloudlet.getTotalExecutionTime()
				+ (cloudlet.getStartTime() - cloudletArrivalTimeMap.get(cloudlet.getId()))));

		col = getTable().newColumn("Cost " + DOLLARS).setFormat(COST_FORMAT);
		addColumn(col, cloudlet -> new CloudletCost(cloudlet, cloudletArrivalTimeMap).getTotalCost());
	}
}
