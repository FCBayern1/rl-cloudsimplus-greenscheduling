package giu.edu.cspg.tables;

import org.cloudsimplus.builders.tables.Table;
import org.cloudsimplus.builders.tables.TableBuilderAbstract;
import org.cloudsimplus.builders.tables.TextTable;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostStateHistoryEntry;

/**
 * Builds a table for printing {@link HostStateHistoryEntry} entries from the
 * {@link Host#getStateHistory()}.
 * It defines a set of default columns but new ones can be added
 * dynamically using the {@code addColumn()} methods.
 *
 * <p>
 * The basic usage of the class is by calling its constructor,
 * giving a Host to print its history, and then
 * calling the {@link #build()} method.
 * </p>
 *
 * @author Manoel Campos da Silva Filho
 * @since CloudSim Plus 2.3.2
 */
public class HostHistoryTableBuilderCsv extends TableBuilderAbstract<HostStateHistoryEntry> {
    private final Host host;

    /**
     * Instantiates a builder to print the history of a Host using the a
     * default {@link TextTable}.
     * To use a different {@link Table}, check the alternative constructors.
     *
     * @param host the Host to get the history to print
     */
    public HostHistoryTableBuilderCsv(final Host host) {
        super(host.getStateHistory());
        this.host = host;
    }

    /**
     * Instantiates a builder to print the history of a Host using the given
     * {@link Table}.
     *
     * @param host  the Host to get the history to print
     * @param table the table to use to print the history
     */
    public HostHistoryTableBuilderCsv(final Host host, final Table table) {
        super(host.getStateHistory(), table);
        this.host = host;
    }

    @Override
    protected void createTableColumns() {
        final var col1 = getTable().newColumn("Time Secs", "", "%5.0f");
        addColumn(col1, HostStateHistoryEntry::time);

        final String format = "%9.0f";
        final var col2 = getTable().newColumn("Total Requested MIPS", "", format);
        addColumn(col2, HostStateHistoryEntry::requestedMips);

        final var col3 = getTable().newColumn("Total Allocated MIPS", "", format);
        addColumn(col3, HostStateHistoryEntry::allocatedMips);

        final var col4 = getTable().newColumn("Used", "", "%3.0f%%");
        addColumn(col4, history -> history.percentUsage() * 100);

        addColumn(getTable().newColumn("Host Active"), HostStateHistoryEntry::active);

        final var col5 = getTable().newColumn("Host Total MIPS", "", format);
        addColumn(col5, history -> host.getTotalMipsCapacity());

        final var col6 = getTable().newColumn("Host Total Usage", "", "%5.1f%%");
        addColumn(col6, history -> history.allocatedMips() / host.getTotalMipsCapacity() * 100);

        // Add power consumption column (new)
        final var col7 = getTable().newColumn("Power (W)", "", "%6.2f");
        addColumn(col7, history -> {
            if (host.getPowerModel() != null) {
                return host.getPowerModel().getPower(history.percentUsage());
            }
            return 0.0;
        });

        // Add instantaneous energy consumption (power * time delta)
        final var col8 = getTable().newColumn("Energy (Wh)", "", "%8.4f");
        addColumn(col8, history -> {
            if (host.getPowerModel() != null) {
                // Energy = Power * Time
                // Get time delta from previous entry (1 second typically)
                double power = host.getPowerModel().getPower(history.percentUsage());
                // Assuming 1 second interval, convert to hours: 1/3600
                return power / 3600.0;
            }
            return 0.0;
        });
    }
}
