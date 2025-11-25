package giu.edu.cspg.utils;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for calculating and reporting energy consumption statistics
 * for Hosts and Datacenters in CloudSim Plus simulations.
 */
public class EnergyConsumptionUtils {
    private static final Logger LOGGER = LoggerFactory.getLogger(EnergyConsumptionUtils.class.getSimpleName());

    /**
     * Data class to hold energy consumption statistics for a single host.
     */
    public static class HostEnergyStats {
        public final long hostId;
        public final double energyConsumptionWh;  // Watt-hours
        public final double averagePowerW;        // Watts
        public final double maxPowerW;            // Watts
        public final double utilizationPercent;   // 0-100

        public HostEnergyStats(long hostId, double energyWh, double avgPower, double maxPower, double utilization) {
            this.hostId = hostId;
            this.energyConsumptionWh = energyWh;
            this.averagePowerW = avgPower;
            this.maxPowerW = maxPower;
            this.utilizationPercent = utilization * 100;
        }
    }

    /**
     * Data class to hold aggregated energy consumption statistics for a datacenter.
     */
    public static class DatacenterEnergyStats {
        public final double totalEnergyConsumptionWh;     // Total Watt-hours
        public final double totalEnergyConsumptionKWh;    // Total Kilowatt-hours
        public final double averagePowerW;                 // Average Watts across all hosts
        public final double peakPowerW;                    // Peak Watts
        public final int numberOfHosts;
        public final double simulationDurationSeconds;
        public final Map<Long, HostEnergyStats> hostStats;

        public DatacenterEnergyStats(double totalEnergyWh, double avgPower, double peakPower,
                                      int numHosts, double duration, Map<Long, HostEnergyStats> hostStats) {
            this.totalEnergyConsumptionWh = totalEnergyWh;
            this.totalEnergyConsumptionKWh = totalEnergyWh / 1000.0;
            this.averagePowerW = avgPower;
            this.peakPowerW = peakPower;
            this.numberOfHosts = numHosts;
            this.simulationDurationSeconds = duration;
            this.hostStats = hostStats;
        }
    }

    /**
     * Calculates detailed energy consumption statistics for all hosts in a datacenter.
     *
     * @param datacenter The datacenter to analyze
     * @param simulationDuration Total simulation duration in seconds
     * @return DatacenterEnergyStats object containing all energy metrics
     */
    public static DatacenterEnergyStats calculateDatacenterEnergy(Datacenter datacenter, double simulationDuration) {
        List<Host> hostList = datacenter.getHostList();

        if (hostList.isEmpty()) {
            LOGGER.warn("Datacenter has no hosts. Returning zero energy consumption.");
            return new DatacenterEnergyStats(0, 0, 0, 0, simulationDuration, new HashMap<>());
        }

        Map<Long, HostEnergyStats> hostStatsMap = new HashMap<>();
        double totalEnergy = 0.0;
        double totalAvgPower = 0.0;
        double maxPower = 0.0;

        for (Host host : hostList) {
            // Get energy consumption using CloudSim Plus built-in method
            // Note: getTotalUpTimeHours() gives simulation time in hours
            double energyWh = 0;
            double hostMaxPower = 0;

            if (host.getPowerModel() != null) {
                // Get total energy consumption from power model
                // PowerModel tracks energy consumption internally
                var powerModel = host.getPowerModel();
                hostMaxPower = powerModel.getPower(1.0); // Max power at 100% utilization

                // Calculate energy based on state history
                // Energy (Wh) = Sum of (Power(t) * duration(t))
                var stateHistory = host.getStateHistory();
                if (!stateHistory.isEmpty()) {
                    double prevTime = 0;
                    for (var historyEntry : stateHistory) {
                        double currentTime = historyEntry.time();
                        double duration = currentTime - prevTime;
                        double utilization = historyEntry.percentUsage();
                        double power = powerModel.getPower(utilization);
                        energyWh += power * (duration / 3600.0); // Convert seconds to hours
                        prevTime = currentTime;
                    }
                }
            }

            // Calculate average power: Energy (Wh) / Time (h)
            double timeHours = simulationDuration / 3600.0;
            double avgPower = timeHours > 0 ? energyWh / timeHours : 0;

            // Calculate average host utilization from state history
            double utilization = 0;
            var stateHistory = host.getStateHistory();
            if (!stateHistory.isEmpty()) {
                double totalUtilization = 0;
                for (var historyEntry : stateHistory) {
                    totalUtilization += historyEntry.percentUsage();
                }
                utilization = totalUtilization / stateHistory.size();
            }

            // Store individual host stats
            HostEnergyStats hostStats = new HostEnergyStats(
                host.getId(),
                energyWh,
                avgPower,
                hostMaxPower,
                utilization
            );
            hostStatsMap.put(host.getId(), hostStats);

            // Accumulate totals
            totalEnergy += energyWh;
            totalAvgPower += avgPower;
            maxPower = Math.max(maxPower, hostMaxPower);

            LOGGER.debug("Host {}: Energy={} Wh, AvgPower={} W, Utilization={}%",
                host.getId(), String.format("%.2f", energyWh),
                String.format("%.2f", avgPower), String.format("%.2f", utilization * 100));
        }

        // Calculate average power across all hosts
        double avgPowerAllHosts = !hostList.isEmpty() ? totalAvgPower / hostList.size() : 0;

        return new DatacenterEnergyStats(
            totalEnergy,
            avgPowerAllHosts,
            maxPower,
            hostList.size(),
            simulationDuration,
            hostStatsMap
        );
    }

    /**
     * Prints a formatted summary of energy consumption statistics.
     *
     * @param stats The DatacenterEnergyStats to print
     */
    public static void printEnergySummary(DatacenterEnergyStats stats) {
        LOGGER.info("===============================================");
        LOGGER.info("       ENERGY CONSUMPTION STATISTICS");
        LOGGER.info("===============================================");
        LOGGER.info("Simulation Duration: {} seconds ({} hours)",
            String.format("%.2f", stats.simulationDurationSeconds),
            String.format("%.4f", stats.simulationDurationSeconds / 3600.0));
        LOGGER.info("Number of Hosts: {}", stats.numberOfHosts);
        LOGGER.info("-----------------------------------------------");
        LOGGER.info("Total Energy Consumption: {} Wh ({} kWh)",
            String.format("%.2f", stats.totalEnergyConsumptionWh),
            String.format("%.4f", stats.totalEnergyConsumptionKWh));
        LOGGER.info("Average Power Consumption: {} W",
            String.format("%.2f", stats.averagePowerW));
        LOGGER.info("Peak Power (Max Host): {} W",
            String.format("%.2f", stats.peakPowerW));

        // Calculate carbon emissions (optional, using average grid emissions)
        // Assuming 0.5 kg CO2 per kWh (varies by region)
        double carbonKg = stats.totalEnergyConsumptionKWh * 0.5;
        LOGGER.info("Estimated CO2 Emissions: {} kg",
            String.format("%.4f", carbonKg));

        LOGGER.info("===============================================");
    }

    /**
     * Calculates the Power Usage Effectiveness (PUE) metric.
     * PUE = Total Datacenter Power / IT Equipment Power
     *
     * @param totalDatacenterPower Total power including cooling, lighting, etc.
     * @param itEquipmentPower Power consumed by IT equipment (servers)
     * @return PUE value (ideal = 1.0, typical = 1.5-2.0)
     */
    public static double calculatePUE(double totalDatacenterPower, double itEquipmentPower) {
        if (itEquipmentPower <= 0) {
            return 0;
        }
        return totalDatacenterPower / itEquipmentPower;
    }

    /**
     * Estimates total datacenter power including infrastructure overhead.
     * Typical PUE for modern datacenters is 1.5-2.0.
     *
     * @param itEquipmentPower Power consumed by servers
     * @param pue Power Usage Effectiveness (default 1.5 for average datacenter)
     * @return Estimated total datacenter power including cooling, lighting, etc.
     */
    public static double estimateTotalDatacenterPower(double itEquipmentPower, double pue) {
        return itEquipmentPower * pue;
    }
}
