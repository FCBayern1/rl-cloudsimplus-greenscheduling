package giu.edu.cspg.utils;

import java.util.ArrayList;
import java.util.Comparator;
import static java.util.Comparator.comparingDouble;
import static java.util.Comparator.comparingLong;
import java.util.List;
import java.util.Map;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.builders.tables.AbstractTable;
import org.cloudsimplus.builders.tables.CsvTable;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostStateHistoryEntry;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmCost;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import giu.edu.cspg.common.CloudletCost;
import giu.edu.cspg.singledc.LoadBalancingBroker;
import giu.edu.cspg.singledc.SimulationCore;
import giu.edu.cspg.tables.CloudletsTableBuilderWithDetails;
import giu.edu.cspg.tables.HostHistoryTableBuilderCsv;
import giu.edu.cspg.tables.TableLogger;
import giu.edu.cspg.tables.VmsTableBuilderWithDetails;
import giu.edu.cspg.utils.EnergyConsumptionUtils;
import giu.edu.cspg.utils.EnergyConsumptionUtils.DatacenterEnergyStats;

public class SimulationResultUtils {
    private static final Logger LOGGER = LoggerFactory.getLogger(SimulationResultUtils.class.getSimpleName());

    /**
     * Prints standard simulation results tables (Cloudlets, VMs, Hosts) and saves
     * them to CSV files.
     * 
     * @param simulationCore The simulation core instance after the simulation has
     *                       finished.
     * @param baseFileName   A base name for the output files (e.g.,
     *                       "SimulationName_PolicyName").
     */
    public static void printAndSaveResults(SimulationCore simulationCore, String baseFileName) {
        LoadBalancingBroker broker = simulationCore.getBroker();
        Map<Long, Double> arrivalTimeMap = broker.getCloudletArrivalTimeMap();
        Datacenter datacenter = simulationCore.getDatacenter();
        double clock = simulationCore.getClock();
        printAndSaveResults(broker, arrivalTimeMap, datacenter, clock, baseFileName);
    }

    /**
     * Prints standard simulation results tables (Cloudlets, VMs, Hosts) and saves
     * them to CSV files.
     * 
     * @param broker         The broker instance after the simulation has finished.
     * @param arrivalTimeMap A map of Cloudlet IDs to their arrival times.
     * @param datacenter     The datacenter instance after the simulation has
     *                       finished.
     * @param clock          The simulation clock time at the end of the simulation.
     * @param baseFileName   A base name for the output files (e.g.,
     *                       "SimulationName_PolicyName").
     */
    public static void printAndSaveResults(DatacenterBrokerSimple broker, Map<Long, Double> arrivalTimeMap,
            Datacenter datacenter, double clock,
            String baseFileName) {
        LOGGER.info("Processing simulation results for {}...", baseFileName);

        if (broker == null) {
            LOGGER.error("Broker is null. Cannot print results.");
            return;
        }

        // --- Cloudlet Results ---
        List<Cloudlet> finishedList = broker.getCloudletFinishedList();
        if (finishedList.isEmpty()) {
            LOGGER.info("No cloudlets finished to print in the results table.");
        } else {
            LOGGER.info("Generating Cloudlet Results Table ({} finished)...", finishedList.size());
            // Sort Cloudlets (by Arrival Time first, then by VM ID, then by Cloudlet ID)
            List<Cloudlet> sortedCloudlets = new ArrayList<>(finishedList);
            final Comparator<Cloudlet> sortByVmId = comparingLong(c -> c.getVm().getId());
            final Comparator<Cloudlet> sortByArrivalTime = comparingDouble(c -> arrivalTimeMap.get(c.getId()));
            sortedCloudlets.sort(sortByArrivalTime.thenComparing(sortByVmId).thenComparing(Cloudlet::getId));

            CloudletsTableBuilderWithDetails cloudletsTable = new CloudletsTableBuilderWithDetails(sortedCloudlets,
                    new CsvTable(), arrivalTimeMap);
            cloudletsTable.build();
            String cloudletPath = String.format("results/%s/cloudlets.csv", baseFileName);
            TableLogger.logAndSaveTable((AbstractTable) cloudletsTable.getTable(), cloudletPath);
            LOGGER.info("Cloudlet results saved to {}", cloudletPath);
        }

        // --- VM Results ---
        List<Vm> vmList = broker.getVmCreatedList();
        if (vmList.isEmpty()) {
            LOGGER.info("No VMs were created or tracked for results.");
        } else {
            LOGGER.info("Generating VM Results Table ({} total)...", vmList.size());
            VmsTableBuilderWithDetails vmsTable = new VmsTableBuilderWithDetails(vmList, new CsvTable());
            vmsTable.build();
            String vmPath = String.format("results/%s/vms.csv", baseFileName);
            TableLogger.logAndSaveTable((AbstractTable) vmsTable.getTable(), vmPath);
            LOGGER.info("VM results saved to {}", vmPath);
        }

        // --- Host Results ---
        List<Host> hostList = datacenter.getHostList();
        if (hostList.isEmpty()) {
            LOGGER.info("No Hosts were created.");
        } else {
            LOGGER.info("Generating Host History Tables ({} hosts)...", hostList.size());
            hostList.forEach(host -> printAndSaveHostHistory(host, baseFileName));
            LOGGER.info("Host history saved.");
        }

        // --- Overall Cost/Stats ---
        showOverallStats(arrivalTimeMap, vmList, finishedList);

        // --- Energy Consumption Stats ---
        LOGGER.info("Calculating energy consumption statistics...");
        DatacenterEnergyStats energyStats = EnergyConsumptionUtils.calculateDatacenterEnergy(datacenter, clock);
        EnergyConsumptionUtils.printEnergySummary(energyStats);

        // Save energy stats to CSV
        String energyPath = String.format("results/%s/energy_consumption.csv", baseFileName);
        saveEnergyStatsToCsv(energyStats, energyPath);
        LOGGER.info("Energy consumption stats saved to {}", energyPath);

        LOGGER.info("Total simulation time: {} seconds", clock);
        LOGGER.info("Result processing finished for {}.", baseFileName);
    }

    /**
     * Saves green energy statistics to a CSV file.
     *
     * @param greenEnergyStats Map containing green energy statistics from LoadBalancerGateway.getGreenEnergyStats()
     * @param baseFileName     A base name for the output file
     */
    public static void saveGreenEnergyStats(Map<String, Double> greenEnergyStats, String baseFileName) {
        if (greenEnergyStats == null || greenEnergyStats.isEmpty()) {
            LOGGER.info("No green energy statistics available to save.");
            return;
        }

        LOGGER.info("Saving green energy statistics...");
        String filePath = String.format("results/%s/green_energy_summary.csv", baseFileName);

        try {
            java.io.File file = new java.io.File(filePath);
            file.getParentFile().mkdirs(); // Create directories if they don't exist

            try (java.io.PrintWriter writer = new java.io.PrintWriter(file)) {
                // Write header
                writer.println("=== GREEN ENERGY SUMMARY ===");
                writer.println();

                // Write main statistics
                writer.println("Metric,Value,Unit");
                writer.println(String.format("Cumulative Green Energy,%.4f,Wh",
                    greenEnergyStats.getOrDefault("cumulative_green_wh", 0.0)));
                writer.println(String.format("Cumulative Brown Energy,%.4f,Wh",
                    greenEnergyStats.getOrDefault("cumulative_brown_wh", 0.0)));
                writer.println(String.format("Total Wasted Green Energy,%.4f,Wh",
                    greenEnergyStats.getOrDefault("total_wasted_green_wh", 0.0)));
                writer.println(String.format("Current Green Power,%.2f,W",
                    greenEnergyStats.getOrDefault("current_green_power_w", 0.0)));
                writer.println(String.format("Green Energy Ratio,%.4f,fraction (0-1)",
                    greenEnergyStats.getOrDefault("green_ratio", 0.0)));

                if (greenEnergyStats.containsKey("overall_green_ratio")) {
                    writer.println(String.format("Overall Green Ratio,%.4f,fraction (0-1)",
                        greenEnergyStats.get("overall_green_ratio")));
                }

                writer.println();

                // Write derived statistics
                double cumulativeGreen = greenEnergyStats.getOrDefault("cumulative_green_wh", 0.0);
                double cumulativeBrown = greenEnergyStats.getOrDefault("cumulative_brown_wh", 0.0);
                double totalEnergy = cumulativeGreen + cumulativeBrown;
                double wastedGreen = greenEnergyStats.getOrDefault("total_wasted_green_wh", 0.0);

                writer.println("=== DERIVED STATISTICS ===");
                writer.println("Metric,Value,Unit");
                writer.println(String.format("Total Energy Consumed,%.4f,Wh", totalEnergy));
                writer.println(String.format("Total Energy Consumed,%.6f,kWh", totalEnergy / 1000.0));
                writer.println(String.format("Green Percentage,%.2f,%%",
                    totalEnergy > 0 ? (cumulativeGreen / totalEnergy * 100.0) : 0.0));
                writer.println(String.format("Brown Percentage,%.2f,%%",
                    totalEnergy > 0 ? (cumulativeBrown / totalEnergy * 100.0) : 0.0));
                writer.println(String.format("Waste Percentage (of green available),%.2f,%%",
                    (cumulativeGreen + wastedGreen) > 0 ? (wastedGreen / (cumulativeGreen + wastedGreen) * 100.0) : 0.0));
                writer.println(String.format("CO2 Emissions Avoided (vs 100%% brown),%.4f,kg",
                    (cumulativeGreen / 1000.0) * 0.5)); // Assuming 0.5 kg CO2 per kWh for brown energy
            }

            LOGGER.info("Green energy statistics saved to {}", filePath);
        } catch (java.io.IOException e) {
            LOGGER.error("Failed to save green energy stats to CSV: {}", e.getMessage(), e);
        }
    }

    /** Prints and saves the state history for a single host. */
    private static void printAndSaveHostHistory(Host host, String baseFileName) {
        // Only print/save if the host was actually used
        final boolean cpuUtilizationNotZero = host.getStateHistory()
                .stream()
                .map(HostStateHistoryEntry::percentUsage)
                .anyMatch(cpuUtilization -> cpuUtilization > 0);

        if (cpuUtilizationNotZero) {
            HostHistoryTableBuilderCsv hostTableBuilder = new HostHistoryTableBuilderCsv(host, new CsvTable());
            hostTableBuilder.build();
            String hostPath = String.format("results/%s/host%d.csv", baseFileName, host.getId());
            TableLogger.logAndSaveTable((AbstractTable) hostTableBuilder.getTable(), hostPath);
        } else {
            LOGGER.info("\tHost {} was not utilized, skipping history table.", host.getId());
        }
    }

    /** Saves energy consumption statistics to CSV file. */
    private static void saveEnergyStatsToCsv(DatacenterEnergyStats stats, String filePath) {
        try {
            java.io.File file = new java.io.File(filePath);
            file.getParentFile().mkdirs(); // Create directories if they don't exist

            try (java.io.PrintWriter writer = new java.io.PrintWriter(file)) {
                // Write summary section
                writer.println("=== DATACENTER ENERGY CONSUMPTION SUMMARY ===");
                writer.println(String.format("Simulation Duration (s),%.2f", stats.simulationDurationSeconds));
                writer.println(String.format("Simulation Duration (h),%.4f", stats.simulationDurationSeconds / 3600.0));
                writer.println(String.format("Number of Hosts,%d", stats.numberOfHosts));
                writer.println(String.format("Total Energy Consumption (Wh),%.2f", stats.totalEnergyConsumptionWh));
                writer.println(String.format("Total Energy Consumption (kWh),%.4f", stats.totalEnergyConsumptionKWh));
                writer.println(String.format("Average Power (W),%.2f", stats.averagePowerW));
                writer.println(String.format("Peak Power (W),%.2f", stats.peakPowerW));
                writer.println(String.format("Estimated CO2 Emissions (kg),%.4f", stats.totalEnergyConsumptionKWh * 0.5));
                writer.println();

                // Write per-host details
                writer.println("=== PER-HOST ENERGY CONSUMPTION ===");
                writer.println("Host ID,Energy (Wh),Average Power (W),Max Power (W),CPU Utilization (%)");

                stats.hostStats.values().stream()
                    .sorted((a, b) -> Long.compare(a.hostId, b.hostId))
                    .forEach(hostStat -> {
                        writer.println(String.format("%d,%.2f,%.2f,%.2f,%.2f",
                            hostStat.hostId,
                            hostStat.energyConsumptionWh,
                            hostStat.averagePowerW,
                            hostStat.maxPowerW,
                            hostStat.utilizationPercent));
                    });
            }
            LOGGER.debug("Energy stats CSV saved successfully to {}", filePath);
        } catch (java.io.IOException e) {
            LOGGER.error("Failed to save energy stats to CSV: {}", e.getMessage(), e);
        }
    }

    /** Calculates and logs overall simulation statistics. */
    private static void showOverallStats(Map<Long, Double> arrivalTimeMap, List<Vm> vmList,
            List<Cloudlet> finishedCloudlets) {
        LOGGER.info("Calculating Overall Simulation Statistics...");

        double totalCloudletCost = 0.0;
        double totalVmCost = 0.0;
        double totalUtilizedVmCost = 0.0;
        double totalCompletionTime = 0.0;
        double totalVmCpuUtilizationSum = 0.0;
        int finishedCount = finishedCloudlets.size();
        int utilizedVmCount = 0;
        int vmCount = 0;

        for (Cloudlet cloudlet : finishedCloudlets) {
            CloudletCost cloudletCost = new CloudletCost(cloudlet, arrivalTimeMap);
            totalCloudletCost += cloudletCost.getTotalCost();

            // Calculate completion time = execution time + wait time
            Double arrival = arrivalTimeMap.get(cloudlet.getId());
            if (arrival != null) {
                final var totalWaitTime = Math.ceil(cloudlet.getStartTime() - arrival);
                totalCompletionTime += (cloudlet.getTotalExecutionTime() + totalWaitTime);
            }
        }

        for (Vm vm : vmList) {
            if (vm.hasStarted() || vm.isFinished()) {
                if (vm.getCpuUtilizationStats().getMean() > 0) {
                    totalUtilizedVmCost += new VmCost(vm).getTotalCost();
                    totalVmCpuUtilizationSum += vm.getCpuUtilizationStats().getMean() * 100.0;
                    utilizedVmCount++;
                }
                totalVmCost += new VmCost(vm).getTotalCost();
                vmCount++;
            }
        }

        LOGGER.info("Total cost of executing {} Cloudlets = ${}", finishedCount,
                String.format("%.2f", totalCloudletCost));
        LOGGER.info("Total cost of running {} VMs = ${}", vmCount, String.format("%.2f", totalVmCost));

        if (finishedCount > 0) {
            LOGGER.info("Mean cost per Cloudlet = ${}", String.format("%.2f", totalCloudletCost / finishedCount));
            LOGGER.info("Mean Completion Time per Cloudlet = {} seconds",
                    String.format("%.2f", totalCompletionTime / finishedCount));
        } else {
            LOGGER.info("No finished Cloudlets to calculate mean costs/time.");
        }

        if (utilizedVmCount > 0) {
            LOGGER.info("Mean cost per Utilized VM = ${}", String.format("%.2f",
                    totalUtilizedVmCost / utilizedVmCount));
            LOGGER.info("Mean CPU Utilization of Utilized VMs = {}%",
                    String.format("%.2f", totalVmCpuUtilizationSum / utilizedVmCount));
        } else {
            LOGGER.info("No VMs were utilized to calculate mean VM costs/utilization.");
        }
    }
}
