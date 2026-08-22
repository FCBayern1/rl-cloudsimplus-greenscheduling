package exe.edu.cspg.energy;

import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

/**
 * Provides green energy data from wind turbine power generation.
 * Supports spline interpolation for continuous power queries.
 *
 * Simplified version - removed prediction model integration.
 * Future forecasts now use God's Eye (ground truth) from CSV data.
 */
public class GreenEnergyProvider {
    private static final Logger LOGGER = LoggerFactory.getLogger(GreenEnergyProvider.class.getSimpleName());

    // Configuration
    private final int turbineId;
    private final String csvFilePath;
    private final TimeScalingMode timeScalingMode;

    // Spline interpolation
    private PolynomialSplineFunction powerSpline;
    private double[] timePoints;
    private double[] powerValues;
    private double minTime;
    private double maxTime;

    // Energy statistics
    private double cumulativeGreenEnergyWh = 0;
    private double cumulativeBrownEnergyWh = 0;
    private double totalWastedGreenWh = 0;
    private double lastUpdateTime = 0;

    // Max power for normalization (computed from CSV data)
    private double maxPowerKw = 1.0;

    // Future trend feature configuration
    private final int shortTermRows;   // Default: 3 rows (30 minutes)
    private final int longTermRows;    // Default: 144 rows (24 hours)

    // Flag for simplified CSV format (only timestamp + power)
    private boolean isSimplifiedFormat = false;

    // Time zone offset for geo-distributed simulation (in rows/simulation seconds)
    // In COMPRESSED mode: 1 row = 10 min real time, so 6 hours = 36 rows
    // This simulates different geographic locations with different local times
    private final int timeZoneOffsetRows;

    // Global simulation warm-up offset (in rows). Added on top of timeZoneOffsetRows
    // so the simulation can start at row (warmup + tz) instead of (tz), giving every
    // DC at least `warmup` rows of real CSV history before sim_step=0 — eliminates
    // cold-start for downstream forecasters (e.g. TimeCAP). Defaults to 0.
    private final int simulationWarmupRows;

    // Divisor applied in COMPRESSED mode to scale raw CSV kW readings into the
    // per-simulation-second power regime. Configurable from config.yml via
    // "compressed_power_divisor"; defaults to 60.0.
    private final double compressedPowerDivisor;
    // Mode-independent multiplier applied AFTER mode handling (default 1.0 = legacy)
    private final double greenPowerScale;

    /**
     * Convert the configured time-zone + warm-up offset (in "rows") into
     * simulation-time seconds.
     *
     * - In {@link TimeScalingMode#COMPRESSED}: simulation time is treated as row index, so 1 row == 1 second.
     * - In {@link TimeScalingMode#REAL_TIME}: wind data points are typically 600s apart, so 1 row == 600 seconds.
     *
     * The warmup offset is added so the same row index is used uniformly across
     * "where is now in the CSV" lookups (current power, future trend features,
     * etc.) — i.e. sim_step=0 corresponds to CSV row (timeZoneOffsetRows + simulationWarmupRows).
     */
    private double getTimeZoneOffsetSeconds() {
        final double rowSeconds = timeScalingMode != null ? timeScalingMode.getTypicalInterval() : 1.0;
        return ((double) (timeZoneOffsetRows + simulationWarmupRows + episodeOffsetRows)) * rowSeconds;
    }

    /** Per-episode window shift (rows), set at reset when green_episode_offset_range
     *  is enabled. Shared by all providers in an episode so cross-DC phase structure
     *  is preserved; it re-bases WHICH slice of the wind year the episode replays.
     *  Without it every episode replays the same trajectory and a trained policy
     *  can memorise the future, making forecast observations redundant. */
    private int episodeOffsetRows = 0;

    public void setEpisodeOffsetRows(int rows) {
        this.episodeOffsetRows = Math.max(0, rows);
    }

    /**
     * Wrap a time value into [minTime, maxTime] assuming cyclic data.
     * Handles both overflow (>max) and underflow (<min).
     */
    private double wrapTimeCyclic(double t) {
        final double dataLength = maxTime - minTime;
        if (dataLength <= 0) return t;

        double shifted = t - minTime;
        shifted = shifted % dataLength;
        if (shifted < 0) shifted += dataLength;
        return minTime + shifted;
    }

    /**
     * Create a green energy provider for a specific wind turbine.
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath) {
        this(turbineId, csvFilePath, TimeScalingMode.REAL_TIME, 3, 144, 0);
    }

    /**
     * Create a green energy provider with specified time scaling mode.
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode) {
        this(turbineId, csvFilePath, timeScalingMode, 3, 144, 0);
    }

    /**
     * Create a green energy provider with full configuration (legacy, no timezone offset).
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode,
                              int shortTermRows, int longTermRows) {
        this(turbineId, csvFilePath, timeScalingMode, shortTermRows, longTermRows, 0);
    }

    /**
     * Create a green energy provider with full configuration including timezone offset.
     *
     * @param turbineId Wind turbine ID
     * @param csvFilePath Path to CSV file or directory
     * @param timeScalingMode COMPRESSED or REAL_TIME
     * @param shortTermRows Rows for short-term prediction features
     * @param longTermRows Rows for long-term prediction features
     * @param timeZoneOffsetRows Offset in rows to simulate different time zones.
     *                           In COMPRESSED mode: 6 rows = 1 hour (each row = 10 min real time)
     *                           Example offsets: US West=0, US East=18 (3h), Europe=54 (9h), APAC=96 (16h)
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode,
                              int shortTermRows, int longTermRows, int timeZoneOffsetRows) {
        this(turbineId, csvFilePath, timeScalingMode, shortTermRows, longTermRows, timeZoneOffsetRows, 60.0);
    }

    /**
     * Full-config constructor (legacy: no simulation warm-up offset). Delegates
     * to the 8-arg constructor with simulationWarmupRows = 0 so existing call
     * sites keep working unchanged.
     *
     * @param compressedPowerDivisor divisor applied to wind power readings
     *                               when {@code timeScalingMode == COMPRESSED}.
     *                               Must be positive; a non-positive value
     *                               falls back to the historical 60.0.
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode,
                              int shortTermRows, int longTermRows, int timeZoneOffsetRows,
                              double compressedPowerDivisor) {
        this(turbineId, csvFilePath, timeScalingMode, shortTermRows, longTermRows,
             timeZoneOffsetRows, compressedPowerDivisor, 0);
    }

    /**
     * Full-config constructor including the global simulation warm-up offset.
     *
     * @param simulationWarmupRows extra row offset added on top of
     *                             timeZoneOffsetRows so sim_step=0 maps to
     *                             CSV row (timeZoneOffsetRows + simulationWarmupRows).
     *                             Recommended: 96 (= 16 h history available
     *                             at episode start). Use 0 for legacy behaviour.
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode,
                              int shortTermRows, int longTermRows, int timeZoneOffsetRows,
                              double compressedPowerDivisor, int simulationWarmupRows) {
        this(turbineId, csvFilePath, timeScalingMode, shortTermRows, longTermRows,
             timeZoneOffsetRows, compressedPowerDivisor, simulationWarmupRows, 1.0);
    }

    /**
     * Full constructor with the mode-independent green power multiplier
     * (testbed 12, Codex 2026-08-22). Applied after REAL_TIME passthrough or
     * the COMPRESSED divisor in every power read path; 1.0 = legacy.
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode,
                              int shortTermRows, int longTermRows, int timeZoneOffsetRows,
                              double compressedPowerDivisor, int simulationWarmupRows,
                              double greenPowerScale) {
        this.greenPowerScale = greenPowerScale > 0 ? greenPowerScale : 1.0;
        this.turbineId = turbineId;
        this.csvFilePath = resolveCsvPath(turbineId, csvFilePath);
        this.timeScalingMode = timeScalingMode;
        this.shortTermRows = shortTermRows;
        this.longTermRows = longTermRows;
        this.timeZoneOffsetRows = timeZoneOffsetRows;
        this.simulationWarmupRows = Math.max(0, simulationWarmupRows);
        this.compressedPowerDivisor = compressedPowerDivisor > 0 ? compressedPowerDivisor : 60.0;

        LOGGER.info("Initialising GreenEnergyProvider for turbine {} with CSV '{}', mode: {}, tzOffset: {} rows, warmup: {} rows, compressedDivisor: {}",
                   turbineId, this.csvFilePath, timeScalingMode.getDescription(),
                   timeZoneOffsetRows, this.simulationWarmupRows, this.compressedPowerDivisor);
        loadAndBuildSpline();
        LOGGER.info("GreenEnergyProvider initialized successfully");
    }

    /**
     * Resolve the actual CSV path to load.
     * Supports both simplified format (2 columns) and legacy format (15 columns).
     * Prefers simplified directory if available.
     */
    private String resolveCsvPath(int turbineId, String csvFilePath) {
        if (csvFilePath == null || csvFilePath.isBlank()) {
            csvFilePath = "windProduction/simplified";
        }

        // If it's a concrete CSV file, use as-is
        String lower = csvFilePath.toLowerCase();
        if (lower.endsWith(".csv")) {
            return csvFilePath;
        }

        // Treat as a directory/base path
        String base = csvFilePath.endsWith("/") ? csvFilePath.substring(0, csvFilePath.length() - 1) : csvFilePath;

        // Try simplified directory first
        String simplifiedPath = base.replace("/split", "/simplified");
        if (!simplifiedPath.contains("simplified")) {
            simplifiedPath = base + "/../simplified";
        }

        String resolved = findTurbineFile(simplifiedPath, turbineId);
        if (resolved != null) {
            isSimplifiedFormat = true;
            LOGGER.info("Using simplified format CSV: {}", resolved);
            return resolved;
        }

        // Fallback to original directory
        resolved = findTurbineFile(base, turbineId);
        if (resolved != null) {
            isSimplifiedFormat = false;
            LOGGER.info("Using legacy format CSV: {}", resolved);
            return resolved;
        }

        // Default fallback
        String fallback = "windProduction/simplified/Turbine_" + turbineId + "_2021.csv";
        LOGGER.warn("Could not find CSV for turbine {}, using fallback: {}", turbineId, fallback);
        isSimplifiedFormat = true;
        return fallback;
    }

    /**
     * Try to find a turbine CSV file with various naming patterns.
     */
    private String findTurbineFile(String baseDir, int turbineId) {
        String[] patterns = {
            baseDir + "/Turbine_" + turbineId + "_2021.csv",
            baseDir + "/Turbine_" + turbineId + "_2020.csv",
            baseDir + "/Turbine_" + turbineId + "_2022.csv",
            baseDir + "/turbine_" + turbineId + "_2021.csv",
            baseDir + "/Turbine_" + turbineId + ".csv",
        };

        for (String pattern : patterns) {
            InputStream is = getClass().getClassLoader().getResourceAsStream(pattern);
            if (is != null) {
                try { is.close(); } catch (Exception ignored) {}
                return pattern;
            }

            java.io.File file = new java.io.File(pattern);
            if (file.exists() && file.isFile()) {
                return pattern;
            }
        }

        return null;
    }

    /**
     * Get current green power at specified simulation time.
     * Applies timezone offset for geo-distributed simulation.
     */
    public double getCurrentPowerW(double simulationTime) {
        if (powerSpline == null) {
            LOGGER.warn("Power spline not initialized, returning 0");
            return 0;
        }

        try {
            // Apply timezone offset to simulate different geographic locations
            double adjustedTime = simulationTime + getTimeZoneOffsetSeconds();
            // Wrap around for cyclic behavior (both directions)
            adjustedTime = wrapTimeCyclic(adjustedTime);

            double powerKW = powerSpline.value(adjustedTime);
            double powerW = Math.max(0, powerKW * 1000);

            if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                powerW = powerW / compressedPowerDivisor;
            }

            return powerW * greenPowerScale;
        } catch (OutOfRangeException e) {
            return 0;
        } catch (Exception e) {
            LOGGER.error("Error interpolating power at time {}: {}", simulationTime, e.getMessage());
            return 0;
        }
    }

    /**
     * Allocate energy between green and brown sources.
     */
    public EnergyAllocation allocateEnergy(double demandPowerW, double currentTime, double timeDelta) {
        double greenPowerW = getCurrentPowerW(currentTime);
        double timeDeltaHours = timeDelta / 3600.0;

        double demandWh = demandPowerW * timeDeltaHours;
        double greenAvailableWh = greenPowerW * timeDeltaHours;

        double greenUsedWh = Math.min(demandWh, greenAvailableWh);
        double brownUsedWh = demandWh - greenUsedWh;
        double wastedGreenWh = greenAvailableWh - greenUsedWh;

        cumulativeGreenEnergyWh += greenUsedWh;
        cumulativeBrownEnergyWh += brownUsedWh;
        totalWastedGreenWh += wastedGreenWh;
        lastUpdateTime = currentTime;

        return new EnergyAllocation(
            greenUsedWh, brownUsedWh, wastedGreenWh,
            greenPowerW, demandPowerW
        );
    }

    /**
     * Get future green power predictions using ground truth (God's Eye).
     * Applies timezone offset for geo-distributed simulation.
     */
    public double[] getFuturePowerPredictions(double currentTime, int[] horizonSeconds) {
        if (powerSpline == null || horizonSeconds == null) {
            return new double[0];
        }

        double[] predictions = new double[horizonSeconds.length];
        final double tzOffsetSec = getTimeZoneOffsetSeconds();

        for (int i = 0; i < horizonSeconds.length; i++) {
            // Apply timezone offset
            double futureTime = currentTime + horizonSeconds[i] + tzOffsetSec;
            // Wrap around for cyclic behavior (both directions)
            futureTime = wrapTimeCyclic(futureTime);

            try {
                double futurePowerKW = powerSpline.value(futureTime);
                double futurePowerW = Math.max(0, futurePowerKW * 1000);

                if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                    futurePowerW = futurePowerW / compressedPowerDivisor;
                }

                predictions[i] = futurePowerW * greenPowerScale;
            } catch (OutOfRangeException e) {
                predictions[i] = 0.0;
            }
        }

        return predictions;
    }

    /**
     * Mean green power (W) over the next {@code horizonSteps} simulation steps,
     * starting at {@code simTime}. Used by the window-aware per-action carbon
     * reward (per_action_window_carbon): for a long cloudlet the carbon it
     * actually causes depends on the green supply over its whole run window,
     * not on the instant it was routed. Unit handling mirrors
     * {@link #getCurrentPowerW}: CSV kW → W, divided by the compressed-power
     * divisor in COMPRESSED mode. In COMPRESSED mode 1 step == 1 CSV row and
     * indices wrap cyclically (same as simTimeToRowIndex). Falls back to the
     * instantaneous value outside COMPRESSED mode or without data.
     */
    public double getMeanFuturePowerW(double simTime, int horizonSteps) {
        if (timeScalingMode != TimeScalingMode.COMPRESSED
                || powerValues == null || powerValues.length == 0 || horizonSteps <= 1) {
            return getCurrentPowerW(simTime);
        }
        int n = powerValues.length;
        int start = simTimeToRowIndex(simTime);
        if (start < 0) {
            return getCurrentPowerW(simTime);
        }
        double sumKw = 0;
        for (int i = 0; i < horizonSteps; i++) {
            sumKw += powerValues[(start + i) % n];
        }
        double meanW = Math.max(0, (sumKw / horizonSteps) * 1000.0);
        return meanW / compressedPowerDivisor * greenPowerScale;
    }

    /**
     * Compute future trend features for RL observation (God's Eye mode).
     */
    public double[] computeFutureTrendFeatures(double simTime) {
        double[] features = new double[4];
        features[0] = 0.5;  // short_mean
        features[1] = 0.0;  // short_trend
        features[2] = 0.5;  // long_mean
        features[3] = 0.5;  // long_peak_timing

        if (powerValues == null || powerValues.length == 0) {
            return features;
        }

        int currentIdx = simTimeToRowIndex(simTime);
        if (currentIdx < 0 || currentIdx >= powerValues.length) {
            return features;
        }

        // Short-term features
        int shortEndIdx = Math.min(currentIdx + shortTermRows, powerValues.length);
        int shortAvailable = shortEndIdx - currentIdx;

        if (shortAvailable > 0) {
            double shortSum = 0;
            for (int i = currentIdx; i < shortEndIdx; i++) {
                shortSum += powerValues[i];
            }
            double shortMean = shortSum / shortAvailable;
            features[0] = Math.min(1.0, Math.max(0.0, shortMean / maxPowerKw));

            double startPower = powerValues[currentIdx];
            double endPower = powerValues[shortEndIdx - 1];
            double shortTrend = (endPower - startPower) / maxPowerKw;
            features[1] = Math.min(1.0, Math.max(-1.0, shortTrend));
        }

        // Long-term features
        int longEndIdx = Math.min(currentIdx + longTermRows, powerValues.length);
        int longAvailable = longEndIdx - currentIdx;

        if (longAvailable > 0) {
            double longSum = 0;
            int peakIdx = currentIdx;
            double peakPower = powerValues[currentIdx];

            for (int i = currentIdx; i < longEndIdx; i++) {
                longSum += powerValues[i];
                if (powerValues[i] > peakPower) {
                    peakPower = powerValues[i];
                    peakIdx = i;
                }
            }
            double longMean = longSum / longAvailable;
            features[2] = Math.min(1.0, Math.max(0.0, longMean / maxPowerKw));

            double peakTiming = (double)(peakIdx - currentIdx) / longAvailable;
            features[3] = Math.min(1.0, Math.max(0.0, peakTiming));
        }

        return features;
    }

    /**
     * Convert simulation time to CSV row index.
     * Applies timezone offset for geo-distributed simulation.
     */
    private int simTimeToRowIndex(double simTime) {
        // Apply timezone offset
        double adjustedTime = simTime + getTimeZoneOffsetSeconds();

        if (timeScalingMode == TimeScalingMode.COMPRESSED) {
            int index = (int) Math.round(adjustedTime);
            // Wrap around for cyclic behavior
            if (powerValues != null && powerValues.length > 0) {
                int n = powerValues.length;
                index = index % n;
                if (index < 0) index += n;
            }
            return index;
        } else {
            if (timePoints == null || timePoints.length == 0) {
                return -1;
            }

            int low = 0;
            int high = timePoints.length - 1;

            while (low < high) {
                int mid = (low + high) / 2;
                if (timePoints[mid] < adjustedTime) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }

            if (low > 0 && Math.abs(timePoints[low - 1] - adjustedTime) < Math.abs(timePoints[low] - adjustedTime)) {
                return low - 1;
            }
            return low;
        }
    }

    // ==================== Statistics Methods ====================

    public double getCumulativeGreenEnergyWh() {
        return cumulativeGreenEnergyWh;
    }

    public double getCumulativeBrownEnergyWh() {
        return cumulativeBrownEnergyWh;
    }

    public double getTotalWastedGreenWh() {
        return totalWastedGreenWh;
    }

    public double getOverallGreenRatio() {
        double total = cumulativeGreenEnergyWh + cumulativeBrownEnergyWh;
        return total > 0 ? cumulativeGreenEnergyWh / total : 0.0;
    }

    public double getMaxPowerKw() {
        return maxPowerKw;
    }

    public void resetStatistics() {
        cumulativeGreenEnergyWh = 0;
        cumulativeBrownEnergyWh = 0;
        totalWastedGreenWh = 0;
        lastUpdateTime = 0;
    }

    public void updateCumulativeStatistics(double greenUsedWh, double greenWastedWh) {
        cumulativeGreenEnergyWh += greenUsedWh;
        totalWastedGreenWh += greenWastedWh;
    }

    // ==================== Multi-Turbine Aggregation ====================

    public static double[] computeAggregatedFutureTrendFeatures(
            List<GreenEnergyProvider> providers, double simTime) {

        double[] aggregated = new double[]{0.5, 0.0, 0.5, 0.5};

        if (providers == null || providers.isEmpty()) {
            return aggregated;
        }

        if (providers.size() == 1) {
            return providers.get(0).computeFutureTrendFeatures(simTime);
        }

        double totalMaxPower = 0.0;
        double weightedShortMean = 0.0;
        double weightedShortTrend = 0.0;
        double weightedLongMean = 0.0;
        double earliestPeakTiming = 1.0;

        for (GreenEnergyProvider provider : providers) {
            double[] features = provider.computeFutureTrendFeatures(simTime);
            double maxPower = provider.getMaxPowerKw();

            totalMaxPower += maxPower;
            weightedShortMean += features[0] * maxPower;
            weightedShortTrend += features[1] * maxPower;
            weightedLongMean += features[2] * maxPower;

            if (features[3] < earliestPeakTiming) {
                earliestPeakTiming = features[3];
            }
        }

        if (totalMaxPower > 0) {
            aggregated[0] = Math.min(1.0, weightedShortMean / totalMaxPower);
            aggregated[1] = Math.max(-1.0, Math.min(1.0, weightedShortTrend / totalMaxPower));
            aggregated[2] = Math.min(1.0, weightedLongMean / totalMaxPower);
        }
        aggregated[3] = earliestPeakTiming;

        return aggregated;
    }

    // ==================== CSV Loading ====================

    /**
     * Load CSV data and build spline interpolation function.
     */
    private void loadAndBuildSpline() {
        try {
            List<WindDataPoint> dataPoints = loadCsvData();
            LOGGER.info("Loaded {} data points for turbine {}", dataPoints.size(), turbineId);

            if (dataPoints.isEmpty()) {
                LOGGER.error("No data points loaded for turbine {} from file: {}", turbineId, csvFilePath);
                return;
            }

            // Extract time and power sequences, removing duplicates
            List<Double> uniqueTimes = new ArrayList<>();
            List<Double> uniquePowers = new ArrayList<>();

            double lastTime = Double.NEGATIVE_INFINITY;
            int duplicatesRemoved = 0;

            for (WindDataPoint dp : dataPoints) {
                if (dp.timestamp > lastTime) {
                    uniqueTimes.add(dp.timestamp);
                    uniquePowers.add(dp.powerKW);
                    lastTime = dp.timestamp;
                } else {
                    duplicatesRemoved++;
                }
            }

            if (duplicatesRemoved > 0) {
                LOGGER.warn("Removed {} duplicate timestamps", duplicatesRemoved);
            }

            timePoints = uniqueTimes.stream().mapToDouble(Double::doubleValue).toArray();
            powerValues = uniquePowers.stream().mapToDouble(Double::doubleValue).toArray();

            minTime = timePoints[0];
            maxTime = timePoints[timePoints.length - 1];

            SplineInterpolator interpolator = new SplineInterpolator();
            powerSpline = interpolator.interpolate(timePoints, powerValues);

            maxPowerKw = java.util.Arrays.stream(powerValues).max().orElse(1.0);
            if (maxPowerKw <= 0) {
                maxPowerKw = 1.0;
            }

            LOGGER.info("Spline built: {} points, time [{}, {}], power [{}, {}] kW",
                       timePoints.length, minTime, maxTime,
                       java.util.Arrays.stream(powerValues).min().orElse(0), maxPowerKw);

        } catch (Exception e) {
            LOGGER.error("Failed to load and build spline: {}", e.getMessage(), e);
        }
    }

    /**
     * Load CSV data - supports both simplified (2 columns) and legacy (15 columns) formats.
     */
    private List<WindDataPoint> loadCsvData() throws IOException {
        List<WindDataPoint> result = new ArrayList<>();

        BufferedReader reader = null;
        try {
            InputStream is = getClass().getClassLoader().getResourceAsStream(csvFilePath);
            if (is != null) {
                reader = new BufferedReader(new InputStreamReader(is));
                LOGGER.info("Loading CSV from resources: {}", csvFilePath);
            } else {
                reader = new BufferedReader(new FileReader(csvFilePath));
                LOGGER.info("Loading CSV from file system: {}", csvFilePath);
            }

            String header = reader.readLine();
            LOGGER.info("CSV Header: {}", header);

            // Auto-detect format from header
            boolean simplified = header != null && header.toLowerCase().contains("power_kw")
                                 && !header.toLowerCase().contains("wspd");
            if (simplified != isSimplifiedFormat) {
                LOGGER.info("Format auto-detected: {}", simplified ? "simplified" : "legacy");
                isSimplifiedFormat = simplified;
            }

            String line;
            long baseTimestamp = -1;
            int lineCount = 0;
            int rowIndex = 0;

            while ((line = reader.readLine()) != null) {
                lineCount++;
                String[] parts = line.split(",");

                double timestamp;
                double powerKW;

                if (isSimplifiedFormat) {
                    // Simplified format: timestamp,power_kw
                    if (parts.length < 2) continue;

                    if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                        if (rowIndex < 12) {
                            rowIndex++;
                            continue;
                        }
                        timestamp = rowIndex - 12;
                        rowIndex++;
                    } else {
                        try {
                            timestamp = parseTimestamp(parts[0].trim(), baseTimestamp);
                            if (baseTimestamp < 0) {
                                // parseTimestamp returns an absolute timestamp in milliseconds
                                // when baseTimestamp is not set yet. Keep baseTimestamp in ms.
                                baseTimestamp = (long) timestamp;
                                timestamp = 0;
                            }
                        } catch (Exception e) {
                            continue;
                        }
                    }

                    powerKW = parseDoubleOrZero(parts[1]);

                } else {
                    // Legacy format: TurbID,Tmstamp,...,Patv (15 columns)
                    if (parts.length < 15) continue;

                    if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                        if (rowIndex < 12) {
                            rowIndex++;
                            continue;
                        }
                        timestamp = rowIndex - 12;
                        rowIndex++;
                    } else {
                        try {
                            timestamp = parseTimestamp(parts[1].trim(), baseTimestamp);
                            if (baseTimestamp < 0) {
                                // parseTimestamp returns an absolute timestamp in milliseconds
                                // when baseTimestamp is not set yet. Keep baseTimestamp in ms.
                                baseTimestamp = (long) timestamp;
                                timestamp = 0;
                            }
                        } catch (Exception e) {
                            continue;
                        }
                    }

                    powerKW = parseDoubleOrZero(parts[14]);  // Patv column
                }

                result.add(new WindDataPoint(timestamp, powerKW));

                if (lineCount % 100000 == 0) {
                    LOGGER.info("Progress: {} lines processed...", lineCount);
                }
            }

            LOGGER.info("CSV loading complete: {} data points from {} lines", result.size(), lineCount);

        } finally {
            if (reader != null) {
                reader.close();
            }
        }

        return result;
    }

    private double parseTimestamp(String timestampStr, long baseTimestamp) {
        Timestamp ts;

        if (timestampStr.contains("/")) {
            timestampStr = timestampStr.replace("/", "-") + ":00";
            String[] dateParts = timestampStr.split(" ")[0].split("-");
            String[] timeParts = timestampStr.split(" ")[1].split(":");
            String formatted = String.format("%04d-%02d-%02d %02d:%02d:%02d",
                Integer.parseInt(dateParts[0]),
                Integer.parseInt(dateParts[1]),
                Integer.parseInt(dateParts[2]),
                Integer.parseInt(timeParts[0]),
                Integer.parseInt(timeParts[1]),
                Integer.parseInt(timeParts[2]));
            ts = Timestamp.valueOf(formatted);
        } else {
            ts = Timestamp.valueOf(timestampStr);
        }

        if (baseTimestamp < 0) {
            return ts.getTime();
        }
        return (ts.getTime() - baseTimestamp) / 1000.0;
    }

    private double parseDoubleOrZero(String value) {
        if (value == null || value.trim().isEmpty()) {
            return 0.0;
        }
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException e) {
            return 0.0;
        }
    }

    /**
     * Simple data structure for storing time-power pairs.
     */
    private static class WindDataPoint {
        final double timestamp;
        final double powerKW;

        WindDataPoint(double timestamp, double powerKW) {
            this.timestamp = timestamp;
            this.powerKW = powerKW;
        }
    }
}
