package giu.edu.cspg.energy;

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
 * Supports spline interpolation for continuous power queries and
 * maintains historical data buffer for prediction inputs.
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

    // Historical data buffer for prediction
    private final CircularBuffer<WindDataFrame> historyBuffer;
    private static final int HISTORY_SIZE = 12;

    // Energy statistics
    private double cumulativeGreenEnergyWh = 0;
    private double cumulativeBrownEnergyWh = 0;
    private double totalWastedGreenWh = 0;
    private double lastUpdateTime = 0;

    // Prediction cache
    private double[] cachedPrediction = null;
    private double predictionCacheTime = -1;
    private static final double PREDICTION_CACHE_DURATION = 600; // 10 minutes

    /**
     * Create a green energy provider for a specific wind turbine.
     * Uses REAL_TIME mode by default (preserves original 600s intervals).
     *
     * @param turbineId   Turbine ID to load data for
     * @param csvFilePath Path to CSV data file (relative to resources or absolute)
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath) {
        this(turbineId, csvFilePath, TimeScalingMode.REAL_TIME);
    }

    /**
     * Create a green energy provider with specified time scaling mode.
     *
     * @param turbineId       Turbine ID to load data for
     * @param csvFilePath     Path to CSV data file (relative to resources or absolute)
     * @param timeScalingMode Time scaling mode (REAL_TIME or COMPRESSED)
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath, TimeScalingMode timeScalingMode) {
        this.turbineId = turbineId;
        this.csvFilePath = csvFilePath;
        this.timeScalingMode = timeScalingMode;
        this.historyBuffer = new CircularBuffer<>(HISTORY_SIZE);

        LOGGER.info("Initialising GreenEnergyProvider for turbine {} with mode: {}...",
                   turbineId, timeScalingMode.getDescription());
        loadAndBuildSpline();
        LOGGER.info("GreenEnergyProvider initialized successfully");
    }

    /**
     * Get current green power at specified simulation time (with spline interpolation).
     *
     * @param simulationTime Simulation time in seconds
     * @return Green power in Watts, or 0 if out of range
     */
    public double getCurrentPowerW(double simulationTime) {
        if (powerSpline == null) {
            LOGGER.warn("Power spline not initialized, returning 0");
            return 0;
        }

        try {
            // Use simulation time directly (assumes time mapping is handled elsewhere if needed)
            double powerKW = powerSpline.value(simulationTime);
            double powerW = Math.max(0, powerKW * 1000);  // kW to W, clip negative values

            // COMPRESSED mode scaling: CSV data represents 600s intervals compressed to 1s
            // Scale power down to match the time compression ratio
            if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                // Real-world data: 1 CSV row = 600 seconds (10 minutes)
                // Simulation: 1 CSV row = 1 second
                // Therefore, scale power by 600 to match energy availability per simulation second
                powerW = powerW / 600.0;
                LOGGER.trace("COMPRESSED mode: Scaled power from {} W to {} W (÷600)",
                           powerKW * 1000, powerW);
            }

            return powerW;
        } catch (OutOfRangeException e) {
            if (simulationTime < minTime || simulationTime > maxTime) {
                LOGGER.debug("Simulation time {} out of data range [{}, {}], returning 0",
                           simulationTime, minTime, maxTime);
            }
            return 0;
        } catch (Exception e) {
            LOGGER.error("Error interpolating power at time {}: {}", simulationTime, e.getMessage());
            return 0;
        }
    }

    /**
     * Update history buffer with current timestep data.
     * Fetches complete feature data from interpolated/cached sources.
     *
     * @param simulationTime Current simulation time
     * @param powerW         Current power in W
     */
    public void updateHistoryBuffer(double simulationTime, double powerW) {
        // For now, create a simple frame with power only
        // In full implementation, would fetch all 16 features
        double[] features = new double[WindDataFrame.NUM_FEATURES];
        features[WindDataFrame.POWER_INDEX] = powerW / 1000.0;  // W to kW

        WindDataFrame frame = new WindDataFrame(simulationTime, features);
        historyBuffer.add(frame);
    }

    /**
     * Get historical data for prediction input.
     *
     * @param lookback Number of historical timesteps to retrieve
     * @return List of historical frames (may be less than lookback if insufficient data)
     */
    public List<WindDataFrame> getHistoricalData(int lookback) {
        return historyBuffer.getLast(lookback);
    }

    /**
     * Get predicted future power values (requires external prediction service).
     * Uses caching to avoid excessive calls.
     *
     * @param windPredictor Prediction service object (Py4J interface)
     * @param horizon       Number of future steps to predict
     * @param currentTime   Current simulation time (for cache validation)
     * @return Array of predicted power values in Watts
     */
    public double[] getPredictedPowerW(Object windPredictor, int horizon, double currentTime) {
        // Check cache validity
        if (cachedPrediction != null &&
            cachedPrediction.length == horizon &&
            (currentTime - predictionCacheTime) < PREDICTION_CACHE_DURATION) {
            LOGGER.debug("Using cached prediction (age: {}s)", currentTime - predictionCacheTime);
            return cachedPrediction;
        }

        // Get historical data
        List<WindDataFrame> history = getHistoricalData(12);
        if (history.size() < 12) {
            LOGGER.warn("Insufficient history for prediction: {}/12, returning zeros", history.size());
            return new double[horizon];
        }

        // Call prediction service (implementation depends on Py4J interface)
        try {
            double[] predictedKW = invokePythonPredictor(windPredictor, history, horizon);

            // Convert kW to W and cache
            cachedPrediction = new double[predictedKW.length];
            for (int i = 0; i < predictedKW.length; i++) {
                cachedPrediction[i] = Math.max(0, predictedKW[i] * 1000);
            }

            predictionCacheTime = currentTime;
            LOGGER.debug("Prediction updated: {} steps ahead", horizon);

            return cachedPrediction;
        } catch (Exception e) {
            LOGGER.error("Prediction failed: {}", e.getMessage(), e);
            return new double[horizon];  // Return zeros on failure
        }
    }

    /**
     * Allocate energy between green and brown sources.
     * Green energy is prioritized; excess is wasted (not stored).
     *
     * @param demandPowerW  Total power demand in W
     * @param currentTime   Current simulation time in seconds
     * @param timeDelta     Time interval in seconds
     * @return EnergyAllocation result
     */
    public EnergyAllocation allocateEnergy(double demandPowerW, double currentTime, double timeDelta) {
        double greenPowerW = getCurrentPowerW(currentTime);
        double timeDeltaHours = timeDelta / 3600.0;

        // Calculate energy amounts (Wh)
        double demandWh = demandPowerW * timeDeltaHours;
        double greenAvailableWh = greenPowerW * timeDeltaHours;

        // Allocation strategy: prioritize green energy
        double greenUsedWh = Math.min(demandWh, greenAvailableWh);
        double brownUsedWh = demandWh - greenUsedWh;
        double wastedGreenWh = greenAvailableWh - greenUsedWh;  // Excess is wasted

        // Update cumulative statistics
        cumulativeGreenEnergyWh += greenUsedWh;
        cumulativeBrownEnergyWh += brownUsedWh;
        totalWastedGreenWh += wastedGreenWh;
        lastUpdateTime = currentTime;

        // Update history buffer
        updateHistoryBuffer(currentTime, greenPowerW);

        return new EnergyAllocation(
            greenUsedWh, brownUsedWh, wastedGreenWh,
            greenPowerW, demandPowerW
        );
    }

    /**
     * Get future green power predictions at specified time horizons.
     * Uses ground truth data from the CSV (perfect prediction).
     *
     * @param currentTime Current simulation time in seconds
     * @param horizonSeconds Array of future time horizons in seconds (e.g., [300, 900, 1800, 3600])
     * @return Array of predicted power values in Watts for each horizon
     */
    public double[] getFuturePowerPredictions(double currentTime, int[] horizonSeconds) {
        if (powerSpline == null || horizonSeconds == null) {
            return new double[0];
        }

        double[] predictions = new double[horizonSeconds.length];

        for (int i = 0; i < horizonSeconds.length; i++) {
            double futureTime = currentTime + horizonSeconds[i];

            // Check if future time is within data range
            if (futureTime > maxTime) {
                // Beyond available data, use last known value
                predictions[i] = getCurrentPowerW(maxTime);
                LOGGER.trace("Future time {} exceeds data range, using last value", futureTime);
            } else {
                // Get power at future time point (ground truth from CSV)
                try {
                    double futurePowerKW = powerSpline.value(futureTime);
                    double futurePowerW = Math.max(0, futurePowerKW * 1000);

                    // Apply same scaling as getCurrentPowerW for consistency
                    if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                        futurePowerW = futurePowerW / 600.0;
                    }

                    predictions[i] = futurePowerW;
                } catch (OutOfRangeException e) {
                    predictions[i] = 0.0;
                    LOGGER.debug("Future time {} out of range, returning 0", futureTime);
                }
            }
        }

        return predictions;
    }

    /**
     * Get future green power predictions with confidence intervals.
     * Uses ground truth mean and estimates uncertainty from historical variance.
     *
     * @param currentTime Current simulation time in seconds
     * @param horizonSeconds Array of future time horizons in seconds
     * @return 2D array [horizon][3] where each row is [lower_95%, mean, upper_95%]
     */
    public double[][] getFuturePowerPredictionsWithCI(double currentTime, int[] horizonSeconds) {
        if (powerSpline == null || horizonSeconds == null) {
            return new double[0][3];
        }

        double[][] predictions = new double[horizonSeconds.length][3];

        // Estimate standard deviation from historical data (simple approach)
        // TODO: Could be improved with rolling window statistics
        double estimatedStdDev = estimateHistoricalStdDev();

        for (int i = 0; i < horizonSeconds.length; i++) {
            double futureTime = currentTime + horizonSeconds[i];
            double meanPower = 0.0;

            if (futureTime > maxTime) {
                meanPower = getCurrentPowerW(maxTime);
            } else {
                try {
                    double futurePowerKW = powerSpline.value(futureTime);
                    meanPower = Math.max(0, futurePowerKW * 1000);

                    if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                        meanPower = meanPower / 600.0;
                    }
                } catch (OutOfRangeException e) {
                    meanPower = 0.0;
                }
            }

            // Calculate 95% confidence interval (1.96 * std dev)
            double margin = 1.96 * estimatedStdDev;
            predictions[i][0] = Math.max(0, meanPower - margin);  // Lower bound
            predictions[i][1] = meanPower;                        // Mean
            predictions[i][2] = meanPower + margin;               // Upper bound
        }

        return predictions;
    }

    /**
     * Estimate standard deviation from historical power data.
     * Simple implementation using overall variance.
     *
     * @return Estimated standard deviation in Watts
     */
    private double estimateHistoricalStdDev() {
        if (powerValues == null || powerValues.length < 2) {
            return 0.0;
        }

        // Calculate mean
        double sum = 0.0;
        for (double powerKW : powerValues) {
            double powerW = powerKW * 1000;
            if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                powerW = powerW / 600.0;
            }
            sum += powerW;
        }
        double mean = sum / powerValues.length;

        // Calculate variance
        double varianceSum = 0.0;
        for (double powerKW : powerValues) {
            double powerW = powerKW * 1000;
            if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                powerW = powerW / 600.0;
            }
            double diff = powerW - mean;
            varianceSum += diff * diff;
        }
        double variance = varianceSum / powerValues.length;

        return Math.sqrt(variance);
    }

    /**
     * Get cumulative green energy consumed.
     *
     * @return Total green energy in Wh
     */
    public double getCumulativeGreenEnergyWh() {
        return cumulativeGreenEnergyWh;
    }

    /**
     * Get cumulative brown energy consumed.
     *
     * @return Total brown energy in Wh
     */
    public double getCumulativeBrownEnergyWh() {
        return cumulativeBrownEnergyWh;
    }

    /**
     * Get total wasted green energy.
     *
     * @return Total wasted energy in Wh
     */
    public double getTotalWastedGreenWh() {
        return totalWastedGreenWh;
    }

    /**
     * Get overall green energy ratio.
     *
     * @return Ratio of green to total energy (0-1)
     */
    public double getOverallGreenRatio() {
        double total = cumulativeGreenEnergyWh + cumulativeBrownEnergyWh;
        return total > 0 ? cumulativeGreenEnergyWh / total : 0.0;
    }

    /**
     * Reset cumulative statistics.
     */
    public void resetStatistics() {
        cumulativeGreenEnergyWh = 0;
        cumulativeBrownEnergyWh = 0;
        totalWastedGreenWh = 0;
        lastUpdateTime = 0;
        historyBuffer.clear();
        cachedPrediction = null;
        predictionCacheTime = -1;
    }

    // ==================== Private Helper Methods ====================

    /**
     * Load CSV data and build spline interpolation function.
     */
    private void loadAndBuildSpline() {
        try {
            LOGGER.info("Starting loadCsvData for turbine {} from file: {}", turbineId, csvFilePath);
            List<WindDataPoint> dataPoints = loadCsvData();
            LOGGER.info("Loaded {} data points for turbine {}", dataPoints.size(), turbineId);

            if (dataPoints.isEmpty()) {
                LOGGER.error(" No data points loaded for turbine {} from file: {}", turbineId, csvFilePath);
                LOGGER.error("   Please verify:");
                LOGGER.error("   1. File exists: {}", csvFilePath);
                LOGGER.error("   2. Turbine ID {} exists in CSV", turbineId);
                LOGGER.error("   3. CSV format is correct (18 columns)");
                return;
            }

            // Extract time and power sequences, removing duplicates
            List<Double> uniqueTimes = new ArrayList<>();
            List<Double> uniquePowers = new ArrayList<>();

            double lastTime = Double.NEGATIVE_INFINITY;
            int duplicatesRemoved = 0;

            for (WindDataPoint dp : dataPoints) {
                if (dp.timestamp > lastTime) {
                    // Strictly increasing timestamp
                    uniqueTimes.add(dp.timestamp);
                    uniquePowers.add(dp.powerKW);
                    lastTime = dp.timestamp;
                } else {
                    // Duplicate or non-monotonic timestamp, skip
                    duplicatesRemoved++;
                    if (duplicatesRemoved <= 5) {
                        LOGGER.warn("Skipping duplicate/non-monotonic timestamp at {} s (power: {} kW)",
                                   dp.timestamp, dp.powerKW);
                    }
                }
            }

            if (duplicatesRemoved > 0) {
                LOGGER.warn("Removed {} duplicate/non-monotonic timestamps from dataset", duplicatesRemoved);
            }

            // Convert to arrays
            timePoints = uniqueTimes.stream().mapToDouble(Double::doubleValue).toArray();
            powerValues = uniquePowers.stream().mapToDouble(Double::doubleValue).toArray();

            minTime = timePoints[0];
            maxTime = timePoints[timePoints.length - 1];

            // Build spline interpolator
            LOGGER.info("Building spline interpolator with {} unique points...", timePoints.length);
            SplineInterpolator interpolator = new SplineInterpolator();
            powerSpline = interpolator.interpolate(timePoints, powerValues);

            LOGGER.info(" Spline interpolation built successfully!");
            LOGGER.info("   Time scaling mode: {}", timeScalingMode.getDescription());
            LOGGER.info("   Data points: {} (original: {})", timePoints.length, dataPoints.size());
            LOGGER.info("   Time range: [{} s, {} s] ({} hours)",
                       minTime, maxTime, (maxTime - minTime) / 3600.0);
            LOGGER.info("   Power range: [{} kW, {} kW]",
                       java.util.Arrays.stream(powerValues).min().orElse(0),
                       java.util.Arrays.stream(powerValues).max().orElse(0));

            // Log data utilization for typical simulation duration
            double typicalSimDuration = 2000.0;  // 2000 seconds typical
            if (timeScalingMode == TimeScalingMode.REAL_TIME) {
                double dataPointsUsed = typicalSimDuration / 600.0;  // One per 10 minutes
                double utilizationPercent = (dataPointsUsed / timePoints.length) * 100.0;
                LOGGER.info("   Data utilization (2000s simulation): ~{} points ({}%)",
                           (int) Math.ceil(dataPointsUsed), utilizationPercent);
            } else if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                double dataPointsUsed = Math.min(typicalSimDuration, timePoints.length);
                double utilizationPercent = (dataPointsUsed / timePoints.length) * 100.0;
                LOGGER.info("   Data utilization (2000s simulation): {} points ({}%)",
                           (int) dataPointsUsed, utilizationPercent);
            }

        } catch (Exception e) {
            LOGGER.error(" Failed to load and build spline: {}", e.getMessage(), e);
            LOGGER.error("   Stack trace:", e);
        }
    }

    /**
     * Load CSV data for specified turbine.
     *
     * @return List of data points
     */
    private List<WindDataPoint> loadCsvData() throws IOException {
        List<WindDataPoint> result = new ArrayList<>();

        // Try to load from resources first, then from absolute path
        BufferedReader reader = null;
        try {
            InputStream is = getClass().getClassLoader().getResourceAsStream(csvFilePath);
            if (is != null) {
                reader = new BufferedReader(new InputStreamReader(is));
                LOGGER.info("Loading CSV from resources: {}", csvFilePath);
            } else {
                LOGGER.warn("CSV not found in resources, trying file system: {}", csvFilePath);
                reader = new BufferedReader(new FileReader(csvFilePath));
                LOGGER.info("Loading CSV from file system: {}", csvFilePath);
            }

            String line = reader.readLine();  // Skip header
            LOGGER.info("CSV Header: {}", line);

            long baseTimestamp = -1;
            int lineCount = 0;
            int matchedLines = 0;
            int skippedColumns = 0;
            int skippedTurbine = 0;
            int rowIndex = 0;  // For COMPRESSED mode

            while ((line = reader.readLine()) != null) {
                lineCount++;
                String[] parts = line.split(",");

                if (parts.length < 15) {
                    skippedColumns++;
                    if (skippedColumns <= 5) {  // Only log first 5 warnings
                        LOGGER.warn("Line {} has insufficient columns ({}), skipping: {}",
                                   lineCount, parts.length, line.substring(0, Math.min(80, line.length())));
                    }
                    continue;
                }

                // Parse turbine ID
                int tid;
                try {
                    tid = Integer.parseInt(parts[0].trim());
                } catch (NumberFormatException e) {
                    LOGGER.debug("Line {}: Invalid turbine ID, skipping", lineCount);
                    continue;  // Skip invalid lines
                }

                if (tid != turbineId) {
                    skippedTurbine++;
                    continue;  // Only load specified turbine
                }

                matchedLines++;

                // Log first matched line for debugging
                if (matchedLines == 1) {
                    LOGGER.info("First matched line for turbine {}: {}", turbineId,
                               line.substring(0, Math.min(100, line.length())));
                }

                // Determine timestamp based on mode
                double timestamp;

                if (timeScalingMode == TimeScalingMode.COMPRESSED) {
                    // COMPRESSED mode: Use row index as timestamp (1 data point = 1 second)
                    // Skip first 12 rows to ensure sufficient lookback for prediction
                    if (rowIndex < 12) {
                        rowIndex++;
                        continue;  // Skip this row, need 12 rows for lookback
                    }
                    timestamp = rowIndex - 12;  // Map: row 12 → timestamp 0, row 13 → timestamp 1, etc.
                    rowIndex++;
                } else {
                    // REAL_TIME mode: Use actual timestamp from CSV
                    try {
                        // Handle multiple timestamp formats:
                        // - "2020-01-03 01:25:00" (SQL format)
                        // - "2020/1/3 1:25" (CSV format from dataset)
                        String timestampStr = parts[1].trim();
                        Timestamp ts;

                        if (timestampStr.contains("/")) {
                            // Convert "2020/1/3 1:25" to SQL format
                            timestampStr = timestampStr
                                .replace("/", "-")  // 2020-1-3 1:25
                                .replaceFirst("(\\d+)-(\\d+)-(\\d+)", "$1-$2-$3")  // Keep format
                                + ":00";  // Add seconds
                            // Parse with flexible format
                            String[] dateParts = timestampStr.split(" ")[0].split("-");
                            String[] timeParts = timestampStr.split(" ")[1].split(":");
                            String formattedTimestamp = String.format("%04d-%02d-%02d %02d:%02d:%02d",
                                Integer.parseInt(dateParts[0]),
                                Integer.parseInt(dateParts[1]),
                                Integer.parseInt(dateParts[2]),
                                Integer.parseInt(timeParts[0]),
                                Integer.parseInt(timeParts[1]),
                                Integer.parseInt(timeParts[2]));
                            ts = Timestamp.valueOf(formattedTimestamp);
                        } else {
                            // Standard SQL timestamp format
                            ts = Timestamp.valueOf(timestampStr);
                        }

                        if (baseTimestamp < 0) baseTimestamp = ts.getTime();
                        timestamp = (ts.getTime() - baseTimestamp) / 1000.0;

                    } catch (Exception e) {
                        if (matchedLines <= 5) {  // Log first few errors for debugging
                            LOGGER.warn("Failed to parse line {}: {} (timestamp: '{}')",
                                       lineCount, e.getMessage(), parts[1]);
                        }
                        continue;  // Skip this line on timestamp parse error
                    }
                }

                // Parse power (Patv) - column 14 (15th column, 0-indexed)
                double powerKW = parseDoubleOrZero(parts[14]);

                result.add(new WindDataPoint(timestamp, powerKW));

                // Progress logging
                if (matchedLines % 100000 == 0) {
                    LOGGER.info("Progress: Loaded {} data points for turbine {}...", matchedLines, turbineId);
                }
            }

            LOGGER.info("CSV loading complete:");
            LOGGER.info("  Total lines processed: {}", lineCount);
            LOGGER.info("  Lines skipped (insufficient columns): {}", skippedColumns);
            LOGGER.info("  Lines skipped (different turbine): {}", skippedTurbine);
            LOGGER.info("  Matched lines for turbine {}: {}", turbineId, matchedLines);

        } finally {
            if (reader != null) {
                reader.close();
            }
        }

        return result;
    }

    /**
     * Parse double value or return 0 on failure.
     */
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
     * Invoke Python prediction service via Py4J.
     * This is a placeholder - actual implementation depends on Py4J interface.
     *
     * @param predictor Python predictor object
     * @param history   Historical data frames
     * @param horizon   Prediction horizon
     * @return Predicted power values in kW
     */
    private double[] invokePythonPredictor(Object predictor, List<WindDataFrame> history, int horizon) {
        // TODO: Implement actual Py4J call
        // For now, return dummy prediction
        LOGGER.warn("Python predictor invocation not yet implemented, returning zeros");
        return new double[horizon];
    }

    /**
     * Simple data structure for storing time-power pairs from CSV.
     */
    private static class WindDataPoint {
        final double timestamp;  // Relative time in seconds
        final double powerKW;    // Power in kW

        WindDataPoint(double timestamp, double powerKW) {
            this.timestamp = timestamp;
            this.powerKW = powerKW;
        }
    }
}
