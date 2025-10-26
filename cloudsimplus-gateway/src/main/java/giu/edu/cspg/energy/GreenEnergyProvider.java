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
     *
     * @param turbineId   Turbine ID to load data for
     * @param csvFilePath Path to CSV data file (relative to resources or absolute)
     */
    public GreenEnergyProvider(int turbineId, String csvFilePath) {
        this.turbineId = turbineId;
        this.csvFilePath = csvFilePath;
        this.historyBuffer = new CircularBuffer<>(HISTORY_SIZE);

        LOGGER.info("Initialising GreenEnergyProvider for turbine {}...", turbineId);
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
            List<WindDataPoint> dataPoints = loadCsvData();

            if (dataPoints.isEmpty()) {
                LOGGER.error("No data points loaded for turbine {}", turbineId);
                return;
            }

            // Extract time and power sequences
            timePoints = new double[dataPoints.size()];
            powerValues = new double[dataPoints.size()];

            for (int i = 0; i < dataPoints.size(); i++) {
                WindDataPoint dp = dataPoints.get(i);
                timePoints[i] = dp.timestamp;
                powerValues[i] = dp.powerKW;
            }

            minTime = timePoints[0];
            maxTime = timePoints[timePoints.length - 1];

            // Build spline interpolator
            SplineInterpolator interpolator = new SplineInterpolator();
            powerSpline = interpolator.interpolate(timePoints, powerValues);

            LOGGER.info("Spline interpolation built: {} data points, time range [{:.1f}s, {:.1f}s]",
                       dataPoints.size(), minTime, maxTime);

        } catch (Exception e) {
            LOGGER.error("Failed to load and build spline: {}", e.getMessage(), e);
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
                reader = new BufferedReader(new FileReader(csvFilePath));
                LOGGER.info("Loading CSV from file system: {}", csvFilePath);
            }

            String line = reader.readLine();  // Skip header
            long baseTimestamp = -1;
            int lineCount = 0;
            int matchedLines = 0;

            while ((line = reader.readLine()) != null) {
                lineCount++;
                String[] parts = line.split(",");

                if (parts.length < 18) {
                    LOGGER.warn("Line {} has insufficient columns ({}), skipping", lineCount, parts.length);
                    continue;
                }

                // Parse turbine ID
                int tid;
                try {
                    tid = Integer.parseInt(parts[0].trim());
                } catch (NumberFormatException e) {
                    continue;  // Skip invalid lines
                }

                if (tid != turbineId) continue;  // Only load specified turbine

                matchedLines++;

                // Parse timestamp
                try {
                    Timestamp ts = Timestamp.valueOf(parts[1].trim());
                    if (baseTimestamp < 0) baseTimestamp = ts.getTime();

                    double relativeSeconds = (ts.getTime() - baseTimestamp) / 1000.0;

                    // Parse power (Patv) - last column
                    double powerKW = parseDoubleOrZero(parts[17]);

                    result.add(new WindDataPoint(relativeSeconds, powerKW));

                } catch (Exception e) {
                    LOGGER.debug("Failed to parse line {}: {}", lineCount, e.getMessage());
                }

                // Progress logging
                if (matchedLines % 100000 == 0) {
                    LOGGER.info("Loaded {} data points for turbine {}...", matchedLines, turbineId);
                }
            }

            LOGGER.info("CSV loading complete: {} total lines, {} matched for turbine {}",
                       lineCount, matchedLines, turbineId);

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
