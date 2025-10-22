package giu.edu.cspg.energy;

import java.util.Arrays;

/**
 * Represents a single timestep of wind turbine data with all features.
 * Used for storing historical data and providing input to prediction models.
 */
public class WindDataFrame {
    private final double timestamp;       // Relative time in seconds
    private final double[] features;      // 16 features: Wspd, Wdir, Etmp, Itmp, Ndir, Pab1-3, Prtv, T2m, Sp, RelH, Wspd_w, Wdir_w, Tp, Patv

    /**
     * Feature names in order
     */
    public static final String[] FEATURE_NAMES = {
        "Wspd", "Wdir", "Etmp", "Itmp", "Ndir",
        "Pab1", "Pab2", "Pab3", "Prtv", "T2m",
        "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp",
        "Patv"
    };

    public static final int POWER_INDEX = 15;  // Index of Patv (power) in features array
    public static final int NUM_FEATURES = 16;

    /**
     * Create a wind data frame.
     *
     * @param timestamp Relative time in seconds
     * @param features  Array of 16 feature values
     */
    public WindDataFrame(double timestamp, double[] features) {
        if (features == null || features.length != NUM_FEATURES) {
            throw new IllegalArgumentException(
                "Features array must have exactly " + NUM_FEATURES + " elements"
            );
        }

        this.timestamp = timestamp;
        this.features = Arrays.copyOf(features, NUM_FEATURES);
    }

    /**
     * Get timestamp in seconds.
     *
     * @return Timestamp
     */
    public double getTimestamp() {
        return timestamp;
    }

    /**
     * Get all features.
     *
     * @return Copy of features array
     */
    public double[] getFeatures() {
        return Arrays.copyOf(features, NUM_FEATURES);
    }

    /**
     * Get a specific feature by index.
     *
     * @param index Feature index (0-15)
     * @return Feature value
     */
    public double getFeature(int index) {
        if (index < 0 || index >= NUM_FEATURES) {
            throw new IndexOutOfBoundsException("Feature index must be 0-15");
        }
        return features[index];
    }

    /**
     * Get power value (Patv) in kW.
     *
     * @return Power in kW
     */
    public double getPowerKW() {
        return features[POWER_INDEX];
    }

    /**
     * Get wind speed (Wspd) in m/s.
     *
     * @return Wind speed
     */
    public double getWindSpeed() {
        return features[0];
    }

    @Override
    public String toString() {
        return String.format("WindDataFrame[t=%.1fs, power=%.2fkW, wspd=%.2fm/s]",
                           timestamp, getPowerKW(), getWindSpeed());
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof WindDataFrame)) return false;

        WindDataFrame other = (WindDataFrame) obj;
        return Double.compare(timestamp, other.timestamp) == 0 &&
               Arrays.equals(features, other.features);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(features) + Double.hashCode(timestamp);
    }
}
