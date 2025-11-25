package giu.edu.cspg.energy;

/**
 * Time scaling modes for mapping wind-data timestamps to simulation time.
 *
 * This allows flexible mapping of wind turbine data (typically sampled at
 * 10-minute intervals) to simulation time, enabling both realistic and
 * compressed time scenarios.
 */
public enum TimeScalingMode {

    /**
     * Real-time mode: Use actual timestamps from CSV data.
     *
     * Each data point preserves its original timestamp interval (e.g., 600s).
     * Suitable for realistic time simulations.
     *
     * Example:
     *   CSV row 0 (2021-01-01 00:00:00) -> t=0s
     *   CSV row 1 (2021-01-01 00:10:00) -> t=600s
     *   CSV row 2 (2021-01-01 00:20:00) -> t=1200s
     *   ...
     *
     * Data utilization for 2000s simulation: ~3-4 data points (0.006%)
     */
    REAL_TIME,

    /**
     * Compressed mode: Map each data point to 1 second of simulation time.
     *
     * Each consecutive data point is treated as 1 second apart, regardless
     * of actual timestamp spacing. Enables efficient use of dataset.
     *
     * Example:
     *   CSV row 0 -> t=0s
     *   CSV row 1 -> t=1s
     *   CSV row 2 -> t=2s
     *   ...
     *   CSV row 1999 -> t=1999s
     *
     * Data utilization for 2000s simulation: 2000 data points (3.8%)
     */
    COMPRESSED;

    /**
     * Get the typical data interval for this mode.
     *
     * @return Interval in seconds between consecutive data points
     */
    public double getTypicalInterval() {
        switch (this) {
            case REAL_TIME:
                return 600.0;  // 10 minutes
            case COMPRESSED:
                return 1.0;    // 1 second
            default:
                return 1.0;
        }
    }

    /**
     * Get a human-readable description of this mode.
     *
     * @return Description string
     */
    public String getDescription() {
        switch (this) {
            case REAL_TIME:
                return "Real-time (preserves original 600s intervals)";
            case COMPRESSED:
                return "Compressed (1 data point = 1 second)";
            default:
                return toString();
        }
    }
}
