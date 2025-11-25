package giu.edu.cspg.singledc;

import lombok.Getter;

/**
 * Represents the result of executing one simulation step, conforming to the
 * Gymnasium API's expected return tuple (observation, reward, terminated,
 * truncated, info).
 */
@Getter
public class SimulationStepResult {
    private final ObservationState observation;
    private final double reward;
    private final boolean terminated;
    private final boolean truncated;
    private final SimulationStepInfo info;

    public SimulationStepResult(ObservationState observation, double reward, boolean terminated, boolean truncated,
            SimulationStepInfo info) {
        this.observation = observation;
        this.reward = reward;
        this.terminated = terminated;
        this.truncated = truncated;
        this.info = info;
    }

    // --- Getters ---
    public boolean isTerminated() {
        return terminated;
    }
    public boolean isTruncated() {
        return truncated;
    }

    @Override
    public String toString() {
        return "SimulationStepResult{" +
                "observation=" + observation +
                ", reward=" + reward +
                ", terminated=" + terminated +
                ", truncated=" + truncated +
                ", info=" + info.toString() +
                '}';
    }
}
