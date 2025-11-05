package giu.edu.cspg.singledc;

/**
 * Represents the result of resetting the simulation environment, conforming to
 * the
 * Gymnasium API's expected return tuple (observation, info).
 */
public class SimulationResetResult {
    private final ObservationState observation;
    private final SimulationStepInfo info;

    public SimulationResetResult(ObservationState observation, SimulationStepInfo info) {
        this.observation = observation;
        this.info = info;
    }

    // --- Getters ---
    public ObservationState getObservation() {
        return observation;
    }

    public SimulationStepInfo getInfo() {
        return info;
    }

    @Override
    public String toString() {
        return "SimulationResetResult{" +
                "info=" + info.toString() +
                '}';
    }
}
