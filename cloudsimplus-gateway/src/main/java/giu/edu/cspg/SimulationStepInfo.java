package giu.edu.cspg;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;

/**
 * Contains auxiliary information about the outcome of a simulation step.
 */
public class SimulationStepInfo {
    private final Gson gson = new Gson();
    // Action Outcome Flags
    private final boolean assignmentSuccess;
    private final boolean createVmAttempted; // Whether a create action was tried
    private final boolean createVmSuccess; // Whether the tried creation was successful
    private final boolean destroyVmAttempted;// Whether a destroy action was tried
    private final boolean destroyVmSuccess; // Whether the tried destruction was successful
    private final boolean invalidActionTaken; // If the action itself was invalid (e.g., bad index)

    // State/Resource Change Info
    private final int hostAffectedId; // Host ID affected by create/destroy (-1 if none/NA)
    private final int coresChanged; // Number of cores added/removed (+ve for create, -ve for destroy, 0 otherwise)

    // Simulation Context
    private final double currentClock;
    private final int[] observationTreeArray;

    // Reward Components
    private final double rewardWaitTimeComponent;
    private final double rewardUnutilizationComponent;
    private final double rewardQueuePenaltyComponent;
    private final double rewardInvalidActionComponent;
    private final double rewardEnergyComponent; // Energy consumption penalty
    private final List<Double> completedCloudletWaitTimes;

    // Energy Metrics
    private final double currentPowerW;          // Current power consumption in Watts
    private final double cumulativeEnergyWh;     // Cumulative energy consumption in Watt-hours
    private final double averageHostUtilization; // Average CPU utilization across all hosts

    // Constructor with all fields
    public SimulationStepInfo(boolean assignmentSuccess, boolean createVmAttempted, boolean createVmSuccess,
            boolean destroyVmAttempted, boolean destroyVmSuccess, boolean invalidActionTaken, int hostAffectedId,
            int coresChanged,
            double currentClock, double rewardWaitTimeComponent,
            double rewardUnutilizationComponent,
            double rewardQueuePenaltyComponent, double rewardInvalidActionComponent,
            double rewardEnergyComponent,
            int[] observationTreeArray, List<Double> completedCloudletWaitTimes,
            double currentPowerW, double cumulativeEnergyWh, double averageHostUtilization) {
        this.assignmentSuccess = assignmentSuccess;
        this.createVmAttempted = createVmAttempted;
        this.createVmSuccess = createVmSuccess;
        this.destroyVmAttempted = destroyVmAttempted;
        this.destroyVmSuccess = destroyVmSuccess;
        this.invalidActionTaken = invalidActionTaken;
        this.hostAffectedId = hostAffectedId;
        this.coresChanged = coresChanged;
        this.currentClock = currentClock;
        this.rewardWaitTimeComponent = rewardWaitTimeComponent;
        this.rewardUnutilizationComponent = rewardUnutilizationComponent;
        this.rewardQueuePenaltyComponent = rewardQueuePenaltyComponent;
        this.rewardInvalidActionComponent = rewardInvalidActionComponent;
        this.rewardEnergyComponent = rewardEnergyComponent;
        this.observationTreeArray = observationTreeArray;
        this.completedCloudletWaitTimes = completedCloudletWaitTimes;
        this.currentPowerW = currentPowerW;
        this.cumulativeEnergyWh = cumulativeEnergyWh;
        this.averageHostUtilization = averageHostUtilization;
    }

    // Simplified constructor for SimulationResetResult where action outcomes aren't
    // relevant
    public SimulationStepInfo(double currentClock) {
        this(false, false, false, false, false, false, -1, 0, currentClock, 0, 0, 0, 0, 0, new int[1],
                new ArrayList<>(), 0, 0, 0);
    }

    // --- Getters ---
    public boolean isAssignmentSuccess() {
        return assignmentSuccess;
    }

    public boolean isCreateVmAttempted() {
        return createVmAttempted;
    }

    public boolean isCreateVmSuccess() {
        return createVmSuccess;
    }

    public boolean isDestroyVmAttempted() {
        return destroyVmAttempted;
    }

    public boolean isDestroyVmSuccess() {
        return destroyVmSuccess;
    }

    public boolean isInvalidActionTaken() {
        return invalidActionTaken;
    }

    public int getHostAffectedId() {
        return hostAffectedId;
    }

    public int getCoresChanged() {
        return coresChanged;
    }

    public double getCurrentClock() {
        return currentClock;
    }

    public double getRewardWaitTimeComponent() {
        return rewardWaitTimeComponent;
    }

    public double getRewardUntilizationComponent() {
        return rewardUnutilizationComponent;
    }

    public double getRewardQueuePenaltyComponent() {
        return rewardQueuePenaltyComponent;
    }

    public double getRewardInvalidActionComponent() {
        return rewardInvalidActionComponent;
    }

    public double getRewardEnergyComponent() {
        return rewardEnergyComponent;
    }

    public double getCurrentPowerW() {
        return currentPowerW;
    }

    public double getCumulativeEnergyWh() {
        return cumulativeEnergyWh;
    }

    public double getAverageHostUtilization() {
        return averageHostUtilization;
    }

    public String getObservationTreeArrayAsJson() {
        return gson.toJson(observationTreeArray);
    }

    public String getCompletedCloudletWaitTimesAsJson() {
        return gson.toJson(completedCloudletWaitTimes);
    }

    /**
     * Converts the info into a Map for easy translation to a Python dictionary by
     * Py4J.
     * 
     * @return A Map representation of the StepInfo.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("assignment_success", this.assignmentSuccess);
        map.put("create_vm_attempted", this.createVmAttempted);
        map.put("create_vm_success", this.createVmSuccess);
        map.put("destroy_vm_attempted", this.destroyVmAttempted);
        map.put("destroy_vm_success", this.destroyVmSuccess);
        map.put("invalid_action_taken", this.invalidActionTaken);
        map.put("host_affected_id", this.hostAffectedId);
        map.put("cores_changed", this.coresChanged);
        map.put("current_clock", this.currentClock);
        map.put("reward_wait_time", this.rewardWaitTimeComponent);
        map.put("reward_unutilization", this.rewardUnutilizationComponent);
        map.put("reward_queue_penalty", this.rewardQueuePenaltyComponent);
        map.put("reward_invalid_action", this.rewardInvalidActionComponent);
        map.put("reward_energy", this.rewardEnergyComponent);
        map.put("current_power_w", this.currentPowerW);
        map.put("cumulative_energy_wh", this.cumulativeEnergyWh);
        map.put("average_host_utilization", this.averageHostUtilization);
        return map;
    }

    @Override
    public String toString() {
        return "SimulationStepInfo{" +
                "assignOK=" + assignmentSuccess +
                ", createAttempt=" + createVmAttempted +
                ", createOK=" + createVmSuccess +
                ", destroyAttempt=" + destroyVmAttempted +
                ", destroyOK=" + destroyVmSuccess +
                ", invalidAction=" + invalidActionTaken +
                ", hostAffected=" + hostAffectedId +
                ", coresChanged=" + coresChanged +
                ", clock=" + currentClock +
                ", rewardWaitTime=" + rewardWaitTimeComponent +
                ", rewardUntilization=" + rewardUnutilizationComponent +
                ", rewardQueuePenalty=" + rewardQueuePenaltyComponent +
                ", rewardInvalidAction=" + rewardInvalidActionComponent +
                '}';
    }
}
