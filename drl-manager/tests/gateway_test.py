import os
from datetime import datetime
import logging
from pprint import PrettyPrinter
import time
import traceback
from py4j.java_gateway import JavaGateway, GatewayParameters

pp = PrettyPrinter(width=200)

# Create base logs directory if it doesn't exist
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Generate timestamp matching Java's format but with MINUTE precision
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')  # Minute precision
run_dir = os.path.join(log_dir, timestamp)

# Create run directory if not exists (may have been created by Java)
os.makedirs(run_dir, exist_ok=True)

# Configure logging to three destinations
handlers = [
    # 1. Current log (overwritten each run)
    logging.FileHandler(os.path.join(log_dir, 'simulation.log'), mode='w'),
    
    # 2. Per-run log in timestamped directory (overwritten if same minute)
    logging.FileHandler(os.path.join(run_dir, 'simulation.log'), mode='w'),
    
    # 3. Console output
    logging.StreamHandler()
]

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=handlers
)

logger = logging.getLogger()

def process_observation_state(java_obs_state):
    """Converts the Java ObservationState object to a Python dictionary."""
    if java_obs_state is None:
        return None
    # Py4J automatically converts Java arrays to Python lists/tuples
    host_loads = list(java_obs_state.getHostLoads())
    host_ram_ratios = list(java_obs_state.getHostRamUsageRatio())
    vm_loads = list(java_obs_state.getVmLoads())
    vm_types = list(java_obs_state.getVmTypes())
    vm_host_map = list(java_obs_state.getVmHostMap())
    infrastructure_observation = list(java_obs_state.getInfrastructureObservation())

    return {
        "host_loads": host_loads,
        "host_ram_usage_ratio": host_ram_ratios,
        "vm_loads": vm_loads,
        "vm_types": vm_types,
        "vm_host_map": vm_host_map,
        "infrastructure_observation": infrastructure_observation,
        "waiting_cloudlets": java_obs_state.getWaitingCloudlets(),
        "next_cloudlet_pes": java_obs_state.getNextCloudletPes(),
        "actual_vm_count": java_obs_state.getActualVmCount(),
        "actual_host_count": java_obs_state.getActualHostCount(),
    }

def process_step_info(java_step_info):
    """Converts the Java StepInfo object's Map to a Python dictionary."""
    if java_step_info is None:
        return None
    # Use the toMap() method defined in Java StepInfo
    java_map = java_step_info.toMap()
    # Py4J automatically converts the Java Map to a Python dict
    return dict(java_map)

def pretty_print_infrastructure(obs: list):
    """
    Given the flat infrastructure_observation list:
      [ totalCores, hostsNum,
        host0Pes, host0VmCount, vm0Pes, vm0JobCount, job0Pes, 0, …,
        vm1Pes, vm1JobCount, …,
        host1Pes, host1VmCount, …, … ]
    prints it with indentation.
    """
    idx = 0
    total_cores = obs[idx]; idx += 1
    hosts_num   = obs[idx]; idx += 1

    logger.info(f"Datacenter total cores: {total_cores}")
    logger.info(f"Hosts: {hosts_num}")

    for h in range(hosts_num):
        host_pes   = obs[idx]; idx += 1
        vm_count   = obs[idx]; idx += 1
        logger.info(f"  Host[{h}]: PEs={host_pes}  VMs={vm_count}")

        for v in range(vm_count):
            vm_pes    = obs[idx]; idx += 1
            job_count = obs[idx]; idx += 1
            logger.info(f"    VM[{v}]: PEs={vm_pes}  Cloudlets={job_count}")

            for j in range(job_count):
                job_pes = obs[idx]; idx += 1
                _       = obs[idx]; idx += 1    # placeholder for the “0”
                logger.info(f"      Cloudlet[{j}]: PEs={job_pes}")


# --- Simulation Parameters ---
# These should ideally match defaults or values you'd put in config.yml
sim_params = {
    "simulation_name": "test1_csv",

    "hosts_count": 8,
    "host_pes": 16,
    "host_pe_mips": 50000, # 50k MIPS per PE
    "host_ram": 65536, # 64 GB
    "host_bw": 50000, # 50 Gbps
    "host_storage": 100000, # 1 TB

    "small_vm_pes": 2,
    "small_vm_ram": 8192, # 8 GB
    "small_vm_bw": 1000,
    "small_vm_storage": 4000, # 4 GB

    "medium_vm_multiplier": 2, # -> 4 PEs
    "large_vm_multiplier": 4,  # -> 8 PEs

    "initial_s_vm_count": 1,
    "initial_m_vm_count": 1,
    "initial_l_vm_count": 1,

    "workload_mode": "CSV",
    "cloudlet_trace_file": "traces/hill_max30.csv",
    "workload_reader_mips": 8000, # Should match Host MIPS usually

    "simulation_timestep": 1.0, # RL step duration
    "min_time_between_events": 0.1,
    "vm_startup_delay": 0.0,
    "vm_shutdown_delay": 0.0,

    "max_episode_length": 60, # Limit simulation steps for testing

    "reward_wait_time_coef": 0.1,
    "reward_unutilization_coef": 0.85,
    "reward_cost_coef": 0.5,
    "reward_queue_penalty_coef": 0.05,
    "reward_invalid_action_coef": 1.0,
}

# --- Connect to Gateway ---
logger.info("Connecting to Java Gateway...")
# Increase timeout if gateway takes longer to start
gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
gateway_entry_point = gateway.entry_point
logger.info("Connection successful.")

# --- Test Loop ---
try:
    # 1. Configure Simulation
    logger.info("\nConfiguring simulation...")
    gateway_entry_point.configureSimulation(sim_params)
    logger.info("Configuration sent.")

    # 2. Reset Simulation
    logger.info("\nResetting simulation...")
    seed = 4567
    reset_result_java = gateway_entry_point.reset(seed)
    initial_obs_state_java = reset_result_java.getObservation()
    initial_info_java = reset_result_java.getInfo()

    initial_obs_dict = process_observation_state(initial_obs_state_java)
    initial_info_dict = process_step_info(initial_info_java)

    logger.info("Reset complete.")
    logger.info("Infrastructure tree:")
    pretty_print_infrastructure(initial_obs_dict["infrastructure_observation"])
    logger.info("Initial Observation:")
    logger.info(pp.pformat(initial_obs_dict))
    logger.info("Initial Info:")
    logger.info(pp.pformat(initial_info_dict))

    # 3. Step through Simulation with hardcoded actions
    current_step = 0
    done = False

    terminated = False
    truncated = False
    current_obs_dict = initial_obs_dict # Start with initial observation

    while not done:
        # ** Hardcoded Action Logic **
        # Action format: [action_type, target_vm_id, target_host_id, vm_type_index]
        # Types: 0=NoOp, 1=AssignCloudlet, 2=CreateVm, 3=DestroyVm
        # VmTypeIndices: 0=S, 1=M, 2=L
        action_list = [0, -1, -1, -1] # Default to NoOp
        action_description = "NoOp"

        # Example sequence:
        if current_step < 5: # Try assigning first few cloudlets (assuming VmIds 0, 1 exist initially)
            if current_obs_dict["waiting_cloudlets"] > 0:
                 target_vm = current_step % 2 # Alternate between VM 0 and 1
                 action_description = f"Assign Cloudlet to VM {target_vm}"
                 action_list = [1, target_vm, -1, -1]
        elif current_step == 10: # Try creating a Medium VM on Host 1
            action_description = "Create Medium VM on Host 1"
            action_list = [2, -1, 1, 1] # Type index 1 = Medium
        elif current_step == 20: # Try creating a Small VM on Host 0
            action_description = "Create Small VM on Host 0"
            action_list = [2, -1, 0, 0] # Type index 0 = Small
        elif current_step == 30: # Try destroying VM 1 (initial Medium VM)
            action_description = "Destroy VM 1"
            action_list = [3, 1, -1, -1]
        elif current_step == 40: # Try destroying VM 0 (initial Small VM)
             action_description = "Destroy VM 0"
             action_list = [3, 0, -1, -1]

        # Determine if we should print this step
        is_noop = (action_list[0] == 0)
        if not is_noop or is_noop:
            logger.info(f"\n--- Step {current_step + 1} ---")
            logger.info(f"Action: {action_description}")

        # Convert Python list to Java ArrayList for Py4J
        java_action = gateway.jvm.java.util.ArrayList(action_list)

        # Execute step
        step_result_java = gateway_entry_point.step(java_action)

        # Process results
        obs_state_java = step_result_java.getObservation()
        reward = step_result_java.getReward()
        terminated = step_result_java.isTerminated()
        truncated = step_result_java.isTruncated()
        info_java = step_result_java.getInfo()

        current_obs_dict = process_observation_state(obs_state_java)
        info_dict = process_step_info(info_java)

        # Print results only if it wasn't a NoOp
        if not is_noop or is_noop:
            logger.info(f"Reward: {reward:.4f}")
            logger.info("Infrastructure tree:")
            pretty_print_infrastructure(current_obs_dict["infrastructure_observation"])
            logger.info("Observation:")
            logger.info(pp.pformat(current_obs_dict))
            logger.info("Info:")
            logger.info(pp.pformat(info_dict))
            logger.info(f"Terminated: {terminated}, Truncated: {truncated}")
            time.sleep(0.1)

        done = terminated or truncated

        current_step += 1

    # 4. Close Simulation (Triggers results table printing in Java)
    logger.info("\nClosing simulation...")
    gateway_entry_point.close()
    logger.info("Close request sent.")

except Exception as e:
    logger.error(f"\nAn error occurred: {e}")
    traceback.print_exc()

finally:
    # Ensure gateway is shut down (optional, as close() should handle it now)
    logger.info("Shutting down gateway connection.")
    gateway.shutdown()
    pass

logger.info("\nTest script finished.")
