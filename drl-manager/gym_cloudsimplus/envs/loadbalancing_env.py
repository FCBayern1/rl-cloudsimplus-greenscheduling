import time
import traceback
import logging
import json
import numpy as np
import gymnasium as gym
from pprint import PrettyPrinter
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JNetworkError

pp = PrettyPrinter(width=200)

logger = logging.getLogger(__name__.split('.')[-1])

# Based on https://gymnasium.farama.org/api/env/
class LoadBalancingEnv(gym.Env):
    """
    Gymnasium Environment for Load Balancing ONLY on fixed infrastructure.
    Agent MUST attempt to assign a cloudlet each step if available.
    Interacts with LoadBalancerGateway.step(targetVmId).

    Action Space (Discrete): Integer representing the target VM ID (0 to num_vms-1)

    Observation Space (Dict): Simplified state for load balancing.
    - vm_loads: Box(num_vms) - CPU Load [0,1]
    - waiting_cloudlets: Box(1) - Normalized count
    - next_cloudlet_pes: Box(1) - Normalized PEs needed
    """
    metadata = {"render_modes": ["human", "ansi"]}
    # Class-level guard to avoid closing the Java gateway multiple times across instances
    _java_gateway_closed = False

    def __init__(self, config_params: dict, render_mode="ansi"):
        super(LoadBalancingEnv, self).__init__()

        logger.info("Initializing LoadBalancingEnv...")
        self.config = config_params
        self._closed = False
        # Episode-level accumulators for logging meaningful per-episode stats
        self._reset_episode_accumulators()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            gym.logger.warn(
                "Invalid render mode" 'Render modes allowed: ["human" | "ansi"]'
            )
        self.render_mode = render_mode

        # --- Connect to Py4J Gateway ---
        gateway_host = self.config.get("gateway_host", "0.0.0.0")
        gateway_port = self.config.get("gateway_port", 25333)
        logger.info(f"Attempting to connect to Gateway on {gateway_host}:{gateway_port}...")
        retries = 5
        while retries > 0:
            try:
                # Set auto_convert=True for easier type handling
                self.gateway = JavaGateway(
                    gateway_parameters=GatewayParameters(auto_convert=True)
                )
                # Test connection
                self.gateway.jvm.System.out.println("Python Env connected!")
                self.loadbalancer_gateway = self.gateway.entry_point
                logger.info("Successfully connected to Java Gateway.")
                break # Exit loop on successful connection
            except (ConnectionRefusedError, Py4JNetworkError) as e:
                retries -= 1
                logger.warning(f"Gateway connection failed: {e}. Retrying in 5 seconds... ({retries} retries left)")
                if retries == 0:
                    logger.error("Max retries reached. Could not connect to Gateway.")
                    raise
                time.sleep(5)

        # --- Configure Simulation (Call Java side) ---
        try:
            logger.info("\nConfiguring simulation...")
            # Convert Python dict to Java HashMap for Py4J
            params = self.gateway.jvm.java.util.HashMap()
            for key, value in self.config.items():
                params.put(key, value)
            self.loadbalancer_gateway.configureSimulation(params)
            logger.info("Java simulation configured via gateway.")
        except Exception as e:
             logger.error(f"Failed to configure simulation via gateway: {e}")
             self.close() # Close gateway connection if config fails
             raise

        # --- Define Spaces (based on config) ---
        self.large_vm_pes = self.config.get("large_vm_pes", 8)
        self.num_vms = self.config.get("initial_s_vm_count", 0) + self.config.get("initial_m_vm_count", 0) + self.config.get("initial_l_vm_count", 0)
        if self.num_vms <= 0:
            raise ValueError("Config 'num_vms' must be positive.")
        logger.info(f"Using num_vms={self.num_vms} for spaces")

        # Action Space: Choose which VM to assign the next cloudlet to
        self.action_space = spaces.Discrete(self.num_vms + 1)
        logger.info(f"Action Space defined: {self.action_space} (0 maps to -1, 1 maps to VM 0, ...)")

        # Observation Space
        self.observation_space = spaces.Dict({
            "vm_loads": spaces.Box(low=0.0, high=1.0, shape=(self.num_vms,), dtype=np.float32),
            "vm_available_pes": spaces.Box(low=0, high=self.large_vm_pes, shape=(self.num_vms,), dtype=np.int32),
            "vm_types": spaces.Box(low=0, high=3, shape=(self.num_vms,), dtype=np.int32),  # 0=None, 1=Small, 2=Medium, 3=Large
            "waiting_cloudlets": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "next_cloudlet_pes": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # Enhanced cloudlet information
            "next_cloudlet_mi": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "next_cloudlet_wait_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "queue_pes_distribution": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.int32),  # [small, medium, large]
            # Historical statistics
            "completed_cloudlets_last_10_steps": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })
        logger.info(f"Observation Space defined: {self.observation_space.spaces.keys()}")

    def reset(self, seed=None, options=None):
        super(LoadBalancingEnv, self).reset(seed=seed, options=options)
        self.current_step = 0

        # Use stored seed or generate one if None
        current_seed = seed if seed is not None else self.config.get("seed", 42)
        logger.info(f"Resetting environment with seed: {current_seed}")

        try:
            reset_result_java = self.loadbalancer_gateway.reset(current_seed)
            observation = self._get_obs(reset_result_java.getObservation())
            info = self._process_info(reset_result_java.getInfo())
            info["actual_vm_count"] = reset_result_java.getObservation().getActualVmCount()
            info["actual_host_count"] = reset_result_java.getObservation().getActualHostCount()

            logger.debug("Reset successful.")
            logger.debug(f"Initial Observation: {observation}") # Can be very verbose
            logger.debug(f"Initial Info: {info}")

            # Store initial state info needed for action masking
            self._update_internal_state(observation)

            return (observation, info)
        except Exception as e:
             logger.error(f"Error during env reset: {e}")
             traceback.print_exc()
             # Return dummy observation/info if reset fails critically
             dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
             dummy_info = {
                 "error": "Reset failed",
                 "reward_wait_time": 0.0,
                 "reward_unutilization": 0.0,
                 "reward_queue_penalty": 0.0,
                 "reward_invalid_action": 0.0,
                 "reward_energy": 0.0,
                 "current_power_w": 0.0,
                 "actual_vm_count": 0,
                 "actual_host_count": 0
             }
             return dummy_obs, dummy_info

    def _get_obs(self, java_obs_state):
        """Converts the Java ObservationState object to a NumPy dictionary."""
        if java_obs_state is None:
            logger.error("Received null Java ObservationState!")
            # Return a zeroed-out observation matching the space structure
            return {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.action.spaces.items()}

        # Extract arrays (Py4J automatically converts Java arrays to Python tuples/lists)
        # Assumes Java side provides vmLoads padded up to maxPotentialVms, slice it
        full_vm_loads = self._to_nparray(java_obs_state.getVmLoads(), dtype=np.float32)
        # Slice using self.num_vms which defines our fixed fleet size
        vm_loads = full_vm_loads[:self.num_vms]

        vm_available_pes = self._to_nparray(java_obs_state.getVmAvailablePes(), dtype=np.int32)

        # Extract VM types (0=None, 1=Small, 2=Medium, 3=Large)
        full_vm_types = self._to_nparray(java_obs_state.getVmTypes(), dtype=np.int32)
        vm_types = full_vm_types[:self.num_vms]

        # Extract raw scalar values
        waiting_cloudlets_raw = np.array([java_obs_state.getWaitingCloudlets()], dtype=np.float32) # Keep as float Box for SB3
        next_cloudlet_pes_raw = np.array([java_obs_state.getNextCloudletPes()], dtype=np.float32) # Keep as float Box for SB3

        # Extract enhanced cloudlet information
        next_cloudlet_mi = np.array([float(java_obs_state.getNextCloudletMi())], dtype=np.float32)
        next_cloudlet_wait_time = np.array([java_obs_state.getNextCloudletWaitTime()], dtype=np.float32)
        queue_pes_dist = self._to_nparray(java_obs_state.getQueuePesDistribution(), dtype=np.int32)

        # Extract historical statistics
        completed_last_10 = np.array([java_obs_state.getCompletedCloudletsLast10Steps()], dtype=np.int32)

        # Ensure correct shape (defensive)
        if vm_loads.shape[0] != self.num_vms: vm_loads = np.pad(vm_loads, (0, self.num_vms - vm_loads.shape[0]))[:self.num_vms]
        if vm_available_pes.shape[0] != self.num_vms: vm_available_pes = np.pad(vm_available_pes, (0, self.num_vms - vm_available_pes.shape[0]))[:self.num_vms]
        if vm_types.shape[0] != self.num_vms: vm_types = np.pad(vm_types, (0, self.num_vms - vm_types.shape[0]))[:self.num_vms]

        obs_dict = {
            "vm_loads": vm_loads,
            "vm_available_pes": vm_available_pes,
            "vm_types": vm_types,  # Added VM type information for optimal resource matching
            "waiting_cloudlets": waiting_cloudlets_raw, # Raw count
            "next_cloudlet_pes": next_cloudlet_pes_raw, # Raw count
            # Enhanced cloudlet information
            "next_cloudlet_mi": next_cloudlet_mi,
            "next_cloudlet_wait_time": next_cloudlet_wait_time,
            "queue_pes_distribution": queue_pes_dist,
            # Historical statistics
            "completed_cloudlets_last_10_steps": completed_last_10,
        }
        logger.debug(f"Processed Observation: {obs_dict}")
        return obs_dict

    def _to_nparray(self, raw_obs, dtype=np.float32):
        obs = list(raw_obs)
        return np.array(obs, dtype=dtype)

    def _process_info(self, java_info_obj):
        """Converts the Java StepInfo object map to a Python dict."""
        if java_info_obj is None:
            return {}
        try:
             # Py4J should convert the map returned by toMap() automatically
            info_map = java_info_obj.toMap()
            return dict(info_map)
        except Exception as e:
            logger.error(f"Error processing info object: {e}")
            return {"error": str(e)} # Return error info

    def _reset_episode_accumulators(self):
        self._ep_steps = 0
        self._ep_sum = {
            "reward_wait_time": 0.0,
            "reward_unutilization": 0.0,
            "reward_queue_penalty": 0.0,
            "reward_invalid_action": 0.0,
            "reward_energy": 0.0,
            "current_power_w": 0.0,
        }

    def step(self, action: int):
        self.current_step += 1

        # --- Map Action ---
        # Agent outputs 0 to num_vms.
        # Map 0 to targetVmId -1 (No assign)
        # Map 1 to targetVmId 0
        # Map n to targetVmId n-1
        target_vm_id = int(action) - 1
        logger.debug(f"Raw Action: {action}, Mapped Target VM ID: {target_vm_id}")

        try:
            step_result_java = self.loadbalancer_gateway.step(target_vm_id)

            observation = self._get_obs(step_result_java.getObservation())
            reward = float(step_result_java.getReward())
            terminated = bool(step_result_java.isTerminated())
            truncated = bool(step_result_java.isTruncated())
            info = self._process_info(step_result_java.getInfo())
            info["actual_vm_count"] = step_result_java.getObservation().getActualVmCount()
            info["actual_host_count"] = step_result_java.getObservation().getActualHostCount()

            # --- Accumulate episode-level stats ---
            self._ep_steps += 1
            for k in list(self._ep_sum.keys()):
                if k in info and isinstance(info[k], (int, float)):
                    self._ep_sum[k] += float(info[k])

            # On episode end, compute episode means and attach to final info
            if terminated or truncated:
                steps = max(1, self._ep_steps)
                info["episode_reward_wait_time_mean"] = self._ep_sum["reward_wait_time"] / steps
                info["episode_reward_unutilization_mean"] = self._ep_sum["reward_unutilization"] / steps
                info["episode_reward_queue_penalty_mean"] = self._ep_sum["reward_queue_penalty"] / steps
                info["episode_reward_invalid_action_mean"] = self._ep_sum["reward_invalid_action"] / steps
                info["episode_reward_energy_mean"] = self._ep_sum["reward_energy"] / steps
                info["episode_avg_power_w"] = self._ep_sum["current_power_w"] / steps
                # Reset accumulators for next episode
                self._reset_episode_accumulators()

            # Update internal state for action masking
            self._update_internal_state(observation)

            if self.render_mode == "human":
                self.render()

            logger.debug(f"Step Result: Obs keys={list(observation.keys())}, Rew={reward:.2f}, Term={terminated}, Trunc={truncated}, Info keys={list(info.keys())}")

            return (observation, reward, terminated, truncated, info)

        except Exception as e:
             logger.error(f"Error during env step: {e}")
             traceback.print_exc()
             # Return dummy observation/info and set done=True if step fails critically
             dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}
             dummy_info = {
                 "error": "Step failed",
                 "reward_wait_time": 0.0,
                 "reward_unutilization": 0.0,
                 "reward_queue_penalty": 0.0,
                 "reward_invalid_action": 0.0,
                 "reward_energy": 0.0,
                 "current_power_w": 0.0
             }
             return dummy_obs, 0.0, True, False, dummy_info

    def _update_internal_state(self, observation: dict):
        """Stores the latest observation dictionary for use in action masking."""
        if observation is None:
             logger.warning("Attempted to update internal state with None observation.")
             return
        self.last_observation = observation

    def render(self):
        """Renders the environment.
        - human: Parses JSON from Java and pretty-prints the dictionary.
        - ansi: Returns the raw JSON string from Java.
        - dict: Parses JSON from Java and returns the Python dictionary.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method "
                "without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="human")'
            )
            return

        try:
            raw_render_info = self.loadbalancer_gateway.getRenderInfo()
            json_string = self.java_gateway_entry_point.getRenderInfoAsJson()

            if self.render_mode == "human":
                try:
                    data = json.loads(json_string)
                    print("--- Render Info ---")
                    pp.pformat(data)
                    print("-------------------")
                except json.JSONDecodeError:
                    print("--- Render Info (Raw) ---")
                    print(raw_render_info)
                    print("-----------------------")
                return

            elif self.render_mode == "ansi":
                return str(json.loads(json_string))
            else:
                return super(LoadBalancingEnv, self).render()

        except Exception as e:
            error_msg = f"Error getting/processing render info: {e}"
            if self.render_mode == "human":
                print(error_msg)
            else:
                return error_msg if self.render_mode == 'ansi' else {"error": error_msg}

    def close(self):
        if self._closed:
            return
        self._closed = True
        logger.info("Closing environment and gateway connection...")
        # Try to close Java side once globally across all env instances
        if not LoadBalancingEnv._java_gateway_closed:
            try:
                if hasattr(self, 'loadbalancer_gateway') and self.loadbalancer_gateway:
                    self.loadbalancer_gateway.close()
            except Py4JNetworkError:
                # Connection already down or closed elsewhere; not an error during shutdown
                logger.info("Java gateway already closed; ignoring during shutdown.")
            except Exception as e:
                # Downgrade to warning to avoid noisy shutdown logs
                logger.warning(f"Non-critical error during gateway close request: {e}")
            finally:
                LoadBalancingEnv._java_gateway_closed = True
        # Always shutdown client side quietly
        try:
            if hasattr(self, 'gateway') and self.gateway:
                self.gateway.shutdown()
                logger.info("Py4J Gateway client shut down.")
        except Exception:
            # Ignore any client-side shutdown issues
            pass

    def action_masks(self) -> list[bool]:
        """
        Generates action masks for the Discrete action space (size num_vms + 1).
        Mask index 0 corresponds to action -1 (No assign).
        Mask indices 1 to num_vms correspond to actions 0 to num_vms-1 (Assign to VM).
        """
        if self.last_observation is None:
            logger.warning("Masking before first observation. Allowing only No Assign (-1).")
            mask = [False] * (self.num_vms + 1)
            mask[0] = True # Allow action -1 (index 0)
            return mask

        # Extract state needed for masking
        vm_available_pes = self.last_observation['vm_available_pes']
        waiting_cloudlets = int(self.last_observation['waiting_cloudlets'][0])
        next_cloudlet_pes = int(self.last_observation['next_cloudlet_pes'][0])

        # Initialize mask - Disallow everything initially
        mask = [False] * (self.num_vms + 1)

        # Check if assignment is possible *at all*
        can_assign_anything = waiting_cloudlets > 0 and next_cloudlet_pes > 0

        if not can_assign_anything:
            # If queue is empty or next job needs 0 PEs, only allow action -1 (No Assign)
            mask[0] = True # Allow action -1 (at index 0)
            logger.debug("Masking: Queue empty or next job invalid. Allowing only No Assign (-1).")
        else:
            # Queue has items, check individual VMs
            # Action -1 (index 0) is generally DISALLOWED if assignment IS possible.
            # The agent *should* pick a VM. If no VM is valid, it gets invalid action penalty.
            mask[0] = False # Disallow explicit "No Assign" if cloudlets are waiting

            active_vm_found_with_capacity = False
            for vm_id in range(self.num_vms):
                # Check if VM ID 'vm_id' has enough PEs
                if vm_available_pes[vm_id] >= next_cloudlet_pes:
                    # Allow action corresponding to this vm_id
                    # Action vm_id maps to mask index vm_id + 1
                    mask[vm_id + 1] = True # <<< Mask index offset by 1
                    active_vm_found_with_capacity = True

            # If NO VM has capacity, what should the agent do?
            # Option A (Current): No action is valid (mask all False). SB3 might complain?
            # Option B: Allow action -1 (index 0) again as a fallback?
            # Option C: Allow ALL VM assignments (mask all True from index 1 onwards) and let Java penalize.
            # Let's go with Option C - allows agent to take an action, simplifies SB3 handling.
            if not active_vm_found_with_capacity:
                 logger.debug(f"Masking: No VM found with enough PEs ({next_cloudlet_pes}). Allowing all VMs (Java handles invalid).")
                 # Allow assigning to any VM (index 1 to num_vms)
                 for i in range(1, self.num_vms + 1):
                     mask[i] = True
                 mask[0] = False # Still disallow explicit No Assign if queue not empty

        # Sanity check length
        if len(mask) != self.action_space.n: # .n for Discrete space
             logger.error(f"Mask length mismatch! Expected {self.action_space.n}, got {len(mask)}.")
             return [True] * self.action_space.n # Fallback

        return mask
