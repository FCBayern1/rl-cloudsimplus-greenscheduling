"""
Hierarchical Multi-Datacenter Reinforcement Learning Environment

This environment implements a two-level hierarchical MARL system:
- Global Level: Routes arriving cloudlets to datacenters (Global Agent)
- Local Level: Schedules cloudlets to VMs within each datacenter (Local Agents)

Architecture:
    Python (Gymnasium) <--> Py4J <--> Java (CloudSim Plus Multi-DC Simulation)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters, Py4JNetworkError

logger = logging.getLogger(__name__)


class HierarchicalMultiDCEnv(gym.Env):
    """
    Hierarchical Multi-Datacenter Load Balancing Environment.

    Two-level decision making:
    1. Global Agent: Routes arriving cloudlets to datacenters
    2. Local Agents: Assign cloudlets to VMs within each datacenter

    Action Space:
        - Global: Discrete(num_datacenters) for each arriving cloudlet
        - Local: Discrete(num_vms_per_dc) for each datacenter

    Observation Space:
        - Global: Aggregated state of all datacenters (green power, queues, utilization)
        - Local: Per-DC state (VM loads, local queues, next cloudlet)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    # Class-level guard to avoid closing the Java gateway multiple times across instances
    _java_gateway_closed = False

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hierarchical multi-datacenter environment.

        Args:
            config: Configuration dictionary containing:
                - multi_datacenter_enabled: bool
                - datacenters: List[dict] of datacenter configurations
                - py4j_port: int (default 25333)
                - global_routing_batch_size: int (cloudlets to route per step, default 5)
                - max_arriving_cloudlets: int (deprecated, for backward compatibility)
                - ... other CloudSim Plus settings
        """
        super(HierarchicalMultiDCEnv, self).__init__()

        self.config = config
        self.py4j_port = config.get("py4j_port", 25333)
        self.num_datacenters = len(config.get("datacenters", [{"datacenter_id": 0}]))
        
        # Fixed batch size for global routing decisions (key parameter)
        self.global_routing_batch_size = config.get("global_routing_batch_size", 5)
        
        # Backward compatibility: if max_arriving_cloudlets is set, use it as batch size
        if "max_arriving_cloudlets" in config and "global_routing_batch_size" not in config:
            logger.warning(
                "'max_arriving_cloudlets' is deprecated. Use 'global_routing_batch_size' instead."
            )
            self.global_routing_batch_size = config.get("max_arriving_cloudlets", 5)

        # Py4J Gateway connection
        self.gateway = None
        self.java_env = None

        # Episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Define observation and action spaces
        self._setup_observation_spaces()
        self._setup_action_spaces()

        logger.info(f"HierarchicalMultiDCEnv initialized with {self.num_datacenters} datacenters")

    def _setup_observation_spaces(self):
        """
        Define observation spaces for global and local agents.
        """
        # Global observation space (aggregated DC-level metrics)
        self.global_observation_space = spaces.Dict({
            # Green energy metrics (W - Watts)
            "dc_current_green_power_w": spaces.Box(
                low=0.0, high=5000000.0,  # 5 MW max (increased to accommodate high wind power)
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_current_power_w": spaces.Box(
                low=0.0, high=10000.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_green_ratio": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_cumulative_wasted_green_wh": spaces.Box(
                low=0.0, high=1e6,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_queue_sizes": spaces.Box(
                low=0, high=10000,
                shape=(self.num_datacenters,),
                dtype=np.int32  # Changed to int32 (queue sizes are integers)
            ),
            "dc_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_available_pes": spaces.Box(
                low=0, high=1000,
                shape=(self.num_datacenters,),
                dtype=np.int32  # Changed to int32 (PEs are integers)
            ),
            "dc_ram_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "upcoming_cloudlets_count": spaces.Discrete(1000),  # Total cloudlets in global waiting queue
            "batch_cloudlet_pes": spaces.Box(
                low=0, high=100,  # Max PEs for a cloudlet
                shape=(self.global_routing_batch_size,),
                dtype=np.int32
            ),
            "batch_cloudlet_mi": spaces.Box(
                low=0, high=1000000,  # Max MI for a cloudlet
                shape=(self.global_routing_batch_size,),
                dtype=np.int64
            ),
            "upcoming_pes_distribution": spaces.Box(
                low=0, high=1000,
                shape=(3,),  # [small (1-2 PEs), medium (3-4 PEs), large (5+ PEs)]
                dtype=np.int32
            ),
            "load_imbalance": spaces.Box(
                low=0.0, high=10.0,
                shape=(1,),
                dtype=np.float32
            ),
            "recent_completed": spaces.Discrete(10000),
        })

        # Local observation spaces (per datacenter)
        # Track per-DC sizes but expose a shared max-sized space for SB3 compatibility
        dc_defaults = {
            "hosts_count": 16,
            "initial_s_vm_count": 10,
            "initial_m_vm_count": 5,
            "initial_l_vm_count": 3,
        }
        dc_configs = self.config.get("datacenters")
        if not dc_configs:
            dc_configs = [dc_defaults.copy()]

        self.dc_host_counts: List[int] = [
            int(dc.get("hosts_count", dc_defaults["hosts_count"])) for dc in dc_configs
        ]
        self.max_hosts = max(self.dc_host_counts) if self.dc_host_counts else dc_defaults["hosts_count"]

        self.dc_vm_counts: List[int] = [
            int(
                dc.get("initial_s_vm_count", dc_defaults["initial_s_vm_count"]) +
                dc.get("initial_m_vm_count", dc_defaults["initial_m_vm_count"]) +
                dc.get("initial_l_vm_count", dc_defaults["initial_l_vm_count"])
            )
            for dc in dc_configs
        ]
        self.max_vms = max(self.dc_vm_counts) if self.dc_vm_counts else (
            dc_defaults["initial_s_vm_count"] +
            dc_defaults["initial_m_vm_count"] +
            dc_defaults["initial_l_vm_count"]
        )

        self.local_observation_space = spaces.Dict({
            "host_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_hosts,),
                dtype=np.float32
            ),
            "host_ram_usage": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_hosts,),
                dtype=np.float32
            ),
            "vm_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_vms,),
                dtype=np.float32
            ),
            "vm_types": spaces.Box(
                low=0, high=3,  # 0=Off, 1=Small, 2=Medium, 3=Large
                shape=(self.max_vms,),
                dtype=np.int32
            ),
            "vm_available_pes": spaces.Box(
                low=0, high=100,
                shape=(self.max_vms,),
                dtype=np.int32
            ),
            "waiting_cloudlets": spaces.Discrete(10000),
            "next_cloudlet_pes": spaces.Discrete(100),
        })

    def _setup_action_spaces(self):
        """
        Define action spaces for global and local agents.
        
        Global Agent: Routes a fixed batch of cloudlets per step.
        - If fewer cloudlets available, extra actions are ignored.
        - If more cloudlets available, they wait in queue for next step.
        
        Local Agents: Assign one cloudlet per DC per step.
        """
        # Global action space: Fixed batch of routing decisions
        # Each action selects a datacenter for one cloudlet
        self.global_action_space = spaces.MultiDiscrete(
            [self.num_datacenters] * self.global_routing_batch_size
        )

        # Local action spaces: Select VM for each datacenter's next cloudlet
        # Each datacenter has its own action space
        # Action space includes: 0 = NoAssign, 1 to max_vms = VM indices
        max_vms = getattr(self, "max_vms", 1)
        self.local_action_space = spaces.Discrete(max_vms + 1)  # +1 for NoAssign option

        # Gymnasium requires self.action_space and self.observation_space
        # Combine global and local spaces into a Dict space
        self.action_space = spaces.Dict({
            "global": self.global_action_space,
            "local": spaces.Dict({
                i: self.local_action_space for i in range(self.num_datacenters)
            })
        })

        self.observation_space = spaces.Dict({
            "global": self.global_observation_space,
            "local": spaces.Dict({
                i: self.local_observation_space for i in range(self.num_datacenters)
            })
        })

    def _connect_to_java(self):
        """
        Establish Py4J connection to Java gateway with retry mechanism.

        Retries connection up to max_retries times with exponential backoff.
        If connection fails after all retries, raises RuntimeError.
        """
        if self.gateway is None:
            max_retries = self.config.get("gateway_max_retries", 5)
            retry_delay = self.config.get("gateway_retry_delay", 5.0)

            logger.info(f"Attempting to connect to Java gateway on port {self.py4j_port}...")

            retries = max_retries
            while retries > 0:
                try:
                    # Attempt connection
                    self.gateway = JavaGateway(
                        gateway_parameters=GatewayParameters(port=self.py4j_port, auto_convert=True)
                    )

                    # Test connection by calling a simple Java method
                    self.gateway.jvm.System.out.println(
                        f"Python HierarchicalMultiDCEnv connected on port {self.py4j_port}!"
                    )

                    self.java_env = self.gateway.entry_point
                    logger.info(f"Successfully connected to Java gateway on port {self.py4j_port}")

                    # Successfully connected, exit retry loop
                    break

                except (ConnectionRefusedError, Py4JNetworkError) as e:
                    retries -= 1
                    if retries > 0:
                        logger.warning(
                            f"Gateway connection failed: {e}. "
                            f"Retrying in {retry_delay} seconds... ({retries} retries left)"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error("Max retries reached. Could not connect to Java gateway.")
                        raise RuntimeError(
                            f"Could not connect to Java gateway on port {self.py4j_port} "
                            f"after {max_retries} attempts. "
                            f"Make sure the Java gateway server is running:\n"
                            f"  cd cloudsimplus-gateway && ./gradlew run"
                        ) from e

                except Exception as e:
                    # Unexpected error, don't retry
                    logger.error(f"Unexpected error connecting to Java gateway: {e}")
                    raise RuntimeError(
                        f"Unexpected error connecting to Java gateway: {e}"
                    ) from e

            # Configure simulation after successful connection
            try:
                logger.info("Configuring multi-datacenter simulation...")
                self.java_env.configureSimulation(self.config)
                logger.info("Multi-datacenter simulation configured successfully")

            except Exception as e:
                logger.error(f"Failed to configure simulation: {e}")
                # Clean up gateway connection on configuration failure
                self._cleanup_gateway()
                raise RuntimeError(
                    f"Failed to configure multi-datacenter simulation. "
                    f"Check Java logs for details."
                ) from e

    def _cleanup_gateway(self):
        """
        Clean up gateway connection resources.
        """
        if self.gateway is not None:
            try:
                self.gateway.close()
                logger.info("Java gateway connection closed")
            except Exception as e:
                logger.warning(f"Error closing gateway: {e}")
            finally:
                self.gateway = None
                self.java_env = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (not used currently)

        Returns:
            observations: Dict with 'global' and 'local' observations
            info: Additional information

        Raises:
            RuntimeError: If connection to Java gateway fails or reset fails
        """
        super().reset(seed=seed)

        # Connect to Java if not already connected (with retry mechanism)
        self._connect_to_java()

        # Reset Java simulation
        try:
            logger.debug(f"Resetting Java simulation with seed {seed}...")
            result = self.java_env.reset(seed if seed is not None else 0)
        except Exception as e:
            logger.error(f"Failed to reset Java simulation: {e}")
            raise RuntimeError(
                f"Failed to reset multi-datacenter simulation. "
                f"Check Java logs for details."
            ) from e

        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Parse observations from HierarchicalResetResult
        try:
            # Reset returns HierarchicalResetResult (only observations and info)
            observations = self._parse_hierarchical_observation_from_reset(result)
            info = self._parse_info_from_reset(result)
        except Exception as e:
            logger.error(f"Failed to parse reset result: {e}")
            raise RuntimeError(
                f"Failed to parse observations from Java. "
                f"Check observation structure compatibility."
            ) from e

        # Store observations for action masking
        self.last_observations = observations

        logger.info(f"Environment reset successfully for episode (seed={seed})")
        return observations, info

    def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one hierarchical step in the environment.

        Args:
            action: Dictionary containing:
                - 'global': List of datacenter indices for arriving cloudlets
                - 'local': Dict mapping datacenter_id -> vm_id

        Returns:
            observations: Dict with 'global' and 'local' observations
            rewards: Dict with 'global' and 'local' rewards
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information

        Raises:
            RuntimeError: If environment not initialized or step execution fails
            ValueError: If action format is invalid
        """
        if self.java_env is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first before calling step()."
            )

        # Validate and extract actions
        try:
            global_actions = action.get("global", [])
            local_actions_map = action.get("local", {})

            if not isinstance(global_actions, (list, np.ndarray)):
                raise ValueError(
                    f"'global' actions must be a list or array, got {type(global_actions)}"
                )
            if not isinstance(local_actions_map, dict):
                raise ValueError(
                    f"'local' actions must be a dict, got {type(local_actions_map)}"
                )
        except Exception as e:
            logger.error(f"Invalid action format: {e}")
            raise ValueError(f"Invalid action format. Expected dict with 'global' and 'local' keys.") from e

        # Get actual number of cloudlets in global waiting queue (batch routing mode)
        try:
            num_available = self.java_env.getGlobalWaitingCloudletsCount()
        except Exception as e:
            logger.error(f"Failed to get global waiting cloudlets count: {e}")
            # Continue with 0 if this fails
            num_available = 0

        # Dynamically trim global actions to actual available cloudlets
        # This allows the agent to output a fixed batch_size, but only route what's available
        if len(global_actions) > num_available:
            logger.debug(f"Trimming global actions from {len(global_actions)} to {num_available} (queue size)")
            global_actions = global_actions[:num_available]

        # Convert local actions dict to Java-compatible format
        # Apply action mapping: agent outputs 0 to num_vms -> Java expects -1 to num_vms-1
        # - action=0 → targetVmId=-1 (NoAssign)
        # - action=1 → targetVmId=0 (VM 0)
        # - action=n → targetVmId=n-1 (VM n-1)
        try:
            local_actions_java = {}
            for dc_id, agent_action in local_actions_map.items():
                # Map agent action to Java targetVmId
                target_vm_id = int(agent_action) - 1  # 0→-1, 1→0, 2→1, ...
                local_actions_java[int(dc_id)] = target_vm_id
                logger.debug(f"DC {dc_id}: agent_action={agent_action} → targetVmId={target_vm_id}")
        except Exception as e:
            logger.error(f"Failed to convert local actions: {e}")
            raise ValueError(f"Invalid local action format. DC IDs and VM IDs must be integers.") from e

        # Convert numpy types to Python native types for Py4J compatibility
        # Py4J cannot serialize numpy.int64, numpy.ndarray, etc.
        if isinstance(global_actions, np.ndarray):
            global_actions = global_actions.tolist()
        global_actions_python = [int(x) for x in global_actions]  # Ensure all elements are Python int
        local_actions_python = {int(k): int(v) for k, v in local_actions_java.items()}

        # Execute step in Java simulation
        try:
            logger.info(f"[STEP {self.current_step + 1}] Calling Java with global_actions={global_actions_python}, local_actions={local_actions_python}")
            print(f"[DEBUG HierarchicalMultiDCEnv] Calling Java step with {len(global_actions_python)} global actions")
            result = self.java_env.step(global_actions_python, local_actions_python)
            print(f"[DEBUG HierarchicalMultiDCEnv] Java step returned successfully")
        except Exception as e:
            logger.error(f"Failed to execute step in Java simulation: {e}")
            print(f"[DEBUG HierarchicalMultiDCEnv] Java step FAILED: {e}")
            raise RuntimeError(
                f"Failed to execute simulation step. Check Java logs for details."
            ) from e

        # Parse results
        try:
            observations = self._parse_hierarchical_observation(result)
            rewards = self._parse_hierarchical_rewards(result)
            terminated = result.isTerminated()
            truncated = result.isTruncated()
            info = self._parse_info(result)
        except Exception as e:
            logger.error(f"Failed to parse step result: {e}")
            raise RuntimeError(
                f"Failed to parse step results from Java. "
                f"Check observation/reward structure compatibility."
            ) from e

        # Update episode state
        self.current_step += 1
        self.episode_reward += rewards["global"]
        self.done = terminated or truncated

        # Store observations for action masking
        self.last_observations = observations

        logger.debug(
            f"Step {self.current_step}: Global reward={rewards['global']:.3f}, "
            f"Terminated={terminated}, Truncated={truncated}"
        )

        return observations, rewards, terminated, truncated, info

    def _parse_hierarchical_observation_from_reset(
        self,
        result  # HierarchicalResetResult from Java
    ) -> Dict[str, Any]:
        """
        Parse HierarchicalResetResult into observation dict.
        This is specifically for reset() which returns HierarchicalResetResult.
        """
        # Parse global observation (GlobalObservationState)
        global_obs_java = result.getGlobalObservation()
        global_obs = self._convert_global_observation(global_obs_java)

        # Parse local observations (Map<Integer, ObservationState>)
        local_obs_java = result.getLocalObservations()
        local_obs = {}
        for dc_id in local_obs_java:
            obs_state = local_obs_java[dc_id]
            if obs_state is not None:
                dc_index = int(dc_id)
                local_obs[dc_index] = self._convert_local_observation(dc_index, obs_state)

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_info_from_reset(
        self,
        result  # HierarchicalResetResult from Java
    ) -> Dict[str, Any]:
        """
        Parse info from HierarchicalResetResult.
        This is specifically for reset() which returns HierarchicalResetResult.
        Ensures all values are Python native types (serializable).
        """
        info_java = result.getInfo()
        info = {}
        for key in info_java:
            value = info_java[key]
            # Convert Java objects to Python native types
            info[str(key)] = self._convert_java_value(value)
        return info

    def _convert_global_observation(self, global_obs_java) -> Dict[str, Any]:
        """
        Convert Java GlobalObservationState to Python dict.
        """
        return {
            # Green energy metrics
            "dc_current_green_power_w": np.array(global_obs_java.getDcCurrentGreenPowerW(), dtype=np.float32),
            "dc_current_power_w": np.array(global_obs_java.getDcCurrentPowerW(), dtype=np.float32),
            "dc_green_ratio": np.array(global_obs_java.getDcGreenRatio(), dtype=np.float32),
            "dc_cumulative_wasted_green_wh": np.array(global_obs_java.getDcCumulativeWastedGreenWh(), dtype=np.float32),
            "dc_queue_sizes": np.array(global_obs_java.getDcQueueSizes(), dtype=np.int32),
            "dc_utilizations": np.array(global_obs_java.getDcUtilizations(), dtype=np.float32),
            "dc_available_pes": np.array(global_obs_java.getDcAvailablePes(), dtype=np.int32),
            "dc_ram_utilizations": np.array(global_obs_java.getDcRamUtilizations(), dtype=np.float32),
            "upcoming_cloudlets_count": global_obs_java.getUpcomingCloudletsCount(),
            "batch_cloudlet_pes": np.array(global_obs_java.getBatchCloudletPes(), dtype=np.int32),
            "batch_cloudlet_mi": np.array(global_obs_java.getBatchCloudletMi(), dtype=np.int64),
            "upcoming_pes_distribution": np.array(global_obs_java.getUpcomingCloudletsPesDistribution(), dtype=np.int32),
            "load_imbalance": np.array([global_obs_java.getLoadImbalance()], dtype=np.float32),
            "recent_completed": global_obs_java.getRecentCompletedCloudlets(),
        }

    def _convert_local_observation(self, dc_id: int, local_obs_java) -> Dict[str, Any]:
        """
        Convert Java ObservationState to Python dict, padding/trimming so each DC
        matches the shared observation space while preserving its own host/VM count.
        """
        host_target = self._get_dc_host_count(dc_id)
        vm_target = self._get_dc_vm_count(dc_id)

        host_loads = np.array(local_obs_java.getHostLoads(), dtype=np.float32)[:host_target]
        host_ram_usage = np.array(local_obs_java.getHostRamUsageRatio(), dtype=np.float32)[:host_target]
        vm_loads = np.array(local_obs_java.getVmLoads(), dtype=np.float32)[:vm_target]
        vm_types = np.array(local_obs_java.getVmTypes(), dtype=np.int32)[:vm_target]
        vm_available_pes = np.array(local_obs_java.getVmAvailablePes(), dtype=np.int32)[:vm_target]

        return {
            "host_loads": self._pad_vector(host_loads, self.max_hosts, 0.0),
            "host_ram_usage": self._pad_vector(host_ram_usage, self.max_hosts, 0.0),
            "vm_loads": self._pad_vector(vm_loads, self.max_vms, 0.0),
            "vm_types": self._pad_vector(vm_types, self.max_vms, 0),
            "vm_available_pes": self._pad_vector(vm_available_pes, self.max_vms, 0),
            "waiting_cloudlets": local_obs_java.getWaitingCloudlets(),
            "next_cloudlet_pes": local_obs_java.getNextCloudletPes(),
        }

    def _get_dc_host_count(self, dc_id: int) -> int:
        """Return configured host count for a datacenter (fallback to max_hosts)."""
        if hasattr(self, "dc_host_counts") and 0 <= dc_id < len(self.dc_host_counts):
            return self.dc_host_counts[dc_id]
        return getattr(self, "max_hosts", 1)

    def _get_dc_vm_count(self, dc_id: int) -> int:
        """Return configured VM count for a datacenter (fallback to max_vms)."""
        if hasattr(self, "dc_vm_counts") and 0 <= dc_id < len(self.dc_vm_counts):
            return self.dc_vm_counts[dc_id]
        return getattr(self, "max_vms", 1)

    @staticmethod
    def _pad_vector(values: np.ndarray, target_len: int, fill_value: float) -> np.ndarray:
        """
        Ensure vectors share a consistent length by trimming overflow and padding
        the tail with a provided fill_value.
        """
        current_len = values.shape[0]
        if current_len == target_len:
            return values

        if current_len > target_len:
            return values[:target_len]

        padded = np.full((target_len,), fill_value, dtype=values.dtype)
        if current_len > 0:
            padded[:current_len] = values
        return padded

    def _parse_hierarchical_observation(
        self,
        result  # HierarchicalStepResult from Java
    ) -> Dict[str, Any]:
        """
        Parse HierarchicalStepResult into observation dict.
        This is specifically for step() which returns HierarchicalStepResult.
        """
        # Parse global observation
        global_obs_java = result.getGlobalObservation()
        global_obs = self._convert_global_observation(global_obs_java)

        # Parse local observations
        local_obs_map_java = result.getLocalObservations()
        local_obs = {}
        for dc_id in range(self.num_datacenters):
            if dc_id in local_obs_map_java:
                obs_state = local_obs_map_java[dc_id]
                if obs_state is not None:
                    local_obs[dc_id] = self._convert_local_observation(dc_id, obs_state)

        return {
            "global": global_obs,
            "local": local_obs
        }

    def _parse_hierarchical_rewards(
        self,
        result
    ) -> Dict[str, Any]:
        """
        Parse hierarchical rewards from step result.

        Returns:
            {
                'global': float,
                'local': {dc_id: float}
            }
        """
        global_reward = result.getGlobalReward()

        local_rewards_java = result.getLocalRewards()
        local_rewards = {
            dc_id: local_rewards_java.get(dc_id, 0.0)
            for dc_id in range(self.num_datacenters)
        }

        return {
            "global": global_reward,
            "local": local_rewards
        }

    def _parse_info(self, result) -> Dict[str, Any]:
        """
        Parse additional info from step result.
        Ensures all values are Python native types (serializable).
        """
        info_java = result.getInfo()

        # Convert Java Map to Python dict with serializable values
        info = {}
        for key in info_java.keySet():
            value = info_java.get(key)
            # Convert Java objects to Python native types
            info[str(key)] = self._convert_java_value(value)

        info["episode_step"] = self.current_step
        info["episode_reward"] = self.episode_reward

        return info
    
    def _convert_java_value(self, value):
        """
        Convert a Java value to a Python native type (for serialization).
        """
        if value is None:
            return None
        
        # Already Python type
        if isinstance(value, (bool, int, float, str)):
            return value
        
        # Try to convert using Python type constructors
        try:
            # Try as int first (for Integer, Long, etc.)
            return int(value)
        except (TypeError, ValueError):
            pass
        
        try:
            # Try as float (for Double, Float, etc.)
            return float(value)
        except (TypeError, ValueError):
            pass
        
        try:
            # Try as bool (for Boolean)
            if str(value).lower() in ('true', 'false'):
                return str(value).lower() == 'true'
        except:
            pass
        
        # For complex objects (Maps, Lists), convert to string
        return str(value)

    def render(self):
        """
        Render the environment (not implemented for this environment).
        """
        pass

    def close(self):
        """
        Close the environment and cleanup resources.

        Safely closes the Java simulation environment and shuts down the Py4J gateway.
        This method is called automatically by Gymnasium when the environment is no longer needed.
        """
        # Close Java simulation environment
        if self.java_env is not None:
            try:
                logger.info("Closing Java simulation environment...")
                self.java_env.close()
                logger.info("Java simulation environment closed successfully")
            except Exception as e:
                logger.warning(f"Error closing Java simulation environment: {e}")

        # Shutdown Py4J gateway
        if self.gateway is not None:
            try:
                logger.info("Shutting down Py4J gateway...")
                self.gateway.shutdown()
                logger.info("Py4J gateway shutdown successfully")
            except Exception as e:
                logger.warning(f"Error shutting down Py4J gateway: {e}")
            finally:
                self.gateway = None
                self.java_env = None

    def get_num_datacenters(self) -> int:
        """Get the number of datacenters in the environment."""
        return self.num_datacenters

    def get_arriving_cloudlets_count(self) -> int:
        """
        Get the number of cloudlets in global waiting queue.
        
        DEPRECATED: This method name is misleading. Use get_global_waiting_cloudlets_count() instead.
        Kept for backward compatibility with tests.
        """
        return self.get_global_waiting_cloudlets_count()
    
    def get_global_waiting_cloudlets_count(self) -> int:
        """Get the number of cloudlets in the global waiting queue (batch routing mode)."""
        if self.java_env is None:
            return 0
        return self.java_env.getGlobalWaitingCloudletsCount()

    def get_local_action_masks(self, dc_id: int) -> np.ndarray:
        """
        Generate action mask for a specific datacenter's local agent.

        Mask logic (consistent with Single-DC environment):
        - If queue is empty: only allow action 0 (NoAssign)
        - If queue has tasks: forbid action 0, allow VMs with enough resources
        - If no VM has enough resources: allow all VMs (Java handles penalty)

        Args:
            dc_id: Datacenter ID

        Returns:
            mask: Boolean array of shape (num_vms+1,) where True = action allowed
        """
        # Fallback: allow all actions if environment not initialized
        if self.java_env is None or dc_id >= self.num_datacenters or dc_id < 0:
            logger.warning(f"Cannot generate mask for DC {dc_id}, allowing all actions")
            return np.ones(self.local_action_space.n, dtype=bool)

        # Get DC state from last observation
        try:
            if not hasattr(self, 'last_observations') or 'local' not in self.last_observations:
                logger.debug(f"No observations available yet, allowing all actions for DC {dc_id}")
                return np.ones(self.local_action_space.n, dtype=bool)

            local_obs = self.last_observations["local"].get(dc_id)
            if local_obs is None:
                logger.warning(f"No observation for DC {dc_id}, allowing all actions")
                return np.ones(self.local_action_space.n, dtype=bool)

            vm_available_pes = local_obs["vm_available_pes"]
            waiting_cloudlets = local_obs["waiting_cloudlets"]
            next_cloudlet_pes = local_obs["next_cloudlet_pes"]

        except Exception as e:
            logger.error(f"Failed to extract state for DC {dc_id}: {e}, allowing all actions")
            return np.ones(self.local_action_space.n, dtype=bool)

        # Get actual VM count for this DC
        dc_vm_count = self._get_dc_vm_count(dc_id)
        
        # Initialize mask (all False)
        mask = np.zeros(self.local_action_space.n, dtype=bool)

        # Case 1: Queue is empty or next task invalid
        if waiting_cloudlets == 0 or next_cloudlet_pes == 0:
            mask[0] = True  # Only allow action 0 (NoAssign)
            logger.debug(f"DC {dc_id}: Queue empty, only NoAssign allowed")
            return mask

        # Case 2: Queue has tasks
        mask[0] = False  # Forbid explicit NoAssign (encourage assignment)

        # Check each VM's resources (only actual VMs, not padding)
        has_valid_vm = False
        for vm_idx in range(min(len(vm_available_pes), dc_vm_count)):
            available_pes = vm_available_pes[vm_idx]
            if available_pes >= next_cloudlet_pes:
                mask[vm_idx + 1] = True  # action (vm_idx+1) → targetVmId (vm_idx)
                has_valid_vm = True

        # Case 3: No VM has enough resources
        # Align with loadbalancing_env.py: Force assignment (disallow NoAssign)
        if not has_valid_vm:
            logger.debug(f"DC {dc_id}: No VM has {next_cloudlet_pes} PEs, allowing all VMs (forcing assignment)")
            mask[0] = False  # Disallow NoAssign
            mask[1:dc_vm_count+1] = True  # Allow all VMs

        logger.debug(f"DC {dc_id}: Mask generated - {np.sum(mask)}/{len(mask)} actions allowed")
        return mask
