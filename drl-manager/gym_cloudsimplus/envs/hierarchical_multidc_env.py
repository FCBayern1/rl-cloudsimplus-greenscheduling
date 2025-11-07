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
                - max_arriving_cloudlets: int (for action space sizing)
                - ... other CloudSim Plus settings
        """
        super(HierarchicalMultiDCEnv, self).__init__()

        self.config = config
        self.py4j_port = config.get("py4j_port", 25333)
        self.num_datacenters = len(config.get("datacenters", [{"datacenter_id": 0}]))
        self.max_arriving_cloudlets = config.get("max_arriving_cloudlets", 50)

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
            "dc_green_power": spaces.Box(
                low=0.0, high=10000.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            # New green energy metrics
            "dc_current_green_power_w": spaces.Box(
                low=0.0, high=10000.0,
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
            "upcoming_cloudlets_count": spaces.Discrete(self.max_arriving_cloudlets + 1),
            "next_cloudlet_pes": spaces.Discrete(100),  # Max PEs for a cloudlet
            "next_cloudlet_mi": spaces.Box(
                low=0, high=1000000,  # Max MI for a cloudlet
                shape=(1,),
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
        # Dynamically sized based on DC configs, but we'll use max sizes for now
        # Todo: This should be dynamic host number for different DC
        max_hosts = max([dc.get("hosts_count", 16) for dc in self.config.get("datacenters", [{"hosts_count": 16}])])
        max_vms = max([
            dc.get("initial_s_vm_count", 10) +
            dc.get("initial_m_vm_count", 5) +
            dc.get("initial_l_vm_count", 3)
            for dc in self.config.get("datacenters", [{"initial_s_vm_count": 10, "initial_m_vm_count": 5, "initial_l_vm_count": 3}])
        ])

        self.local_observation_space = spaces.Dict({
            "host_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(max_hosts,),
                dtype=np.float32
            ),
            "host_ram_usage": spaces.Box(
                low=0.0, high=1.0,
                shape=(max_hosts,),
                dtype=np.float32
            ),
            "vm_loads": spaces.Box(
                low=0.0, high=1.0,
                shape=(max_vms,),
                dtype=np.float32
            ),
            "vm_types": spaces.Box(
                low=0, high=3,  # 0=Off, 1=Small, 2=Medium, 3=Large
                shape=(max_vms,),
                dtype=np.int32
            ),
            "vm_available_pes": spaces.Box(
                low=0, high=100,
                shape=(max_vms,),
                dtype=np.int32
            ),
            "waiting_cloudlets": spaces.Discrete(10000),
            "next_cloudlet_pes": spaces.Discrete(100),
        })

    def _setup_action_spaces(self):
        """
        Define action spaces for global and local agents.
        """
        # Global action space: Select datacenter for each arriving cloudlet
        # Dynamic size based on actual arriving cloudlets
        self.global_action_space = spaces.MultiDiscrete(
            [self.num_datacenters] * self.max_arriving_cloudlets
        )

        # Local action spaces: Select VM for each datacenter's next cloudlet
        # Each datacenter has its own action space
        # For simplicity, we use a fixed max VM count
        max_vms = max([
            dc.get("initial_s_vm_count", 10) +
            dc.get("initial_m_vm_count", 5) +
            dc.get("initial_l_vm_count", 3)
            for dc in self.config.get("datacenters", [{"initial_s_vm_count": 10, "initial_m_vm_count": 5, "initial_l_vm_count": 3}])
        ])
        self.local_action_space = spaces.Discrete(max_vms)

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

        # Parse observations
        try:
            observations = self._parse_hierarchical_observation(result)
            info = self._parse_info(result)
        except Exception as e:
            logger.error(f"Failed to parse reset result: {e}")
            raise RuntimeError(
                f"Failed to parse observations from Java. "
                f"Check observation structure compatibility."
            ) from e

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

        # Get actual number of arriving cloudlets
        try:
            num_arriving = self.java_env.getArrivingCloudletsCount()
        except Exception as e:
            logger.error(f"Failed to get arriving cloudlets count: {e}")
            # Continue with 0 if this fails
            num_arriving = 0

        # Trim global actions to actual arriving count
        if len(global_actions) > num_arriving:
            global_actions = global_actions[:num_arriving]

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
                logger.trace(f"DC {dc_id}: agent_action={agent_action} → targetVmId={target_vm_id}")
        except Exception as e:
            logger.error(f"Failed to convert local actions: {e}")
            raise ValueError(f"Invalid local action format. DC IDs and VM IDs must be integers.") from e

        # Execute step in Java simulation
        try:
            logger.debug(f"Executing step {self.current_step + 1}...")
            result = self.java_env.step(global_actions, local_actions_java)
        except Exception as e:
            logger.error(f"Failed to execute step in Java simulation: {e}")
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

        logger.debug(
            f"Step {self.current_step}: Global reward={rewards['global']:.3f}, "
            f"Terminated={terminated}, Truncated={truncated}"
        )

        return observations, rewards, terminated, truncated, info

    def _parse_hierarchical_observation(
        self,
        result
    ) -> Dict[str, Any]:
        """
        Parse hierarchical step result into observation dict.

        Returns:
            {
                'global': global observation dict,
                'local': {dc_id: local observation dict}
            }
        """
        # Parse global observation (GlobalObservationState)
        global_obs_java = result.getGlobalObservation()
        global_obs = {
            "dc_green_power": np.array(global_obs_java.getDcGreenPower(), dtype=np.float32),
            # New green energy metrics
            "dc_current_green_power_w": np.array(global_obs_java.getDcCurrentGreenPowerW(), dtype=np.float32),
            "dc_current_power_w": np.array(global_obs_java.getDcCurrentPowerW(), dtype=np.float32),
            "dc_green_ratio": np.array(global_obs_java.getDcGreenRatio(), dtype=np.float32),
            "dc_cumulative_wasted_green_wh": np.array(global_obs_java.getDcCumulativeWastedGreenWh(), dtype=np.float32),
            "dc_queue_sizes": np.array(global_obs_java.getDcQueueSizes(), dtype=np.int32),
            "dc_utilizations": np.array(global_obs_java.getDcUtilizations(), dtype=np.float32),
            "dc_available_pes": np.array(global_obs_java.getDcAvailablePes(), dtype=np.int32),
            "dc_ram_utilizations": np.array(global_obs_java.getDcRamUtilizations(), dtype=np.float32),
            "upcoming_cloudlets_count": global_obs_java.getUpcomingCloudletsCount(),
            "next_cloudlet_pes": global_obs_java.getNextCloudletPes(),
            "next_cloudlet_mi": np.array([global_obs_java.getNextCloudletMi()], dtype=np.int64),
            "upcoming_pes_distribution": np.array(global_obs_java.getUpcomingCloudletsPesDistribution(), dtype=np.int32),
            "load_imbalance": np.array([global_obs_java.getLoadImbalance()], dtype=np.float32),
            "recent_completed": global_obs_java.getRecentCompletedCloudlets(),
        }

        # Parse local observations
        local_obs_map_java = result.getLocalObservations()
        local_obs = {}

        for dc_id in range(self.num_datacenters):
            if dc_id in local_obs_map_java:
                local_obs_java = local_obs_map_java[dc_id]
                local_obs[dc_id] = {
                    "host_loads": np.array(local_obs_java.getHostLoads(), dtype=np.float32),
                    "host_ram_usage": np.array(local_obs_java.getHostRamUsageRatio(), dtype=np.float32),
                    "vm_loads": np.array(local_obs_java.getVmLoads(), dtype=np.float32),
                    "vm_types": np.array(local_obs_java.getVmTypes(), dtype=np.int32),
                    "vm_available_pes": np.array(local_obs_java.getVmAvailablePes(), dtype=np.int32),
                    "waiting_cloudlets": local_obs_java.getWaitingCloudlets(),
                    "next_cloudlet_pes": local_obs_java.getNextCloudletPes(),
                }

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
        """
        info_java = result.getInfo()

        # Convert Java Map to Python dict
        info = {}
        for key in info_java.keySet():
            info[str(key)] = info_java.get(key)

        info["episode_step"] = self.current_step
        info["episode_reward"] = self.episode_reward

        return info

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
        """Get the number of cloudlets arriving in the current timestep."""
        if self.java_env is None:
            return 0
        return self.java_env.getArrivingCloudletsCount()

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

        # Initialize mask (all False)
        mask = np.zeros(self.local_action_space.n, dtype=bool)

        # Case 1: Queue is empty or next task invalid
        if waiting_cloudlets == 0 or next_cloudlet_pes == 0:
            mask[0] = True  # Only allow action 0 (NoAssign)
            logger.trace(f"DC {dc_id}: Queue empty, only NoAssign allowed")
            return mask

        # Case 2: Queue has tasks
        mask[0] = False  # Forbid explicit NoAssign (encourage assignment)

        # Check each VM's resources
        has_valid_vm = False
        for vm_idx, available_pes in enumerate(vm_available_pes):
            if available_pes >= next_cloudlet_pes:
                mask[vm_idx + 1] = True  # action (vm_idx+1) → targetVmId (vm_idx)
                has_valid_vm = True

        # Case 3: No VM has enough resources (fallback: allow all VMs, Java handles penalty)
        if not has_valid_vm:
            logger.trace(f"DC {dc_id}: No VM has {next_cloudlet_pes} PEs, allowing all VMs")
            mask[1:] = True  # Allow all VM actions
            mask[0] = False  # Still forbid NoAssign

        logger.trace(f"DC {dc_id}: Mask generated - {np.sum(mask)}/{len(mask)} actions allowed")
        return mask
