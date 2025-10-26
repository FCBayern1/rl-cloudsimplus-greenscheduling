"""
Hierarchical Multi-Datacenter Reinforcement Learning Environment

This environment implements a two-level hierarchical MARL system:
- Global Level: Routes arriving cloudlets to datacenters (Global Agent)
- Local Level: Schedules cloudlets to VMs within each datacenter (Local Agents)

Architecture:
    Python (Gymnasium) <--> Py4J <--> Java (CloudSim Plus Multi-DC Simulation)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters

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

    metadata = {"render_modes": []}

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
        super().__init__()

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
            "dc_queue_sizes": spaces.Box(
                low=0, high=10000,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_utilizations": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "dc_available_pes": spaces.Box(
                low=0, high=1000,
                shape=(self.num_datacenters,),
                dtype=np.float32
            ),
            "upcoming_cloudlets_count": spaces.Discrete(self.max_arriving_cloudlets + 1),
            "next_cloudlet_pes": spaces.Discrete(100),  # Max PEs for a cloudlet
        })

        # Local observation spaces (per datacenter)
        # Dynamically sized based on DC configs, but we'll use max sizes for now
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

    def _connect_to_java(self):
        """
        Establish Py4J connection to Java gateway.
        """
        if self.gateway is None:
            try:
                self.gateway = JavaGateway(
                    gateway_parameters=GatewayParameters(port=self.py4j_port, auto_convert=True)
                )
                self.java_env = self.gateway.entry_point
                logger.info(f"Connected to Java gateway on port {self.py4j_port}")

                # Configure simulation
                self.java_env.configureSimulation(self.config)
                logger.info("Java simulation configured successfully")

            except Exception as e:
                logger.error(f"Failed to connect to Java gateway: {e}")
                raise RuntimeError(
                    f"Could not connect to Java gateway on port {self.py4j_port}. "
                    f"Make sure the Java gateway server is running."
                ) from e

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns:
            observations: Dict with 'global' and 'local' observations
            info: Additional information
        """
        super().reset(seed=seed)

        # Connect to Java if not already connected
        self._connect_to_java()

        # Reset Java simulation
        result = self.java_env.reset(seed if seed is not None else 0)

        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

        # Parse observations
        observations = self._parse_hierarchical_observation(result)
        info = self._parse_info(result)

        logger.info(f"Environment reset for episode with seed {seed}")
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
        """
        if self.java_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Extract actions
        global_actions = action.get("global", [])
        local_actions_map = action.get("local", {})

        # Get actual number of arriving cloudlets
        num_arriving = self.java_env.getArrivingCloudletsCount()

        # Trim global actions to actual arriving count
        if len(global_actions) > num_arriving:
            global_actions = global_actions[:num_arriving]

        # Convert local actions dict to Java-compatible format
        local_actions_java = {}
        for dc_id, vm_id in local_actions_map.items():
            local_actions_java[int(dc_id)] = int(vm_id)

        # Execute step in Java simulation
        result = self.java_env.step(global_actions, local_actions_java)

        # Parse results
        observations = self._parse_hierarchical_observation(result)
        rewards = self._parse_hierarchical_rewards(result)
        terminated = result.isTerminated()
        truncated = result.isTruncated()
        info = self._parse_info(result)

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
        # Parse global observation
        global_obs_java = result.getGlobalObservation()
        global_obs = {
            "dc_green_power": np.array(global_obs_java.getVmLoads(), dtype=np.float32),  # Reused field
            "dc_queue_sizes": np.array(global_obs_java.getHostRamUsageRatio(), dtype=np.float32),  # Reused
            "dc_utilizations": np.array(global_obs_java.getHostLoads(), dtype=np.float32),
            "dc_available_pes": np.array(global_obs_java.getVmAvailablePes(), dtype=np.float32),
            "upcoming_cloudlets_count": global_obs_java.getWaitingCloudlets(),
            "next_cloudlet_pes": global_obs_java.getNextCloudletPes(),
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
        """
        if self.java_env is not None:
            try:
                self.java_env.close()
                logger.info("Java environment closed")
            except Exception as e:
                logger.warning(f"Error closing Java environment: {e}")

        if self.gateway is not None:
            try:
                self.gateway.shutdown()
                logger.info("Py4J gateway shutdown")
            except Exception as e:
                logger.warning(f"Error shutting down gateway: {e}")

        self.java_env = None
        self.gateway = None

    def get_num_datacenters(self) -> int:
        """Get the number of datacenters in the environment."""
        return self.num_datacenters

    def get_arriving_cloudlets_count(self) -> int:
        """Get the number of cloudlets arriving in the current timestep."""
        if self.java_env is None:
            return 0
        return self.java_env.getArrivingCloudletsCount()
