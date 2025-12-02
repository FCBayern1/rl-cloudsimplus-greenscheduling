"""
PettingZoo ParallelEnv wrapper for Hierarchical Multi-Datacenter Environment.

This module provides a PettingZoo-compatible interface for the hierarchical
multi-datacenter MARL system, enabling compatibility with PettingZoo-based
training frameworks (RLlib, CleanRL, etc.) while preserving all existing
functionality.

Architecture:
    PettingZoo ParallelEnv (this file)
        | wraps
    HierarchicalMultiDCEnv (existing, unchanged)
        | wraps
    Java CloudSim Plus Simulation (Py4J)

Agents:
    - "global_agent": Routes arriving cloudlets to datacenters
    - "local_agent_0", "local_agent_1", ...: VM scheduling per datacenter

Key Features:
    - Zero modifications to existing code
    - Zero Java changes required
    - Standard PettingZoo API compliance
    - Parameter sharing support via policy mapping
    - Action masking support
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .hierarchical_multidc_env import HierarchicalMultiDCEnv

logger = logging.getLogger(__name__)


class HierarchicalMultiDCParallelEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for hierarchical multi-datacenter MARL.

    This environment wraps the existing HierarchicalMultiDCEnv and provides
    a standard PettingZoo API, making it compatible with various MARL frameworks.

    Agents:
        - "global_agent": Handles datacenter routing decisions
        - "local_agent_0", ..., "local_agent_N": Handle VM scheduling per DC

    All agents act simultaneously in each timestep (parallel execution).

    Example:
        >>> config = {...}  # Your environment config
        >>> env = HierarchicalMultiDCParallelEnv(config)
        >>> observations, infos = env.reset()
        >>>
        >>> actions = {
        ...     "global_agent": np.array([0, 1, 2, 0, 1]),  # Route 5 cloudlets
        ...     "local_agent_0": 3,  # Assign to VM 3 in DC 0
        ...     "local_agent_1": 1,  # Assign to VM 1 in DC 1
        ...     "local_agent_2": 5   # Assign to VM 5 in DC 2
        ... }
        >>> observations, rewards, terminations, truncations, infos = env.step(actions)
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "hierarchical_multidc_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        """
        Initialize PettingZoo wrapper.

        Args:
            config: Configuration dictionary for HierarchicalMultiDCEnv
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()

        self.render_mode = render_mode

        # Wrap the base hierarchical environment (no modifications to original)
        logger.info("Creating base HierarchicalMultiDCEnv...")
        base_env = HierarchicalMultiDCEnv(config=config)

        # Wrap with wind prediction if enabled
        base_env = self._wrap_with_prediction_if_enabled(base_env, config)

        self.base_env = base_env

        self.num_datacenters = self.base_env.num_datacenters
        self.global_routing_batch_size = self.base_env.global_routing_batch_size

        # Define agent names (PettingZoo requirement: flat namespace)
        self.possible_agents = self._create_agent_list()
        self.agents = self.possible_agents.copy()

        # Define observation and action spaces for each agent
        self._observation_spaces = self._create_observation_spaces()
        self._action_spaces = self._create_action_spaces()

        # Store last observations for action masking
        self._last_observations = None

        logger.info(
            f"HierarchicalMultiDCParallelEnv initialized with {len(self.agents)} agents: "
            f"{self.agents}"
        )

    def _create_agent_list(self) -> List[str]:
        """
        Create flat list of agent names.

        Returns:
            List of agent names: ["global_agent", "local_agent_0", ...]
        """
        agents = ["global_agent"]
        agents.extend([f"local_agent_{i}" for i in range(self.num_datacenters)])
        return agents

    def _create_observation_spaces(self) -> Dict[str, spaces.Space]:
        """
        Create observation space dict for all agents.

        For action masking support, observation space is a Dict containing:
        - "observation": the original observation space
        - "action_mask": binary mask of valid actions

        Returns:
            Dict mapping agent_name -> observation_space (Dict space with mask)
        """
        obs_spaces = {}

        # Global agent observation space (NO action mask).
        # The global policy currently uses DictObsModel, which ignores action masks,
        # so we only expose the underlying observation.
        obs_spaces["global_agent"] = spaces.Dict({
            "observation": self.base_env.global_observation_space,
        })

        # Local agents observation spaces (each DC has its own obs and action mask size)
        for i in range(self.num_datacenters):
            dc_vm_count = self.base_env._get_dc_vm_count(i)
            dc_host_count = self.base_env._get_dc_host_count(i)
            num_local_actions = dc_vm_count + 1  # NoAssign + VMs

            # Each DC has different number of VMs/hosts, so create custom obs space

            local_obs_space = spaces.Dict({
                "host_loads": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(dc_host_count,),
                    dtype=np.float32
                ),
                "host_ram_usage": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(dc_host_count,),
                    dtype=np.float32
                ),
                "vm_loads": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(dc_vm_count,),
                    dtype=np.float32
                ),
                "vm_types": spaces.Box(
                    low=0, high=3,  # 0=Off, 1=Small, 2=Medium, 3=Large
                    shape=(dc_vm_count,),
                    dtype=np.int32
                ),
                "vm_available_pes": spaces.Box(
                    low=0, high=100,
                    shape=(dc_vm_count,),
                    dtype=np.int32
                ),
                "waiting_cloudlets": spaces.Discrete(100000),  # Increased for large workloads
                "next_cloudlet_pes": spaces.Discrete(256),  # Increased for cloudlets with more PEs
            })

            obs_spaces[f"local_agent_{i}"] = spaces.Dict({
                "observation": local_obs_space,
                "action_mask": spaces.Box(
                    low=0, high=1,
                    shape=(num_local_actions,),
                    dtype=np.float32
                )
            })

        return obs_spaces

    def _create_action_spaces(self) -> Dict[str, spaces.Space]:
        """
        Create action space dict for all agents.

        Each DC gets its own action space sized to its actual VM count.
        This eliminates invalid actions without needing action masking.

        Returns:
            Dict mapping agent_name -> action_space
        """
        action_spaces = {
            "global_agent": self.base_env.global_action_space
        }

        # Each local agent gets its own action space based on actual VM count
        for i in range(self.num_datacenters):
            dc_vm_count = self.base_env._get_dc_vm_count(i)
            # Action space: NoAssign (0) + VM indices (1 to dc_vm_count)
            action_spaces[f"local_agent_{i}"] = spaces.Discrete(dc_vm_count + 1)
            logger.info(f"DC {i}: {dc_vm_count} VMs -> action space Discrete({dc_vm_count + 1})")

        return action_spaces

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """
        PettingZoo API: Get observation spaces for all agents.

        Returns:
            Dict of observation spaces
        """
        return self._observation_spaces

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """
        PettingZoo API: Get action spaces for all agents.

        Returns:
            Dict of action spaces
        """
        return self._action_spaces

    def observation_space(self, agent: str) -> spaces.Space:
        """
        Get observation space for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Observation space for this agent
        """
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """
        Get action space for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Action space for this agent
        """
        return self._action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observations: Dict[agent_name, observation]
            infos: Dict[agent_name, info_dict]
        """
        logger.debug(f"Resetting PettingZoo environment (seed={seed})...")

        # Reset base environment
        hierarchical_obs, hierarchical_info = self.base_env.reset(seed=seed, options=options)

        # Convert hierarchical format to flat agent dict format
        observations = self._hierarchical_to_flat_observations(hierarchical_obs)

        # Replicate info for all agents (or customize per agent if needed)
        infos = {agent: hierarchical_info.copy() for agent in self.agents}

        # Store for action masking
        self._last_observations = observations

        # Reset agent list (all agents are active)
        self.agents = self.possible_agents.copy()

        logger.info("PettingZoo environment reset complete")
        return observations, infos

    def step(
        self,
        actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],  # observations
        Dict[str, float],  # rewards
        Dict[str, bool],  # terminations
        Dict[str, bool],  # truncations
        Dict[str, Dict]  # infos
    ]:
        """
        Execute one environment step with actions from all agents.

        Args:
            actions: Dict[agent_name, action]
                Example: {
                    "global_agent": np.array([0, 1, 2, 0, 1]),
                    "local_agent_0": 3,
                    "local_agent_1": 1,
                    "local_agent_2": 5
                }

        Returns:
            observations: Dict[agent_name, observation]
            rewards: Dict[agent_name, reward]
            terminations: Dict[agent_name, bool] - natural episode end
            truncations: Dict[agent_name, bool] - time limit reached
            infos: Dict[agent_name, info_dict]
        """
        # Convert flat agent actions to hierarchical format
        hierarchical_actions = self._flat_to_hierarchical_actions(actions)

        logger.debug(
            f"Step with actions: global={len(hierarchical_actions['global'])} cloudlets, "
            f"local={list(hierarchical_actions['local'].values())}"
        )

        # Execute step in base environment
        (
            hierarchical_obs,
            hierarchical_rewards,
            terminated,
            truncated,
            hierarchical_info
        ) = self.base_env.step(hierarchical_actions)

        # Convert results to PettingZoo format
        observations = self._hierarchical_to_flat_observations(hierarchical_obs)
        rewards = self._hierarchical_to_flat_rewards(hierarchical_rewards)

        # All agents share the same termination/truncation status
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # Replicate info for all agents
        infos = {agent: hierarchical_info.copy() for agent in self.agents}

        # Store for action masking
        self._last_observations = observations

        logger.debug(
            f"Step result: rewards={[f'{k}:{v:.2f}' for k, v in rewards.items()]}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        return observations, rewards, terminations, truncations, infos

    def _hierarchical_to_flat_observations(
        self,
        hierarchical_obs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert hierarchical observation format to flat agent dict with action masks.

        For RLlib action masking support, each observation is a dict with:
        - "observation": the actual observation array
        - "action_mask": binary mask of valid actions (1=valid, 0=invalid)

        Args:
            hierarchical_obs: {
                "global": {...},
                "local": {0: {...}, 1: {...}, ...}
            }

        Returns:
            {
                "global_agent": {"observation": ..., "action_mask": ...},
                "local_agent_0": {"observation": ..., "action_mask": ...},
                ...
            }
        """
        flat_obs = {}

        # Global agent observation (no action mask for global routing)
        flat_obs["global_agent"] = {
            "observation": hierarchical_obs["global"],
        }

        # Local agents observations with action masks
        for dc_id_raw, local_obs in hierarchical_obs["local"].items():
            # Ensure dc_id is Python int (Java may return Integer object)
            dc_id = int(dc_id_raw)
            agent_name = f"local_agent_{dc_id}"

            # Get actual VM and host counts for this DC
            dc_vm_count = self.base_env._get_dc_vm_count(dc_id)
            dc_host_count = self.base_env._get_dc_host_count(dc_id)

            # Debug: log observation sizes before trimming
            logger.debug(
                f"{agent_name}: Original obs sizes - "
                f"hosts={len(local_obs['host_loads'])}, vms={len(local_obs['vm_loads'])}, "
                f"Expected - hosts={dc_host_count}, vms={dc_vm_count}"
            )

            # Trim padded observation arrays to actual DC size
            # The base env pads to max_vms/max_hosts, we need to trim to actual size
            trimmed_obs = {
                "host_loads": local_obs["host_loads"][:dc_host_count],
                "host_ram_usage": local_obs["host_ram_usage"][:dc_host_count],
                "vm_loads": local_obs["vm_loads"][:dc_vm_count],
                "vm_types": local_obs["vm_types"][:dc_vm_count],
                "vm_available_pes": local_obs["vm_available_pes"][:dc_vm_count],
                "waiting_cloudlets": local_obs["waiting_cloudlets"],
                "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
            }

            # Verify trimmed sizes match expected
            if len(trimmed_obs["host_loads"]) != dc_host_count:
                logger.error(
                    f"{agent_name}: Trimmed host_loads size mismatch! "
                    f"Got {len(trimmed_obs['host_loads'])}, expected {dc_host_count}"
                )
            if len(trimmed_obs["vm_loads"]) != dc_vm_count:
                logger.error(
                    f"{agent_name}: Trimmed vm_loads size mismatch! "
                    f"Got {len(trimmed_obs['vm_loads'])}, expected {dc_vm_count}"
                )

            # Get action mask for this local agent
            try:
                action_mask = self.base_env.get_local_action_masks(dc_id)
                # Trim action mask to actual DC action space size
                action_mask = action_mask[:dc_vm_count + 1]  # +1 for NoAssign
                # Convert to float32 for RLlib compatibility
                action_mask = action_mask.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to get action mask for {agent_name}: {e}")
                # Fallback: allow all actions (with correct size)
                action_mask = np.ones(dc_vm_count + 1, dtype=np.float32)

            flat_obs[agent_name] = {
                "observation": trimmed_obs,
                "action_mask": action_mask
            }

        return flat_obs

    def _flat_to_hierarchical_actions(
        self,
        flat_actions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert flat agent actions to hierarchical format.

        Args:
            flat_actions: {
                "global_agent": action,
                "local_agent_0": action,
                "local_agent_1": action,
                ...
            }

        Returns:
            {
                "global": action,
                "local": {0: action, 1: action, ...}
            }
        """
        # Extract global action
        global_action = flat_actions.get("global_agent")

        # Extract local actions
        local_actions = {}
        for i in range(self.num_datacenters):
            agent_name = f"local_agent_{i}"
            if agent_name in flat_actions:
                local_actions[i] = flat_actions[agent_name]

        hierarchical_actions = {
            "global": global_action,
            "local": local_actions
        }

        return hierarchical_actions

    def _hierarchical_to_flat_rewards(
        self,
        hierarchical_rewards: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Convert hierarchical reward format to flat agent dict.

        Args:
            hierarchical_rewards: {
                "global": reward,
                "local": {0: reward, 1: reward, ...}
            }

        Returns:
            {
                "global_agent": reward,
                "local_agent_0": reward,
                "local_agent_1": reward,
                ...
            }
        """
        flat_rewards = {
            "global_agent": float(hierarchical_rewards["global"])
        }

        for dc_id, reward in hierarchical_rewards["local"].items():
            agent_name = f"local_agent_{dc_id}"
            flat_rewards[agent_name] = float(reward)

        return flat_rewards

    def get_action_mask(self, agent: str) -> Optional[np.ndarray]:
        """
        Get action mask for a specific agent.

        This method provides action masking information for invalid actions,
        which is useful for masked PPO algorithms.

        Args:
            agent: Agent name

        Returns:
            Boolean array where True = valid action, False = invalid action
            Returns None for global agent (no masking needed)
        """
        if agent == "global_agent":
            # Global agent doesn't need action masking
            return None

        if self._last_observations is None:
            logger.warning(f"Action mask requested for {agent} before first observation")
            return None

        # Extract DC ID from agent name
        if not agent.startswith("local_agent_"):
            logger.warning(f"Unknown agent for action masking: {agent}")
            return None

        try:
            dc_id = int(agent.split("_")[-1])
        except (ValueError, IndexError):
            logger.error(f"Invalid agent name format: {agent}")
            return None

        # Get mask from base environment
        try:
            mask = self.base_env.get_local_action_masks(dc_id)
            return mask
        except Exception as e:
            logger.error(f"Failed to get action mask for {agent}: {e}")
            return None

    def get_all_action_masks(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get action masks for all agents.

        Returns:
            Dict mapping agent_name -> action_mask
        """
        masks = {}

        for agent in self.agents:
            masks[agent] = self.get_action_mask(agent)

        return masks

    def render(self) -> Optional[Any]:
        """
        Render the environment.

        Returns:
            Render output (format depends on render_mode)
        """
        if self.render_mode is None:
            return None

        # Delegate to base environment
        return self.base_env.render()

    def close(self):
        """
        Close environment and cleanup resources.
        """
        logger.info("Closing PettingZoo environment...")
        if hasattr(self, 'base_env') and self.base_env is not None:
            self.base_env.close()
        logger.info("PettingZoo environment closed")

    def state(self) -> np.ndarray:
        """
        Get global state (optional method for centralized training).

        Returns:
            Global state array combining all observations
        """
        if self._last_observations is None:
            return np.array([])

        # Concatenate global and all local observations
        state_parts = []

        # Add global observation
        global_obs = self._last_observations.get("global_agent", {})
        state_parts.extend(self._flatten_observation(global_obs))

        # Add local observations
        for i in range(self.num_datacenters):
            local_obs = self._last_observations.get(f"local_agent_{i}", {})
            state_parts.extend(self._flatten_observation(local_obs))

        return np.array(state_parts, dtype=np.float32)

    def _flatten_observation(self, obs: Dict[str, Any]) -> List[float]:
        """
        Flatten a dictionary observation into a 1D list.

        Args:
            obs: Observation dictionary

        Returns:
            Flattened list of values
        """
        values = []

        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                values.extend(value.flatten().tolist())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))

        return values

    def _wrap_with_prediction_if_enabled(
        self,
        base_env: Any,
        config: Dict[str, Any]
    ) -> Any:
        """
        Wrap environment with wind prediction if enabled in config.

        Args:
            base_env: Base environment to wrap
            config: Configuration dictionary

        Returns:
            Wrapped or original environment
        """
        wind_pred_config = config.get('wind_prediction', {})

        if not wind_pred_config.get('enabled', False):
            logger.info("Wind prediction disabled (config: wind_prediction.enabled = false)")
            return base_env

        logger.info("Wrapping environment with wind power prediction...")

        # Import here to avoid circular dependency
        from gym_cloudsimplus.wrappers import WindPredictionWrapper

        # Parse turbine_csv_paths (REQUIRED - convert list of dicts to dict)
        csv_paths_config = wind_pred_config.get('turbine_csv_paths')
        if csv_paths_config is None:
            logger.error(
                "Wind prediction enabled but 'turbine_csv_paths' not configured! "
                "Please add turbine_csv_paths to config.yml under wind_prediction section."
            )
            raise ValueError("turbine_csv_paths is required when wind_prediction is enabled")

        if isinstance(csv_paths_config, dict):
            # Already a dict, ensure keys are ints
            turbine_csv_paths = {int(k): v for k, v in csv_paths_config.items()}
        elif isinstance(csv_paths_config, list):
            # List of {turbine_id: path} dicts, merge them
            turbine_csv_paths = {}
            for item in csv_paths_config:
                turbine_csv_paths.update({int(k): v for k, v in item.items()})
        else:
            raise ValueError(f"Invalid turbine_csv_paths format: {type(csv_paths_config)}")

        wrapped_env = WindPredictionWrapper(
            env=base_env,
            model_checkpoint=wind_pred_config.get('model_checkpoint'),
            scalers_path=wind_pred_config.get('scalers_path'),
            data_path=wind_pred_config.get('data_path'),
            turbine_ids=wind_pred_config.get('turbine_ids', [1, 57, 124]),
            turbine_csv_paths=turbine_csv_paths,
            prediction_horizon=wind_pred_config.get('horizon', 8),
            device=wind_pred_config.get('device', 'cpu'),
            enable_logging=wind_pred_config.get('enable_logging', True),
            csv_start_offset=wind_pred_config.get('csv_start_offset', 12)
        )

        logger.info(
            f"Wind prediction enabled for PettingZoo environment: "
            f"horizon={wind_pred_config.get('horizon', 8)}, "
            f"turbines={wind_pred_config.get('turbine_ids', [1, 57, 124])}, "
            f"mode=13-feature CSV"
        )

        return wrapped_env


# Convenience function for creating the environment
def make_env(config: Dict[str, Any], **kwargs) -> HierarchicalMultiDCParallelEnv:
    """
    Factory function to create PettingZoo environment.

    Args:
        config: Environment configuration dictionary
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        HierarchicalMultiDCParallelEnv instance
    """
    return HierarchicalMultiDCParallelEnv(config=config, **kwargs)
