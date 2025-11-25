"""
Joint Training Environment for Hierarchical Multi-Datacenter MARL.

This wrapper enables joint training of the Global Agent and Local Agents
by managing their sequential execution within a single episode.

Design:
- Global Agent decides datacenter routing
- Local Agents decide VM scheduling (with parameter sharing)
- All agents are trained simultaneously in the same episode
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union
import logging
import yaml
import os

from .hierarchical_multidc_env import HierarchicalMultiDCEnv

logger = logging.getLogger(__name__)


class JointTrainingEnv(gym.Env):
    """
    Gymnasium wrapper for joint training of Global and Local agents.

    This environment wraps HierarchicalMultiDCEnv and provides
    a unified interface for training both agent types simultaneously.

    Training Flow:
    1. Global Agent receives global observation and selects datacenters
    2. Each Local Agent receives local observation and selects VMs
    3. Combined actions are executed in the simulation
    4. Each agent receives its respective reward

    Parameter Sharing:
    - All Local Agents share the same neural network
    - Different observations are fed to the shared network
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Union[str, Dict[str, Any]],
        mode: str = "training",
        render_mode: Optional[str] = None
    ):
        """
        Initialize joint training environment.

        Args:
            config: Either a path to configuration YAML file (str) or a config dictionary
            mode: 'training' or 'evaluation'
            render_mode: Rendering mode (if any)
        """
        super().__init__()

        self.mode = mode
        self.render_mode = render_mode

        # Load config if it's a path
        if isinstance(config, str):
            self.config_path = config
            if not os.path.exists(config):
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = config
            self.config_path = None

        # Create base hierarchical environment
        self.base_env = HierarchicalMultiDCEnv(config=self.config)

        self.num_datacenters = self.base_env.num_datacenters
        self.global_routing_batch_size = self.base_env.global_routing_batch_size

        # Define observation and action spaces
        # We'll expose separate spaces for global and local agents
        self.global_observation_space = self.base_env.global_observation_space
        self.local_observation_space = self.base_env.local_observation_space
        self.global_action_space = self.base_env.global_action_space
        self.local_action_space = self.base_env.local_action_space

        # For Gymnasium compatibility, set the primary spaces
        # (Some wrappers may expect single observation/action space)
        self.observation_space = spaces.Dict({
            "global": self.global_observation_space,
            "local": spaces.Dict({
                f"dc_{i}": self.local_observation_space
                for i in range(self.num_datacenters)
            })
        })

        self.action_space = spaces.Dict({
            "global": self.global_action_space,
            "local": spaces.Dict({
                f"dc_{i}": self.local_action_space
                for i in range(self.num_datacenters)
            })
        })

        # Episode tracking
        self.current_step = 0
        self.episode_reward = {
            "global": 0.0,
            "local": {i: 0.0 for i in range(self.num_datacenters)}
        }

        logger.info(
            f"JointTrainingEnv initialized with {self.num_datacenters} datacenters"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.

        Returns:
            observations: Hierarchical observations
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset base environment
        observations, info = self.base_env.reset(seed=seed, options=options)

        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = {
            "global": 0.0,
            "local": {i: 0.0 for i in range(self.num_datacenters)}
        }

        logger.debug("JointTrainingEnv reset complete")

        return observations, info

    def step(
        self,
        actions: Dict[str, Any]
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary with 'global' and 'local' actions
                {
                    'global': np.array([dc_id, dc_id, ...]),
                    'local': {dc_id: vm_action, ...}
                }

        Returns:
            observations: Hierarchical observations
            rewards: Rewards for global and local agents
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate actions structure
        if "global" not in actions or "local" not in actions:
            raise ValueError(
                "Actions must contain 'global' and 'local' keys. "
                f"Got: {actions.keys()}"
            )

        # Execute step in base environment
        observations, rewards, terminated, truncated, info = self.base_env.step(actions)

        # Update episode tracking
        self.current_step += 1
        self.episode_reward["global"] += rewards["global"]
        for dc_id, reward in rewards["local"].items():
            self.episode_reward["local"][dc_id] += reward

        # Add episode info
        info["episode_step"] = self.current_step
        info["episode_reward"] = self.episode_reward.copy()

        # Add per-step reward info for callbacks and Monitor
        info["global_reward"] = rewards["global"]
        if rewards["local"]:
            info["local_reward"] = np.mean(list(rewards["local"].values()))
            info["total_reward"] = rewards["global"] + info["local_reward"]
        else:
            info["local_reward"] = 0.0
            info["total_reward"] = rewards["global"]

        # Add energy/datacenter metrics from base env info if available
        if "global_energy_stats" in info:
            stats = info["global_energy_stats"]
            # Check if stats is a dict (might be string due to Java conversion)
            if isinstance(stats, dict):
                info["green_energy_ratio"] = stats.get("green_energy_ratio", 0.0)
                info["brown_energy_wh"] = stats.get("total_brown_energy_wh", 0.0)
                info["wasted_green_wh"] = stats.get("total_wasted_green_wh", 0.0)
            else:
                # Skip if stats is not a dict (Java conversion issue)
                info["green_energy_ratio"] = 0.0
                info["brown_energy_wh"] = 0.0
                info["wasted_green_wh"] = 0.0

        if terminated or truncated:
            # Log episode summary
            avg_local_reward = np.mean(list(self.episode_reward["local"].values())) if self.episode_reward["local"] else 0
            logger.info(
                f"Episode ended at step {self.current_step}: "
                f"Global reward={self.episode_reward['global']:.2f}, "
                f"Avg local reward={avg_local_reward:.2f}"
            )

        return observations, rewards, terminated, truncated, info

    def get_action_masks(self) -> Dict[str, Any]:
        """
        Get action masks for all agents.

        Returns:
            {
                'global': None,  # No masking for global agent
                'local': {dc_id: mask_array, ...}
            }
        """
        action_masks = {
            "global": None,
            "local": {}
        }

        # Get action masks for each datacenter's local agent
        for dc_id in range(self.num_datacenters):
            try:
                mask = self.base_env.get_local_action_masks(dc_id)
                action_masks["local"][dc_id] = mask
            except Exception as e:
                logger.warning(
                    f"Failed to get action mask for DC {dc_id}: {e}. "
                    f"Using all-valid mask."
                )
                # Fallback: allow all actions
                action_masks["local"][dc_id] = np.ones(
                    self.local_action_space.n,
                    dtype=bool
                )

        return action_masks

    def get_local_action_masks(self, dc_id: int) -> np.ndarray:
        """
        Get action mask for a specific datacenter's local agent.

        Args:
            dc_id: Datacenter ID

        Returns:
            Boolean array of valid actions for this datacenter
        """
        return self.base_env.get_local_action_masks(dc_id)

    def get_global_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current global observation.

        Returns:
            Global observation dictionary
        """
        if hasattr(self.base_env, "last_observations"):
            return self.base_env.last_observations["global"]
        else:
            logger.warning("No observations available. Call reset() first.")
            return {}

    def get_local_observations(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Get current local observations for all datacenters.

        Returns:
            {dc_id: local_observation_dict, ...}
        """
        if hasattr(self.base_env, "last_observations"):
            return self.base_env.last_observations["local"]
        else:
            logger.warning("No observations available. Call reset() first.")
            return {}

    def render(self):
        """Render environment (if supported)."""
        if self.render_mode == "human":
            # Print current state
            print(f"\n=== Step {self.current_step} ===")
            print(f"Global reward: {self.episode_reward['global']:.2f}")
            print(f"Local rewards: {self.episode_reward['local']}")

    def close(self):
        """Clean up environment resources."""
        if self.base_env is not None:
            self.base_env.close()
            logger.info("JointTrainingEnv closed")


class ParameterSharingWrapper(gym.Wrapper):
    """
    Wrapper to facilitate parameter sharing among Local Agents.

    This wrapper flattens the local observations into a batch,
    making it easier to feed them into a shared neural network.

    Usage:
        - During training, collect all local observations
        - Feed them as a batch to the shared local agent network
        - Distribute actions back to respective datacenters
    """

    def __init__(self, env: JointTrainingEnv):
        """
        Initialize parameter sharing wrapper.

        Args:
            env: JointTrainingEnv instance
        """
        super().__init__(env)
        self.env = env
        self.num_datacenters = env.num_datacenters
        self.global_routing_batch_size = env.global_routing_batch_size

    # --- Passthrough attributes for convenience ---
    @property
    def global_observation_space(self):
        return self.env.global_observation_space

    @property
    def global_action_space(self):
        return self.env.global_action_space

    @property
    def local_observation_space(self):
        return self.env.local_observation_space

    @property
    def local_action_space(self):
        return self.env.local_action_space

    def get_action_masks(self) -> Dict[str, Any]:
        """
        Get action masks from the underlying environment.
        
        Returns:
            Action masks dict: {'global': ..., 'local': {dc_id: mask, ...}}
        """
        return self.env.get_action_masks()

    def get_batched_local_observations(self) -> np.ndarray:
        """
        Get local observations as a batch for parameter sharing.

        Returns:
            Batched observations: shape (num_datacenters, obs_dim)
        """
        local_obs_dict = self.env.get_local_observations()

        # Convert dict observations to arrays
        obs_batch = []
        for dc_id in range(self.num_datacenters):
            if dc_id in local_obs_dict:
                obs = local_obs_dict[dc_id]
                # Flatten the observation dict into a single array
                obs_array = self._flatten_observation(obs)
                obs_batch.append(obs_array)
            else:
                logger.warning(
                    f"Missing observation for DC {dc_id}. Using zeros."
                )
                # Use zero observation as placeholder
                obs_batch.append(np.zeros(self._get_obs_dim()))

        return np.array(obs_batch)

    def get_batched_action_masks(self) -> np.ndarray:
        """
        Get action masks as a batch for parameter sharing.

        Returns:
            Batched masks: shape (num_datacenters, action_dim)
        """
        action_masks_dict = self.env.get_action_masks()

        masks_batch = []
        for dc_id in range(self.num_datacenters):
            if dc_id in action_masks_dict["local"]:
                mask = action_masks_dict["local"][dc_id]
                masks_batch.append(mask)
            else:
                # Fallback: allow all actions
                masks_batch.append(
                    np.ones(self.env.local_action_space.n, dtype=bool)
                )

        return np.array(masks_batch)

    def _flatten_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten a dictionary observation into a 1D array.

        Args:
            obs: Observation dictionary

        Returns:
            Flattened observation array
        """
        arrays = []

        # Ensure consistent ordering
        keys = sorted(obs.keys())

        for key in keys:
            value = obs[key]
            if isinstance(value, np.ndarray):
                arrays.append(value.flatten())
            elif isinstance(value, (int, float, np.integer, np.floating)):
                arrays.append(np.array([value]))
            else:
                logger.warning(
                    f"Unexpected observation type for key '{key}': {type(value)}"
                )

        return np.concatenate(arrays) if arrays else np.array([])

    def _get_obs_dim(self) -> int:
        """
        Calculate the dimensionality of flattened observation.

        Returns:
            Observation dimension
        """
        # Get a sample observation from the base environment's observation space
        sample_obs = self.env.local_observation_space.sample()
        flattened = self._flatten_observation(sample_obs)
        return len(flattened)
