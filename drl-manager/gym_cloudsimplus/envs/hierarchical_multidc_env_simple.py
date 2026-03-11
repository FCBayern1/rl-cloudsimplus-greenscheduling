"""
tests/verify_action_mask_logic.py, tests/test_reset_gymnasium_compliance.py, tests/test_green_energy.py -- 测试

hierarchical_multidc_pettingzoo_simple.py 直接使用

Simplified Hierarchical Multi-Datacenter Environment WITHOUT God's Eye Features.

This environment is identical to HierarchicalMultiDCEnv but excludes the
"God's Eye" future prediction features from the global observation space.

Removed Features (God's Eye):
    - dc_future_short_mean: Short-term (30min) average green energy
    - dc_future_short_trend: Short-term trend direction
    - dc_future_long_mean: Long-term (24h) average green energy
    - dc_future_long_peak_timing: Long-term peak timing

This version is used for fair comparison experiments where the agent
cannot see future energy availability.

Usage:
    >>> config = {...}  # Your environment config
    >>> env = HierarchicalMultiDCEnvSimple(config=config)
    >>> obs, info = env.reset()
    >>> # obs["global"] will NOT contain future prediction features
"""

import logging
from typing import Dict, Any

import numpy as np
from gymnasium import spaces

from .hierarchical_multidc_env import HierarchicalMultiDCEnv

logger = logging.getLogger(__name__)

# Features to exclude from global observation (God's Eye features)
GODS_EYE_FEATURES = [
    "dc_future_short_mean",
    "dc_future_short_trend",
    "dc_future_long_mean",
    "dc_future_long_peak_timing",
]


class HierarchicalMultiDCEnvSimple(HierarchicalMultiDCEnv):
    """
    Simplified Hierarchical Multi-Datacenter Environment.

    This environment removes God's Eye future prediction features from the
    global observation space, providing a fairer comparison scenario where
    agents cannot see future energy availability.

    Inherits all functionality from HierarchicalMultiDCEnv but overrides:
    - global_observation_space: Excludes God's Eye features
    - _convert_global_observation: Excludes God's Eye features from observations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simplified environment.

        Args:
            config: Configuration dictionary (same as HierarchicalMultiDCEnv)
        """
        # Call parent constructor first
        super().__init__(config=config)

        # Override global observation space to exclude God's Eye features
        self._override_global_observation_space()

        logger.info("HierarchicalMultiDCEnvSimple initialized")
        logger.info("  God's Eye features DISABLED:")
        for feature in GODS_EYE_FEATURES:
            logger.info(f"    - {feature}")

    def _override_global_observation_space(self):
        """
        Override the global observation space to exclude God's Eye features.
        """
        # Get the parent's global observation space (Dict space)
        parent_space = self.global_observation_space

        # Create new space dict without God's Eye features
        new_spaces = {}
        for key, space in parent_space.spaces.items():
            if key not in GODS_EYE_FEATURES:
                new_spaces[key] = space

        # Replace global observation space
        self.global_observation_space = spaces.Dict(new_spaces)

        logger.info(
            f"Global observation space updated: removed {len(GODS_EYE_FEATURES)} God's Eye features"
        )
        logger.info(
            f"New global observation space has {len(new_spaces)} features"
        )

    def _convert_global_observation(self, global_obs_java) -> Dict[str, Any]:
        """
        Convert Java GlobalObservationState to Python dict, excluding God's Eye features.

        This override removes the future prediction features from the observation.
        """
        # Get full observation from parent
        full_obs = super()._convert_global_observation(global_obs_java)

        # Remove God's Eye features
        simple_obs = {
            key: value
            for key, value in full_obs.items()
            if key not in GODS_EYE_FEATURES
        }

        return simple_obs
