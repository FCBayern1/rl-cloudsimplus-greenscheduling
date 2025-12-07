"""
PettingZoo ParallelEnv wrapper for Simplified Hierarchical Multi-Datacenter Environment.

This is the MARL version of HierarchicalMultiDCEnvSimple, which removes God's Eye
future prediction features for fair comparison experiments.

Removed Features (God's Eye):
    - dc_future_short_mean
    - dc_future_short_trend
    - dc_future_long_mean
    - dc_future_long_peak_timing

Usage:
    >>> config = {...}  # Your environment config
    >>> env = HierarchicalMultiDCParallelEnvSimple(config)
    >>> observations, infos = env.reset()
    >>> # observations will NOT contain future prediction features
"""

import logging
from typing import Dict, Any, Optional

from .hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv
from .hierarchical_multidc_env_simple import HierarchicalMultiDCEnvSimple

logger = logging.getLogger(__name__)


class HierarchicalMultiDCParallelEnvSimple(HierarchicalMultiDCParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for simplified multi-DC environment.

    This environment removes God's Eye future prediction features from the
    global observation space, providing a fairer comparison scenario where
    agents cannot see future energy availability.

    Inherits all functionality from HierarchicalMultiDCParallelEnv but uses
    HierarchicalMultiDCEnvSimple as the base environment.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "hierarchical_multidc_simple_v0",
        "is_parallelizable": True,
    }

    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        """
        Initialize simplified PettingZoo wrapper.

        Args:
            config: Configuration dictionary for HierarchicalMultiDCEnvSimple
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        # Don't call parent __init__ directly - we need to override base_env creation
        # Instead, replicate the initialization with our simplified base env

        from pettingzoo import ParallelEnv
        ParallelEnv.__init__(self)

        self.render_mode = render_mode

        # Use SIMPLIFIED base environment (no God's Eye features)
        logger.info("Creating base HierarchicalMultiDCEnvSimple (no God's Eye features)...")
        base_env = HierarchicalMultiDCEnvSimple(config=config)

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
            f"HierarchicalMultiDCParallelEnvSimple initialized with {len(self.agents)} agents"
        )
        logger.info("  God's Eye features DISABLED:")
        logger.info("    - dc_future_short_mean")
        logger.info("    - dc_future_short_trend")
        logger.info("    - dc_future_long_mean")
        logger.info("    - dc_future_long_peak_timing")


# Convenience function for creating the environment
def make_env_simple(config: Dict[str, Any], **kwargs) -> HierarchicalMultiDCParallelEnvSimple:
    """
    Factory function to create simplified PettingZoo environment.

    Args:
        config: Environment configuration dictionary
        **kwargs: Additional arguments passed to environment constructor

    Returns:
        HierarchicalMultiDCParallelEnvSimple instance
    """
    return HierarchicalMultiDCParallelEnvSimple(config=config, **kwargs)
