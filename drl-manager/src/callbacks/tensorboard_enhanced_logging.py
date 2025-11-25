"""
Enhanced TensorBoard logging callback for PPO training metrics.

This callback logs detailed training metrics including:
- Policy loss
- Value loss
- Entropy loss
- KL divergence
- Clip fraction
- Learning rate
- And more PPO-specific metrics
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class EnhancedTensorBoardCallback(BaseCallback):
    """
    Enhanced callback for detailed TensorBoard logging.
    
    Logs all PPO training metrics including:
    - Policy network loss (policy_loss)
    - Value network loss (value_loss)
    - Entropy loss (entropy_loss)
    - KL divergence (approx_kl)
    - Clip fraction (clip_fraction)
    - Explained variance (explained_variance)
    - Learning rate
    
    Args:
        agent_name: Name of the agent ("global" or "local")
        verbose: Verbosity level
    """

    def __init__(
        self,
        agent_name: str = "agent",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.step_count = 0
        
        logger.info(f"EnhancedTensorBoardCallback initialized for {agent_name}")

    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Extracts and logs PPO training metrics from self.model.
        """
        self.step_count += 1
        
        # Only log after training updates (not every env step)
        # PPO updates happen every n_steps (e.g., 2048)
        if self.step_count % self.model.n_steps != 0:
            return True
        
        # Try to extract training metrics from the model
        # These are populated after PPO's train() call
        try:
            # Check if logger exists
            if self.logger is None:
                return True
            
            # Get the model's logger to extract metrics
            # SB3 stores training metrics in model.logger.name_to_value
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                metrics = self.model.logger.name_to_value
                
                # Log PPO-specific metrics with agent prefix
                ppo_metrics = {
                    "policy_loss": "train/policy_gradient_loss",
                    "value_loss": "train/value_loss",
                    "entropy_loss": "train/entropy_loss",
                    "approx_kl": "train/approx_kl",
                    "clip_fraction": "train/clip_fraction",
                    "explained_variance": "train/explained_variance",
                    "learning_rate": "train/learning_rate",
                    "n_updates": "train/n_updates",
                    "loss": "train/loss"  # Total loss
                }
                
                for metric_key, metric_name in ppo_metrics.items():
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        # Record with agent-specific prefix
                        self.logger.record(
                            f"{self.agent_name}/train/{metric_key}",
                            value
                        )
                        
                        if self.verbose > 1 and metric_key in ["policy_loss", "value_loss"]:
                            logger.debug(
                                f"[{self.agent_name}] {metric_key}: {value:.6f}"
                            )
            
        except Exception as e:
            logger.warning(f"Error logging PPO metrics for {self.agent_name}: {e}")
        
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout (after collecting n_steps).
        
        This is a good place to log rollout-specific statistics.
        """
        try:
            if self.logger is None:
                return
            
            # Log rollout statistics if available
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer
                
                # Calculate mean episode statistics
                if len(ep_info) > 0:
                    ep_rewards = [info['r'] for info in ep_info]
                    ep_lengths = [info['l'] for info in ep_info]
                    
                    self.logger.record(
                        f"{self.agent_name}/rollout/mean_ep_reward",
                        np.mean(ep_rewards)
                    )
                    self.logger.record(
                        f"{self.agent_name}/rollout/mean_ep_length",
                        np.mean(ep_lengths)
                    )
                    
                    if self.verbose > 0:
                        logger.info(
                            f"[{self.agent_name}] Rollout complete - "
                            f"Mean reward: {np.mean(ep_rewards):.2f}, "
                            f"Mean length: {np.mean(ep_lengths):.1f}"
                        )
        
        except Exception as e:
            logger.warning(f"Error logging rollout stats for {self.agent_name}: {e}")

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.verbose > 0:
            logger.info(f"[{self.agent_name}] Training started - Enhanced logging enabled")
            logger.info(f"  Logging to: {self.logger.dir if self.logger else 'N/A'}")


class SeparateAgentMetricsCallback(BaseCallback):
    """
    Callback to separately track and log metrics for Global vs Local agents.
    
    Works with hierarchical training to ensure metrics don't get mixed.
    
    Args:
        agent_type: "global" or "local"
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        agent_type: str,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.agent_type = agent_type
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        logger.info(f"SeparateAgentMetricsCallback initialized for {agent_type} agent")
    
    def _on_step(self) -> bool:
        """Track episode statistics separately for each agent."""
        
        # Accumulate reward for current episode
        if "rewards" in self.locals and len(self.locals["rewards"]) > 0:
            self.current_episode_reward += self.locals["rewards"][0]
            self.current_episode_length += 1
        
        # Check if episode ended
        if "dones" in self.locals and len(self.locals["dones"]) > 0:
            if self.locals["dones"][0]:
                # Episode ended - record stats
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                # Log to TensorBoard
                if self.logger is not None:
                    self.logger.record(
                        f"{self.agent_type}_agent/episode/reward",
                        self.current_episode_reward
                    )
                    self.logger.record(
                        f"{self.agent_type}_agent/episode/length",
                        self.current_episode_length
                    )
                    
                    # Log rolling mean (last 100 episodes)
                    if len(self.episode_rewards) >= 10:
                        window = min(100, len(self.episode_rewards))
                        mean_reward = np.mean(self.episode_rewards[-window:])
                        mean_length = np.mean(self.episode_lengths[-window:])
                        
                        self.logger.record(
                            f"{self.agent_type}_agent/episode/mean_reward_100",
                            mean_reward
                        )
                        self.logger.record(
                            f"{self.agent_type}_agent/episode/mean_length_100",
                            mean_length
                        )
                
                if self.verbose > 0:
                    logger.info(
                        f"[{self.agent_type.upper()}] Episode complete - "
                        f"Reward: {self.current_episode_reward:.2f}, "
                        f"Length: {self.current_episode_length}"
                    )
                
                # Reset for next episode
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
        
        return True





