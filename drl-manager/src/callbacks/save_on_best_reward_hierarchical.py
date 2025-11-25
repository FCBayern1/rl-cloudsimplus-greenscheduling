"""
Callback for saving hierarchical multi-agent models based on training reward.

This callback tracks both global and local agent performance, and saves
the best models when combined reward improves.
"""

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results

logger = logging.getLogger(__name__)


class SaveOnBestRewardHierarchicalCallback(BaseCallback):
    """
    Callback for saving hierarchical multi-agent models based on training reward.

    Tracks global and local agent rewards separately and saves best models
    when the combined reward improves.

    Args:
        log_dir: Directory for saving models and logs
        global_model: Global agent model (PPO)
        local_model: Local agent model (MaskablePPO)
        save_freq: Frequency (in timesteps) to check for new best model
        verbose: Verbosity level
    """

    def __init__(
        self,
        log_dir: str,
        global_model: Any,
        local_model: Any,
        save_freq: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.global_model = global_model
        self.local_model = local_model
        self.save_freq = save_freq

        # Best reward tracking
        self.best_mean_reward = -np.inf
        self.best_global_reward = -np.inf
        self.best_local_reward = -np.inf

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_global_rewards = []
        self.episode_local_rewards = []

        # Current episode data
        self._reset_episode_data()

        # Model save paths
        self.best_global_model_path = self.log_dir / "best_global_model"
        self.best_local_model_path = self.log_dir / "best_local_model"

        # CSV paths
        self.progress_csv = self.log_dir / "training_progress.csv"
        self.best_episode_csv = self.log_dir / "best_episode_details.csv"

        logger.info(f"SaveOnBestRewardHierarchicalCallback initialized")
        logger.info(f"  Log dir: {self.log_dir}")
        logger.info(f"  Save frequency: {save_freq} timesteps")

    def _reset_episode_data(self):
        """Reset data collectors for new episode."""
        self.current_episode_rewards = []
        self.current_episode_global_rewards = []
        self.current_episode_local_rewards = []
        self.current_episode_timesteps = []
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called after each environment step."""

        # Show progress dot every 50 steps
        if self.num_timesteps % 50 == 0:
            print(".", end="", flush=True)

        # Show detailed progress every 100 steps
        if self.num_timesteps > 0 and self.num_timesteps % 100 == 0:
            print()  # Newline after dots
            logger.info(f"[Progress] Timesteps: {self.num_timesteps:5d} | Episodes: {self.episode_count:3d}")

        # Extract info from environment
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]

            # Collect step data
            # FIX: Use total_reward from info instead of current agent's reward
            # During alternating training, self.locals["rewards"] returns:
            #   - global_reward when training Global Agent
            #   - local_reward when training Local Agent
            # This causes episode_reward to oscillate between +201 and -1300
            # The correct total_reward = global_reward + local_reward is in info
            total_reward = info.get("total_reward", 0.0)
            global_reward = info.get("global_reward", 0.0)
            local_reward = info.get("local_reward", 0.0)

            self.current_episode_rewards.append(total_reward)
            self.current_episode_global_rewards.append(global_reward)
            self.current_episode_local_rewards.append(local_reward)
            self.current_episode_timesteps.append(self.num_timesteps)
            self.current_episode_length += 1

            # Check if episode ended
            if self.locals.get("dones", [False])[0]:
                self._on_episode_end()

        return True

    def _on_episode_end(self):
        """Called when an episode ends."""
        self.episode_count += 1

        # Calculate episode statistics
        episode_reward = np.sum(self.current_episode_rewards)
        episode_global_reward = np.mean(self.current_episode_global_rewards) if self.current_episode_global_rewards else 0
        episode_local_reward = np.mean(self.current_episode_local_rewards) if self.current_episode_local_rewards else 0

        # Store episode rewards
        self.episode_rewards.append(episode_reward)
        self.episode_global_rewards.append(episode_global_reward)
        self.episode_local_rewards.append(episode_local_reward)

        # Calculate rolling mean (last 100 episodes)
        window_size = min(100, len(self.episode_rewards))
        mean_reward = np.mean(self.episode_rewards[-window_size:])
        mean_global_reward = np.mean(self.episode_global_rewards[-window_size:])
        mean_local_reward = np.mean(self.episode_local_rewards[-window_size:])

        # Log episode statistics
        if self.verbose > 0:
            logger.info("=" * 60)
            logger.info(f"Episode {self.episode_count} completed (Timestep: {self.num_timesteps}, Length: {self.current_episode_length} steps)")
            logger.info(f"  Cumulative Global Reward: {episode_reward:.3f}")
            logger.info(f"  Mean Global Reward/step: {episode_global_reward:.3f}")
            logger.info(f"  Mean Local Reward/step: {episode_local_reward:.3f}")
            logger.info(f"  Mean Cumulative Reward (last {window_size} eps): {mean_reward:.3f}")
            logger.info(f"  Best Mean Cumulative Reward: {self.best_mean_reward:.3f}")
            logger.info("=" * 60)

        # Record to TensorBoard
        if self.logger:
            self.logger.record("episode/reward", episode_reward)
            self.logger.record("episode/global_reward", episode_global_reward)
            self.logger.record("episode/local_reward", episode_local_reward)
            self.logger.record("episode/mean_reward", mean_reward)
            self.logger.record("episode/mean_global_reward", mean_global_reward)
            self.logger.record("episode/mean_local_reward", mean_local_reward)
            self.logger.record("episode/length", self.current_episode_length)
            self.logger.record("episode/number", self.episode_count)

        # Save progress to CSV
        self._save_progress_csv(
            episode_reward, episode_global_reward, episode_local_reward,
            mean_reward, mean_global_reward, mean_local_reward
        )

        # Check if this is a new best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.best_global_reward = mean_global_reward
            self.best_local_reward = mean_local_reward

            if self.verbose > 0:
                logger.info("=" * 60)
                logger.info("*** NEW BEST MEAN REWARD! Saving models...")
                logger.info("=" * 60)

            self._save_best_models()
            self._save_best_episode_details()

        # Reset episode data
        self._reset_episode_data()

    def _save_progress_csv(
        self,
        episode_reward: float,
        episode_global_reward: float,
        episode_local_reward: float,
        mean_reward: float,
        mean_global_reward: float,
        mean_local_reward: float
    ):
        """Save training progress to CSV."""
        data = {
            "timestep": self.num_timesteps,
            "episode": self.episode_count,
            "episode_reward": episode_reward,
            "episode_global_reward": episode_global_reward,
            "episode_local_reward": episode_local_reward,
            "mean_reward": mean_reward,
            "mean_global_reward": mean_global_reward,
            "mean_local_reward": mean_local_reward,
            "best_mean_reward": self.best_mean_reward
        }

        df = pd.DataFrame([data])

        # Append to CSV (create if doesn't exist)
        if self.progress_csv.exists():
            df.to_csv(self.progress_csv, mode='a', header=False, index=False)
        else:
            df.to_csv(self.progress_csv, mode='w', header=True, index=False)

    def _save_best_models(self):
        """Save both global and local models as best models."""
        try:
            # Save global model
            self.global_model.save(str(self.best_global_model_path))
            logger.info(f"  [SAVED] Best global model -> {self.best_global_model_path}.zip")

            # Save local model
            self.local_model.save(str(self.best_local_model_path))
            logger.info(f"  [SAVED] Best local model -> {self.best_local_model_path}.zip")

        except Exception as e:
            logger.error(f"Error saving best models: {e}", exc_info=True)

    def _save_best_episode_details(self):
        """Save details of the best episode to CSV."""
        try:
            details = {
                "timestep": self.current_episode_timesteps,
                "reward": self.current_episode_rewards,
                "global_reward": self.current_episode_global_rewards,
                "local_reward": self.current_episode_local_rewards
            }

            df = pd.DataFrame(details)
            df.to_csv(self.best_episode_csv, index=False)

            if self.verbose > 0:
                logger.info(f"  [SAVED] Best episode details -> {self.best_episode_csv}")

        except Exception as e:
            logger.error(f"Error saving best episode details: {e}", exc_info=True)

    def _on_training_end(self):
        """Called when training ends."""
        logger.info("Training completed!")
        logger.info(f"  Total episodes: {self.episode_count}")
        logger.info(f"  Best mean reward: {self.best_mean_reward:.3f}")
        logger.info(f"  Best global reward: {self.best_global_reward:.3f}")
        logger.info(f"  Best local reward: {self.best_local_reward:.3f}")
