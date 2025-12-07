"""
Custom Callbacks for Hierarchical Multi-Datacenter MARL Training.

This module provides callbacks for monitoring green energy metrics,
performance metrics, and other training statistics.
"""

import os
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

logger = logging.getLogger(__name__)


class GreenEnergyMonitorCallback(BaseCallback):
    """
    Callback for monitoring green energy metrics during training.

    This callback tracks:
    - Green energy usage ratio
    - Brown energy consumption
    - Wasted green energy
    - Carbon emissions
    - Task completion rate
    - Load balance metrics
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 100,
        verbose: int = 0
    ):
        """
        Initialize green energy monitor callback.

        Args:
            log_dir: Directory to save detailed logs (CSV format)
            log_freq: Logging frequency (every N steps)
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.log_dir = Path(log_dir) if log_dir else None
        self.log_freq = log_freq

        # Metrics accumulation
        self.episode_green_energy = []
        self.episode_brown_energy = []
        self.episode_wasted_green = []
        self.episode_green_ratio = []
        self.episode_carbon_emissions = []
        self.episode_completion_rate = []
        self.episode_load_balance = []

        # CSV logging
        self.csv_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_file = self.log_dir / "green_energy_metrics.csv"
            self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        if self.csv_file:
            with open(self.csv_file, 'w') as f:
                f.write(
                    "timestep,episode,green_energy_ratio,brown_energy_wh,"
                    "wasted_green_wh,carbon_kg,completion_rate,load_balance\n"
                )

    def _on_step(self) -> bool:
        """
        Called at every step during training.

        Returns:
            True to continue training
        """
        # Extract green energy metrics from environment info
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            # Check if episode ended
            if info.get("episode"):
                self._log_episode_metrics(info)

        # Periodic logging
        if self.num_timesteps % self.log_freq == 0:
            self._log_current_metrics()

        return True

    def _log_episode_metrics(self, info: Dict[str, Any]):
        """
        Log metrics at the end of an episode.

        Args:
            info: Episode info dictionary
        """
        episode_info = info.get("episode", {})

        # Extract green energy metrics if available
        green_ratio = info.get("green_energy_ratio", 0.0)
        brown_energy = info.get("brown_energy_wh", 0.0)
        wasted_green = info.get("wasted_green_wh", 0.0)
        carbon_kg = info.get("carbon_emissions_kg", 0.0)
        completion_rate = info.get("completion_rate", 0.0)
        load_balance = info.get("load_balance", 0.0)

        # Accumulate metrics
        self.episode_green_ratio.append(green_ratio)
        self.episode_brown_energy.append(brown_energy)
        self.episode_wasted_green.append(wasted_green)
        self.episode_carbon_emissions.append(carbon_kg)
        self.episode_completion_rate.append(completion_rate)
        self.episode_load_balance.append(load_balance)

        # Write to CSV
        if self.csv_file:
            with open(self.csv_file, 'a') as f:
                f.write(
                    f"{self.num_timesteps},{len(self.episode_green_ratio)},"
                    f"{green_ratio:.4f},{brown_energy:.2f},{wasted_green:.2f},"
                    f"{carbon_kg:.4f},{completion_rate:.4f},{load_balance:.4f}\n"
                )

        # Log to TensorBoard
        self.logger.record("green_energy/ratio", green_ratio)
        self.logger.record("green_energy/brown_wh", brown_energy)
        self.logger.record("green_energy/wasted_wh", wasted_green)
        self.logger.record("green_energy/carbon_kg", carbon_kg)
        self.logger.record("performance/completion_rate", completion_rate)
        self.logger.record("performance/load_balance", load_balance)

        if self.verbose >= 1:
            logger.info(
                f"Episode {len(self.episode_green_ratio)}: "
                f"Green ratio={green_ratio:.2%}, "
                f"Carbon={carbon_kg:.2f}kg, "
                f"Completion={completion_rate:.2%}"
            )

    def _log_current_metrics(self):
        """Log averaged metrics over recent episodes."""

        if len(self.episode_green_ratio) == 0:
            return

        # Calculate averages over last 10 episodes
        window = 10
        recent_green_ratio = np.mean(self.episode_green_ratio[-window:])
        recent_carbon = np.mean(self.episode_carbon_emissions[-window:])
        recent_completion = np.mean(self.episode_completion_rate[-window:])

        self.logger.record("green_energy/recent_ratio_avg", recent_green_ratio)
        self.logger.record("green_energy/recent_carbon_avg", recent_carbon)
        self.logger.record("performance/recent_completion_avg", recent_completion)

    def _on_training_end(self) -> None:
        """Called at the end of training."""

        if len(self.episode_green_ratio) == 0:
            logger.warning("No episode metrics collected during training")
            return

        # Log final summary
        avg_green_ratio = np.mean(self.episode_green_ratio)
        avg_carbon = np.mean(self.episode_carbon_emissions)
        avg_completion = np.mean(self.episode_completion_rate)

        logger.info("=" * 60)
        logger.info("Training Summary - Green Energy Metrics")
        logger.info("=" * 60)
        logger.info(f"Average Green Energy Ratio: {avg_green_ratio:.2%}")
        logger.info(f"Average Carbon Emissions: {avg_carbon:.2f} kg")
        logger.info(f"Average Completion Rate: {avg_completion:.2%}")
        logger.info(f"Total Episodes: {len(self.episode_green_ratio)}")
        logger.info("=" * 60)


class MultiAgentMetricsCallback(BaseCallback):
    """
    Callback for monitoring multi-agent specific metrics.

    Tracks:
    - Individual agent rewards
    - Policy entropy for each agent type
    - Coordination metrics
    """

    def __init__(
        self,
        num_datacenters: int,
        log_freq: int = 100,
        verbose: int = 0
    ):
        """
        Initialize multi-agent metrics callback.

        Args:
            num_datacenters: Number of datacenters (local agents)
            log_freq: Logging frequency
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.num_datacenters = num_datacenters
        self.log_freq = log_freq

        # Metrics tracking
        self.global_rewards = []
        self.local_rewards = {i: [] for i in range(num_datacenters)}

    def _on_step(self) -> bool:
        """Called at every step."""

        # Extract rewards from info
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            if "episode_reward" in info:
                episode_reward = info["episode_reward"]

                # Log global reward
                if "global" in episode_reward:
                    self.global_rewards.append(episode_reward["global"])
                    self.logger.record("agent/global_reward", episode_reward["global"])

                # Log local rewards
                if "local" in episode_reward:
                    for dc_id, reward in episode_reward["local"].items():
                        self.local_rewards[dc_id].append(reward)
                        self.logger.record(f"agent/local_{dc_id}_reward", reward)

        # Periodic summary
        if self.num_timesteps % self.log_freq == 0:
            self._log_summary()

        return True

    def _log_summary(self):
        """Log summary statistics."""

        if len(self.global_rewards) > 0:
            avg_global = np.mean(self.global_rewards[-10:])
            self.logger.record("agent/recent_global_avg", avg_global)

        for dc_id, rewards in self.local_rewards.items():
            if len(rewards) > 0:
                avg_local = np.mean(rewards[-10:])
                self.logger.record(f"agent/recent_local_{dc_id}_avg", avg_local)


class ActionDistributionCallback(BaseCallback):
    """
    Callback for monitoring action distributions.

    Tracks:
    - Datacenter selection distribution (global agent)
    - VM selection distribution (local agents)
    - Action masking statistics
    """

    def __init__(
        self,
        num_datacenters: int,
        num_vms: int,
        log_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Initialize action distribution callback.

        Args:
            num_datacenters: Number of datacenters
            num_vms: Number of VMs per datacenter
            log_freq: Logging frequency
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.num_datacenters = num_datacenters
        self.num_vms = num_vms
        self.log_freq = log_freq

        # Action counters
        self.global_action_counts = np.zeros(num_datacenters)
        self.local_action_counts = np.zeros(num_vms + 1)  # +1 for NoAssign
        self.masked_action_counts = 0
        self.total_actions = 0

    def _on_step(self) -> bool:
        """Called at every step."""

        # Extract actions from locals
        if "actions" in self.locals:
            actions = self.locals["actions"]

            self.total_actions += 1

            # Track global actions
            if isinstance(actions, dict) and "global" in actions:
                global_action = actions["global"]
                if isinstance(global_action, np.ndarray):
                    for action in global_action.flatten():
                        if 0 <= action < self.num_datacenters:
                            self.global_action_counts[action] += 1

            # Track local actions
            if isinstance(actions, dict) and "local" in actions:
                for dc_id, local_action in actions["local"].items():
                    if 0 <= local_action <= self.num_vms:
                        self.local_action_counts[local_action] += 1

        # Periodic logging
        if self.num_timesteps % self.log_freq == 0:
            self._log_distributions()

        return True

    def _log_distributions(self):
        """Log action distributions."""

        if self.total_actions == 0:
            return

        # Global action distribution
        global_dist = self.global_action_counts / max(self.global_action_counts.sum(), 1)
        for dc_id, prob in enumerate(global_dist):
            self.logger.record(f"actions/global_dc_{dc_id}_prob", prob)

        # Local action distribution
        local_dist = self.local_action_counts / max(self.local_action_counts.sum(), 1)
        self.logger.record("actions/local_noassign_prob", local_dist[0])
        for vm_id in range(1, len(local_dist)):
            self.logger.record(f"actions/local_vm_{vm_id-1}_prob", local_dist[vm_id])

        # Entropy (measure of exploration)
        global_entropy = -np.sum(global_dist * np.log(global_dist + 1e-10))
        local_entropy = -np.sum(local_dist * np.log(local_dist + 1e-10))

        self.logger.record("actions/global_entropy", global_entropy)
        self.logger.record("actions/local_entropy", local_entropy)

        if self.verbose >= 1:
            logger.info(
                f"Action entropy - Global: {global_entropy:.3f}, "
                f"Local: {local_entropy:.3f}"
            )


def create_callbacks(
    num_datacenters: int,
    num_vms: int,
    log_dir: str,
    checkpoint_freq: int = 10000
) -> list:
    """
    Create a list of callbacks for joint training.

    Args:
        num_datacenters: Number of datacenters
        num_vms: Number of VMs per datacenter
        log_dir: Directory for logs
        checkpoint_freq: Checkpoint saving frequency

    Returns:
        List of callbacks
    """
    from stable_baselines3.common.callbacks import CheckpointCallback

    callbacks = [
        GreenEnergyMonitorCallback(
            log_dir=log_dir,
            log_freq=100,
            verbose=1
        ),
        MultiAgentMetricsCallback(
            num_datacenters=num_datacenters,
            log_freq=100,
            verbose=1
        ),
        ActionDistributionCallback(
            num_datacenters=num_datacenters,
            num_vms=num_vms,
            log_freq=1000,
            verbose=1
        ),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=log_dir,
            name_prefix="checkpoint",
            verbose=1
        )
    ]

    return callbacks
