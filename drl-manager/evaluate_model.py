#!/usr/bin/env python3
"""
Evaluate trained RLlib model for Multi-Datacenter Load Balancing.

Supports both the GTrXL RLModule pipeline and legacy RLlib pipeline.
Auto-detects parameter sharing (shared_local_policy vs local_policy_{dc_id}).

Usage:
    # Evaluate on simulation environment
    python evaluate_model.py --checkpoint /path/to/checkpoint --experiment experiment_multi_5dc_carbon --episodes 10

    # Test inference only (no Java required)
    python evaluate_model.py --checkpoint /path/to/checkpoint --experiment experiment_multi_5dc_carbon --mode test
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import ray
from ray.rllib.algorithms.algorithm import Algorithm

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _detect_parameter_sharing(algorithm) -> bool:
    """Detect if the checkpoint was trained with shared_local_policy."""
    try:
        policy_ids = set(algorithm.config.policies or [])
    except Exception:
        policy_ids = set()

    if "shared_local_policy" in policy_ids:
        return True
    if any(pid.startswith("local_policy_") for pid in policy_ids):
        return False

    # Fallback: try computing with shared_local_policy
    return True


def _local_policy_id(dc_id: int, use_shared: bool) -> str:
    return "shared_local_policy" if use_shared else f"local_policy_{dc_id}"


class MultiDCScheduler:
    """
    Trained Multi-Datacenter Scheduler for inference.
    Wraps a trained RLlib checkpoint and provides scheduling decisions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "../config.yml",
        experiment_id: str = "experiment_multi_5dc_carbon"
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.experiment_id = experiment_id

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False)

        all_config = load_config(config_path)
        if experiment_id not in all_config:
            raise ValueError(f"Experiment '{experiment_id}' not found in config")

        self.exp_config = all_config[experiment_id]
        self.num_datacenters = len(self.exp_config.get('datacenters', []))

        logger.info("Restoring model from %s ...", checkpoint_path)
        self.algorithm = Algorithm.from_checkpoint(checkpoint_path)
        logger.info("Model loaded successfully! Datacenters: %d", self.num_datacenters)

        self.use_shared = _detect_parameter_sharing(self.algorithm)
        logger.info(
            "Policy mapping: %s",
            "shared_local_policy" if self.use_shared else "local_policy_{dc_id}"
        )

    def get_global_action(self, global_obs: Dict[str, Any]) -> np.ndarray:
        action = self.algorithm.compute_single_action(
            observation=global_obs,
            policy_id="global_policy",
            explore=False
        )
        return action

    def get_local_action(self, dc_id: int, local_obs: Dict[str, Any]) -> int:
        policy_id = _local_policy_id(dc_id, self.use_shared)
        action = self.algorithm.compute_single_action(
            observation=local_obs,
            policy_id=policy_id,
            explore=False
        )
        return int(action)

    def get_all_actions(
        self,
        observations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        actions = {}
        if "global_agent" in observations:
            actions["global_agent"] = self.get_global_action(observations["global_agent"])
        for dc_id in range(self.num_datacenters):
            agent_name = f"local_agent_{dc_id}"
            if agent_name in observations:
                actions[agent_name] = self.get_local_action(dc_id, observations[agent_name])
        return actions

    def close(self):
        if hasattr(self, 'algorithm'):
            self.algorithm.stop()


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    experiment_id: str,
    num_episodes: int = 10,
) -> Dict[str, Any]:
    """Evaluate trained model on simulation environment (requires Java Gateway)."""
    logger.info("=" * 70)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 70)

    scheduler = MultiDCScheduler(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        experiment_id=experiment_id
    )

    logger.info("Creating evaluation environment...")
    env = HierarchicalMultiDCParallelEnv(scheduler.exp_config)

    episode_rewards = []
    episode_lengths = []
    episode_metrics = []

    for ep in range(num_episodes):
        logger.info("\n--- Episode %d/%d ---", ep + 1, num_episodes)

        observations, infos = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            actions = scheduler.get_all_actions(observations)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            episode_length += 1
            done = any(terminations.values()) or any(truncations.values())

        final_info = infos.get("global_agent", {})
        global_energy_stats = final_info.get("global_energy_stats", {})

        metrics = {
            "episode": ep + 1,
            "reward": episode_reward,
            "length": episode_length,
            "green_ratio": global_energy_stats.get("green_energy_ratio", 0),
            "carbon_kg": global_energy_stats.get("total_carbon_emission_kg", 0),
            "brown_used_wh": global_energy_stats.get("total_brown_energy_wh", 0),
            "completion_rate_mi": global_energy_stats.get("completion_rate_mi", 0),
        }
        episode_metrics.append(metrics)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        logger.info("  Reward: %.2f", episode_reward)
        logger.info("  Length: %d", episode_length)
        logger.info("  Green Ratio: %.2f%%", metrics['green_ratio'] * 100)
        logger.info("  Carbon: %.6f kg", metrics['carbon_kg'])

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Mean Reward: %.2f +/- %.2f", np.mean(episode_rewards), np.std(episode_rewards))
    logger.info("  Mean Length: %.1f", np.mean(episode_lengths))
    logger.info("  Mean Green Ratio: %.2f%%", np.mean([m['green_ratio'] for m in episode_metrics]) * 100)
    logger.info("  Mean Carbon: %.6f kg", np.mean([m['carbon_kg'] for m in episode_metrics]))

    env.close()
    scheduler.close()

    return {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_carbon_kg": float(np.mean([m['carbon_kg'] for m in episode_metrics])),
        "episode_metrics": episode_metrics
    }


def find_latest_checkpoint(experiment_dir: str) -> Optional[str]:
    """Find the latest checkpoint in an experiment directory."""
    experiment_path = Path(experiment_dir)
    checkpoint_dirs = []
    for path in experiment_path.rglob("checkpoint_*"):
        if path.is_dir():
            checkpoint_dirs.append(path)
    if not checkpoint_dirs:
        return None
    checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoint_dirs[0])


def test_model_inference(checkpoint_path: str, config_path: str, experiment_id: str):
    """
    Test model inference without Java environment.
    Creates random observations to verify checkpoint loading and action computation.
    """
    logger.info("=" * 70)
    logger.info("Testing Model Inference (No Java Required)")
    logger.info("=" * 70)

    all_config = load_config(config_path)
    exp_config = all_config[experiment_id]
    num_dcs = len(exp_config.get('datacenters', []))
    batch_size = exp_config.get('global_routing_batch_size', 5)

    logger.info("  Datacenters: %d", num_dcs)
    logger.info("  Batch size: %d", batch_size)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    algorithm = Algorithm.from_checkpoint(checkpoint_path)
    logger.info("Checkpoint loaded successfully!")

    use_shared = _detect_parameter_sharing(algorithm)
    logger.info(
        "Detected policy mapping: %s",
        "shared_local_policy" if use_shared else "local_policy_{dc_id}"
    )

    # Test global policy
    logger.info("\n--- Testing Global Policy ---")
    mock_global_obs = {
        "observation": {
            "dc_green_power": np.random.rand(num_dcs).astype(np.float32),
            "dc_power_consumption": np.random.rand(num_dcs).astype(np.float32) * 1000,
            "dc_green_ratio": np.random.rand(num_dcs).astype(np.float32),
            "dc_wasted_green": np.random.rand(num_dcs).astype(np.float32) * 100,
            "dc_future_short_mean": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_short_trend": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_long_mean": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_long_peak": np.random.rand(num_dcs).astype(np.float32),
            "dc_queue_sizes": np.random.randint(0, 100, num_dcs).astype(np.int32),
            "dc_utilizations": np.random.rand(num_dcs).astype(np.float32),
            "dc_available_pes": np.random.randint(0, 100, num_dcs).astype(np.int32),
            "dc_ram_utilizations": np.random.rand(num_dcs).astype(np.float32),
            "upcoming_cloudlets": np.array(batch_size, dtype=np.int32),
            "batch_cloudlet_pes": np.random.randint(1, 8, batch_size).astype(np.int32),
            "batch_cloudlet_mi": np.random.randint(1000, 10000, batch_size).astype(np.int64),
            "upcoming_pes_distribution": np.random.rand(8).astype(np.float32),
            "load_imbalance": np.array(0.1, dtype=np.float32),
            "recent_completed": np.array(10, dtype=np.int32),
            "num_datacenters": np.array(num_dcs, dtype=np.int32),
            "current_time": np.array(100.0, dtype=np.float32),
        },
        "action_mask": np.ones((num_dcs + 1) * batch_size, dtype=np.float32)
    }

    try:
        global_action = algorithm.compute_single_action(
            observation=mock_global_obs,
            policy_id="global_policy",
            explore=False
        )
        logger.info("  Global action: %s  shape: %s", global_action, np.array(global_action).shape)
        logger.info("  Global policy inference OK")
    except Exception as e:
        logger.error("  Global policy failed: %s", e)

    # Test local policies
    logger.info("\n--- Testing Local Policies ---")
    for dc_id in range(num_dcs):
        dc_config = exp_config.get('datacenters', [])[dc_id]
        vm_count = (
            dc_config.get('initial_s_vm_count', 10) +
            dc_config.get('initial_m_vm_count', 10) +
            dc_config.get('initial_l_vm_count', 10)
        )

        mock_local_obs = {
            "observation": {
                "host_loads": np.random.rand(20).astype(np.float32),
                "host_ram_usage": np.random.rand(20).astype(np.float32),
                "vm_loads": np.random.rand(vm_count).astype(np.float32),
                "vm_types": np.random.randint(0, 4, vm_count).astype(np.int32),
                "vm_available_pes": np.random.randint(0, 8, vm_count).astype(np.int32),
                "waiting_cloudlets": np.array(5, dtype=np.int32),
                "next_cloudlet_pes": np.array(2, dtype=np.int32),
            },
            "action_mask": np.ones(vm_count + 1, dtype=np.float32)
        }

        policy_id = _local_policy_id(dc_id, use_shared)
        try:
            local_action = algorithm.compute_single_action(
                observation=mock_local_obs,
                policy_id=policy_id,
                explore=False
            )
            logger.info("  DC %d (%s): action=%s (VM count=%d)", dc_id, policy_id, local_action, vm_count)
        except Exception as e:
            logger.error("  DC %d (%s): Failed - %s", dc_id, policy_id, e)

    logger.info("\n" + "=" * 70)
    logger.info("Model inference test complete!")
    logger.info("=" * 70)

    algorithm.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Multi-DC scheduling model"
    )
    parser.add_argument(
        '--checkpoint', '-c', type=str,
        help='Path to checkpoint directory (or experiment directory to find latest)'
    )
    parser.add_argument(
        '--config', type=str, default='../config.yml',
        help='Path to config.yml'
    )
    parser.add_argument(
        '--experiment', '-e', type=str, default='experiment_multi_5dc_carbon',
        help='Experiment ID in config'
    )
    parser.add_argument(
        '--episodes', '-n', type=int, default=10,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--mode', '-m', type=str, choices=['evaluate', 'test'], default='evaluate',
        help='Mode: "evaluate" (full eval with Java) or "test" (inference test only)'
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_dir = f"../logs/{args.experiment}"
        logger.info("No checkpoint specified, searching in %s ...", default_dir)
        checkpoint_path = find_latest_checkpoint(default_dir)
        if checkpoint_path is None:
            logger.error("No checkpoints found in %s", default_dir)
            sys.exit(1)

    if Path(checkpoint_path).is_dir() and not checkpoint_path.endswith("checkpoint"):
        found = find_latest_checkpoint(checkpoint_path)
        if found:
            checkpoint_path = found

    logger.info("Using checkpoint: %s", checkpoint_path)

    if args.mode == 'test':
        test_model_inference(
            checkpoint_path=checkpoint_path,
            config_path=args.config,
            experiment_id=args.experiment
        )
    else:
        logger.info("\nFull evaluation requires Java Gateway running!")
        logger.info("Start: cd cloudsimplus-gateway && ./gradlew run -PappMainClass=exe.edu.cspg.MainMultiDC\n")
        evaluate_model(
            checkpoint_path=checkpoint_path,
            config_path=args.config,
            experiment_id=args.experiment,
            num_episodes=args.episodes,
        )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
