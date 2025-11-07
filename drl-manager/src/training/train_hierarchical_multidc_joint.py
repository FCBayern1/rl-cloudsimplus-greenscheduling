"""
Joint Training Script for Hierarchical Multi-Datacenter MARL.

This script trains both the Global Agent and Local Agents simultaneously
using Stable-Baselines3 with parameter sharing for Local Agents.

Features:
- Global Agent: PPO for datacenter routing
- Local Agents: MaskablePPO with parameter sharing for VM scheduling
- Green energy optimization reward
- Joint training in same episode
- Checkpoint saving and loading
- TensorBoard logging
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

# Add drl-manager root to path (to import gym_cloudsimplus and src packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gym_cloudsimplus.envs.joint_training_env import JointTrainingEnv, ParameterSharingWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JointTrainingManager:
    """
    Manages joint training of Global and Local agents.

    This class coordinates the training loop, handles model creation,
    and manages checkpoints and logging.
    """

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        global_model_config: Dict[str, Any],
        local_model_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ):
        """
        Initialize joint training manager.

        Args:
            config_path: Path to environment configuration
            output_dir: Directory for outputs (models, logs)
            global_model_config: Configuration for global agent model
            local_model_config: Configuration for local agent model
            training_config: Training hyperparameters
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_model_config = global_model_config
        self.local_model_config = local_model_config
        self.training_config = training_config

        # Create environment
        logger.info(f"Creating joint training environment from {config_path}")
        self.env = self._create_environment()

        # Create models
        self.global_model = None
        self.local_model = None
        self._create_models()

        # Training state
        self.total_timesteps = training_config.get("total_timesteps", 100000)
        self.current_timestep = 0

        logger.info("JointTrainingManager initialized successfully")

    def _create_environment(self) -> JointTrainingEnv:
        """
        Create and wrap the training environment.

        Returns:
            Wrapped joint training environment
        """
        env = JointTrainingEnv(
            config_path=self.config_path,
            mode="training"
        )

        # Wrap with parameter sharing for easier batch processing
        env = ParameterSharingWrapper(env)

        logger.info(
            f"Environment created with {env.num_datacenters} datacenters"
        )

        return env

    def _create_models(self):
        """Create Global and Local agent models."""

        # === Global Agent Model (PPO) ===
        logger.info("Creating Global Agent model (PPO)...")

        global_policy = self.global_model_config.get("policy", "MlpPolicy")
        global_lr = self.global_model_config.get("learning_rate", 3e-4)
        global_gamma = self.global_model_config.get("gamma", 0.99)
        global_n_steps = self.global_model_config.get("n_steps", 2048)
        global_batch_size = self.global_model_config.get("batch_size", 64)

        # Create a simple wrapper that exposes only global observation/action
        class GlobalAgentEnv:
            """Wrapper to expose only global agent's view."""

            def __init__(self, base_env):
                self.base_env = base_env
                self.observation_space = base_env.global_observation_space
                self.action_space = base_env.global_action_space

            def reset(self, **kwargs):
                obs, info = self.base_env.reset(**kwargs)
                return obs["global"], info

            def step(self, action):
                # Need local actions - use random for now during global-only training
                # In joint training, we'll handle this differently
                local_actions = {}
                for dc_id in range(self.base_env.num_datacenters):
                    local_actions[dc_id] = self.base_env.local_action_space.sample()

                obs, rewards, terminated, truncated, info = self.base_env.step({
                    "global": action,
                    "local": local_actions
                })

                return obs["global"], rewards["global"], terminated, truncated, info

        global_env = GlobalAgentEnv(self.env)

        self.global_model = PPO(
            policy=global_policy,
            env=global_env,
            learning_rate=global_lr,
            gamma=global_gamma,
            n_steps=global_n_steps,
            batch_size=global_batch_size,
            verbose=1,
            tensorboard_log=str(self.output_dir / "tensorboard" / "global"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info("Global Agent model created")

        # === Local Agent Model (MaskablePPO with Parameter Sharing) ===
        logger.info("Creating Local Agent model (MaskablePPO)...")

        local_policy = self.local_model_config.get("policy", "MlpPolicy")
        local_lr = self.local_model_config.get("learning_rate", 3e-4)
        local_gamma = self.local_model_config.get("gamma", 0.99)
        local_n_steps = self.local_model_config.get("n_steps", 2048)
        local_batch_size = self.local_model_config.get("batch_size", 64)

        # Create a wrapper that exposes only local observation/action
        class LocalAgentEnv:
            """Wrapper to expose only local agent's view (for one DC)."""

            def __init__(self, base_env, dc_id=0):
                self.base_env = base_env
                self.dc_id = dc_id
                self.observation_space = base_env.local_observation_space
                self.action_space = base_env.local_action_space

            def reset(self, **kwargs):
                obs, info = self.base_env.reset(**kwargs)
                # Return observation for first DC
                local_obs_dict = obs.get("local", {})
                dc_obs = local_obs_dict.get(self.dc_id, {})
                return dc_obs, info

            def step(self, action):
                # Need global actions - use random for now
                global_action = self.base_env.global_action_space.sample()

                # Build local actions (only for this DC, others random)
                local_actions = {}
                for dc_id in range(self.base_env.num_datacenters):
                    if dc_id == self.dc_id:
                        local_actions[dc_id] = action
                    else:
                        local_actions[dc_id] = self.base_env.local_action_space.sample()

                obs, rewards, terminated, truncated, info = self.base_env.step({
                    "global": global_action,
                    "local": local_actions
                })

                # Return local observation and reward for this DC
                local_obs_dict = obs.get("local", {})
                dc_obs = local_obs_dict.get(self.dc_id, {})
                dc_reward = rewards.get("local", {}).get(self.dc_id, 0.0)

                return dc_obs, dc_reward, terminated, truncated, info

            def action_masks(self):
                """Return action mask for this DC."""
                masks = self.base_env.get_action_masks()
                return masks["local"].get(self.dc_id, None)

        local_env = LocalAgentEnv(self.env, dc_id=0)

        self.local_model = MaskablePPO(
            policy=local_policy,
            env=local_env,
            learning_rate=local_lr,
            gamma=local_gamma,
            n_steps=local_n_steps,
            batch_size=local_batch_size,
            verbose=1,
            tensorboard_log=str(self.output_dir / "tensorboard" / "local"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info("Local Agent model created")

    def train(self):
        """
        Run joint training loop.

        This method alternates between training global and local agents,
        or trains them simultaneously depending on configuration.
        """
        logger.info("=" * 60)
        logger.info("Starting Joint Training")
        logger.info("=" * 60)

        training_strategy = self.training_config.get("strategy", "alternating")

        if training_strategy == "alternating":
            self._train_alternating()
        elif training_strategy == "simultaneous":
            self._train_simultaneous()
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")

        logger.info("Training completed!")

    def _train_alternating(self):
        """
        Alternating training: Train global agent, then local agents.
        """
        global_steps = self.training_config.get("global_steps_per_cycle", 10000)
        local_steps = self.training_config.get("local_steps_per_cycle", 10000)
        num_cycles = self.training_config.get("num_cycles", 10)

        logger.info(
            f"Alternating training: {num_cycles} cycles of "
            f"{global_steps} global + {local_steps} local steps"
        )

        for cycle in range(num_cycles):
            logger.info(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")

            # Train Global Agent
            logger.info("Training Global Agent...")
            self.global_model.learn(
                total_timesteps=global_steps,
                reset_num_timesteps=False
            )

            # Save checkpoint
            global_path = self.output_dir / f"global_cycle_{cycle + 1}.zip"
            self.global_model.save(global_path)
            logger.info(f"Global model saved to {global_path}")

            # Train Local Agents
            logger.info("Training Local Agents...")
            self.local_model.learn(
                total_timesteps=local_steps,
                reset_num_timesteps=False
            )

            # Save checkpoint
            local_path = self.output_dir / f"local_cycle_{cycle + 1}.zip"
            self.local_model.save(local_path)
            logger.info(f"Local model saved to {local_path}")

    def _train_simultaneous(self):
        """
        Simultaneous training: Custom training loop with both agents.

        NOTE: This is a simplified implementation. For production,
        consider using more sophisticated multi-agent training libraries.
        """
        logger.warning(
            "Simultaneous training is experimental. "
            "Using simplified alternating mini-batches."
        )

        # Simplified: Alternate at mini-batch level
        batch_size = self.training_config.get("batch_size", 1000)
        num_batches = self.total_timesteps // batch_size

        for batch in range(num_batches):
            logger.info(f"Batch {batch + 1}/{num_batches}")

            # Train global for half batch
            self.global_model.learn(
                total_timesteps=batch_size // 2,
                reset_num_timesteps=False
            )

            # Train local for half batch
            self.local_model.learn(
                total_timesteps=batch_size // 2,
                reset_num_timesteps=False
            )

            # Save periodic checkpoints
            if (batch + 1) % 10 == 0:
                self.save_checkpoint(f"batch_{batch + 1}")

    def save_checkpoint(self, name: str = "final"):
        """
        Save model checkpoints.

        Args:
            name: Checkpoint name suffix
        """
        global_path = self.output_dir / f"global_{name}.zip"
        local_path = self.output_dir / f"local_{name}.zip"

        self.global_model.save(global_path)
        self.local_model.save(local_path)

        logger.info(f"Checkpoint saved: {name}")

    def load_checkpoint(self, global_path: str, local_path: str):
        """
        Load model checkpoints.

        Args:
            global_path: Path to global model
            local_path: Path to local model
        """
        self.global_model = PPO.load(global_path)
        self.local_model = MaskablePPO.load(local_path)

        logger.info("Checkpoints loaded successfully")


def main():
    """Main training entry point."""

    parser = argparse.ArgumentParser(
        description="Joint training for hierarchical multi-datacenter MARL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config.yml",
        help="Path to environment configuration YAML"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../logs/joint_training",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=100000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="alternating",
        choices=["alternating", "simultaneous"],
        help="Training strategy"
    )

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Configuration for models
    global_model_config = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
    }

    local_model_config = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
    }

    training_config = {
        "total_timesteps": args.total_timesteps,
        "strategy": args.strategy,
        "global_steps_per_cycle": 10000,
        "local_steps_per_cycle": 10000,
        "num_cycles": 10,
    }

    # Create manager and train
    manager = JointTrainingManager(
        config_path=args.config,
        output_dir=str(output_dir),
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config
    )

    try:
        manager.train()
        manager.save_checkpoint("final")
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
