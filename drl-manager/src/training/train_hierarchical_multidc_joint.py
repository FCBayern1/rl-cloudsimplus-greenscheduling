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
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import gymnasium as gym

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Add drl-manager root to path (to import gym_cloudsimplus and src packages)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gym_cloudsimplus.envs.joint_training_env import JointTrainingEnv, ParameterSharingWrapper
from src.callbacks.save_on_best_reward_hierarchical import SaveOnBestRewardHierarchicalCallback
from src.callbacks.tensorboard_enhanced_logging import (
    EnhancedTensorBoardCallback,
    SeparateAgentMetricsCallback
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_random_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - Environment variables

    Args:
        seed: Random seed value
    """
    try:
        seed = int(seed)

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            # Additional settings for deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)

        logger.info(f"[OK] Random seed set to: {seed}")
        logger.info(f"   - Python random: {seed}")
        logger.info(f"   - NumPy: {seed}")
        logger.info(f"   - PyTorch: {seed}")
        if torch.cuda.is_available():
            logger.info(f"   - CUDA: {seed} (deterministic mode enabled)")

    except Exception as e:
        logger.error(f"Failed to set random seeds: {e}", exc_info=True)
        raise


class JointTrainingManager:
    """
    Manages joint training of Global and Local agents.

    This class coordinates the training loop, handles model creation,
    and manages checkpoints and logging.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        global_model_config: Dict[str, Any],
        local_model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        config_path: Optional[str] = None
    ):
        """
        Initialize joint training manager.

        Args:
            config: Environment configuration dictionary
            output_dir: Directory for outputs (models, logs)
            global_model_config: Configuration for global agent model
            local_model_config: Configuration for local agent model
            training_config: Training hyperparameters
        """
        self.config = config
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_model_config = global_model_config
        self.local_model_config = local_model_config
        self.training_config = training_config

        # Create environment
        if config_path:
            logger.info(f"Creating joint training environment from {config_path}")
        else:
            logger.info("Creating joint training environment from provided config dict")
        self.env = self._create_environment()

        # Store reference to base environment (before Monitor wrapper)
        # This is needed to access custom attributes like global_observation_space
        if hasattr(self.env, 'env'):
            # If wrapped by Monitor, get the base environment
            self.base_env = self.env.env
        else:
            self.base_env = self.env

        # Determine usable devices for SB3 policies
        self._cuda_usable = self._check_cuda_usable()
        base_device_pref = self.config.get("device", "auto")
        global_pref = self.config.get("global_agent", {}).get("device", base_device_pref)
        local_pref = self.config.get("local_agents", {}).get("device", base_device_pref)

        self.global_device = self._resolve_device(global_pref, agent_label="global")
        self.local_device = self._resolve_device(local_pref, agent_label="local")

        logger.info(
            "Device selection -> global: %s | local: %s",
            self.global_device,
            self.local_device
        )

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
            config=self.config,
            mode="training"
        )

        # Wrap with parameter sharing for easier batch processing
        env = ParameterSharingWrapper(env)

        # Wrap with Monitor for episode statistics tracking
        monitor_dir = self.output_dir / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)

        info_keywords = (
            "global_reward", "local_reward", "total_reward",
            "cloudlets_routed", "cloudlets_completed",
            "green_energy_ratio", "brown_energy_wh", "wasted_green_wh"
        )

        env = Monitor(
            env,
            str(monitor_dir),
            info_keywords=info_keywords
        )

        logger.info(
            f"Environment created with {env.env.num_datacenters} datacenters"
        )
        logger.info(f"Monitor wrapper added, logging to {monitor_dir}")

        return env

    def _resolve_device(self, preference: Any, agent_label: str) -> str:
        """
        Resolve desired torch device based on preference and CUDA availability.
        """
        pref = str(preference).strip().lower() if preference is not None else "auto"
        if pref not in {"auto", "cpu", "cuda"}:
            logger.warning(
                "Unknown device preference '%s' for %s agent. Falling back to 'auto'.",
                preference,
                agent_label
            )
            pref = "auto"

        if pref == "cpu":
            return "cpu"

        if pref == "cuda":
            if self._cuda_usable:
                return "cuda"
            logger.warning(
                "CUDA requested for %s agent but current PyTorch build cannot use the GPU. "
                "Falling back to CPU.",
                agent_label
            )
            return "cpu"

        # auto: prefer CUDA when fully supported
        return "cuda" if self._cuda_usable else "cpu"

    def _check_cuda_usable(self) -> bool:
        """
        Determine whether the current PyTorch build can execute kernels on the detected GPU.
        """
        if not torch.cuda.is_available():
            logger.info("CUDA not available; running entirely on CPU.")
            return False

        try:
            device_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            device_capability = f"sm_{major}{minor}"
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
            supported_caps = {arch.replace("sm_", "") for arch in arch_list}

            if supported_caps and f"{major}{minor}" not in supported_caps:
                logger.warning(
                    "Detected GPU '%s' with capability %s, "
                    "but this PyTorch build only includes kernels for: %s. "
                    "GPU execution will be disabled.",
                    device_name,
                    device_capability,
                    ", ".join(sorted(f"sm_{cap}" for cap in supported_caps)) or "unknown"
                )
                return False

            logger.info(
                "CUDA device detected: %s (capability %s). Using GPU for compatible agents.",
                device_name,
                device_capability
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to inspect CUDA device (%s). Falling back to CPU execution.",
                exc
            )
            return False

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
        class GlobalAgentEnv(gym.Env):
            """Wrapper to expose only global agent's view."""

            def __init__(self, base_env):
                super().__init__()
                self.base_env = base_env
                self.observation_space = base_env.global_observation_space
                # Use MultiDiscrete action space: agent selects DC for each cloudlet in batch
                self.action_space = base_env.global_action_space
                self.batch_size = base_env.global_routing_batch_size
                self.num_datacenters = base_env.num_datacenters
                self.metadata = {"render_modes": []}
                
                logger.info(f"GlobalAgentEnv initialized:")
                logger.info(f"  action_space: {self.action_space}")
                logger.info(f"  batch_size: {self.batch_size}")
                logger.info(f"  num_datacenters: {self.num_datacenters}")

            def reset(self, **kwargs):
                obs, info = self.base_env.reset(**kwargs)
                return obs["global"], info

            def step(self, action):
                # Action is now an array of DC choices, one per cloudlet in the batch
                # Convert to list of integers
                logger.debug(f"[GlobalAgentEnv.step] Received action: {action}, type: {type(action)}")
                if isinstance(action, np.ndarray):
                    global_actions = action.flatten().astype(int).tolist()
                else:
                    # Handle single integer (shouldn't happen with MultiDiscrete)
                    global_actions = [int(action)]
                
                logger.debug(f"[GlobalAgentEnv.step] Converted to global_actions: {global_actions}, len: {len(global_actions)}, batch_size: {self.batch_size}")

                # Ensure we have exactly batch_size actions
                if len(global_actions) != self.batch_size:
                    logger.warning(
                        f"Expected {self.batch_size} actions, got {len(global_actions)}. "
                        f"Padding/truncating."
                    )
                    if len(global_actions) < self.batch_size:
                        # Pad with last action or random
                        last_action = global_actions[-1] if global_actions else 0
                        global_actions.extend([last_action] * (self.batch_size - len(global_actions)))
                    else:
                        global_actions = global_actions[:self.batch_size]

                # Need local actions - use TRAINED LOCAL MODEL instead of random
                local_actions = {}
                # Get current observations from base environment
                try:
                    # Try to get full observation through internal state
                    if hasattr(self.base_env, 'env'):
                        current_obs = self.base_env.env.last_obs if hasattr(self.base_env.env, 'last_obs') else None
                    else:
                        current_obs = None
                except:
                    current_obs = None
                
                for dc_id in range(self.base_env.num_datacenters):
                    if current_obs and "local" in current_obs:
                        dc_obs = current_obs["local"].get(dc_id, {})
                    else:
                        dc_obs = None
                    # Use trained local model to predict action
                    action_mask = self.base_env.env.get_local_action_masks(dc_id)
                    if hasattr(self, 'local_model') and self.local_model is not None and dc_obs is not None:
                        # Use trained local model
                        try:
                            local_action, _ = self.local_model.predict(
                                dc_obs, 
                                action_masks=action_mask,
                                deterministic=False  # Keep some exploration
                            )
                            local_actions[dc_id] = int(local_action)
                        except Exception as e:
                            logger.debug(f"Failed to use local model for DC {dc_id}: {e}, using random")
                            valid_actions = np.where(action_mask)[0]
                            local_actions[dc_id] = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                    else:
                        # Fallback: random with masking
                        valid_actions = np.where(action_mask)[0]
                        local_actions[dc_id] = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

                obs, rewards, terminated, truncated, info = self.base_env.step({
                    "global": global_actions,
                    "local": local_actions
                })

                return obs["global"], rewards["global"], terminated, truncated, info

            def render(self):
                return None

            def close(self):
                self.base_env.close()

        self.global_env = GlobalAgentEnv(self.base_env)

        self.global_model = PPO(
            policy=global_policy,
            env=self.global_env,
            learning_rate=global_lr,
            gamma=global_gamma,
            n_steps=global_n_steps,
            batch_size=global_batch_size,
            verbose=1,
            tensorboard_log=str(self.output_dir / "tensorboard" / "global"),
            device=self.global_device
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
        class LocalAgentEnv(gym.Env):
            """Wrapper to expose only local agent's view (for one DC)."""

            def __init__(self, base_env, dc_id=0):
                super().__init__()
                self.base_env = base_env
                self.dc_id = dc_id
                self.observation_space = base_env.local_observation_space
                self.action_space = base_env.local_action_space
                self.metadata = {"render_modes": []}

            def reset(self, **kwargs):
                obs, info = self.base_env.reset(**kwargs)
                # Return observation for first DC
                local_obs_dict = obs.get("local", {})
                dc_obs = local_obs_dict.get(self.dc_id, {})
                return dc_obs, info

            def step(self, action):
                # Need global actions - use TRAINED GLOBAL MODEL instead of random
                batch_size = self.base_env.global_routing_batch_size
                # Get current observation
                try:
                    if hasattr(self.base_env, 'env'):
                        current_obs = self.base_env.env.last_obs if hasattr(self.base_env.env, 'last_obs') else None
                    else:
                        current_obs = None
                except:
                    current_obs = None
                
                global_obs = current_obs["global"] if current_obs and "global" in current_obs else None
                
                if hasattr(self, 'global_model') and self.global_model is not None and global_obs is not None:
                    # Use trained global model
                    try:
                        global_action, _ = self.global_model.predict(
                            global_obs,
                            deterministic=False  # Keep some exploration
                        )
                        # Convert to list format
                        if isinstance(global_action, np.ndarray):
                            global_actions = global_action.flatten().astype(int).tolist()
                        else:
                            global_actions = [int(global_action)] * batch_size
                        # Ensure correct length
                        if len(global_actions) < batch_size:
                            global_actions.extend([global_actions[-1]] * (batch_size - len(global_actions)))
                        elif len(global_actions) > batch_size:
                            global_actions = global_actions[:batch_size]
                    except Exception as e:
                        logger.warning(f"Failed to use global model: {e}, using random")
                        global_action_scalar = int(self.base_env.global_action_space.sample().flatten()[0])
                        global_actions = [global_action_scalar] * batch_size
                else:
                    # Fallback: random
                    global_action = self.base_env.global_action_space.sample()
                    if isinstance(global_action, np.ndarray):
                        global_action_scalar = int(global_action.flatten()[0])
                    else:
                        global_action_scalar = int(global_action)
                    global_actions = [global_action_scalar] * batch_size
                    
                if isinstance(action, np.ndarray):
                    action_scalar = int(action.flatten()[0])
                else:
                    action_scalar = int(action)

                # Build local actions (only for this DC, others use trained local model if available)
                local_actions = {}
                for dc_id in range(self.base_env.num_datacenters):
                    if dc_id == self.dc_id:
                        local_actions[dc_id] = action_scalar
                    else:
                        # Use trained local model for other DCs
                        if current_obs and "local" in current_obs:
                            dc_obs = current_obs["local"].get(dc_id, {})
                        else:
                            dc_obs = None
                        action_mask = self.base_env.env.get_local_action_masks(dc_id)
                        if hasattr(self, 'local_model') and self.local_model is not None and dc_obs is not None:
                            try:
                                other_action, _ = self.local_model.predict(
                                    dc_obs,
                                    action_masks=action_mask,
                                    deterministic=False
                                )
                                local_actions[dc_id] = int(other_action)
                            except Exception as e:
                                valid_actions = np.where(action_mask)[0]
                                local_actions[dc_id] = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
                        else:
                            # Fallback: random with masking
                            valid_actions = np.where(action_mask)[0]
                            local_actions[dc_id] = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

                obs, rewards, terminated, truncated, info = self.base_env.step({
                    "global": global_actions,
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

            def render(self):
                return None

            def close(self):
                self.base_env.close()

        self.local_env = LocalAgentEnv(self.base_env, dc_id=0)

        self.local_model = MaskablePPO(
            policy=local_policy,
            env=self.local_env,
            learning_rate=local_lr,
            gamma=local_gamma,
            n_steps=local_n_steps,
            batch_size=local_batch_size,
            verbose=1,
            tensorboard_log=str(self.output_dir / "tensorboard" / "local"),
            device=self.local_device
        )

        logger.info("Local Agent model created")
        
        # Link models to environments for cooperative training
        logger.info("Linking models for cooperative alternating training...")
        self.global_env.local_model = self.local_model
        self.local_env.global_model = self.global_model
        self.local_env.local_model = self.local_model  # For other DCs
        logger.info("Models linked successfully")

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

        # Create callbacks
        save_best_callback = SaveOnBestRewardHierarchicalCallback(
            log_dir=str(self.output_dir),
            global_model=self.global_model,
            local_model=self.local_model,
            save_freq=1000,
            verbose=1
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=str(self.output_dir / "checkpoints"),
            name_prefix="model",
            verbose=1
        )
        
        # Enhanced logging callbacks for detailed metrics
        global_tb_callback = EnhancedTensorBoardCallback(
            agent_name="global",
            verbose=1
        )
        
        local_tb_callback = EnhancedTensorBoardCallback(
            agent_name="local",
            verbose=1
        )
        
        # Separate metrics tracking
        global_metrics_callback = SeparateAgentMetricsCallback(
            agent_type="global",
            verbose=1
        )
        
        local_metrics_callback = SeparateAgentMetricsCallback(
            agent_type="local",
            verbose=1
        )

        # Combine all callbacks
        global_callback_list = CallbackList([
            save_best_callback, 
            checkpoint_callback,
            global_tb_callback,
            global_metrics_callback
        ])
        
        local_callback_list = CallbackList([
            save_best_callback,
            checkpoint_callback,
            local_tb_callback,
            local_metrics_callback
        ])

        for cycle in range(num_cycles):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"  Cycle {cycle + 1}/{num_cycles}")
            logger.info(f"  Current total timesteps: {self.global_model.num_timesteps}")
            logger.info("=" * 70)

            # Train Global Agent
            logger.info("")
            logger.info(f"Training Global Agent... (Target: {global_steps} steps)")
            logger.info(f"  From timestep {self.global_model.num_timesteps} to {self.global_model.num_timesteps + global_steps}")
            self.global_model.learn(
                total_timesteps=global_steps,
                reset_num_timesteps=False,
                callback=global_callback_list,
                progress_bar=True

            )
            logger.info(f"Global Agent training complete. Total timesteps: {self.global_model.num_timesteps}")

            # Save checkpoint
            global_path = self.output_dir / f"global_cycle_{cycle + 1}.zip"
            self.global_model.save(str(global_path))
            logger.info(f"[SAVED] Global model checkpoint -> {global_path}")

            # Train Local Agents
            logger.info("")
            logger.info(f"Training Local Agents... (Target: {local_steps} steps)")
            logger.info(f"  From timestep {self.local_model.num_timesteps} to {self.local_model.num_timesteps + local_steps}")
            self.local_model.learn(
                total_timesteps=local_steps,
                reset_num_timesteps=False,
                callback=local_callback_list,
                progress_bar=True
            )
            logger.info(f"Local Agent training complete. Total timesteps: {self.local_model.num_timesteps}")

            # Save checkpoint
            local_path = self.output_dir / f"local_cycle_{cycle + 1}.zip"
            self.local_model.save(str(local_path))
            logger.info(f"[SAVED] Local model checkpoint -> {local_path}")

        # Final save
        logger.info("")
        logger.info("=" * 70)
        logger.info("Training completed! Saving final models...")
        logger.info("=" * 70)
        self.global_model.save(str(self.output_dir / "final_global_model"))
        self.local_model.save(str(self.output_dir / "final_local_model"))
        logger.info(f"[SAVED] Final models -> {self.output_dir}")

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
        "--experiment",
        type=str,
        default=None,
        help="Experiment key inside the config YAML (e.g., experiment_multi_dc_3)"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config or random)"
    )

    args = parser.parse_args()

    # Load configuration and select experiment (if provided)
    with open(args.config, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    if args.experiment:
        experiment_config = full_config.get(args.experiment)
        if experiment_config is None:
            raise ValueError(
                f"Experiment '{args.experiment}' not found in {args.config}"
            )
        logger.info(f"Loaded experiment '{args.experiment}' from config")
        
        # Merge with common config (common as base, experiment overrides)
        if "common" in full_config:
            merged_config = full_config["common"].copy()
            merged_config.update(experiment_config)
            experiment_config = merged_config
            logger.info("Merged experiment config with common config")
    else:
        experiment_config = full_config
        logger.info("No experiment specified; using full config as environment definition")

    if not isinstance(experiment_config, dict):
        raise TypeError(
            "Selected experiment configuration must be a dictionary. "
            f"Got {type(experiment_config)} instead."
        )

    # === Set Random Seed for Reproducibility ===
    # Priority: 1) Command line, 2) Config file, 3) Random
    if args.seed is not None:
        seed = args.seed
        logger.info(f"Using seed from command line: {seed}")
    elif "seed" in experiment_config:
        seed_value = experiment_config["seed"]
        if isinstance(seed_value, str) and seed_value.lower() == "random":
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Config specified 'random', generated seed: {seed}")
        else:
            seed = int(seed_value)
            logger.info(f"Using seed from config: {seed}")
    else:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"No seed specified, generated random seed: {seed}")

    # Set the seed globally
    set_random_seed(seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Save the seed to output directory for reproducibility
    seed_file = output_dir / "seed_used.txt"
    with open(seed_file, 'w') as f:
        f.write(f"{seed}\n")
    logger.info(f"Seed saved to: {seed_file}")

    # Configuration for models
    global_model_config = {
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
    }

    local_model_config = {
        "policy": "MultiInputPolicy",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
    }

    # Extract training configuration from experiment config
    joint_training_config = experiment_config.get("joint_training", {})
    alternating_config = joint_training_config.get("alternating", {})

    # Get total timesteps from config or args
    total_timesteps = experiment_config.get("timesteps", args.total_timesteps)
    strategy = args.strategy if args.strategy else joint_training_config.get("strategy", "alternating")

    # For alternating strategy, calculate cycle parameters from total_timesteps if not specified
    if strategy == "alternating" and not alternating_config:
        # If alternating config not specified, auto-calculate from total_timesteps
        # Default: 10 cycles with equal global and local steps
        num_cycles = 10
        steps_per_agent_per_cycle = total_timesteps // (num_cycles * 2)  # Divide by 2 (global + local)
        logger.info(f"Auto-calculating cycle parameters from timesteps={total_timesteps}")
        logger.info(f"  Using {num_cycles} cycles with {steps_per_agent_per_cycle} steps per agent")

        training_config = {
            "total_timesteps": total_timesteps,
            "strategy": strategy,
            "num_cycles": num_cycles,
            "global_steps_per_cycle": steps_per_agent_per_cycle,
            "local_steps_per_cycle": steps_per_agent_per_cycle,
            "checkpoint_freq": joint_training_config.get("checkpoint_freq", 10000),
            "log_freq": joint_training_config.get("log_freq", 100),
        }
    else:
        # Use explicit configuration from config file
        training_config = {
            # Total timesteps (for simultaneous strategy or as fallback)
            "total_timesteps": total_timesteps,

            # Strategy
            "strategy": strategy,

            # Alternating training parameters (from config or defaults)
            "num_cycles": alternating_config.get("num_cycles", 10),
            "global_steps_per_cycle": alternating_config.get("global_steps_per_cycle", 10000),
            "local_steps_per_cycle": alternating_config.get("local_steps_per_cycle", 10000),

            # Checkpoint and logging (from config or defaults)
            "checkpoint_freq": joint_training_config.get("checkpoint_freq", 10000),
            "log_freq": joint_training_config.get("log_freq", 100),
        }

    # Log training configuration
    logger.info("=" * 70)
    logger.info("Training Configuration:")
    logger.info(f"  Strategy: {training_config['strategy']}")
    logger.info(f"  Total timesteps: {training_config['total_timesteps']}")
    logger.info(f"  Num cycles: {training_config['num_cycles']}")
    logger.info(f"  Global steps per cycle: {training_config['global_steps_per_cycle']}")
    logger.info(f"  Local steps per cycle: {training_config['local_steps_per_cycle']}")
    total_steps = training_config['num_cycles'] * (
        training_config['global_steps_per_cycle'] +
        training_config['local_steps_per_cycle']
    )
    logger.info(f"  Estimated total training steps: {total_steps}")
    logger.info("=" * 70)

    # Create manager and train
    manager = JointTrainingManager(
        config=experiment_config,
        output_dir=str(output_dir),
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config,
        config_path=args.config
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
