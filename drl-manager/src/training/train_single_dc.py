import os
import time
import logging
import torch
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    StopTrainingOnNoModelImprovement,
)

from src.callbacks.save_on_best_reward import SaveOnBestTrainingRewardCallback
import gym_cloudsimplus  # noqa: F401

logger = logging.getLogger(__name__)

def train(params: dict):
    """
    Trains an RL agent based on the provided parameters.

    Args:
        params (dict): Dictionary containing configuration parameters from config.yml.
    """
    logger.info("Starting training process...")
    logger.info(f"Parameters: {params}")

    # --- Device Setup ---
    device_name = params.get("device", "auto") # Get device from config, default to auto
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    # --- Environment Setup ---
    try:
        logger.info(f"Creating Gym environment: {params.get('env_id', 'LoadBalancingScaling-v0')}")
        # Pass only necessary params (or all params) to the env constructor
        env = gym.make(params.get('env_id', 'LoadBalancingScaling-v0'), config_params=params)
        logger.info("Environment created successfully.")
    except Exception as e:
        logger.error(f"Failed to create Gym environment: {e}", exc_info=True)
        return # Exit if env creation fails

    # --- Logging and Monitoring ---
    log_destination = ["stdout"]
    log_dir = params.get("log_dir") # Get log_dir set by entrypoint.py
    callbacks = []

    if params.get("save_experiment", False) and log_dir:
        logger.info(f"Saving experiment data to: {log_dir}")
        log_destination.extend(["csv", "tensorboard"]) # Add CSV and TensorBoard output formats

        # Setup Monitor wrapper
        # Use info_keywords to log specific values from the info dict to progress.csv
        # Adjust keywords based on keys returned by SimulationStepInfo.toMap()
        info_keywords_to_log = (
            "reward_wait_time", "reward_unutilization",
            "reward_queue_penalty", "reward_invalid_action",
            "reward_energy",  # Energy consumption reward component
            "current_power_w", "cumulative_energy_wh", "average_host_utilization",  # Energy metrics
            "cumulative_green_energy_wh", "cumulative_brown_energy_wh",  # Green energy consumption
            "total_wasted_green_wh", "current_green_power_w", "green_ratio",  # Green energy details
            "assignment_success",
            "invalid_action_taken",
            "actual_vm_count", "current_clock"
        )
        # Episode-aggregated metrics (computed in env at episode end)
        info_keywords_to_log += (
            "episode_reward_wait_time_mean",
            "episode_reward_unutilization_mean",
            "episode_reward_queue_penalty_mean",
            "episode_reward_invalid_action_mean",
            "episode_reward_energy_mean",
            "episode_avg_power_w",
        )
        env = Monitor(env, log_dir, info_keywords=info_keywords_to_log)
        logger.info(f"Environment wrapped with Monitor, logging to {log_dir}")

        # Setup Best Model Callback
        # The callback writes all the other .csv files and saves the model (with replay buffer) when the reward is the best
        save_best_callback = SaveOnBestTrainingRewardCallback(
            log_dir=log_dir,
            verbose=1,
            save_replay_buffer=params.get("save_replay_buffer", False) # Make saving buffer configurable
        )
        logger.info("SaveOnBestTrainingRewardCallback initialized.")
        callbacks.append(save_best_callback)
    else:
        logger.info("Experiment saving disabled or log_dir not specified.")
        # Wrap with Monitor even if not saving to log reward/length easily
        env = Monitor(env)
        logger.info("Environment wrapped with Monitor (no file logging).")

    # --- Vectorized Environment ---
    # DummyVecEnv is usually fine for single environment setups. If performance becomes an issue or A2C is used, switch to SubprocVecEnv.
    # Note: SubprocVecEnv might have issues on Windows or with complex objects
    # See https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
    if params["algorithm"] == "A2C":
        device = "cpu"
        env = SubprocVecEnv([lambda: env], start_method="fork")
    if params["algorithm"] == "MaskablePPO":
            device = "cpu"
            env = DummyVecEnv([lambda: env])
    else:
        env = DummyVecEnv([lambda: env])
        logger.info("Environment wrapped with DummyVecEnv.")

    # --- Evaluation / Early Stopping Setup ---
    # Optional early stopping using SB3 EvalCallback
    early_cfg = params.get("early_stop", {}) or {}
    if early_cfg.get("enabled", False):
        try:
            # Separate eval env (no vectorization needed here; EvalCallback handles it)
            eval_env = gym.make(params.get('env_id', 'LoadBalancingScaling-v0'), config_params=params)
            eval_log_dir = None
            if log_dir:
                eval_log_dir = os.path.join(log_dir, "eval")
                os.makedirs(eval_log_dir, exist_ok=True)
            eval_env = Monitor(eval_env, eval_log_dir) if eval_log_dir else Monitor(eval_env)

            eval_freq = int(early_cfg.get("eval_freq", 5000))
            n_eval_episodes = int(early_cfg.get("n_eval_episodes", 5))

            stop_type = str(early_cfg.get("type", "no_improvement")).lower()
            if stop_type == "reward_threshold":
                threshold = float(early_cfg.get("reward_threshold", 200.0))
                stop_cb = StopTrainingOnRewardThreshold(reward_threshold=threshold, verbose=1)
                eval_cb = EvalCallback(
                    eval_env,
                    best_model_save_path=log_dir,
                    log_path=eval_log_dir or log_dir,
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    callback_on_new_best=stop_cb,
                    verbose=1,
                )
            else:
                # Default: stop when no improvement after patience eval rounds
                patience = int(early_cfg.get("patience", 5))
                min_evals = int(early_cfg.get("min_evals", 5))
                stop_cb = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=patience,
                    min_evals=min_evals,
                    verbose=1,
                )
                eval_cb = EvalCallback(
                    eval_env,
                    best_model_save_path=log_dir,
                    log_path=eval_log_dir or log_dir,
                    eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True,
                    callback_after_eval=stop_cb,
                    verbose=1,
                )

            callbacks.append(eval_cb)
            logger.info("EvalCallback for early stopping initialized.")
        except Exception as e:
            logger.error(f"Failed to set up early stopping EvalCallback: {e}", exc_info=True)

    # --- Algorithm Selection ---
    # Assuming MaskablePPO for now due to action masking requirement
    algorithm_name = params.get("algorithm", "PPO")
    ModelClass = None
    if algorithm_name == "MaskablePPO":
        try:
            ModelClass = MaskablePPO
            logger.info("Using RL Algorithm: MaskablePPO (from sb3_contrib)")
        except ImportError:
            logger.error("sb3_contrib not installed. Cannot use MaskablePPO. Install with 'pip install sb3-contrib'")
            env.close(); return
    elif algorithm_name == "RecurrentPPO":
        if RecurrentPPO is None:
            logger.error("sb3_contrib RecurrentPPO not available. Please install/upgrade sb3-contrib.")
            env.close(); return
        ModelClass = RecurrentPPO
        logger.info("Using RL Algorithm: RecurrentPPO (from sb3_contrib)")
    else:
        try:
            algo_map = {"PPO": sb3.PPO, "A2C": sb3.A2C, "DQN": sb3.DQN, "SAC": sb3.SAC, "TD3": sb3.TD3}
            ModelClass = algo_map.get(algorithm_name)
            if ModelClass:
                logger.info(f"Using RL Algorithm: {algorithm_name} (from stable_baselines3)")
                if algorithm_name != "PPO" and algorithm_name != "A2C": # Only PPO/A2C handle MultiDiscrete well by default
                     logger.warning(f"Algorithm {algorithm_name} might not inherently support MultiDiscrete action spaces or action masking well.")
            else:
                logger.error(f"Algorithm '{algorithm_name}' not found in stable_baselines3 or sb3_contrib.")
                env.close(); return
        except ImportError:
             logger.error("stable_baselines3 not installed correctly.")
             env.close(); return

    # --- Model Instantiation ---
    # Define policy: prefer explicit config, else infer by observation space
    policy = params.get("policy") or ("MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else "MlpPolicy")

    # Define hyperparameters, taking from params or using defaults
    policy_kwargs = params.get("policy_kwargs", None) # e.g., dict(net_arch=[128, 128])

    # Filter model_params based on the selected algorithm's supported arguments
    # This avoids passing unsupported args like 'use_sde' to DQN
    # (More robust would be to use inspect module, but manual filtering is ok for fewer algos)
    common_params = {
        "policy": policy,
        "env": env,
        "seed": params.get("seed"),
        "learning_rate": float(params.get("learning_rate", 3e-4)),
        "policy_kwargs": policy_kwargs,
        "verbose": params.get("verbose", 0), # Set verbose 0 to reduce console spam, rely on logger/callback
        "device": device,
        "tensorboard_log": log_dir if "tensorboard" in log_destination else None
    }
    if common_params["policy_kwargs"] is None:
        del common_params["policy_kwargs"]

    specific_params = {}
    if (ModelClass == MaskablePPO) or (ModelClass == sb3.PPO) or (RecurrentPPO is not None and ModelClass == RecurrentPPO):
        specific_params = {
            "n_steps": int(params.get("n_steps", 2048)),
            "batch_size": int(params.get("batch_size", 64)),
            "n_epochs": int(params.get("n_epochs", 10)),
            "gamma": float(params.get("gamma", 0.99)),
            "gae_lambda": float(params.get("gae_lambda", 0.95)),
            "clip_range": float(params.get("clip_range", 0.2)),
            "ent_coef": float(params.get("ent_coef", 0.0)),
            "vf_coef": float(params.get("vf_coef", 0.5)),
            "max_grad_norm": float(params.get("max_grad_norm", 0.5)),
        }
    elif ModelClass == sb3.A2C:
         specific_params = {
             "n_steps": int(params.get("n_steps", 5)), # A2C default
             "gamma": float(params.get("gamma", 0.99)),
             "gae_lambda": float(params.get("gae_lambda", 1.0)), # A2C default
             "ent_coef": float(params.get("ent_coef", 0.0)),
             "vf_coef": float(params.get("vf_coef", 0.5)),
             "max_grad_norm": float(params.get("max_grad_norm", 0.5)),
             "use_rms_prop": bool(params.get("use_rms_prop", True)), # A2C default
             "use_sde": bool(params.get("use_sde", False)),
             "sde_sample_freq": int(params.get("sde_sample_freq", -1)),
         }
    # Add more elif blocks for DQN, SAC, TD3 parameters if needed

    model_params = {**common_params, **specific_params}

    try:
        model = ModelClass(**model_params)
        logger.info(f"Model {ModelClass.__name__} instantiated successfully.")
        # logger.info(f"Model Policy Architecture:\n{model.policy}") # Print network structure
    except Exception as e:
        logger.error(f"Failed to instantiate model '{ModelClass.__name__}': {e}", exc_info=True)
        env.close()
        return

    # --- Configure Logger (Optional, SB3 does basic logging via Monitor) ---
    # If using the SB3 configure logger explicitly:
    logger_sb3 = configure(log_dir, log_destination)
    model.set_logger(logger_sb3)
    logger.info("SB3 logger configured.")

    # --- Training ---
    total_timesteps = int(params.get("timesteps", 2048))
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=params.get("log_interval", 1), # Log stats every episode
            callback=(callbacks if len(callbacks) > 0 else None),
            progress_bar=True, # Show progress bar
        )
        training_duration = time.time() - start_time
        logger.info(f"Training finished in {training_duration:.2f} seconds.")

        # Save the final model if saving is enabled
        if params.get("save_experiment", False) and log_dir:
            final_model_path = os.path.join(log_dir, "final_model")
            logger.info(f"Saving final model to {final_model_path}")
            model.save(final_model_path)
            # Save replay buffer if applicable (e.g., for DQN, SAC, TD3)
            if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
                 replay_buffer_path = os.path.join(log_dir, "final_model_replay_buffer")
                 logger.info(f"Saving final replay buffer to {replay_buffer_path}")
                 model.save_replay_buffer(replay_buffer_path)

    except Exception as e:
        logger.error(f"Error during model learning: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        logger.info("Closing environment...")
        env.close() # This should call LoadBalancingEnv.close() -> gateway.close()
        if 'eval_env' in locals():
            try:
                eval_env.close()
            except Exception:
                pass
        logger.info("Training script finished.")
        # Delete the model from memory
        if 'model' in locals():
            del model


def main(params: dict):
    """Entrypoint wrapper for compatibility with src.training package.

    Args:
        params: Configuration dictionary passed from callers (e.g., entrypoint).
    """
    return train(params)
