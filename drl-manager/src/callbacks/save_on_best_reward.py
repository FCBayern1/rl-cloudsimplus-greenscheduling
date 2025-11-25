import os
import numpy as np
import pandas as pd
import logging
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__.split('.')[-1])

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model based on the training reward reported by the Monitor wrapper.
    Additionally, logs detailed information about the best episode found so far.

    :param log_dir: Path to the folder where the model will be saved and logs stored.
                    It must contain the file created by the ``Monitor`` wrapper.
    :param save_replay_buffer: Whether to save the replay buffer (if applicable) for the best model.
    :param verbose: Verbosity level.
    """
    def __init__(
        self,
        log_dir: str,
        save_replay_buffer: bool = False, # Changed default to False
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_replay_buffer = save_replay_buffer
        self.model_save_path = os.path.join(log_dir, "best_model")
        self.replay_buffer_save_path = os.path.join(log_dir, "best_model_replay_buffer")
        self.best_mean_reward = -np.inf
        self.best_episode_filename_prefix = "best_episode_details"
        self.current_episode_num_monitor = 0 # Tracks episodes seen by Monitor
        self._clear_episode_details()
        # Check if log_dir exists
        if self.log_dir is None:
             logger.warning("log_dir is None. Cannot save best model or episode details.")


    def _clear_episode_details(self) -> None:
        """Clears the lists storing details for the current episode."""
        # self.observations = [] # Optional: Store observations? Can be large.
        self.actions = []
        self.rewards = [] # Total reward per step
        # Specific components from StepInfo
        self.reward_wait_times = []
        self.reward_unutilizations = []
        self.reward_queue_penalties = []
        self.reward_invalid_actions = []
        self.reward_energies = []  # Energy reward component
        self.assignment_successes = []
        self.invalid_actions_taken = []
        self.actual_vm_counts = []
        self.current_clocks = []
        # Green energy metrics
        self.cumulative_green_energy_whs = []
        self.cumulative_brown_energy_whs = []
        self.total_wasted_green_whs = []
        self.current_green_power_ws = []
        self.green_ratios = []
        self.current_episode_length = 0

    def _save_timestep_details(self) -> None:
        """Saves the details from the current timestep's info dictionary."""
        if not self.locals.get("infos"):
            logger.warning("Callback: 'infos' not found in locals. Cannot save step details.")
            return

        info = self.locals["infos"][0] # Assuming DummyVecEnv (index 0)

        # self.observations.append(self._extract_observation_from_locals("obs_tensor")) # Optional
        self.actions.append(self.locals["actions"][0])
        self.rewards.append(self.locals["rewards"][0])
        # self.new_observations.append(self._extract_observation_from_locals("new_obs")) # Optional

        # Extract metrics from info dict (use .get() with defaults)
        self.reward_wait_times.append(info.get("reward_wait_time", 0.0))
        self.reward_unutilizations.append(info.get("reward_unutilization", 0.0))
        self.reward_queue_penalties.append(info.get("reward_queue_penalty", 0.0))
        self.reward_invalid_actions.append(info.get("reward_invalid_action", 0.0))
        self.reward_energies.append(info.get("reward_energy", 0.0))
        self.assignment_successes.append(info.get("assignment_success", False))
        self.invalid_actions_taken.append(info.get("invalid_action_taken", False))
        self.actual_vm_counts.append(info.get("actual_vm_count", 0))
        self.current_clocks.append(info.get("current_clock", 0.0))
        # Green energy metrics
        self.cumulative_green_energy_whs.append(info.get("cumulative_green_energy_wh", 0.0))
        self.cumulative_brown_energy_whs.append(info.get("cumulative_brown_energy_wh", 0.0))
        self.total_wasted_green_whs.append(info.get("total_wasted_green_wh", 0.0))
        self.current_green_power_ws.append(info.get("current_green_power_w", 0.0))
        self.green_ratios.append(info.get("green_ratio", 0.0))

        self.current_episode_length += 1


    def _create_episode_details_dict(self) -> dict:
        """Creates a dictionary of the collected episode details."""
        # Timesteps for this episode (approximate calculation)
        ep_last_timestep = self.num_timesteps
        ep_first_timestep = ep_last_timestep - self.current_episode_length + 1
        timesteps = np.arange(ep_first_timestep, ep_last_timestep + 1)

        details = {
            "timestep": timesteps,
            "clock": self.current_clocks,
            "target_vm_id": self.actions,
            "reward_total": self.rewards,
            "reward_wait_time": self.reward_wait_times,
            "reward_unutilization": self.reward_unutilizations,
            "reward_queue_penalty": self.reward_queue_penalties,
            "reward_invalid_action": self.reward_invalid_actions,
            "reward_energy": self.reward_energies,
            "assignment_success": self.assignment_successes,
            "invalid_action_taken": self.invalid_actions_taken,
            "actual_vm_count": self.actual_vm_counts,
            # Green energy metrics
            "cumulative_green_energy_wh": self.cumulative_green_energy_whs,
            "cumulative_brown_energy_wh": self.cumulative_brown_energy_whs,
            "total_wasted_green_wh": self.total_wasted_green_whs,
            "current_green_power_w": self.current_green_power_ws,
            "green_ratio": self.green_ratios,
        }
        # Ensure all lists have the same length (robustness)
        expected_len = self.current_episode_length
        for key, value in details.items():
             if len(value) != expected_len:
                  logger.warning(f"Length mismatch for '{key}' in episode details! Expected {expected_len}, got {len(value)}. Padding/truncating.")
                  # Simple padding/truncating - more sophisticated handling might be needed
                  if len(value) > expected_len:
                      details[key] = value[:expected_len]
                  else:
                      padding_val = 0 if np.issubdtype(np.array(value).dtype, np.number) else None
                      details[key] = value + [padding_val] * (expected_len - len(value))

        return details


    def _save_best_episode_details(self, episode_num: int) -> None:
        """Saves the details of the best episode found so far to a CSV."""
        if not self.log_dir: return

        # Delete previous best episode file(s) if they exist
        try:
            for f in os.listdir(self.log_dir):
                if f.startswith(self.best_episode_filename_prefix) and f.endswith(".csv"):
                    os.remove(os.path.join(self.log_dir, f))
                    logger.debug(f"Removed previous best episode file: {f}")
        except OSError as e:
             logger.error(f"Error removing previous best episode file: {e}")

        # Save new best episode
        episode_details_path = os.path.join(
            self.log_dir,
            f"{self.best_episode_filename_prefix}_{episode_num}.csv",
        )
        try:
            df_dict = self._create_episode_details_dict()
            df = pd.DataFrame(df_dict)
            df.to_csv(episode_details_path, index=False)
            if self.verbose > 0:
                logger.info(f"Saved best episode details to {episode_details_path}")
        except Exception as e:
            logger.error(f"Error saving best episode details to {episode_details_path}: {e}", exc_info=True)


    def _save_best_model(self) -> None:
        """Saves the current model as the best model found so far."""
        if not self.log_dir: return

        if self.verbose > 0:
            logger.info(f"Saving new best model to {self.model_save_path}.zip")
        self.model.save(self.model_save_path)

        if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            if self.verbose > 0:
                logger.info(f"Saving best model replay buffer to {self.replay_buffer_save_path}")
            try:
                 self.model.save_replay_buffer(self.replay_buffer_save_path)
            except NotImplementedError:
                 logger.warning(f"Algorithm {type(self.model).__name__} does not support saving replay buffer.")
            except Exception as e:
                logger.error(f"Error saving replay buffer: {e}", exc_info=True)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # 1. Save details for the current timestep
        self._save_timestep_details()

        # 2. Check if the episode finished (using the 'dones' flag from VecEnv)
        if self.locals["dones"][0]:
            self.current_episode_num_monitor += 1 # Increment episode count

            # --- Log episode stats using SB3 Logger (reads from Monitor buffer) ---
            # This relies on the Monitor wrapper writing to progress.csv
            # We just need to trigger the dump if necessary.
            if self.logger is not None:
                 # Record custom metrics maybe? Monitor already handles reward/length.
                 self.logger.record("rollout/ep_num_monitor", self.current_episode_num_monitor)

                 # === Log Energy Consumption Statistics ===
                 # Get final (cumulative) values from the last step of the episode
                 if len(self.cumulative_green_energy_whs) > 0:
                     final_green_wh = self.cumulative_green_energy_whs[-1]
                     final_brown_wh = self.cumulative_brown_energy_whs[-1]
                     final_total_wh = final_green_wh + final_brown_wh
                     final_wasted_wh = self.total_wasted_green_whs[-1]
                     final_clock = self.current_clocks[-1] if len(self.current_clocks) > 0 else 0

                     # Calculate metrics
                     green_ratio = final_green_wh / final_total_wh if final_total_wh > 0 else 0
                     total_green_available = final_green_wh + final_wasted_wh
                     green_utilization = final_green_wh / total_green_available if total_green_available > 0 else 0

                     # Calculate average power
                     episode_duration_h = (final_clock / 3600.0) if final_clock > 0 else 1e-6
                     avg_power_w = final_total_wh / episode_duration_h
                     avg_green_power_w = np.mean(self.current_green_power_ws) if len(self.current_green_power_ws) > 0 else 0

                     # Record to TensorBoard/CSV
                     self.logger.record("energy/total_energy_wh", final_total_wh)
                     self.logger.record("energy/total_energy_kwh", final_total_wh / 1000.0)
                     self.logger.record("energy/green_energy_wh", final_green_wh)
                     self.logger.record("energy/green_energy_kwh", final_green_wh / 1000.0)
                     self.logger.record("energy/brown_energy_wh", final_brown_wh)
                     self.logger.record("energy/brown_energy_kwh", final_brown_wh / 1000.0)
                     self.logger.record("energy/wasted_green_wh", final_wasted_wh)
                     self.logger.record("energy/wasted_green_kwh", final_wasted_wh / 1000.0)
                     self.logger.record("energy/green_ratio", green_ratio)
                     self.logger.record("energy/green_ratio_percentage", green_ratio * 100)
                     self.logger.record("energy/green_utilization", green_utilization)
                     self.logger.record("energy/green_utilization_percentage", green_utilization * 100)
                     self.logger.record("energy/avg_power_w", avg_power_w)
                     self.logger.record("energy/avg_power_kw", avg_power_w / 1000.0)
                     self.logger.record("energy/avg_green_power_w", avg_green_power_w)
                     self.logger.record("energy/avg_green_power_kw", avg_green_power_w / 1000.0)
                     self.logger.record("energy/episode_duration_s", final_clock)
                     self.logger.record("energy/episode_duration_h", episode_duration_h)

                     # Log summary to console for visibility
                     if self.verbose > 0:
                         logger.info(f"  Energy Summary - Total: {final_total_wh:.2f} Wh, Green: {green_ratio*100:.1f}%, Wasted: {final_wasted_wh:.2f} Wh")

                 # Let SB3 core handle logger.dump to avoid resetting CSV headers early


            # --- Check for new best model based on Monitor's smoothed reward ---
            if self.log_dir:
                # Load Monitor results
                try:
                    monitor_results = load_results(self.log_dir)
                    if len(monitor_results) > 0:
                         # Get the mean reward (usually smoothed over 100 episodes by Monitor)
                        mean_reward = np.mean(monitor_results.r.iloc[-100:]) # Use last 100 episodes rolling mean
                        if self.verbose > 0:
                            logger.info(f"Episode {self.current_episode_num_monitor} finished.")
                            logger.info(f"Num timesteps: {self.num_timesteps}")
                            logger.info(f"Best mean reward: {self.best_mean_reward:.3f} - Last mean reward per episode: {mean_reward:.3f}")

                        # New best model found
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            if self.verbose > 0:
                                logger.info(f"New best mean reward: {self.best_mean_reward:.3f}")
                            self._save_best_model()
                            # Save details of this potentially best episode
                            self._save_best_episode_details(self.current_episode_num_monitor)

                except Exception as e:
                    logger.error(f"Error loading Monitor results or saving best model: {e}", exc_info=True)

            # Clear lists for the next episode
            self._clear_episode_details()

        return True # Continue training
