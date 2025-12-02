"""
RLlib callback for tracking green energy metrics during training.

This callback logs green energy waste and other metrics at the end of each episode,
allowing visualization of how the agent's policy improves green energy efficiency
over time.
"""

import os
import csv
import json
import ast
import logging
from typing import Dict, Optional
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy

logger = logging.getLogger(__name__)


def safe_convert_to_dict(obj, key_name="object"):
    """
    Safely convert an object to a dictionary, handling:
    - Python dict (return as-is)
    - JSON string (parse it)
    - Java Map (convert to dict)
    - None or other types (return empty dict)

    Args:
        obj: Object to convert
        key_name: Name of the object (for logging)

    Returns:
        Dictionary representation of the object
    """
    # Already a dict
    if isinstance(obj, dict):
        return obj

    # None or empty
    if not obj:
        return {}

    # Try parsing as string (Python dict repr or JSON)
    if isinstance(obj, str):
        # First try ast.literal_eval for Python dict string representation (with single quotes)
        # Example: "{'key': 'value'}" -> {'key': 'value'}
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, dict):
                logger.debug(f"[CALLBACK] Successfully parsed {key_name} from Python dict string")
                return parsed
            else:
                logger.warning(f"[CALLBACK] {key_name} parsed but not a dict: {type(parsed)}")
        except (ValueError, SyntaxError) as e:
            # Not a valid Python literal, try JSON next
            logger.debug(f"[CALLBACK] {key_name} not a Python literal, trying JSON: {e}")

        # Try JSON parsing (with double quotes)
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                logger.debug(f"[CALLBACK] Successfully parsed {key_name} from JSON string")
                return parsed
            else:
                logger.warning(f"[CALLBACK] {key_name} JSON parsed but not a dict: {type(parsed)}")
                return {}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[CALLBACK] Failed to parse {key_name} as JSON: {e}")
            logger.error(f"[CALLBACK] {key_name} value: {obj[:200]}...")  # First 200 chars
            return {}

    # Try converting Java Map to dict
    try:
        # Check if it's a Java object with dict-like methods
        if hasattr(obj, 'get') and hasattr(obj, 'keySet'):
            logger.debug(f"[CALLBACK] Converting {key_name} from Java Map to dict")
            result = {}
            for key in obj.keySet():
                result[key] = obj.get(key)
            return result
    except Exception as e:
        logger.error(f"[CALLBACK] Failed to convert {key_name} from Java Map: {e}")

    logger.warning(f"[CALLBACK] {key_name} has unexpected type: {type(obj)}")
    return {}


class GreenEnergyLoggerCallback(DefaultCallbacks):
    """
    RLlib callback for logging green energy metrics per episode.

    Logs:
    - Episode number
    - Total green energy wasted (Wh)
    - Total green energy used (Wh)
    - Total brown energy used (Wh)
    - Green energy ratio (0-1)
    - Waste ratio (wasted / available)
    - Episode reward
    - Episode length
    """

    def __init__(self, log_dir: str = None):
        super().__init__()
        self.episode_counter = 0
        self.log_dir = log_dir
        self.csv_file = None
        self.csv_initialized = False
        self.best_reward = float('-inf')
        self.best_episode_data = None

        # Initialize best_episode_file if log_dir is provided
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.best_episode_file = os.path.join(self.log_dir, "best_episode_details.csv")
            logger.info(f"[INIT] Log directory: {self.log_dir}")
        else:
            self.best_episode_file = None

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Called when an episode is done.

        Args:
            worker: Reference to the current rollout worker
            base_env: BaseEnv running the episode
            policies: Mapping of policy id to policy objects
            episode: Episode object that contains info about the episode
            env_index: Index of the environment
        """
        # Initialize CSV file on first episode (worker-safe)
        if not self.csv_initialized:
            logger.info(f"[CALLBACK DEBUG] Initializing CSV for worker {worker.worker_index}")
            self._init_csv(worker)
            self.csv_initialized = True
            logger.info(f"[CALLBACK DEBUG] CSV initialized. File: {self.csv_file}")

        # Extract green energy metrics from episode info
        # For multi-agent environments, we need to get info for a specific agent
        # All agents share the same info dict, so we can use any agent (e.g., "global_agent")
        last_info = None

        # Try to get info from global_agent first
        if hasattr(episode, 'last_info_for'):
            try:
                last_info = episode.last_info_for("global_agent")
            except (KeyError, TypeError):
                # If global_agent doesn't exist, try without agent_id
                last_info = episode.last_info_for()

        # Fallback: try to get from episode history
        if last_info is None or len(last_info) == 0:
            logger.error(f"[CALLBACK DEBUG] No info from last_info_for(). Trying episode history...")

            # Try to access episode's info history
            if hasattr(episode, 'agent_to_last_info'):
                agent_infos = episode.agent_to_last_info
                logger.info(f"[CALLBACK DEBUG] agent_to_last_info keys: {list(agent_infos.keys())}")

                # Get info from any agent (they all have the same info)
                if len(agent_infos) > 0:
                    first_agent = list(agent_infos.keys())[0]
                    last_info = agent_infos[first_agent]
                    logger.info(f"[CALLBACK DEBUG] Got info from agent: {first_agent}")

        if last_info is None or len(last_info) == 0:
            logger.error(f"[CALLBACK DEBUG] Failed to get episode info! Episode length: {episode.length}, Total reward: {episode.total_reward}")
            logger.error(f"[CALLBACK DEBUG] Episode attributes: {dir(episode)}")
            return

        logger.info(f"[CALLBACK DEBUG] Episode ended! Length: {episode.length}, Reward: {episode.total_reward}")
        logger.info(f"[CALLBACK DEBUG] last_info keys: {list(last_info.keys())}")

        # Get global energy stats from info (handle string/Java Map/dict)
        global_energy_stats_raw = last_info.get('global_energy_stats', {})
        logger.info(f"[CALLBACK DEBUG] global_energy_stats type: {type(global_energy_stats_raw)}")

        global_energy_stats = safe_convert_to_dict(global_energy_stats_raw, "global_energy_stats")

        if not global_energy_stats:
            logger.error(f"[CALLBACK DEBUG] No global_energy_stats in episode info")
            logger.error(f"[CALLBACK DEBUG] Available keys in last_info: {list(last_info.keys())}")
            return

        green_waste = global_energy_stats.get('total_wasted_green_wh', 0.0)
        green_used = global_energy_stats.get('total_green_energy_wh', 0.0)
        brown_used = global_energy_stats.get('total_brown_energy_wh', 0.0)
        total_energy = green_used + brown_used
        green_ratio = global_energy_stats.get('green_energy_ratio', 0.0)

        # Calculate waste ratio
        available_green = green_used + green_waste
        waste_ratio = green_waste / available_green if available_green > 0 else 0.0

        # Extract carbon emission metrics
        total_carbon_kg = global_energy_stats.get('total_carbon_emission_kg', 0.0)
        carbon_intensity = global_energy_stats.get('carbon_intensity_kg_per_kwh', 0.0)

        # Episode metrics
        episode_reward = episode.total_reward
        episode_length = episode.length

        # Extract per-agent rewards for hierarchical MARL analysis
        global_agent_reward = 0.0
        local_agent_rewards = {}  # Changed to dict for per-DC tracking

        # Access agent_rewards dict from episode
        # In RLlib, agent_rewards keys are tuples: (agent_id, policy_id)
        if hasattr(episode, 'agent_rewards'):
            agent_rewards_dict = episode.agent_rewards
            logger.info(f"[CALLBACK DEBUG] Agent rewards: {agent_rewards_dict}")

            # Extract global agent reward
            # Keys are tuples: (agent_id, policy_id)
            for (agent_id, policy_id), reward in agent_rewards_dict.items():
                if agent_id == 'global_agent':
                    global_agent_reward = reward
                    logger.info(f"[CALLBACK DEBUG] Global agent reward: {global_agent_reward}")
                elif agent_id.startswith('local_agent_'):
                    # Extract DC ID from agent_id (e.g., "local_agent_0" -> 0)
                    try:
                        dc_id = int(agent_id.split('_')[-1])
                        local_agent_rewards[dc_id] = reward
                        logger.info(f"[CALLBACK DEBUG] {agent_id} (DC {dc_id}) reward: {reward}")
                    except (ValueError, IndexError):
                        logger.warning(f"[CALLBACK DEBUG] Could not parse DC ID from {agent_id}")

        # Calculate average local agent reward
        local_agents_avg_reward = sum(local_agent_rewards.values()) / len(local_agent_rewards) if local_agent_rewards else 0.0
        logger.info(f"[CALLBACK DEBUG] Average local agent reward: {local_agents_avg_reward} (from {len(local_agent_rewards)} agents)")

        # Calculate task completion rate
        # Note: This relies on global_energy_stats having completed/created cloudlet counts
        # which should be added to the CloudSim info map in Java
        total_cloudlets_created = global_energy_stats.get('total_created_cloudlets', 0)
        total_cloudlets_finished = global_energy_stats.get('total_finished_cloudlets', 0)
        
        if total_cloudlets_created > 0:
            completion_rate = total_cloudlets_finished / total_cloudlets_created
        else:
            completion_rate = 0.0
            
        logger.info(f"[CALLBACK DEBUG] Cloudlets: Finished {total_cloudlets_finished} / Created {total_cloudlets_created} (Rate: {completion_rate:.2%})")

        # Increment episode counter
        self.episode_counter += 1

        # Track best episode
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_episode_data = {
                'episode': self.episode_counter,
                'reward': episode_reward,
                'length': episode_length,
                'green_waste_wh': green_waste,
                'green_used_wh': green_used,
                'brown_used_wh': brown_used,
                'total_energy_wh': total_energy,
                'green_ratio': green_ratio,
                'waste_ratio': waste_ratio,
                'total_carbon_kg': total_carbon_kg,
                'carbon_intensity_kg_per_kwh': carbon_intensity,
                'global_agent_reward': global_agent_reward,
                'local_agents_avg_reward': local_agents_avg_reward,
                'completion_rate': completion_rate
            }
            self._save_best_episode()

        # Add per-DC energy metrics for separate policy analysis
        dc_energy_metrics_raw = last_info.get('datacenter_energy_metrics', {})
        dc_energy_metrics = safe_convert_to_dict(dc_energy_metrics_raw, "datacenter_energy_metrics")

        # Store per-DC metrics for CSV output
        per_dc_mean_completion_times = {}
        per_dc_cloudlets_finished = {}

        if dc_energy_metrics:
            for dc_id_str, dc_metrics_raw in dc_energy_metrics.items():
                # Convert dc_id to int (Java returns as Integer object key)
                try:
                    dc_id = int(dc_id_str) if isinstance(dc_id_str, str) else dc_id_str
                except (ValueError, TypeError):
                    continue

                # Convert DC metrics to dict (handle string/Java Map/dict)
                dc_metrics = safe_convert_to_dict(dc_metrics_raw, f"dc_{dc_id}_metrics")

                # Extract DC metrics from dict
                dc_green = dc_metrics.get('cumulative_green_wh', 0.0)
                dc_brown = dc_metrics.get('cumulative_brown_wh', 0.0)
                dc_wasted = dc_metrics.get('total_wasted_green_wh', 0.0)
                dc_green_ratio = dc_metrics.get('green_energy_ratio', 0.0)

                # Extract cloudlet completion metrics (NEW)
                dc_cloudlets_finished = dc_metrics.get('cloudlets_finished', 0)
                dc_mean_completion_time = dc_metrics.get('mean_completion_time', 0.0)

                # Store for CSV output
                per_dc_mean_completion_times[dc_id] = dc_mean_completion_time
                per_dc_cloudlets_finished[dc_id] = dc_cloudlets_finished

                # Record per-DC metrics to TensorBoard
                # These will show up as "dc_0/green_used_wh", "dc_1/green_used_wh", etc.
                episode.custom_metrics[f"dc_{dc_id}/green_used_wh"] = dc_green
                episode.custom_metrics[f"dc_{dc_id}/brown_used_wh"] = dc_brown
                episode.custom_metrics[f"dc_{dc_id}/green_wasted_wh"] = dc_wasted
                episode.custom_metrics[f"dc_{dc_id}/green_ratio"] = dc_green_ratio
                episode.custom_metrics[f"dc_{dc_id}/total_energy_wh"] = dc_green + dc_brown
                episode.custom_metrics[f"dc_{dc_id}/cloudlets_finished"] = dc_cloudlets_finished
                episode.custom_metrics[f"dc_{dc_id}/mean_completion_time"] = dc_mean_completion_time

        # Write to monitor.csv (episode-by-episode metrics)
        try:
            # Determine number of DCs from available metrics
            num_dcs = max(len(local_agent_rewards), len(per_dc_mean_completion_times), 10)

            # Build row with base metrics
            row = [
                self.episode_counter,
                green_waste,
                green_used,
                brown_used,
                total_energy,
                green_ratio,
                waste_ratio,
                total_carbon_kg,
                carbon_intensity,
                episode_reward,
                episode_length,
                global_agent_reward,
                local_agents_avg_reward,
                completion_rate
            ]

            # Add per-DC local rewards (local_reward_0, local_reward_1, ..., local_reward_9)
            for dc_id in range(num_dcs):
                row.append(local_agent_rewards.get(dc_id, 0.0))

            # Add per-DC mean completion times (mean_completion_time_dc_0, ..., mean_completion_time_dc_9)
            for dc_id in range(num_dcs):
                row.append(per_dc_mean_completion_times.get(dc_id, 0.0))

            # Add per-DC cloudlets finished (cloudlets_finished_dc_0, ..., cloudlets_finished_dc_9)
            for dc_id in range(num_dcs):
                row.append(per_dc_cloudlets_finished.get(dc_id, 0))

            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to write to monitor CSV: {e}")

        # Add custom metrics to episode (will be aggregated by RLlib)
        episode.custom_metrics["green_waste_wh"] = green_waste
        episode.custom_metrics["green_used_wh"] = green_used
        episode.custom_metrics["brown_used_wh"] = brown_used
        episode.custom_metrics["green_ratio"] = green_ratio
        episode.custom_metrics["waste_ratio"] = waste_ratio
        episode.custom_metrics["total_carbon_kg"] = total_carbon_kg
        episode.custom_metrics["carbon_intensity_kg_per_kwh"] = carbon_intensity
        episode.custom_metrics["global_agent_reward"] = global_agent_reward
        episode.custom_metrics["local_agents_avg_reward"] = local_agents_avg_reward
        episode.custom_metrics["completion_rate"] = completion_rate

        # Log to console (only worker 0 to avoid spam)
        if worker.worker_index == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {self.episode_counter} finished:")
            logger.info(f"  Green Waste:  {green_waste:.2f} Wh")
            logger.info(f"  Green Used:   {green_used:.2f} Wh")
            logger.info(f"  Brown Used:   {brown_used:.2f} Wh")
            logger.info(f"  Green Ratio:  {green_ratio:.2%}")
            logger.info(f"  Waste Ratio:  {waste_ratio:.2%}")
            logger.info(f"  Episode Reward: {episode_reward:.2f}")
            logger.info(f"  Episode Length: {episode_length}")
            logger.info(f"{'='*60}\n")

    def _init_csv(self, worker: RolloutWorker):
        """
        Initialize CSV file with headers.

        Args:
            worker: RLlib worker (used to determine output directory)
        """
        # Get log directory
        if self.log_dir:
            log_dir = self.log_dir
        elif hasattr(worker, 'io_context') and hasattr(worker.io_context, 'log_dir'):
            log_dir = worker.io_context.log_dir
        else:
            log_dir = './logs'

        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # monitor.csv - episode-by-episode metrics (only worker 0)
        if worker.worker_index == 0:
            self.csv_file = os.path.join(log_dir, "monitor.csv")
            # best_episode_file already initialized in __init__
            if not hasattr(self, 'best_episode_file') or not self.best_episode_file:
                self.best_episode_file = os.path.join(log_dir, "best_episode_details.csv")
        else:
            # Other workers save to separate files
            self.csv_file = os.path.join(log_dir, f"monitor_worker{worker.worker_index}.csv")
            if not hasattr(self, 'best_episode_file'):
                self.best_episode_file = None

        # Write monitor.csv headers
        try:
            # Base headers
            headers = [
                'episode',
                'green_waste_wh',
                'green_used_wh',
                'brown_used_wh',
                'total_energy_wh',
                'green_ratio',
                'waste_ratio',
                'total_carbon_kg',
                'carbon_intensity_kg_per_kwh',
                'episode_reward',
                'episode_length',
                'global_agent_reward',
                'local_agents_avg_reward',
                'completion_rate'
            ]

            # Add per-DC local reward headers
            num_dcs = 10  # Default to 10 DCs
            for dc_id in range(num_dcs):
                headers.append(f'local_reward_{dc_id}')

            # Add per-DC mean completion time headers
            for dc_id in range(num_dcs):
                headers.append(f'mean_completion_time_dc_{dc_id}')

            # Add per-DC cloudlets finished headers
            for dc_id in range(num_dcs):
                headers.append(f'cloudlets_finished_dc_{dc_id}')

            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info(f"Initialized monitor.csv with {len(headers)} columns: {self.csv_file}")
        except Exception as e:
            logger.error(f"Failed to initialize monitor CSV: {e}")

        # Initialize best_episode_details.csv (only worker 0)
        if worker.worker_index == 0 and self.best_episode_file:
            try:
                with open(self.best_episode_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'episode',
                        'reward',
                        'length',
                        'green_waste_wh',
                        'green_used_wh',
                        'brown_used_wh',
                        'total_energy_wh',
                        'green_ratio',
                        'waste_ratio',
                        'total_carbon_kg',
                        'carbon_intensity_kg_per_kwh',
                        'global_agent_reward',
                        'local_agents_avg_reward',
                        'completion_rate'  # Added completion rate header
                    ])
                logger.info(f"Initialized best_episode_details.csv: {self.best_episode_file}")
            except Exception as e:
                logger.error(f"Failed to initialize best episode CSV: {e}")

    def _save_best_episode(self):
        """Save best episode details to CSV."""
        if not self.best_episode_file or not self.best_episode_data:
            return

        try:
            # Overwrite file with current best episode
            with open(self.best_episode_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    'episode',
                    'reward',
                    'length',
                    'green_waste_wh',
                    'green_used_wh',
                    'brown_used_wh',
                    'total_energy_wh',
                    'green_ratio',
                    'waste_ratio',
                    'total_carbon_kg',
                    'carbon_intensity_kg_per_kwh',
                    'global_agent_reward',
                    'local_agents_avg_reward'
                ])
                # Write data
                writer.writerow([
                    self.best_episode_data['episode'],
                    self.best_episode_data['reward'],
                    self.best_episode_data['length'],
                    self.best_episode_data['green_waste_wh'],
                    self.best_episode_data['green_used_wh'],
                    self.best_episode_data['brown_used_wh'],
                    self.best_episode_data['total_energy_wh'],
                    self.best_episode_data['green_ratio'],
                    self.best_episode_data['waste_ratio'],
                    self.best_episode_data['total_carbon_kg'],
                    self.best_episode_data['carbon_intensity_kg_per_kwh'],
                    self.best_episode_data['global_agent_reward'],
                    self.best_episode_data['local_agents_avg_reward']
                ])
            logger.info(f"Updated best episode: Episode {self.best_episode_data['episode']} with reward {self.best_reward:.2f}")
        except Exception as e:
            logger.error(f"Failed to save best episode: {e}")

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """
        Called at the end of Algorithm.train().

        NOTE:
        We now rely on RLlib/Ray Tune's built-in logging for policy/value losses,
        and do NOT write additional loss metrics into the result dict.

        This hook is kept only for optional console debugging, without touching
        TensorBoard tags such as policy_loss/vf_loss.

        Args:
            algorithm: Current algorithm instance
            result: Training result dict from the iteration
        """
        try:
            iteration = result.get("training_iteration", 0)

            # Optional: print loss stats for debugging without altering result dict
            learner_info = result.get("info", {}).get("learner", {})
            if not learner_info:
                return

            # Global agent (console only)
            if "global_policy" in learner_info:
                stats = learner_info["global_policy"].get("learner_stats", {})
                pl = stats.get("policy_loss", None)
                vl = stats.get("vf_loss", None)
                if pl is not None and vl is not None:
                    logger.info(f"[{iteration}] GlobalPolicy: policy_loss={pl:.5f}, value_loss={vl:.5f}")

            # Local agents (console only)
            for policy_id, policy_info in learner_info.items():
                if policy_id.startswith("local_policy_"):
                    stats = policy_info.get("learner_stats", {})
                    pl = stats.get("policy_loss", None)
                    vl = stats.get("vf_loss", None)
                    if pl is not None and vl is not None:
                        logger.info(f"[{iteration}] {policy_id}: policy_loss={pl:.5f}, value_loss={vl:.5f}")

        except Exception as e:
            logger.error(f"Failed to read learner stats: {e}", exc_info=True)
