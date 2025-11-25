"""
Wind Prediction Wrapper for Hierarchical Multi-DC Environment

Adds future wind power predictions to observations, giving the RL agent
forecasting capabilities for better decision-making.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, List, Optional
import logging

from src.prediction.wind_prediction_service import WindPredictionService
from src.prediction.wind_predictor import MultiTurbinePredictor
from src.prediction.csv_feature_loader import CSVFeatureLoader

logger = logging.getLogger(__name__)


class WindPredictionWrapper(gym.Wrapper):
    """
    Wraps HierarchicalMultiDCEnv to add wind power predictions to observations.

    The wrapper:
    1. Maintains history buffers for each datacenter's green power
    2. Runs rolling predictions (every step predicts next N steps)
    3. Adds 'dc_predicted_green_power_w' to global observation
    4. Updates observation_space accordingly
    """

    def __init__(
        self,
        env: gym.Env,
        model_checkpoint: str,
        scalers_path: str,
        data_path: str,
        turbine_ids: List[int],
        turbine_csv_paths: Dict[int, str],
        prediction_horizon: int = 8,
        device: str = 'cpu',
        enable_logging: bool = True,
        csv_start_offset: int = 12
    ):
        """
        Initialize the wrapper.

        Args:
            env: Base HierarchicalMultiDCEnv to wrap
            model_checkpoint: Path to trained CViTRNN model checkpoint
            scalers_path: Path to feature scalers
            data_path: Path to npz data file (contains turbine_positions)
            turbine_ids: List of turbine IDs for each datacenter (e.g., [1, 57, 124])
            turbine_csv_paths: Dict mapping turbine_id → CSV file path (REQUIRED)
                               Example: {1: "path/to/turbine_001.csv", 57: "..."}
            prediction_horizon: Number of future steps to predict
            device: Device for model inference ('cpu' or 'cuda')
            enable_logging: Whether to log prediction statistics
            csv_start_offset: CSV row offset (default: 12, Java skips first 12 rows)
        """
        super().__init__(env)

        self.prediction_horizon = prediction_horizon
        self.turbine_ids = turbine_ids
        self.enable_logging = enable_logging

        # Get number of datacenters from base environment
        self.num_datacenters = env.config.get('multi_dc', {}).get('num_datacenters', 3)

        # Validate turbine IDs
        if len(turbine_ids) != self.num_datacenters:
            raise ValueError(
                f"Number of turbine IDs ({len(turbine_ids)}) must match "
                f"number of datacenters ({self.num_datacenters})"
            )

        # Initialize predictor
        logger.info(f"Loading wind power prediction model from {model_checkpoint}")
        self.predictor = MultiTurbinePredictor(
            checkpoint_path=model_checkpoint,
            scalers_path=scalers_path,
            data_path=data_path,
            device=device
        )

        # Initialize CSVFeatureLoader (required)
        logger.info(
            f"Initializing CSVFeatureLoader with {len(turbine_csv_paths)} turbine CSVs "
            f"(start_offset={csv_start_offset})"
        )
        feature_loader = CSVFeatureLoader(
            turbine_csv_paths=turbine_csv_paths,
            csv_start_offset=csv_start_offset
        )
        logger.info("Using full 13-feature CSV prediction mode")

        # Initialize prediction service
        self.prediction_service = WindPredictionService(
            predictor=self.predictor,
            num_datacenters=self.num_datacenters,
            turbine_ids=turbine_ids,
            feature_loader=feature_loader,
            prediction_horizon=prediction_horizon,
            history_length=12,  # CViTRNN requires 12 timesteps
            cache_predictions=True
        )

        # Update observation space to include predictions
        self._update_observation_space()

        # Statistics for logging
        self.step_count = 0
        self.prediction_count = 0

        logger.info(
            f"WindPredictionWrapper initialized: "
            f"{self.num_datacenters} DCs, turbines {turbine_ids}, "
            f"horizon={prediction_horizon}"
        )

    def _update_observation_space(self):
        """
        Update observation space to include predicted green power.
        """
        # Get original observation space
        original_space = self.env.observation_space

        if isinstance(original_space, spaces.Dict):
            # Copy original spaces
            new_spaces = dict(original_space.spaces)

            # Add prediction space to global observations
            if 'global' in new_spaces and isinstance(new_spaces['global'], spaces.Dict):
                global_spaces = dict(new_spaces['global'].spaces)

                # Add predicted green power
                global_spaces['dc_predicted_green_power_w'] = spaces.Box(
                    low=0.0,
                    high=1000000.0,  # 1 MW max
                    shape=(self.num_datacenters, self.prediction_horizon),
                    dtype=np.float32
                )

                new_spaces['global'] = spaces.Dict(global_spaces)

                # Update observation space
                self.observation_space = spaces.Dict(new_spaces)

                logger.info(
                    f"Added 'dc_predicted_green_power_w' to observation space: "
                    f"shape=({self.num_datacenters}, {self.prediction_horizon})"
                )
            else:
                logger.warning("No 'global' dict found in observation space, keeping original")
                self.observation_space = original_space
        else:
            logger.warning("Observation space is not Dict type, keeping original")
            self.observation_space = original_space

    def reset(self, **kwargs):
        """
        Reset environment and prediction service.
        """
        # Reset base environment
        obs, info = self.env.reset(**kwargs)

        # Reset prediction service
        self.prediction_service.reset()
        self.step_count = 0
        self.prediction_count = 0

        # Add zero predictions (no history yet)
        obs = self._add_predictions_to_obs(obs, initialize=True)

        logger.info("WindPredictionWrapper reset")

        return obs, info

    def step(self, action):
        """
        Step environment and add wind power predictions to observation.
        """
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add predictions to observation
        obs = self._add_predictions_to_obs(obs, initialize=False)

        # Advance prediction service timestep (clears cache)
        self.prediction_service.step()

        self.step_count += 1

        # Log statistics periodically
        if self.enable_logging and self.step_count % 100 == 0:
            self._log_statistics()

        return obs, reward, terminated, truncated, info

    def _add_predictions_to_obs(
        self,
        obs: Dict[str, Any],
        initialize: bool = False
    ) -> Dict[str, Any]:
        """
        Add wind power predictions to observation.

        Args:
            obs: Original observation dict
            initialize: If True, return zeros (no history yet)

        Returns:
            Modified observation with predictions
        """
        if 'global' not in obs:
            logger.warning("No 'global' key in observation, skipping prediction")
            return obs

        if initialize:
            # No history yet, return zeros
            predictions = np.zeros(
                (self.num_datacenters, self.prediction_horizon),
                dtype=np.float32
            )
        else:
            # Extract current simulation time from observation
            current_sim_time = None
            if 'simulation_time' in obs['global']:
                current_sim_time = float(obs['global']['simulation_time'])

            # Get predictions for all datacenters
            predictions = self.prediction_service.predict_all_datacenters(
                current_time=current_sim_time
            )
            self.prediction_count += 1

        # Add to global observation
        obs['global']['dc_predicted_green_power_w'] = predictions

        return obs

    def _log_statistics(self):
        """
        Log prediction statistics for monitoring.
        """
        logger.info(f"=== Wind Prediction Statistics (Step {self.step_count}) ===")
        logger.info(f"Total predictions made: {self.prediction_count}")

        for dc_id in range(self.num_datacenters):
            turbine_id = self.turbine_ids[dc_id]
            logger.info(f"  DC {dc_id} (Turbine {turbine_id}): Using CSV features")

    def close(self):
        """
        Clean up resources.
        """
        logger.info("Closing WindPredictionWrapper")
        super().close()
