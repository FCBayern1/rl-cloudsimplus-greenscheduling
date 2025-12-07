"""
Wind Power Prediction Service for RL Training

This module provides rolling wind power predictions integrated with the RL environment.
Each datacenter gets independent predictions based on its assigned turbine.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

from .wind_predictor import MultiTurbinePredictor
from .csv_feature_loader import CSVFeatureLoader

logger = logging.getLogger(__name__)


class WindPredictionService:
    """
    Manages rolling wind power predictions for multiple datacenters.

    Features:
    - Maintains separate history buffers for each datacenter
    - Performs rolling predictions (predict future 8 steps every step)
    - Handles feature loading and normalization
    - Caches predictions to avoid redundant computation
    """

    def __init__(
        self,
        predictor: MultiTurbinePredictor,
        num_datacenters: int,
        turbine_ids: List[int],
        feature_loader: CSVFeatureLoader,
        prediction_horizon: int = 8,
        history_length: int = 12,
        cache_predictions: bool = True
    ):
        """
        Initialize the prediction service.

        Args:
            predictor: Trained MultiTurbinePredictor instance
            num_datacenters: Number of datacenters
            turbine_ids: List of turbine IDs for each datacenter (e.g., [1, 57, 124])
            feature_loader: CSVFeatureLoader for complete 13-feature input (REQUIRED)
            prediction_horizon: Number of future steps to predict
            history_length: Number of historical steps required (should be 12 for CViTRNN)
            cache_predictions: Whether to cache predictions to avoid redundant calls
        """
        self.predictor = predictor
        self.num_datacenters = num_datacenters
        self.turbine_ids = turbine_ids
        self.prediction_horizon = prediction_horizon
        self.history_length = history_length
        self.cache_predictions = cache_predictions
        self.feature_loader = feature_loader

        # Validate turbine IDs
        if len(turbine_ids) != num_datacenters:
            raise ValueError(
                f"Number of turbine IDs ({len(turbine_ids)}) must match "
                f"number of datacenters ({num_datacenters})"
            )

        # Prediction cache: {dc_id: (last_update_time, predictions_array)}
        self.prediction_cache = {}

        # Step counter for cache validation
        self.current_step = 0

        logger.info(
            f"WindPredictionService initialized: {num_datacenters} DCs, "
            f"turbines {turbine_ids}, horizon={prediction_horizon}, "
            f"using 13-feature CSV mode"
        )

    def predict_future_power(
        self,
        dc_id: int,
        current_time: float
    ) -> np.ndarray:
        """
        Predict future green power for a datacenter using CSV features.

        Args:
            dc_id: Datacenter ID (0-indexed)
            current_time: Current simulation time in seconds (required)

        Returns:
            Array of predicted power in Watts, shape (prediction_horizon,)
            Returns zeros if insufficient CSV data.
        """
        # Get turbine ID for this datacenter
        turbine_id = self.turbine_ids[dc_id]

        # Check cache
        if self.cache_predictions and dc_id in self.prediction_cache:
            cached_step, cached_predictions = self.prediction_cache[dc_id]
            if cached_step == self.current_step:
                logger.debug(f"DC {dc_id}: Using cached predictions")
                return cached_predictions

        # Create spatial frames for prediction from CSV
        try:
            frames = self._create_prediction_frames(turbine_id, current_time)

            if frames is None:
                # Insufficient CSV data
                return np.zeros(self.prediction_horizon, dtype=np.float32)

            # Run prediction
            predictions_kw, _ = self.predictor.predict_from_frames(
                frames,
                horizon=self.prediction_horizon
            )

            # Extract predictions for this turbine
            h, w = self.predictor.turbine_positions[turbine_id]
            turbine_predictions_kw = predictions_kw[:, h, w]

            # Convert kW to W
            turbine_predictions_w = turbine_predictions_kw * 1000.0

            # Cache the result
            if self.cache_predictions:
                self.prediction_cache[dc_id] = (self.current_step, turbine_predictions_w)

            logger.debug(
                f"DC {dc_id} (Turbine {turbine_id}): Predicted {turbine_predictions_w.mean():.2f} W "
                f"(range: {turbine_predictions_w.min():.2f}-{turbine_predictions_w.max():.2f})"
            )

            return turbine_predictions_w

        except Exception as e:
            logger.error(f"DC {dc_id}: Prediction failed: {e}", exc_info=True)
            return np.zeros(self.prediction_horizon, dtype=np.float32)

    def _create_prediction_frames(
        self,
        turbine_id: int,
        current_time: float
    ) -> Optional[np.ndarray]:
        """
        Create spatial frames for prediction from CSV features.

        Args:
            turbine_id: Turbine ID for spatial positioning
            current_time: Current simulation time (seconds)

        Returns:
            Spatial frames, shape (history_length, H, W, C), or None if insufficient data
        """
        H, W = self.predictor.grid_shape
        C = self.predictor.num_features  # Should be 13

        # Get turbine position
        h, w = self.predictor.turbine_positions[turbine_id]

        # Load historical features from CSV
        features = self.feature_loader.get_historical_features(
            turbine_id=turbine_id,
            current_sim_time=current_time,
            lookback_steps=self.history_length
        )

        if features is None:
            logger.warning(
                f"Turbine {turbine_id}: Insufficient CSV data at sim_time={current_time}s"
            )
            return None

        # Create empty frames
        frames = np.zeros((self.history_length, H, W, C), dtype=np.float32)

        # features shape: (history_length, 13)
        # Normalize and fill into frames
        for t in range(self.history_length):
            for feat_idx, feat_name in enumerate(self.feature_loader.feature_columns):
                if feat_name in self.predictor.scalers:
                    # Normalize feature
                    normalized_value = self.predictor.scalers[feat_name].transform(
                        [[features[t, feat_idx]]]
                    )[0, 0]
                    frames[t, h, w, feat_idx] = normalized_value
                else:
                    # Feature not in scalers, use raw value
                    logger.warning(f"Feature {feat_name} not in scalers, using raw value")
                    frames[t, h, w, feat_idx] = features[t, feat_idx]

        logger.debug(
            f"Turbine {turbine_id}: Loaded 13 features from CSV at sim_time={current_time}s"
        )

        return frames

    def predict_all_datacenters(self, current_time: float) -> np.ndarray:
        """
        Predict future power for all datacenters.

        Args:
            current_time: Current simulation time in seconds (required)

        Returns:
            Array of predictions, shape (num_datacenters, prediction_horizon)
        """
        predictions = np.zeros(
            (self.num_datacenters, self.prediction_horizon),
            dtype=np.float32
        )

        for dc_id in range(self.num_datacenters):
            predictions[dc_id] = self.predict_future_power(dc_id, current_time)

        return predictions

    def step(self):
        """
        Advance to next timestep. Clears prediction cache.
        """
        self.current_step += 1
        if self.cache_predictions:
            self.prediction_cache.clear()

    def reset(self):
        """
        Reset service (clear cache).
        Called at episode start.
        """
        self.prediction_cache.clear()
        self.current_step = 0

        logger.info("WindPredictionService reset")
