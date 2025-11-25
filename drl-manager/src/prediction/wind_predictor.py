"""
Multi-Turbine Wind Power Predictor using CViTRNN

This module provides a predictor class for the 13-feature CViTRNN model.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add SWF_Prediction models to path
SWF_PREDICTION_PATH = Path('D:/SWF_Prediction')
if SWF_PREDICTION_PATH.exists():
    sys.path.insert(0, str(SWF_PREDICTION_PATH))
    from models.CViTRNN import CViTRNN
else:
    print(f"Warning: SWF_Prediction not found at {SWF_PREDICTION_PATH}")


class MultiTurbinePredictor:
    """13-feature Multi-Turbine Wind Power Predictor"""

    def __init__(self, checkpoint_path: str, scalers_path: str, data_path: str, device: str = 'cpu'):
        """
        Initialize the predictor.

        Args:
            checkpoint_path: Path to model checkpoint (.pth)
            scalers_path: Path to scalers (.pkl)
            data_path: Path to npz data file (contains turbine_positions)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device

        # 1. Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 2。 Build and load model
        self.model = self._build_model()
        self.model.eval()

        # 3. Load scalers
        print(f"Loading scalers: {scalers_path}")
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)

        # 4. Load turbine positions from npz data file
        print(f"Loading turbine positions from: {data_path}")
        data = np.load(data_path, allow_pickle=True)
        self.turbine_positions = data['turbine_positions'].item()  # Convert from numpy array to dict

        # 5. select 13 features (matching training)
        self.feature_columns = [
            'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
            'Pab1', 'Prtv', 'T2m',
            'Sp', 'RelH', 'Wspd_w', 'Wdir_w',
            'Patv'
        ]

        # 6. Patv is the last feature (index 12)
        self.patv_idx = 12

        # 7. Get grid shape and features from checkpoint
        model_cfg = self.checkpoint['model_config']
        if 'spatial_size' in model_cfg:
            self.grid_shape = model_cfg['spatial_size']
        elif 'grid_shape' in model_cfg:
            self.grid_shape = model_cfg['grid_shape']
        else:
            self.grid_shape = (28, 6)

        if 'in_channels' in model_cfg:
            self.num_features = model_cfg['in_channels']
        elif 'num_features' in model_cfg:
            self.num_features = model_cfg['num_features']
        else:
            self.num_features = 13

        print(f"Loaded {len(self.turbine_positions)} turbine positions")

        print(f"Model loaded successfully:")
        print(f"  Features: {self.num_features} ({', '.join(self.feature_columns)})")
        print(f"  Grid shape: {self.grid_shape}")
        print(f"  Patv channel index: {self.patv_idx}")

    def _build_model(self):
        """Build CViTRNN model from checkpoint config"""
        config = self.checkpoint['model_config']

        # Handle different checkpoint formats
        # Newer format: spatial_size, in_channels
        # Older format: grid_shape, num_features
        model_config = {}

        if 'spatial_size' in config:
            model_config['spatial_size'] = config['spatial_size']
        elif 'grid_shape' in config:
            model_config['spatial_size'] = config['grid_shape']
        else:
            model_config['spatial_size'] = (28, 6)  # Default

        if 'in_channels' in config:
            model_config['in_channels'] = config['in_channels']
        elif 'num_features' in config:
            model_config['in_channels'] = config['num_features']
        else:
            model_config['in_channels'] = 13  # Default for 13-feature model

        # Copy other required config parameters
        model_config['embed_dim'] = config['embed_dim']
        model_config['num_cells'] = config['num_cells']
        model_config['num_heads'] = config['num_heads']
        model_config['num_encoder_layers'] = config['num_encoder_layers']
        model_config['dropout'] = 0.0  # No dropout during inference

        # CViTRNN expects a config dict
        model = CViTRNN(model_config)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.to(self.device)
        return model

    def predict_from_frames(self, frames: np.ndarray, horizon: int = 8) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Predict future power from spatial frames.

        Args:
            frames: Input frames (lookback, H, W, C)
            horizon: Prediction horizon (default 8)

        Returns:
            predictions_kw: Predicted power in kW (horizon, H, W)
            turbine_predictions: Dict {turbine_id: np.array([p_t0, ..., p_t_horizon])}
        """
        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(frames).unsqueeze(0).float().to(self.device)
        # Shape: (1, lookback, H, W, C)

        # Run inference
        # CViTRNN forward returns (warm_up_outputs, predictions)
        with torch.no_grad():
            _, predictions = self.model(input_tensor, target_len=horizon)
        # Shape: (1, horizon, H, W, C)

        # Extract Patv channel and denormalize
        predictions_kw = self._denormalize_predictions(predictions)

        # Extract by turbine
        turbine_predictions = self._extract_by_turbine(predictions_kw)

        return predictions_kw, turbine_predictions

    def _denormalize_predictions(self, predictions: torch.Tensor) -> np.ndarray:
        """Denormalize Patv predictions to kW"""
        pred_np = predictions.squeeze(0).cpu().numpy()  # (horizon, H, W, C)
        patv_pred = pred_np[:, :, :, self.patv_idx]  # (horizon, H, W)

        # Denormalize
        H, W = self.grid_shape
        horizon = pred_np.shape[0]
        pred_kw = np.zeros((horizon, H, W))

        patv_scaler = self.scalers['Patv']
        for t in range(horizon):
            for h in range(H):
                for w in range(W):
                    pred_kw[t, h, w] = patv_scaler.inverse_transform(
                        [[patv_pred[t, h, w]]]
                    )[0, 0]

        return pred_kw

    def _extract_by_turbine(self, pred_kw: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract predictions by turbine ID"""
        turbine_preds = {}
        for turbine_id, (h, w) in self.turbine_positions.items():
            turbine_preds[turbine_id] = pred_kw[:, h, w]
        return turbine_preds

    def load_test_data(self, test_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data from npz file.

        Args:
            test_data_path: Path to sdwpf_data.npz

        Returns:
            test_frames: Test input frames (N, H, W, C)
            test_targets: Test target frames (N, H, W, C)
        """
        print(f"Loading test data: {test_data_path}")
        data = np.load(test_data_path)

        test_frames = data['test_frames']  # (N, H, W, C)

        # If test_targets exist, load them
        if 'test_targets' in data:
            test_targets = data['test_targets']
        else:
            # For old format, targets are not separate
            test_targets = None

        print(f"Test frames shape: {test_frames.shape}")
        if test_targets is not None:
            print(f"Test targets shape: {test_targets.shape}")

        return test_frames, test_targets

    def evaluate(self, frames: np.ndarray, ground_truth: np.ndarray, horizon: int = 8) -> Dict[str, float]:
        """
        Evaluate prediction performance.

        Args:
            frames: Input frames (lookback, H, W, C)
            ground_truth: Ground truth future frames (horizon, H, W, C)
            horizon: Prediction horizon

        Returns:
            metrics: Dict with RMSE, MAE, R2
        """
        # Predict
        predictions_kw, _ = self.predict_from_frames(frames, horizon)

        # Extract ground truth Patv and denormalize
        gt_patv = ground_truth[:, :, :, self.patv_idx]
        H, W = self.grid_shape
        gt_kw = np.zeros_like(gt_patv)

        patv_scaler = self.scalers['Patv']
        for t in range(horizon):
            for h in range(H):
                for w in range(W):
                    gt_kw[t, h, w] = patv_scaler.inverse_transform(
                        [[gt_patv[t, h, w]]]
                    )[0, 0]

        # Calculate metrics
        mse = np.mean((predictions_kw - gt_kw) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_kw - gt_kw))

        # R2 score
        ss_res = np.sum((gt_kw - predictions_kw) ** 2)
        ss_tot = np.sum((gt_kw - np.mean(gt_kw)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


def find_latest_checkpoint(experiments_dir: str = 'D:/SWF_Prediction/experiments') -> Optional[str]:
    """
    Find the latest model checkpoint.

    Args:
        experiments_dir: Directory containing experiment folders

    Returns:
        Path to best_model.pth or None if not found
    """
    exp_path = Path(experiments_dir)
    if not exp_path.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return None

    # Find all improved_cvitrnn_* directories
    exp_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('improved_cvitrnn_')],
                     reverse=True)

    if not exp_dirs:
        print("No experiment directories found")
        return None

    # Check for best_model.pth in latest directory
    latest_dir = exp_dirs[0]
    checkpoint_path = latest_dir / 'best_model.pth'

    if checkpoint_path.exists():
        print(f"Found checkpoint: {latest_dir.name}/best_model.pth")
        return str(checkpoint_path)
    else:
        print(f"No best_model.pth found in {latest_dir.name}")
        return None
