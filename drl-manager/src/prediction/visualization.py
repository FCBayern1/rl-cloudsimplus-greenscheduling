"""
Visualization tools for wind power predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import matplotlib.gridspec as gridspec


def plot_predictions(
    turbine_predictions: Dict[int, np.ndarray],
    turbine_ground_truth: Dict[int, np.ndarray],
    turbine_ids: List[int] = [1, 57, 124],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot predicted vs actual power for multiple turbines.

    Args:
        turbine_predictions: Dict {turbine_id: predictions (horizon,)}
        turbine_ground_truth: Dict {turbine_id: ground_truth (horizon,)}
        turbine_ids: List of turbine IDs to plot
        save_path: Path to save the figure
        show: Whether to show the plot
    """
    num_turbines = len(turbine_ids)
    horizon = len(turbine_predictions[turbine_ids[0]])

    fig = plt.figure(figsize=(15, 4 * num_turbines))
    gs = gridspec.GridSpec(num_turbines, 2, figure=fig, width_ratios=[2, 1])

    for idx, turbine_id in enumerate(turbine_ids):
        # Get data
        pred = turbine_predictions[turbine_id]
        gt = turbine_ground_truth[turbine_id]
        timesteps = np.arange(horizon)

        # Left plot: Time series
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(timesteps, gt, 'b-o', label='Ground Truth', linewidth=2, markersize=6)
        ax1.plot(timesteps, pred, 'r--s', label='Prediction', linewidth=2, markersize=6, alpha=0.7)
        ax1.set_xlabel('Future Timestep', fontsize=12)
        ax1.set_ylabel('Power (kW)', fontsize=12)
        ax1.set_title(f'Turbine {turbine_id} - Power Prediction', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Calculate metrics
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        mae = np.mean(np.abs(pred - gt))
        mape = np.mean(np.abs((gt - pred) / (gt + 1e-6))) * 100

        # Add metrics text
        metrics_text = f'RMSE: {rmse:.2f} kW\nMAE: {mae:.2f} kW\nMAPE: {mape:.2f}%'
        ax1.text(0.02, 0.98, metrics_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)

        # Right plot: Scatter plot
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.scatter(gt, pred, alpha=0.6, s=100, edgecolors='black', linewidth=1)

        # Perfect prediction line
        min_val = min(gt.min(), pred.min())
        max_val = max(gt.max(), pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction', linewidth=2)

        ax2.set_xlabel('Ground Truth (kW)', fontsize=12)
        ax2.set_ylabel('Prediction (kW)', fontsize=12)
        ax2.set_title(f'Turbine {turbine_id} - Prediction Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_error_distribution(
    turbine_predictions: Dict[int, np.ndarray],
    turbine_ground_truth: Dict[int, np.ndarray],
    turbine_ids: List[int] = [1, 57, 124],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot error distribution across turbines.

    Args:
        turbine_predictions: Dict {turbine_id: predictions (horizon,)}
        turbine_ground_truth: Dict {turbine_id: ground_truth (horizon,)}
        turbine_ids: List of turbine IDs to analyze
        save_path: Path to save the figure
        show: Whether to show the plot
    """
    fig, axes = plt.subplots(1, len(turbine_ids), figsize=(6 * len(turbine_ids), 5))

    if len(turbine_ids) == 1:
        axes = [axes]

    for idx, turbine_id in enumerate(turbine_ids):
        pred = turbine_predictions[turbine_id]
        gt = turbine_ground_truth[turbine_id]
        errors = pred - gt

        # Histogram
        axes[idx].hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[idx].set_xlabel('Prediction Error (kW)', fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].set_title(f'Turbine {turbine_id} - Error Distribution', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        stats_text = f'Mean: {mean_error:.2f} kW\nStd: {std_error:.2f} kW'
        axes[idx].text(0.02, 0.98, stats_text,
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                      fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_multiple_samples(
    all_predictions: List[Dict[int, np.ndarray]],
    all_ground_truth: List[Dict[int, np.ndarray]],
    turbine_id: int = 1,
    num_samples: int = 3,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot multiple prediction samples for a single turbine.

    Args:
        all_predictions: List of prediction dicts
        all_ground_truth: List of ground truth dicts
        turbine_id: Turbine ID to plot
        num_samples: Number of samples to show
        save_path: Path to save the figure
        show: Whether to show the plot
    """
    num_samples = min(num_samples, len(all_predictions))

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for idx in range(num_samples):
        pred = all_predictions[idx][turbine_id]
        gt = all_ground_truth[idx][turbine_id]
        horizon = len(pred)
        timesteps = np.arange(horizon)

        axes[idx].plot(timesteps, gt, 'b-o', label='Ground Truth', linewidth=2, markersize=6)
        axes[idx].plot(timesteps, pred, 'r--s', label='Prediction', linewidth=2, markersize=6, alpha=0.7)
        axes[idx].set_xlabel('Future Timestep', fontsize=12)
        axes[idx].set_ylabel('Power (kW)', fontsize=12)
        axes[idx].set_title(f'Sample {idx+1} - Turbine {turbine_id}', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=11)
        axes[idx].grid(True, alpha=0.3)

        # Metrics
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        mae = np.mean(np.abs(pred - gt))
        metrics_text = f'RMSE: {rmse:.2f} kW | MAE: {mae:.2f} kW'
        axes[idx].text(0.98, 0.02, metrics_text,
                      transform=axes[idx].transAxes,
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple samples plot saved to: {save_path}")

    if show:
        plt.show()

    return fig
