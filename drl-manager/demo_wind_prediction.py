"""
Wind Power Prediction Demo Script

This script demonstrates the complete inference pipeline for the 13-feature
CViTRNN wind power prediction model.

Features:
- Loads latest trained model
- Runs predictions on test data
- Evaluates performance metrics
- Visualizes results for multiple turbines

Usage:
    python demo_wind_prediction.py
    python demo_wind_prediction.py --num-samples 5 --turbines 1 57 124
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prediction import MultiTurbinePredictor, plot_predictions, plot_error_distribution
from prediction.wind_predictor import find_latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Wind Power Prediction Demo')
    parser.add_argument('--checkpoint', type=str,
                       default='D:/SWF_Prediction/experiments/improved_cvitrnn_20251115_184804/best_model.pth',
                       help='Path to model checkpoint (default: 2025-11-15 stable model)')
    parser.add_argument('--data', type=str,
                       default='D:/SWF_Prediction/Data/sdwpf_data.npz',
                       help='Path to test data')
    parser.add_argument('--scalers', type=str,
                       default='D:/SWF_Prediction/Data/sdwpf_scalers.pkl',
                       help='Path to scalers')
    parser.add_argument('--turbines', type=int, nargs='+', default=[1, 57, 124],
                       help='Turbine IDs to visualize')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Number of test samples to evaluate')
    parser.add_argument('--horizon', type=int, default=8,
                       help='Prediction horizon (timesteps)')
    parser.add_argument('--output-dir', type=str, default='examples',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')

    args = parser.parse_args()

    print("=" * 80)
    print(" " * 20 + "Wind Power Prediction Demo")
    print("=" * 80)
    print()

    # ========== Step 1: Find or load checkpoint ==========
    print(f"[1/5] Using checkpoint: {args.checkpoint}")

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please specify a valid checkpoint path with --checkpoint")
        return

    print()

    # ========== Step 2: Load model ==========
    print("[2/5] Loading model...")
    predictor = MultiTurbinePredictor(
        checkpoint_path=args.checkpoint,
        scalers_path=args.scalers,
        data_path=args.data,
        device=args.device
    )
    print()

    # ========== Step 3: Load test data ==========
    print("[3/5] Loading test data...")
    test_frames, test_targets = predictor.load_test_data(args.data)
    num_test_samples = test_frames.shape[0]
    print(f"Total test samples: {num_test_samples}")
    print()

    # ========== Step 4: Run predictions on samples ==========
    print(f"[4/5] Running predictions on {args.num_samples} samples...")
    print("-" * 80)

    all_predictions = []
    all_ground_truths = []
    overall_rmse = []
    overall_mae = []
    overall_r2 = []

    # Randomly select samples
    np.random.seed(42)
    sample_indices = np.random.choice(num_test_samples - args.horizon,
                                     size=min(args.num_samples, num_test_samples - args.horizon),
                                     replace=False)

    for sample_idx in sample_indices:
        # Get input (12 timesteps history)
        input_frame = test_frames[sample_idx]  # (H, W, C)

        # Get ground truth (8 timesteps future)
        # Note: If test_targets is available, use it. Otherwise, use next frames from test_frames
        if test_targets is not None:
            ground_truth = test_targets[sample_idx]  # (horizon, H, W, C)
        else:
            # Use subsequent frames as ground truth
            ground_truth = test_frames[sample_idx + 1: sample_idx + 1 + args.horizon]  # (horizon, H, W, C)

        # Add lookback dimension if needed
        # For CViTRNN, we need (lookback=12, H, W, C) as input
        # But sdwpf_data.npz might have single frames
        # Let's assume test_frames is already in correct format or we need to reconstruct

        # Check if we need to create lookback sequence
        if input_frame.ndim == 3:  # (H, W, C) - single frame
            # Need to create lookback sequence
            if sample_idx < 11:
                continue  # Skip if not enough history
            input_sequence = test_frames[sample_idx - 11: sample_idx + 1]  # (12, H, W, C)
        else:
            input_sequence = input_frame  # Already in sequence format

        # Predict
        pred_kw, turbine_preds = predictor.predict_from_frames(input_sequence, horizon=args.horizon)

        # Extract ground truth for selected turbines
        turbine_gt = {}
        for turbine_id in args.turbines:
            h, w = predictor.turbine_positions[turbine_id]
            gt_patv_norm = ground_truth[:, h, w, predictor.patv_idx]

            # Denormalize
            patv_scaler = predictor.scalers['Patv']
            gt_kw = np.array([
                patv_scaler.inverse_transform([[val]])[0, 0]
                for val in gt_patv_norm
            ])
            turbine_gt[turbine_id] = gt_kw

        all_predictions.append(turbine_preds)
        all_ground_truths.append(turbine_gt)

        # Calculate metrics for this sample
        sample_errors = []
        for turbine_id in args.turbines:
            pred = turbine_preds[turbine_id]
            gt = turbine_gt[turbine_id]
            error = np.sqrt(np.mean((pred - gt) ** 2))
            sample_errors.append(error)

        avg_rmse = np.mean(sample_errors)
        overall_rmse.append(avg_rmse)

        print(f"Sample {len(all_predictions):2d} | Avg RMSE: {avg_rmse:6.2f} kW")

    print("-" * 80)
    print(f"Overall Average RMSE: {np.mean(overall_rmse):.2f} kW")
    print()

    # ========== Step 5: Visualize results ==========
    print("[5/5] Generating visualizations...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot predictions for first sample
    if len(all_predictions) > 0:
        # Predictions comparison
        output_path = output_dir / 'prediction_comparison.png'
        plot_predictions(
            turbine_predictions=all_predictions[0],
            turbine_ground_truth=all_ground_truths[0],
            turbine_ids=args.turbines,
            save_path=str(output_path),
            show=False
        )

        # Error distribution
        output_path = output_dir / 'error_distribution.png'
        plot_error_distribution(
            turbine_predictions=all_predictions[0],
            turbine_ground_truth=all_ground_truths[0],
            turbine_ids=args.turbines,
            save_path=str(output_path),
            show=False
        )

        print(f"Visualizations saved to: {output_dir}/")
    else:
        print("No predictions generated. Check test data format.")

    print()
    print("=" * 80)
    print(" " * 25 + "Demo Complete!")
    print("=" * 80)

    # Print summary
    print("\nSummary:")
    print(f"  Model: {Path(args.checkpoint).parent.name}")
    print(f"  Test samples evaluated: {len(all_predictions)}")
    print(f"  Turbines analyzed: {args.turbines}")
    print(f"  Average RMSE: {np.mean(overall_rmse):.2f} kW")
    print(f"  Output directory: {output_dir}/")
    print()


if __name__ == '__main__':
    main()
