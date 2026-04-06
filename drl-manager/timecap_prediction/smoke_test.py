"""
Smoke test for TimeCAP_GreenPredictor — no Java env required.

Run from drl-manager/:
    python -m timecap_prediction.smoke_test \
        --checkpoint timecap_prediction/checkpoints/ckpt_best.pth \
        --csv /path/to/Turbine_1_2021.csv \
        --turbine-id 1

Without a trained checkpoint, use --dry-run to test the data pipeline only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Make sure drl-manager root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timecap_prediction.predictor import TimeCAP_GreenPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="", help="Path to ckpt_best.pth")
    p.add_argument("--csv", required=True, help="Path to turbine split CSV (13-feature)")
    p.add_argument("--turbine-id", type=int, default=1, help="Turbine ID")
    p.add_argument("--steps", type=int, default=110, help="Simulated env steps to run")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model load; test only CSV loading and feature computation",
    )
    return p.parse_args()


def dry_run(csv_path: str, turbine_id: int, steps: int):
    """Test CSVFeatureLoader and compute_god_eye_features without a real model."""
    print("\n=== DRY RUN (no model loaded) ===")
    from src.prediction.csv_feature_loader import CSVFeatureLoader
    from timecap_prediction.predictor import _DEFAULT_FEATURE_COLUMNS

    loader = CSVFeatureLoader(
        turbine_csv_paths={turbine_id: csv_path},
        csv_start_offset=12,
        feature_columns=_DEFAULT_FEATURE_COLUMNS,
    )

    df = loader.turbine_data.get(turbine_id)
    if df is None:
        print(f"ERROR: could not load turbine {turbine_id} from {csv_path}")
        sys.exit(1)

    print(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
    patv_max = float(df["Patv"].max())
    print(f"max Patv = {patv_max:.2f} kW")

    # Simulate a fake constant prediction and verify feature output
    from timecap_prediction.predictor import TimeCAP_GreenPredictor

    # Minimal stub — we only need compute_god_eye_features, which is a pure function
    fake_pred = np.random.rand(144).astype(np.float32) * patv_max
    # Build a minimal predictor-like object just to call compute_god_eye_features
    class _Stub:
        turbine_ids = [turbine_id]
        max_power_kw = {turbine_id: patv_max}
        short_term_steps = 3
        long_term_steps = 144

    stub = _Stub()
    feats = TimeCAP_GreenPredictor.compute_god_eye_features(stub, fake_pred, patv_max)
    print(f"\nFake prediction → God's Eye features:")
    for k, v in feats.items():
        print(f"  {k:30s}: {v:.4f}")
    print("\nDRY RUN passed.")


def full_run(checkpoint: str, csv_path: str, turbine_id: int, steps: int, device: str):
    print(f"\n=== FULL RUN  checkpoint={checkpoint} ===")
    predictor = TimeCAP_GreenPredictor(
        checkpoint_path=checkpoint,
        turbine_csv_paths={turbine_id: csv_path},
        device=device,
    )

    predictor.reset()

    for step in range(steps):
        predictor.update(step)

        if step < 5 or step % 20 == 0 or step == steps - 1:
            pred = predictor.predict()
            if pred is None:
                print(f"  step {step:4d}: predict() returned None (model error)")
                continue

            feats = predictor.compute_god_eye_features(pred)
            print(
                f"  step {step:4d} | "
                f"short_mean={feats['short_mean']:.3f}  "
                f"short_trend={feats['short_trend']:+.3f}  "
                f"long_mean={feats['long_mean']:.3f}  "
                f"peak_timing={feats['peak_timing']:.3f}  "
                f"| pred[0]={pred[0]:.1f} kW  pred[-1]={pred[-1]:.1f} kW"
            )

    print("\nFULL RUN passed.")


def main():
    args = parse_args()
    csv_path = args.csv

    if args.dry_run:
        dry_run(csv_path, args.turbine_id, args.steps)
    else:
        if not args.checkpoint:
            print("ERROR: --checkpoint is required unless --dry-run is set.")
            sys.exit(1)
        full_run(args.checkpoint, csv_path, args.turbine_id, args.steps, args.device)


if __name__ == "__main__":
    main()
