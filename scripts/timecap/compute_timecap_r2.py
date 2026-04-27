#!/usr/bin/env python3
"""
Compute R^2 (coefficient of determination) and verify MSE/MAE/RMSE on the
TimeCAP test-set predictions saved by exp_TimeCAP.Inference().

Reads pred.npy / true.npy via mmap and accumulates statistics in chunks,
so it works on the full 1.4M-sample test set without loading both arrays
into RAM.

Usage:
    python compute_timecap_r2.py [TEST_DIR] [--chunk-rows N]

TEST_DIR defaults to:
    drl-manager/timecap_prediction/TimeCAP/test/finetune_TimeCAP_custom_sl96
"""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_DIR = Path(
    "drl-manager/timecap_prediction/TimeCAP/test/finetune_TimeCAP_custom_sl96"
)


def streaming_mean(arr_mm, chunk_rows: int) -> float:
    """Compute global mean over a 3D mmapped array using a one-pass sum."""
    n = arr_mm.shape[0]
    elements_per_row = int(np.prod(arr_mm.shape[1:]))
    total = 0.0
    count = 0
    for s in range(0, n, chunk_rows):
        block = np.asarray(arr_mm[s:s + chunk_rows], dtype=np.float64)
        total += block.sum()
        count += block.size
    assert count == n * elements_per_row
    return total / count


def streaming_metrics(pred_mm, true_mm, true_mean: float, chunk_rows: int):
    """Single pass: SSE, SAE, SST, max_abs_err."""
    assert pred_mm.shape == true_mm.shape, (
        f"shape mismatch: pred={pred_mm.shape} true={true_mm.shape}"
    )
    n = pred_mm.shape[0]
    sse = 0.0   # sum (true-pred)^2  -> MSE
    sae = 0.0   # sum |true-pred|    -> MAE
    sst = 0.0   # sum (true - mean)^2 -> R^2 denominator
    max_abs = 0.0
    count = 0

    for s in range(0, n, chunk_rows):
        p = np.asarray(pred_mm[s:s + chunk_rows], dtype=np.float64)
        t = np.asarray(true_mm[s:s + chunk_rows], dtype=np.float64)
        diff = t - p
        sse += float(np.sum(diff * diff))
        sae += float(np.sum(np.abs(diff)))
        dm = t - true_mean
        sst += float(np.sum(dm * dm))
        block_max = float(np.max(np.abs(diff)))
        if block_max > max_abs:
            max_abs = block_max
        count += t.size

    mse = sse / count
    mae = sae / count
    rmse = np.sqrt(mse)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {
        "n_elements": count,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "max_abs_error": max_abs,
        "SSE": sse,
        "SST": sst,
        "true_mean": true_mean,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test_dir", nargs="?", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--chunk-rows", type=int, default=20000,
                    help="Rows per chunk; ~20k × 144 × 1 ≈ 23 MB float64.")
    args = ap.parse_args()

    pred_path = args.test_dir / "pred.npy"
    true_path = args.test_dir / "true.npy"
    metrics_path = args.test_dir / "metrics.npy"

    for p in (pred_path, true_path):
        if not p.is_file():
            raise SystemExit(f"Missing file: {p}")

    print(f"Loading (mmap):")
    print(f"  pred: {pred_path}  ({pred_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  true: {true_path}  ({true_path.stat().st_size / 1e9:.2f} GB)")

    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")
    print(f"  pred.shape={pred.shape} dtype={pred.dtype}")
    print(f"  true.shape={true.shape} dtype={true.dtype}")

    # Some checkpoints save pred with all encoder channels; trim to last
    # channel so it matches the (uni-channel) ground truth target.
    if pred.shape[-1] != true.shape[-1]:
        if pred.shape[-1] >= true.shape[-1] and pred.shape[:-1] == true.shape[:-1]:
            keep = true.shape[-1]
            print(f"[info] pred has {pred.shape[-1]} channels, true has {keep}. "
                  f"Slicing pred[..., -{keep}:] to align.")
            pred = pred[..., -keep:]
        else:
            raise SystemExit(
                f"Cannot reconcile shapes: pred={pred.shape} true={true.shape}"
            )

    print("\n[pass 1/2] computing global mean of `true` ...")
    t_mean = streaming_mean(true, args.chunk_rows)
    print(f"  true.mean() = {t_mean:.6f}")

    print("\n[pass 2/2] computing SSE / SAE / SST ...")
    m = streaming_metrics(pred, true, t_mean, args.chunk_rows)

    print("\n" + "=" * 60)
    print(f"Test set elements : {m['n_elements']:,}")
    print(f"true mean         : {m['true_mean']:.6f}")
    print(f"MSE               : {m['MSE']:.6f}")
    print(f"RMSE              : {m['RMSE']:.6f}")
    print(f"MAE               : {m['MAE']:.6f}")
    print(f"R^2               : {m['R2']:.6f}")
    print(f"max |error|       : {m['max_abs_error']:.6f}")
    print("=" * 60)

    # Cross-check with the saved metrics.npy from training
    if metrics_path.is_file():
        saved = np.load(metrics_path)
        # metric() returns: mae, mse, rmse, mape, mspe
        print("\nSaved metrics.npy [mae, mse, rmse, mape, mspe]:")
        print(f"  MAE  = {saved[0]:.6f}")
        print(f"  MSE  = {saved[1]:.6f}")
        print(f"  RMSE = {saved[2]:.6f}")
        print(f"  MAPE = {saved[3]:.6f}")
        print(f"  MSPE = {saved[4]:.6f}")

        if abs(saved[1] - m["MSE"]) < 1e-3 and abs(saved[0] - m["MAE"]) < 1e-3:
            print("  ✓ matches recomputed MSE/MAE")
        else:
            print("  ✗ discrepancy between recomputed and saved metrics")


if __name__ == "__main__":
    main()
