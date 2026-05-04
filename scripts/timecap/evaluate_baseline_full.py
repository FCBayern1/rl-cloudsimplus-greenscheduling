#!/usr/bin/env python3
"""
Full evaluation of the saved TimeCAP baseline test artifacts.

Reads pred.npy / true.npy from the test directory (default: baseline_4358062),
denormalises from z-score back to kW using the train-portion (first 70%) Patv
mean/std, and reports:

  - Overall MAE / RMSE / R^2 in BOTH z-score and kW
  - Per-horizon decay at h = 1, 6, 24, 72, 144 (10 min steps each)
  - True-power bucket breakdown (low / medium / high regimes)
  - True-mean baseline R^2 (sanity floor)

No GPU required. Works on the existing 1.4M-window mmapped artifacts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = (
    REPO_ROOT / "drl-manager" / "timecap_prediction" / "TimeCAP" / "test"
    / "finetune_TimeCAP_custom_sl96_baseline_4358062"
)
DEFAULT_DATA_CSV = (
    REPO_ROOT / "drl-manager" / "timecap_prediction" / "data" / "turbines_all134_2021.csv"
)


def load_patv_stats(csv_path: Path):
    df = pd.read_csv(csv_path, usecols=["Patv"])
    n = len(df)
    n_train = int(n * 0.7)
    train_patv = df["Patv"].iloc[:n_train].to_numpy(dtype=np.float64)
    return float(train_patv.mean()), float(train_patv.std(ddof=0)), float(df["Patv"].max())


def streaming_overall(pred, true, true_mean_z, chunk_rows: int):
    n = pred.shape[0]
    sse = sae = sst = 0.0
    max_abs = 0.0
    count = 0
    for s in range(0, n, chunk_rows):
        p = np.asarray(pred[s:s + chunk_rows], dtype=np.float64)
        t = np.asarray(true[s:s + chunk_rows], dtype=np.float64)
        diff = t - p
        sse += float(np.sum(diff * diff))
        sae += float(np.sum(np.abs(diff)))
        dm = t - true_mean_z
        sst += float(np.sum(dm * dm))
        block_max = float(np.max(np.abs(diff)))
        if block_max > max_abs:
            max_abs = block_max
        count += t.size
    mse = sse / count
    return {
        "MSE": mse,
        "MAE": sae / count,
        "RMSE": np.sqrt(mse),
        "R2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "max_abs": max_abs,
        "n": count,
    }


def streaming_per_horizon(pred, true, horizons, chunk_rows: int):
    """Per-horizon SSE / SAE / SST sums (one bucket per requested horizon idx)."""
    n = pred.shape[0]
    H = pred.shape[1]
    horizons = [h for h in horizons if h < H]
    sse = {h: 0.0 for h in horizons}
    sae = {h: 0.0 for h in horizons}
    sst = {h: 0.0 for h in horizons}
    sum_t = {h: 0.0 for h in horizons}
    cnt = {h: 0 for h in horizons}

    # First pass: per-horizon means
    for s in range(0, n, chunk_rows):
        t = np.asarray(true[s:s + chunk_rows], dtype=np.float64)
        for h in horizons:
            sum_t[h] += float(t[:, h, 0].sum())
            cnt[h] += t.shape[0]
    means = {h: sum_t[h] / cnt[h] for h in horizons}

    # Second pass: per-horizon SSE / SAE / SST
    for s in range(0, n, chunk_rows):
        p = np.asarray(pred[s:s + chunk_rows], dtype=np.float64)
        t = np.asarray(true[s:s + chunk_rows], dtype=np.float64)
        diff = t - p
        for h in horizons:
            d = diff[:, h, 0]
            sse[h] += float(np.sum(d * d))
            sae[h] += float(np.sum(np.abs(d)))
            dm = t[:, h, 0] - means[h]
            sst[h] += float(np.sum(dm * dm))

    out = []
    for h in horizons:
        n_h = cnt[h]
        mse = sse[h] / n_h
        out.append({
            "h": h,
            "MAE": sae[h] / n_h,
            "RMSE": np.sqrt(mse),
            "R2": 1.0 - sse[h] / sst[h] if sst[h] > 0 else float("nan"),
            "true_mean": means[h],
        })
    return out


def streaming_power_buckets(pred, true, mean_kw, std_kw, max_kw, chunk_rows: int):
    """Bucket by TRUE power (kW): low [0,200), mid [200,800), high [800,maxP].
       All horizons & windows pooled."""
    edges_kw = [0.0, 200.0, 800.0, max_kw + 1.0]
    labels = ["low (<200 kW)", "mid (200-800 kW)", f"high (>=800 kW, max {max_kw:.0f})"]
    buckets = {i: {"sae": 0.0, "sse": 0.0, "n": 0} for i in range(len(labels))}
    n = pred.shape[0]
    for s in range(0, n, chunk_rows):
        p_z = np.asarray(pred[s:s + chunk_rows], dtype=np.float64)
        t_z = np.asarray(true[s:s + chunk_rows], dtype=np.float64)
        # Denormalize once
        p_kw = p_z * std_kw + mean_kw
        t_kw = t_z * std_kw + mean_kw
        diff_kw = t_kw - p_kw
        flat_t = t_kw.reshape(-1)
        flat_d = diff_kw.reshape(-1)
        for i in range(len(labels)):
            mask = (flat_t >= edges_kw[i]) & (flat_t < edges_kw[i + 1])
            if not mask.any():
                continue
            d = flat_d[mask]
            buckets[i]["sae"] += float(np.sum(np.abs(d)))
            buckets[i]["sse"] += float(np.sum(d * d))
            buckets[i]["n"] += int(mask.sum())
    rows = []
    for i, lab in enumerate(labels):
        b = buckets[i]
        if b["n"] == 0:
            rows.append({"bucket": lab, "n": 0, "MAE_kW": float("nan"),
                         "RMSE_kW": float("nan"), "share_%": 0.0})
            continue
        mse = b["sse"] / b["n"]
        rows.append({
            "bucket": lab,
            "n": b["n"],
            "MAE_kW": b["sae"] / b["n"],
            "RMSE_kW": np.sqrt(mse),
            "share_%": 0.0,  # filled in after total known
        })
    total = sum(b["n"] for b in buckets.values())
    for r in rows:
        r["share_%"] = 100.0 * r["n"] / total if total else 0.0
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    ap.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    ap.add_argument("--chunk-rows", type=int, default=20000)
    ap.add_argument("--horizons", type=str, default="0,5,23,71,143",
                    help="Indices into pred_len (0-based). 0=10min, 5=1h, 23=4h, 71=12h, 143=24h")
    args = ap.parse_args()

    pred_path = args.test_dir / "pred.npy"
    true_path = args.test_dir / "true.npy"
    for p in (pred_path, true_path):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    print(f"=== TimeCAP baseline full evaluation ===")
    print(f"test_dir : {args.test_dir}")
    print(f"data_csv : {args.data_csv}")

    pred = np.load(pred_path, mmap_mode="r")
    true = np.load(true_path, mmap_mode="r")
    print(f"pred shape = {pred.shape} dtype={pred.dtype}")
    print(f"true shape = {true.shape} dtype={true.dtype}")

    # 1. Patv normalization stats from train portion
    mean_kw, std_kw, max_kw = load_patv_stats(args.data_csv)
    print(f"\nPatv (train 70%): mean = {mean_kw:.3f} kW   std = {std_kw:.3f} kW")
    print(f"Patv (full):      max  = {max_kw:.3f} kW")

    # 2. Overall metrics in z-score
    print(f"\n[1/3] computing overall metrics (z-score) ...")
    # First pass: true mean
    n = true.shape[0]
    s_t = 0.0; cnt = 0
    for s in range(0, n, args.chunk_rows):
        b = np.asarray(true[s:s + args.chunk_rows], dtype=np.float64)
        s_t += float(b.sum()); cnt += b.size
    true_mean_z = s_t / cnt
    print(f"  true.mean (z) = {true_mean_z:.6f}  (~ {true_mean_z * std_kw + mean_kw:.2f} kW)")

    overall = streaming_overall(pred, true, true_mean_z, args.chunk_rows)
    print(f"\n  Overall (z-score domain):")
    print(f"    MAE  = {overall['MAE']:.4f}    RMSE = {overall['RMSE']:.4f}")
    print(f"    R^2  = {overall['R2']:.4f}    max|err| = {overall['max_abs']:.4f}")
    # Convert to kW (linear scale, so MAE/RMSE simply * std)
    print(f"\n  Overall (kW domain — linear denorm):")
    print(f"    MAE  = {overall['MAE'] * std_kw:.2f} kW   "
          f"({overall['MAE'] * std_kw / max_kw * 100:.2f}% of maxP)")
    print(f"    RMSE = {overall['RMSE'] * std_kw:.2f} kW   "
          f"({overall['RMSE'] * std_kw / max_kw * 100:.2f}% of maxP)")
    print(f"    R^2  = {overall['R2']:.4f}   (R^2 invariant under linear rescale)")

    # 3. Per-horizon decay
    print(f"\n[2/3] computing per-horizon decay ...")
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    rows = streaming_per_horizon(pred, true, horizons, args.chunk_rows)
    print(f"\n  Per-horizon table:")
    print(f"  {'h':>4} {'time':>9} {'MAE_kW':>10} {'RMSE_kW':>10} {'R^2':>8}")
    print(f"  {'-'*45}")
    for r in rows:
        minutes = (r["h"] + 1) * 10
        if minutes < 60:
            tlbl = f"{minutes}min"
        else:
            tlbl = f"{minutes/60:.1f}h"
        print(f"  {r['h']:>4} {tlbl:>9} {r['MAE']*std_kw:>10.2f} "
              f"{r['RMSE']*std_kw:>10.2f} {r['R2']:>8.4f}")

    # 4. Power-bucket breakdown
    print(f"\n[3/3] computing true-power-bucket breakdown (kW domain, all horizons pooled) ...")
    rows = streaming_power_buckets(pred, true, mean_kw, std_kw, max_kw, args.chunk_rows)
    print(f"\n  {'bucket':<32} {'count':>14} {'share':>8} {'MAE_kW':>10} {'RMSE_kW':>10}")
    print(f"  {'-'*78}")
    for r in rows:
        print(f"  {r['bucket']:<32} {r['n']:>14,} {r['share_%']:>7.2f}% "
              f"{r['MAE_kW']:>10.2f} {r['RMSE_kW']:>10.2f}")

    # 5. Sanity: how good is "predict the mean" baseline?
    print(f"\n  Sanity floor — constant 'predict global mean' MAE on this set:")
    # Using overall true_mean_z, MAE of zeros minus (t - mean) is same as |t-mean|.
    s_a = 0.0
    for s in range(0, n, args.chunk_rows):
        b = np.asarray(true[s:s + args.chunk_rows], dtype=np.float64)
        s_a += float(np.sum(np.abs(b - true_mean_z)))
    mae_const = s_a / cnt
    print(f"    MAE (constant)   = {mae_const * std_kw:.2f} kW   "
          f"({mae_const * std_kw / max_kw * 100:.2f}% of maxP)")
    print(f"    MAE (TimeCAP)    = {overall['MAE'] * std_kw:.2f} kW")
    skill = 1.0 - overall['MAE'] / mae_const
    print(f"    Skill score (1 - MAE/MAE_const) = {skill*100:.1f}%   "
          f"(higher = more useful than the trivial baseline)")
    print(f"\n=== done ===")


if __name__ == "__main__":
    main()
