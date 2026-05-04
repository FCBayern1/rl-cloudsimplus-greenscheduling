#!/usr/bin/env python3
"""
diagnose_ar_os_disagreement.py
==============================
Tests whether |AR - OS| head disagreement correlates with |OS - truth| error
on the baseline TimeCAP checkpoint.

If the correlation is strong (Pearson r > 0.5, Spearman > 0.5) and the
calibration curve is monotonic, then |AR - OS| is a usable free uncertainty
proxy and can be added to the RL state without retraining the model.

Sampling
--------
Picks N random (turbine, csv_row) pairs from the last 20% of each turbine
CSV (the held-out test region used during fine-tuning). For each pair,
runs one forward pass capturing BOTH AR and OS heads (a single forward
gives both — TimeCAP returns them as a tuple), then compares against
ground-truth Patv from the same CSV.

Run
---
    cd drl-manager
    python ../scripts/timecap/diagnose_ar_os_disagreement.py \\
        --n-samples 2000 --device cuda

    sbatch ...   # if you need a GPU node, wrap with a small sbatch
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# --- repo bootstrap ---
_REPO = Path(__file__).resolve().parents[2]
_DRLMANAGER = _REPO / "drl-manager"
for _p in (str(_DRLMANAGER), str(_DRLMANAGER / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from timecap_prediction.predictor import TimeCAP_GreenPredictor  # noqa: E402

# --- defaults ---
DEFAULT_CKPT = (_DRLMANAGER / "timecap_prediction/TimeCAP/model"
                / "finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth")
DEFAULT_CSV_DIR = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/split"
DEFAULT_OUT = _DRLMANAGER / "timecap_prediction/figures/ar_os_disagreement.png"

# Match training default: train [0,70%], val [70%,80%], test [80%,100%]
TEST_FRAC_START = 0.80
SEQ_LEN  = 96
PRED_LEN = 144

# normalised display only (correlation is scale-invariant)
PATV_MAX_KW = 1561.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--turbines", type=str, default="1,15,30,46,60,90,120,134",
                   help="Comma-separated turbine IDs to sample from")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=64,
                   help="Forward batch size (auto-shrinks to fit n-samples)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib plot generation")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(args.seed)

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    if not args.csv_dir.exists():
        sys.exit(f"csv directory not found: {args.csv_dir}")

    turbine_ids = [int(t) for t in args.turbines.split(",")]
    csv_paths = {tid: str(args.csv_dir / f"Turbine_{tid}_2021.csv") for tid in turbine_ids}
    for tid, path in csv_paths.items():
        if not Path(path).exists():
            sys.exit(f"missing turbine CSV: {path}")

    print("=" * 72)
    print("AR-OS head disagreement diagnostic")
    print("=" * 72)
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {args.device}")
    print(f"  turbines    : {turbine_ids}")
    print(f"  n samples   : {args.n_samples}")
    print()

    print("Loading TimeCAP predictor ...")
    predictor = TimeCAP_GreenPredictor(
        checkpoint_path=str(args.checkpoint),
        turbine_csv_paths=csv_paths,
        device=args.device,
    )
    device = torch.device(args.device)
    patv_idx = predictor.patv_idx
    num_features = predictor.num_features

    # Use the first turbine's frame to derive the test region row range.
    # All SDWPF turbine CSVs cover the same 2021 timeline, so this is fine.
    df0 = predictor.feature_loader.turbine_data[turbine_ids[0]]
    n_rows = len(df0)
    row_min = max(int(n_rows * TEST_FRAC_START), SEQ_LEN)
    row_max = n_rows - PRED_LEN
    if row_min >= row_max:
        sys.exit(f"no valid test windows: row_min={row_min} row_max={row_max}")
    print(f"  test rows   : [{row_min}, {row_max})  (csv length {n_rows})")
    print()

    # Sample (turbine, row) pairs
    chosen_t = rng.choice(turbine_ids, size=args.n_samples)
    chosen_r = rng.integers(row_min, row_max, size=args.n_samples)

    # Buffers
    ar_pred  = np.zeros((args.n_samples, PRED_LEN), dtype=np.float32)
    os_pred  = np.zeros((args.n_samples, PRED_LEN), dtype=np.float32)
    truth    = np.zeros((args.n_samples, PRED_LEN), dtype=np.float32)

    BATCH = max(1, min(args.batch, args.n_samples))
    n_batches = (args.n_samples + BATCH - 1) // BATCH
    print(f"Running {n_batches} batches of up to {BATCH} samples on {args.device} ...")

    for b in range(n_batches):
        i0 = b * BATCH
        i1 = min(i0 + BATCH, args.n_samples)
        bs = i1 - i0

        x = np.zeros((bs, SEQ_LEN, num_features), dtype=np.float32)
        y = np.zeros((bs, PRED_LEN), dtype=np.float32)

        for j in range(bs):
            tid = int(chosen_t[i0 + j])
            r = int(chosen_r[i0 + j])
            df = predictor.feature_loader.turbine_data[tid]
            # input  = rows [r-SEQ_LEN, r)   — past SEQ_LEN steps
            # truth  = rows [r, r+PRED_LEN)  — next PRED_LEN steps of Patv
            x[j] = df.iloc[r - SEQ_LEN : r].values.astype(np.float32)
            y[j] = df.iloc[r : r + PRED_LEN]['Patv'].values.astype(np.float32)

        x_t = torch.from_numpy(x).to(device)
        with torch.no_grad():
            dec_AR, dec_OS, _ = predictor.model(x_t, activate_os_head=True)
        if dec_AR is None or dec_OS is None:
            sys.exit("model did not return both heads — check task_name='finetune' in config")

        ar_pred[i0:i1] = dec_AR[:, :, patv_idx].cpu().numpy()
        os_pred[i0:i1] = dec_OS[:, :, patv_idx].cpu().numpy()
        truth[i0:i1]   = y

        if (b + 1) % max(1, n_batches // 10) == 0 or b + 1 == n_batches:
            print(f"  batch {b+1:3d}/{n_batches}")

    # TimeCAP applies instance norm + denorm internally → outputs are in input units (kW)
    # Clip negative AR/OS to 0 (Patv is physically non-negative)
    ar_pred = np.clip(ar_pred, 0.0, None)
    os_pred = np.clip(os_pred, 0.0, None)

    # Disagreement (uncertainty proxy) and actual error of the deployed head (OS)
    disagreement = np.abs(ar_pred - os_pred)   # (N, pred_len), kW
    error        = np.abs(os_pred - truth)     # (N, pred_len), kW

    d = disagreement.ravel()
    e = error.ravel()

    # --- Correlations ---
    pearson = float(np.corrcoef(d, e)[0, 1])
    # Spearman (rank correlation) without scipy: rank-then-pearson
    def _ranks(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        return ranks
    spearman = float(np.corrcoef(_ranks(d), _ranks(e))[0, 1])

    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)
    print(f"  total points         : {len(d):,}  ({args.n_samples} windows × {PRED_LEN} horizon)")
    print(f"  mean |AR - OS|       : {d.mean():7.2f} kW   ({d.mean()/PATV_MAX_KW*100:5.2f}% maxP)")
    print(f"  mean |OS - truth|    : {e.mean():7.2f} kW   ({e.mean()/PATV_MAX_KW*100:5.2f}% maxP)")
    print(f"  Pearson r (d, e)     : {pearson:+7.4f}")
    print(f"  Spearman ρ (d, e)    : {spearman:+7.4f}")

    # Per-horizon breakdown
    print()
    print("  Per-horizon correlation (does the signal hold at long horizons?)")
    print(f"    {'horizon':>10s}  {'mean|d|':>10s}  {'mean|e|':>10s}  {'Pearson':>10s}  {'Spearman':>10s}")
    for h in (1, 6, 24, 72, 144):
        d_h = disagreement[:, :h].ravel()
        e_h = error[:, :h].ravel()
        if len(d_h) < 2 or d_h.std() < 1e-9 or e_h.std() < 1e-9:
            continue
        pr = float(np.corrcoef(d_h, e_h)[0, 1])
        sr = float(np.corrcoef(_ranks(d_h), _ranks(e_h))[0, 1])
        print(f"    {h:>7d}     {d_h.mean():7.2f} kW {e_h.mean():7.2f} kW   {pr:+7.4f}    {sr:+7.4f}")

    # Calibration: bin by disagreement decile, report mean |error| per bin
    print()
    print("  Calibration (does higher disagreement → higher error in practice?)")
    deciles = np.percentile(d, np.linspace(0, 100, 11))
    print(f"    {'decile':>6s}  {'|AR-OS| range (kW)':<28s}  {'mean |OS-truth| (kW)':>22s}")
    monotonic_increase = True
    prev_e = -np.inf
    for i in range(10):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (d >= lo) & (d < hi) if i < 9 else (d >= lo)
        if not mask.any():
            continue
        mean_e = float(e[mask].mean())
        if mean_e < prev_e - 1.0:  # allow 1 kW noise
            monotonic_increase = False
        prev_e = mean_e
        print(f"    {i+1:>4d}    [{lo:7.2f}, {hi:7.2f}]            {mean_e:7.2f}")

    # --- Verdict ---
    print()
    print("=" * 72)
    print("Verdict")
    print("=" * 72)
    if spearman > 0.5 and monotonic_increase:
        verdict = "STRONG  ✓  use |AR-OS| as a free uncertainty signal in RL state"
    elif spearman > 0.3:
        verdict = "WEAK    ~  signal exists but noisy — try MC Dropout for cleaner σ"
    else:
        verdict = "NONE    ✗  AR-OS too aligned (self-distill λ3 dominates) — need MC Dropout / ensemble"
    print(f"  Spearman ρ   = {spearman:+.4f}")
    print(f"  monotonic    = {monotonic_increase}")
    print(f"  → {verdict}")
    print()

    # --- Plot ---
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Hexbin: disagreement vs error
        ax = axes[0]
        # Subsample if too many points for hexbin
        if len(d) > 200000:
            idx = rng.choice(len(d), size=200000, replace=False)
            d_p, e_p = d[idx], e[idx]
        else:
            d_p, e_p = d, e
        hb = ax.hexbin(d_p, e_p, gridsize=60, bins="log", cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="log(count)")
        m = max(d_p.max(), e_p.max())
        ax.plot([0, m], [0, m], "r--", alpha=0.5, label="y = x")
        ax.set_xlabel("|AR - OS| disagreement (kW)")
        ax.set_ylabel("|OS - truth| error (kW)")
        ax.set_title(f"Disagreement vs Error\nPearson r = {pearson:+.3f},  Spearman ρ = {spearman:+.3f}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Calibration: decile centers vs mean error
        ax = axes[1]
        centers, means, p10s, p90s = [], [], [], []
        for i in range(10):
            lo, hi = deciles[i], deciles[i + 1]
            mask = (d >= lo) & (d < hi) if i < 9 else (d >= lo)
            if not mask.any():
                continue
            centers.append(0.5 * (lo + hi))
            means.append(float(e[mask].mean()))
            p10s.append(float(np.percentile(e[mask], 10)))
            p90s.append(float(np.percentile(e[mask], 90)))
        ax.plot(centers, means, "o-", color="#2b6cb0", label="mean |error|")
        ax.fill_between(centers, p10s, p90s, alpha=0.2, color="#2b6cb0", label="10–90 pct")
        ax.set_xlabel("|AR - OS| disagreement decile center (kW)")
        ax.set_ylabel("|OS - truth| error in bin (kW)")
        ax.set_title("Calibration: monotonic = good signal")
        ax.grid(alpha=0.3)
        ax.legend()

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"plot saved: {args.out}")


if __name__ == "__main__":
    main()
