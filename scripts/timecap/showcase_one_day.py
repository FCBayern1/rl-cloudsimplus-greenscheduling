#!/usr/bin/env python3
"""
Showcase: predict one full day of turbine Patv with TimeCAP and plot it
against the ground-truth values read directly from the same CSV.

Pipeline
--------
1. Pick a turbine CSV and a target date (YYYY-MM-DD).
2. Locate the first CSV row of that date  → T
3. Build TimeCAP_GreenPredictor for that single turbine.
4. Push the 96 rows preceding T into the predictor's history buffer
   (rows [T-96, T-1] → simulation_steps [T-108, T-13]).
5. Call predict() → (144,) kW forecast covering rows [T, T+143] (the full day).
6. Read the same 144 rows of Patv from the CSV as ground truth.
7. Plot both curves on hour-of-day axis and save as PNG.

Usage (from repo root)
----------------------
    python scripts/timecap/showcase_one_day.py
    python scripts/timecap/showcase_one_day.py --turbine 46 --date 2021-06-15
    python scripts/timecap/showcase_one_day.py \\
        --checkpoint drl-manager/timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: <repo>/scripts/timecap/showcase_one_day.py → repo = parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRLMANAGER = _REPO_ROOT / "drl-manager"
if str(_DRLMANAGER) not in sys.path:
    sys.path.insert(0, str(_DRLMANAGER))

from timecap_prediction.predictor import TimeCAP_GreenPredictor  # noqa: E402


SEQ_LEN = 96            # 16 h history (10 min × 96)
PRED_LEN = 144          # 24 h forecast (10 min × 144)
CSV_OFFSET = 12         # predictor.update(sim_step) reads CSV row sim_step + 12

DEFAULT_CHECKPOINT = (
    _REPO_ROOT
    / "drl-manager/timecap_prediction/TimeCAP/model"
    / "finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth"
)
DEFAULT_CSV = (
    _REPO_ROOT
    / "cloudsimplus-gateway/bin/main/windProduction/split/Turbine_46_2021.csv"
)
DEFAULT_OUT_DIR = _REPO_ROOT / "drl-manager/timecap_prediction/figures"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--turbine", type=int, default=46,
                   help="Turbine ID (just a label; the CSV is what matters)")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="Path to the turbine's CSV (SDWPF split format with Tmstamp + 13 raw columns)")
    p.add_argument("--date", type=str, default="2021-06-15",
                   help="Target date YYYY-MM-DD (must lie inside the CSV's coverage)")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Path to ckpt_best.pth (model_args.json must sit next to it)")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cpu, cuda, cuda:0")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG. Default: drl-manager/timecap_prediction/figures/showcase_<turbine>_<date>.png")
    p.add_argument("--no-show-history", action="store_true",
                   help="Hide the 16h history segment on the left of the plot")
    return p.parse_args()


def find_target_row(df: pd.DataFrame, date: str) -> int:
    """Return the index of the first row whose Tmstamp falls on the given date."""
    if "Tmstamp" not in df.columns:
        raise ValueError(f"CSV is missing 'Tmstamp' column. Found: {list(df.columns)}")
    mask = df["Tmstamp"].astype(str).str.startswith(date)
    if not mask.any():
        raise ValueError(f"Date {date} not found in CSV. "
                         f"Coverage: {df['Tmstamp'].iloc[0]} → {df['Tmstamp'].iloc[-1]}")
    return int(df.index[mask][0])


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    abs_err = np.abs(err)
    out = {
        "MAE":  float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MaxAE": float(np.max(abs_err)),
        "Pred_kWh":  float(np.sum(pred)) / 6.0,   # 10-min sums → kWh (×1/6 h)
        "True_kWh":  float(np.sum(true)) / 6.0,
        "Energy_err_pct": float(
            (np.sum(pred) - np.sum(true)) / max(np.sum(true), 1e-6) * 100.0
        ),
    }
    if np.var(true) > 1e-9:
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true - np.mean(true)) ** 2))
        out["R2"] = 1.0 - ss_res / ss_tot
    else:
        out["R2"] = float("nan")
    return out


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    print(f"=== TimeCAP one-day showcase ===")
    print(f"  turbine     : {args.turbine}")
    print(f"  csv         : {args.csv}")
    print(f"  date        : {args.date}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {args.device}")

    # Load CSV (only for index lookup + ground truth slicing — the predictor
    # has its own loader that re-reads the file).
    df = pd.read_csv(args.csv)
    T = find_target_row(df, args.date)
    print(f"\n  target_row T = {T}  (Tmstamp: {df['Tmstamp'].iloc[T]})")

    if T < SEQ_LEN + CSV_OFFSET:
        sys.exit(f"Date too early: need at least {SEQ_LEN + CSV_OFFSET} rows of history before T={T}")
    if T + PRED_LEN > len(df):
        sys.exit(f"Date too late: need {PRED_LEN} rows after T={T} but CSV has {len(df)}")

    # Build predictor for a single turbine
    predictor = TimeCAP_GreenPredictor(
        checkpoint_path=str(args.checkpoint),
        turbine_csv_paths={args.turbine: str(args.csv)},
        device=args.device,
    )

    # Push history rows [T-96, T-1] into the buffer.
    # update(sim_step) reads CSV row sim_step + CSV_OFFSET, so:
    #     csv_row = T-96 .. T-1  ⇔  sim_step = T-96-12 .. T-1-12
    predictor.reset()
    sim_step_first = T - SEQ_LEN - CSV_OFFSET
    sim_step_last  = T - 1 - CSV_OFFSET
    for s in range(sim_step_first, sim_step_last + 1):
        predictor.update(s)
    print(f"  pushed history rows [{T - SEQ_LEN}, {T - 1}] "
          f"(sim_steps [{sim_step_first}, {sim_step_last}], buffer={SEQ_LEN})")

    # Forecast
    print(f"  running TimeCAP one-shot forecast …")
    pred_kw = predictor.predict()
    if pred_kw is None:
        sys.exit("predict() returned None — see logs")
    assert pred_kw.shape == (PRED_LEN,), f"unexpected pred shape {pred_kw.shape}"

    # Ground truth: rows [T, T+143] of Patv
    true_kw = df["Patv"].iloc[T : T + PRED_LEN].astype(np.float32).values
    assert true_kw.shape == (PRED_LEN,)

    metrics = compute_metrics(pred_kw, true_kw)
    print(f"\n  --- metrics on day {args.date} ({PRED_LEN} steps) ---")
    print(f"  MAE              : {metrics['MAE']:7.2f} kW")
    print(f"  RMSE             : {metrics['RMSE']:7.2f} kW")
    print(f"  Max abs error    : {metrics['MaxAE']:7.2f} kW")
    print(f"  Predicted energy : {metrics['Pred_kWh']:7.1f} kWh")
    print(f"  True energy      : {metrics['True_kWh']:7.1f} kWh")
    print(f"  Energy error     : {metrics['Energy_err_pct']:+7.2f} %")
    print(f"  R²               : {metrics['R2']:7.3f}")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc_hours = np.arange(PRED_LEN) * 10.0 / 60.0   # 0 → 23.83 h

    if args.no_show_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(fc_hours, true_kw, label="Ground truth", color="#2b6cb0", lw=2)
        ax.plot(fc_hours, pred_kw, label="TimeCAP forecast", color="#dd6b20", lw=2, ls="--")
        ax.fill_between(fc_hours, pred_kw, true_kw, alpha=0.15, color="#dd6b20")
        ax.set_xlim(0, 24)
    else:
        # Show 16h history (read raw Patv from CSV so we don't recompute via predictor)
        hist_kw = df["Patv"].iloc[T - SEQ_LEN : T].astype(np.float32).values
        hist_hours = -np.arange(SEQ_LEN, 0, -1) * 10.0 / 60.0   # -16 → -10/60

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_hours, hist_kw, label="History (16 h)", color="#4a5568", lw=1.6)
        ax.plot(fc_hours, true_kw, label="Ground truth (next 24 h)", color="#2b6cb0", lw=2)
        ax.plot(fc_hours, pred_kw, label="TimeCAP forecast", color="#dd6b20", lw=2, ls="--")
        ax.fill_between(fc_hours, pred_kw, true_kw, alpha=0.15, color="#dd6b20")
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.text(0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                "  forecast →", fontsize=9, va="top")
        ax.set_xlim(-16, 24)

    ax.set_xlabel("Hours from forecast start (target day midnight)")
    ax.set_ylabel("Patv (kW)")
    ax.set_title(
        f"Turbine {args.turbine}  |  forecast for {args.date}  |  ckpt={args.checkpoint.parent.name}\n"
        f"MAE = {metrics['MAE']:.1f} kW · RMSE = {metrics['RMSE']:.1f} kW · "
        f"R² = {metrics['R2']:.3f} · energy err = {metrics['Energy_err_pct']:+.1f}%"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    out_path = args.output or (
        DEFAULT_OUT_DIR / f"showcase_T{args.turbine}_{args.date}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\n  saved figure → {out_path}")


if __name__ == "__main__":
    main()
