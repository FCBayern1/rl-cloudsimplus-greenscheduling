"""
engineer_features.py
====================
Phase-1 feature engineering for TimeCAP fine-tuning.

Reads a baseline merged CSV (output of prepare_turbine_data.py, with the
13-feature schema {Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Prtv, T2m, Sp,
RelH, Wspd_w, Wdir_w, Patv}) and rewrites it with:

  * sin/cos encoding for direction columns (Wdir, Ndir, Wdir_w)
  * physics-prior feature: Wspd_cubed (= Wspd ** 3)
  * cyclical time-of-day / day-of-year / day-of-week features

Why this matters:
  - The TimeCAP forward path ignores ``batch_x_mark``; the only way to
    actually feed the model time information is to add it as input
    columns. (See drl-manager/Code/models/TimeCAP.py.)
  - Raw degrees are discontinuous at 0/360, hurting attention. sin/cos
    pairs make the angle locally linear.
  - Wind power scales ~ V^3 (Betz), so Wspd^3 is a strong physical prior.

Output column order (date first, target last, as required by Dataset_Custom):
    date,
    Wspd, Wspd_cubed, Etmp, Itmp, Pab1, Prtv, T2m, Sp, RelH, Wspd_w,
    Wdir_sin, Wdir_cos, Ndir_sin, Ndir_cos, Wdir_w_sin, Wdir_w_cos,
    hour_sin, hour_cos, doy_sin, doy_cos, dow_sin, dow_cos,
    Patv
  → 22 features + 1 target = enc_in = 23

Usage:
    python -m timecap_prediction.engineer_features \\
        --input  timecap_prediction/data/turbines_all134_2021.csv \\
        --output timecap_prediction/data/turbines_all134_2021_v2.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ANGLE_COLS = ["Wdir", "Ndir", "Wdir_w"]
KEEP_RAW_COLS = ["Wspd", "Etmp", "Itmp", "Pab1", "Prtv", "T2m", "Sp", "RelH", "Wspd_w"]
TARGET = "Patv"


def add_angle_sin_cos(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace each angle column (degrees, 0-360) with a (sin, cos) pair."""
    for c in cols:
        if c not in df.columns:
            print(f"[WARN] angle column missing: {c}")
            continue
        rad = np.deg2rad(df[c].astype(np.float64).values)
        df[f"{c}_sin"] = np.sin(rad).astype(np.float32)
        df[f"{c}_cos"] = np.cos(rad).astype(np.float32)
    return df


def add_wspd_cubed(df: pd.DataFrame) -> pd.DataFrame:
    if "Wspd" not in df.columns:
        print("[WARN] Wspd column missing; skipping Wspd_cubed")
        return df
    w = df["Wspd"].astype(np.float64).values
    df["Wspd_cubed"] = (w ** 3).astype(np.float32)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encodings for hour-of-day / day-of-year / day-of-week."""
    dt = pd.to_datetime(df["date"])
    hour = dt.dt.hour + dt.dt.minute / 60.0          # fractional hour
    doy  = dt.dt.dayofyear.astype(np.float64)        # 1..366
    dow  = dt.dt.dayofweek.astype(np.float64)        # 0..6

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df["doy_sin"]  = np.sin(2 * np.pi * doy  / 366.0).astype(np.float32)
    df["doy_cos"]  = np.cos(2 * np.pi * doy  / 366.0).astype(np.float32)
    df["dow_sin"]  = np.sin(2 * np.pi * dow  / 7.0).astype(np.float32)
    df["dow_cos"]  = np.cos(2 * np.pi * dow  / 7.0).astype(np.float32)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    angle_pairs = [f"{c}_{s}" for c in ANGLE_COLS for s in ("sin", "cos")]
    time_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"]

    new_order = (
        ["date"]
        + ["Wspd", "Wspd_cubed"]
        + [c for c in KEEP_RAW_COLS if c != "Wspd" and c in df.columns]
        + [c for c in angle_pairs if c in df.columns]
        + [c for c in time_cols if c in df.columns]
        + [TARGET]
    )
    missing = [c for c in new_order if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns after engineering: {missing}")
    return df[new_order]


def engineer(input_csv: Path, output_csv: Path) -> None:
    print(f"[load]  {input_csv}")
    df = pd.read_csv(input_csv)
    n_rows, n_cols_in = df.shape
    print(f"        rows={n_rows:,}  cols_in={n_cols_in}")

    if "date" not in df.columns or TARGET not in df.columns:
        raise SystemExit(f"Input CSV must contain 'date' and '{TARGET}' columns.")

    df = add_angle_sin_cos(df, ANGLE_COLS)
    df = add_wspd_cubed(df)
    df = add_time_features(df)

    # drop original angle columns after sin/cos encoding
    df = df.drop(columns=[c for c in ANGLE_COLS if c in df.columns])

    df = reorder_columns(df)

    print(f"[stats] new columns: {list(df.columns)}")
    print(f"        rows={len(df):,}  cols_out={df.shape[1]}  enc_in={df.shape[1] - 1}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    sz_mb = output_csv.stat().st_size / 1e6
    print(f"[save]  {output_csv}  ({sz_mb:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="Path to baseline merged CSV.")
    p.add_argument("--output", type=Path, required=True,
                   help="Path to write the engineered CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engineer(args.input, args.output)
