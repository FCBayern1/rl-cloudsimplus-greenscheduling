"""Forecaster-accuracy diagnostic: how well does TimeCAP predict per-DC green vs ground truth?

Builds the same TimeCAPGodEyeProvider the env uses, walks forward, and at each forecast
compares the predicted pred_len trajectory against the actual CSV future trajectory.
High Pearson r => forecaster is good => timecap underperformance is an AGENT/credit problem
(warm-start / EU-CRD). Low r => forecaster quality problem. best-lag != 0 => time misalignment.
"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent          # drl-manager
os.chdir(REPO); sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from prediction.timecap_godeye_provider import TimeCAPGodEyeProvider

CSV_DIR = REPO.parent / "cloudsimplus-gateway/src/main/resources/windProduction/split"
CKPT = REPO / "timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth"
DC_ASSIGN = {0: [12, 36], 1: [95, 91], 2: [96]}
TZ = {0: 0, 1: 18, 2: 54}
WARMUP = 0
YEAR = 2021
N_STEPS = 700          # forecast points to sweep
PRED = 144

# Ground-truth per-turbine Patv (kW) series from the raw CSVs
patv = {}
for tids in DC_ASSIGN.values():
    for t in tids:
        df = pd.read_csv(CSV_DIR / f"Turbine_{t}_{YEAR}.csv")
        patv[t] = df["Patv"].to_numpy(dtype=float)

paths = {t: str(CSV_DIR / f"Turbine_{t}_{YEAR}.csv") for tids in DC_ASSIGN.values() for t in tids}
prov = TimeCAPGodEyeProvider(
    dc_assignments=DC_ASSIGN, turbine_csv_paths=paths, checkpoint_path=str(CKPT),
    forecast_every=6, device="cpu", csv_start_offset=0, dc_tz_offsets=TZ,
    simulation_warmup_rows=WARMUP, forecast_shift=None,
)
prov.reset(); prov.warmup(start_step=0)

def actual_dc_traj(dc, t0, h):
    """actual summed Patv (kW) for dc's turbines over rows [base+t0 .. +h)."""
    out = np.zeros(h)
    for tid in DC_ASSIGN[dc]:
        base = TZ[dc] + WARMUP + t0
        seg = patv[tid][base: base + h]
        if seg.size < h:
            seg = np.pad(seg, (0, h - seg.size))
        out += seg
    return out

# Sweep: at each step push the row, read the latest raw forecast, align with actual future
pred_by_dc = {d: [] for d in DC_ASSIGN}
act_by_dc = {d: [] for d in DC_ASSIGN}
for T in range(prov.seq_len, prov.seq_len + N_STEPS):
    prov.step_and_get(T)          # fires the TimeCAP forward (every forecast_every)
    raw = prov._last_per_t_pred
    if raw is None:
        continue
    for dc in DC_ASSIGN:
        # predicted DC trajectory (kW): sum turbines
        pred = np.zeros(PRED)
        ok = True
        for tid in DC_ASSIGN[dc]:
            p = raw.get(tid)
            if p is None or p.size < PRED:
                ok = False; break
            pred += p[:PRED]
        if not ok:
            continue
        act = actual_dc_traj(dc, T + 1, PRED)        # actual future right after now
        pred_by_dc[dc].append(pred); act_by_dc[dc].append(act)

def pearson(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

print(f"\n=== TimeCAP forecast accuracy (n forecast points per DC ~ {len(pred_by_dc[0])}) ===")
print(f"{'DC':>3} {'pearson_r':>10} {'best_lag':>9} {'r@best_lag':>11} {'nMAE':>8}  verdict")
for dc in DC_ASSIGN:
    P = np.array(pred_by_dc[dc]); A = np.array(act_by_dc[dc])
    if P.size == 0:
        print(f"{dc:>3}  (no forecasts)"); continue
    r = pearson(P, A)
    # best lag over a small window (detect systematic time shift)
    pm = P.mean(0); am_full = A.mean(0)
    best_lag, best_r = 0, -2.0
    for lag in range(-12, 13):
        if lag >= 0:
            x, y = pm[:PRED - lag], am_full[lag:]
        else:
            x, y = pm[-lag:], am_full[:PRED + lag]
        rr = pearson(x, y)
        if rr == rr and rr > best_r:
            best_r, best_lag = rr, lag
    scale = A.mean() if A.mean() > 1e-6 else 1.0
    nmae = float(np.mean(np.abs(P - A)) / scale)
    verdict = "GOOD forecaster" if r > 0.6 else ("WEAK" if r > 0.3 else "POOR/noise")
    print(f"{dc:>3} {r:>10.3f} {best_lag:>9d} {best_r:>11.3f} {nmae:>8.2f}  {verdict}")
print("\nr>0.6 => agent/credit problem (warm-start/EU-CRD). r<0.3 => forecaster-quality problem.")
print("best_lag far from 0 => systematic time misalignment (tz/offset bug).")
