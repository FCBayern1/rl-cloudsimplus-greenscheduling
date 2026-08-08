#!/usr/bin/env python3
"""Decompose the measured -8-step TimeCAP lag: model contract vs cache staleness.

The nominal contract: input = last seq_len rows up to T0, output pred[h] = power
at T0+1+h. The original diagnostic read the CACHED forecast every step, mixing
generation-time alignment with up-to-(forecast_every-1) steps of staleness, so
its best_lag cannot tell a mislabelled model window from ordinary cache age.

This variant compares ONLY at generation events (zero staleness): detect when
_last_per_t_pred changes identity, anchor at that step T0, compare pred against
actual[T0+1 : T0+1+PRED]. best_lag ~= 0 here => contract is fine and the whole
-8 is staleness/mixing. best_lag << 0 here => real model/index misalignment.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
os.chdir(REPO); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))
from prediction.timecap_godeye_provider import TimeCAPGodEyeProvider  # noqa

CSV_DIR = REPO.parent / "cloudsimplus-gateway/src/main/resources/windProduction/split"
CKPT = REPO / "timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth"
DC_ASSIGN = {0: [12, 36], 1: [95, 91], 2: [96]}
TZ = {0: 0, 1: 18, 2: 54}
N_STEPS, PRED = 700, 144

patv = {t: pd.read_csv(CSV_DIR / f"Turbine_{t}_2021.csv")["Patv"].to_numpy(dtype=float)
        for tids in DC_ASSIGN.values() for t in tids}
paths = {t: str(CSV_DIR / f"Turbine_{t}_2021.csv") for tids in DC_ASSIGN.values() for t in tids}
prov = TimeCAPGodEyeProvider(dc_assignments=DC_ASSIGN, turbine_csv_paths=paths,
                             checkpoint_path=str(CKPT), forecast_every=6, device="cpu",
                             csv_start_offset=0, dc_tz_offsets=TZ,
                             simulation_warmup_rows=0, forecast_shift=None)

def actual(dc, t0, h):
    out = np.zeros(h)
    for tid in DC_ASSIGN[dc]:
        seg = patv[tid][TZ[dc] + t0: TZ[dc] + t0 + h]
        out += np.pad(seg, (0, h - seg.size))
    return out

pred_by, act_by = {d: [] for d in DC_ASSIGN}, {d: [] for d in DC_ASSIGN}
last_id = None
for T in range(prov.seq_len, prov.seq_len + N_STEPS):
    prov.step_and_get(T)
    raw = prov._last_per_t_pred
    if raw is None or id(raw) == last_id:
        continue                      # not a fresh forecast this step
    last_id = id(raw)
    for dc in DC_ASSIGN:
        p = np.zeros(PRED); ok = True
        for tid in DC_ASSIGN[dc]:
            arr = raw.get(tid)
            if arr is None or arr.size < PRED: ok = False; break
            p += arr[:PRED]
        if ok:
            pred_by[dc].append(p)
            act_by[dc].append(actual(dc, T + 1, PRED))

print(f"{'DC':>3} {'n_fresh':>7} {'r@lag0':>8} {'best_lag':>9} {'r@best':>8}")
for dc in DC_ASSIGN:
    P = np.array(pred_by[dc]); A = np.array(act_by[dc])
    r0 = float(np.corrcoef(P.ravel(), A.ravel())[0, 1])
    best = (0, r0)
    for lag in range(-20, 21):
        if lag < 0:  a, b = P[:, -lag:], A[:, :lag]
        elif lag > 0: a, b = P[:, :-lag], A[:, lag:]
        else: a, b = P, A
        r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        if r > best[1]: best = (lag, r)
    print(f"{dc:>3} {len(P):>7} {r0:>8.3f} {best[0]:>9} {best[1]:>8.3f}")
print("\nRead: best_lag ~ 0 => model contract fine (the -8 was cache staleness + mixed sampling).")
print("      best_lag clearly negative => genuine model/wrapper misalignment exists.")
