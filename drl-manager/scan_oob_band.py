#!/usr/bin/env python3
"""Score SPECIFIC wind bands with the pre-registered analytic gamble criterion
(godeye-vs-reactive saving, design_temporal_gamble.py machinery) at the DEPLOYED
episode offsets (1009*k mod 4800, k=0..9).

Purpose (2026-08-09): the OOB verdict came back three-way flat (oracle == nofc_s2
== nofc_s1 at 0.0167, green 76%) -- suspicion: the 2020 band chosen by surface
statistics (mean/std/trough%) has NO gamble structure, so it cannot discriminate.
This scores: (1) the 2021 training band, (2) the 2020 S=17512 OOB band, and
(3) scans all 2020 bands for ones where the gamble structurally exists
(candidates for OOB-v2)."""
import csv
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
BASE_TIDS = [12, 36, 95, 91, 96, 101, 103]
STRETCHED = [7012, 7036, 7095, 7091, 7096, 7101, 7103]
T, NJOBS, STRIDE, RHO = 7200, 1000, 5, 1.29
OFFSETS = [(1009 * k) % 4800 for k in range(10)]

rng = np.random.default_rng(7)
if len(sys.argv) > 1 and sys.argv[1] == "trace":
    # REAL deployed trace (v2026_gamble_n2000.csv): L = MI/ref_mips, latest
    # start = deadline - L. This is what the RL arms actually scheduled.
    tr = list(csv.DictReader(open(REPO / "cloudsimplus-gateway/src/main/resources/traces/v2026_gamble_n2000.csv")))
    L = np.array([max(1, round(float(r["length"]) / 40000.0)) for r in tr])
    arr = np.array([int(r["arrival_time"]) for r in tr])
    latest = np.array([int(r["deadline"]) for r in tr]) - L
    NJOBS = len(tr)
else:
    tab = pq.read_table(REPO / "data/v2026_gpu/asi_opensource_job_execution_summary/part-000.parquet",
                        columns=["duration_hours", "priority_class"])
    d = tab.column("duration_hours").to_numpy(zero_copy_only=False).astype(float)
    pc = tab.column("priority_class").to_numpy(zero_copy_only=False)
    d = d[(pc == "LP") & np.isfinite(d) & (d > 0)] * 3600.0
    L = np.clip(np.round(rng.choice(d, NJOBS) / 20.0), 1, 1000).astype(int)
    slack = rng.integers(1200, 3001, NJOBS)
    arr = np.array([rng.integers(0, max(1, T - int(l) - int(sl))) for l, sl in zip(L, slack)])
    latest = arr + slack
demand_mean = L.sum() / T

def band_2021():
    tot = None
    for t in STRETCHED:
        v = np.array([float(r["power_kw"]) for r in csv.DictReader(open(GATE / f"Turbine_{t}_2021.csv"))])
        tot = v if tot is None else tot + v
    return tot

BASE20 = {t: np.array([float(r["power_kw"]) for r in csv.DictReader(open(GATE / f"Turbine_{t}_2020.csv"))])
          for t in BASE_TIDS}

def band_2020(S):
    agg = sum(BASE20[t][S:S + 1500] for t in BASE_TIDS)
    return np.concatenate([np.zeros(12), np.repeat(agg, 10)])

def score_band(gs, windows=OFFSETS):
    scale = RHO * demand_mean / max(1e-9, gs.mean())
    sav = []
    for off in windows:
        g = gs[off:off + T] * scale
        if g.size < T:
            g = np.pad(g, (0, T - g.size))
        # reactive
        load = np.zeros(T + 1001)
        order = sorted(range(NJOBS), key=lambda j: arr[j])
        waiting, pi = [], 0
        for t in range(T):
            while pi < NJOBS and arr[order[pi]] <= t:
                waiting.append(order[pi]); pi += 1
            waiting.sort(key=lambda j: latest[j])
            still = []
            for j in waiting:
                if t >= latest[j] or load[t] + 1.0 <= g[t]:
                    load[t:t + L[j]] += 1.0
                else:
                    still.append(j)
            waiting = still
        c_react = np.maximum(0.0, load[:T] - g).sum()
        # godeye (urgency order + one replan sweep)
        def pick(j, load):
            f = np.minimum(1.0, np.maximum(0.0, load[:T] + 1.0 - g))
            cs = np.concatenate([[0.0], np.cumsum(f)])
            lo, hi = int(arr[j]), int(latest[j])
            cands = list(range(lo, hi + 1, STRIDE))
            if cands[-1] != hi:
                cands.append(hi)
            return cands[int(np.argmin([cs[min(s + L[j], T)] - cs[s] for s in cands]))]
        gorder = np.argsort(latest)
        load = np.zeros(T + 1001); starts = {}
        for j in gorder:
            s = pick(j, load); starts[j] = s; load[s:s + L[j]] += 1.0
        for j in gorder:
            s0 = starts[j]; load[s0:s0 + L[j]] -= 1.0
            s = pick(j, load); starts[j] = s; load[s:s + L[j]] += 1.0
        c_god = np.maximum(0.0, load[:T] - g).sum()
        sav.append((c_react - c_god) / max(1e-9, c_react))
    return float(np.mean(sav)), float(np.min(sav))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode in ("check", "trace"):
        m, w = score_band(band_2021())
        print(f"2021 TRAINING band : godeye-vs-reactive mean={m*100:5.1f}% worst={w*100:5.1f}%")
        m, w = score_band(band_2020(17512))
        print(f"2020 OOB band S=17512: godeye-vs-reactive mean={m*100:5.1f}% worst={w*100:5.1f}%")
    else:  # scan
        print("scanning 2020 bands (4-window prefilter, top-8 rescored on 10)...")
        pre = []
        for S in range(12, len(BASE20[12]) - 1500, 250):
            m, w = score_band(band_2020(S), windows=OFFSETS[:4])
            pre.append((m, w, S))
            if m > 0.15:
                print(f"  S={S:6d} mean4={m*100:5.1f}% worst4={w*100:5.1f}%")
        pre.sort(reverse=True)
        print("--- top-8 full 10-window scores")
        for m4, w4, S in pre[:8]:
            m, w = score_band(band_2020(S))
            agg = sum(BASE20[t][S:S + 1500] for t in BASE_TIDS)
            print(f"  S={S:6d} mean={m*100:5.1f}% worst={w*100:5.1f}% bandmean={agg.mean():.0f}kW trough%={(agg < 0.2*agg.mean()).mean()*100:4.1f}")
