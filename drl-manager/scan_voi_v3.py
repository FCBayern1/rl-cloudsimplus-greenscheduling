#!/usr/bin/env python3
"""VoI scanner v3 -- the corrected pre-registration instrument.

Lessons baked in (2026-08-10 autopsy of six failed testbeds):
  1. METRIC: pooled ABSOLUTE carbon gap between clairvoyant and the BEST blind
     policy, over the 10 deployment windows. Never mean-of-per-window-ratios
     (near-zero brown denominators inflated the historical 73.9% to fantasy).
  2. BLIND CLASS must be strong: best of {drain, reactive, threshold sweep,
     climatology planner}. A heuristic-grade blind arm overstates VoI (the
     twice-stepped-on rake).
  3. COST MODEL includes the cheap-brown cushion: green at 0.01, brown at the
     green-DC brown intensity (axis). The old scanner counted brown units,
     ignoring that DC0's 0.08 floor made wrong timing nearly free.
  4. Drop-mode does not change the optima (starting at latest-start equals the
     forced fallback); it is a training-shaping device, so it is NOT an axis.

Axes: stretch x Lmin x slack x brown intensity. PASS gate: VoI >= 30% pooled
absolute cost AND blind completion feasible (all jobs start by latest).
"""
import csv
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
BASE_TIDS = [12, 36, 95, 91, 96, 101, 103]
T, NJOBS, STRIDE = 7200, 2000, 5
RHO = 1.29
GREEN_COST, KAPPA_GREEN = 0.01, 0.01
OFFSETS = [(1009 * k) % 4800 for k in range(10)]
rng = np.random.default_rng(11)

BASE = {t: np.array([float(r["power_kw"]) for r in csv.DictReader(open(GATE / f"Turbine_{t}_2021.csv"))])
        for t in BASE_TIDS}

import pyarrow.parquet as pq
tab = pq.read_table(REPO / "data/v2026_gpu/asi_opensource_job_execution_summary/part-000.parquet",
                    columns=["duration_hours", "priority_class"])
dur = tab.column("duration_hours").to_numpy(zero_copy_only=False).astype(float)
pc = tab.column("priority_class").to_numpy(zero_copy_only=False)
DUR_LP = dur[(pc == "LP") & np.isfinite(dur) & (dur > 0)] * 3600.0

def make_band(stretch):
    agg = sum(BASE[t][12:12 + 15000 // stretch + 800] for t in BASE_TIDS)
    return np.repeat(agg, stretch)

def make_jobs(lmin_s, slack_lo, slack_hi):
    d = DUR_LP[DUR_LP >= lmin_s] if lmin_s > 0 else DUR_LP
    L = np.clip(np.round(rng.choice(d, NJOBS) / 20.0), 1, 1000).astype(int)
    slack = rng.integers(slack_lo, slack_hi + 1, NJOBS)
    arr = np.array([rng.integers(0, max(1, T - int(l) - int(sl))) for l, sl in zip(L, slack)])
    return L, arr, arr + slack

def cost_of(load, g, brown_i):
    green_used = np.minimum(load[:T], g).sum()
    brown_used = np.maximum(0.0, load[:T] - g).sum()
    return KAPPA_GREEN * green_used + brown_i * brown_used

def run_drain(g, L, arr, latest):
    load = np.zeros(T + 1100)
    for j in range(NJOBS): load[arr[j]:arr[j] + L[j]] += 1.0
    return load

def run_reactive(g, L, arr, latest, thresh=1.0):
    load = np.zeros(T + 1100)
    order = sorted(range(NJOBS), key=lambda j: arr[j]); waiting, pi = [], 0
    for t in range(T):
        while pi < NJOBS and arr[order[pi]] <= t: waiting.append(order[pi]); pi += 1
        waiting.sort(key=lambda j: latest[j]); still = []
        for j in waiting:
            if t >= latest[j] or load[t] + thresh <= g[t]: load[t:t + L[j]] += 1.0
            else: still.append(j)
        waiting = still
    return load

def run_climatology(g_hist_mean, g, L, arr, latest):
    """Blind planner on the band's mean daily profile (knows climate, not weather)."""
    prof = g_hist_mean
    load = np.zeros(T + 1100)
    def pick(j, load):
        f = np.minimum(1.0, np.maximum(0.0, load[:T] + 1.0 - prof))
        cs = np.concatenate([[0.0], np.cumsum(f)])
        lo, hi = int(arr[j]), int(latest[j]); cands = list(range(lo, hi + 1, STRIDE))
        if cands[-1] != hi: cands.append(hi)
        return cands[int(np.argmin([cs[min(s + L[j], T)] - cs[s] for s in cands]))]
    for j in np.argsort(latest):
        s = pick(j, load); load[s:s + L[j]] += 1.0
    return load


def run_hazard(gs_band, scale, g, L, arr, latest, theta):
    """Strongest blind: knows the BAND's peak-duration statistics (not the
    realization). At a peak onset, commit job j only if the empirical
    probability of the peak lasting >= L_j is >= theta; otherwise hold for the
    next onset, falling back to latest-start. Sweeps theta outside."""
    import numpy as np
    demand = g.mean() / 1.0
    thr = np.median(gs_band) * scale
    above = gs_band * scale >= thr
    runs, cur = [], 0
    for x in above:
        if x: cur += 1
        elif cur: runs.append(cur); cur = 0
    if cur: runs.append(cur)
    runs = np.sort(np.array(runs if runs else [1]))
    def p_dur_ge(l):
        return (runs >= l).mean()
    load = np.zeros(T + 1100)
    onset = np.zeros(T, dtype=bool)
    ab = g >= thr
    onset[1:] = ab[1:] & ~ab[:-1]; onset[0] = ab[0]
    order = sorted(range(NJOBS), key=lambda j: arr[j]); waiting, pi = [], 0
    for t in range(T):
        while pi < NJOBS and arr[order[pi]] <= t: waiting.append(order[pi]); pi += 1
        waiting.sort(key=lambda j: latest[j]); still = []
        for j in waiting:
            commit = t >= latest[j] or (ab[t] and p_dur_ge(L[j]) >= theta and load[t] + 1.0 <= g[t])
            if commit: load[t:t + L[j]] += 1.0
            else: still.append(j)
        waiting = still
    return load

def run_clairvoyant(g, L, arr, latest):
    load = np.zeros(T + 1100); st = {}
    def pick(j, load):
        f = np.minimum(1.0, np.maximum(0.0, load[:T] + 1.0 - g))
        cs = np.concatenate([[0.0], np.cumsum(f)])
        lo, hi = int(arr[j]), int(latest[j]); cands = list(range(lo, hi + 1, STRIDE))
        if cands[-1] != hi: cands.append(hi)
        return cands[int(np.argmin([cs[min(s + L[j], T)] - cs[s] for s in cands]))]
    order = np.argsort(latest)
    for j in order: s = pick(j, load); st[j] = s; load[s:s + L[j]] += 1.0
    for j in order:
        s0 = st[j]; load[s0:s0 + L[j]] -= 1.0
        s = pick(j, load); st[j] = s; load[s:s + L[j]] += 1.0
    return load

def scan_one(stretch, lmin, slack_lo, slack_hi, brown_i):
    gs = make_band(stretch)
    L, arr, latest = make_jobs(lmin, slack_lo, slack_hi)
    demand_mean = L.sum() / T
    scale = RHO * demand_mean / gs.mean()
    costs = {"drain": 0.0, "react1": 0.0, "react2": 0.0, "clim": 0.0, "hz3": 0.0, "hz5": 0.0, "hz7": 0.0, "oracle": 0.0}
    prof_mean = None
    for off in OFFSETS:
        g = gs[off:off + T] * scale
        if g.size < T: g = np.pad(g, (0, T - g.size))
        if prof_mean is None:
            prof_mean = np.mean([gs[o:o + T] * scale for o in OFFSETS if gs[o:o+T].size == T], axis=0)
        costs["drain"]  += cost_of(run_drain(g, L, arr, latest), g, brown_i)
        costs["react1"] += cost_of(run_reactive(g, L, arr, latest, 1.0), g, brown_i)
        costs["react2"] += cost_of(run_reactive(g, L, arr, latest, 0.5), g, brown_i)
        costs["clim"]   += cost_of(run_climatology(prof_mean, g, L, arr, latest), g, brown_i)
        costs["hz3"]  += cost_of(run_hazard(gs, scale, g, L, arr, latest, 0.3), g, brown_i)
        costs["hz5"]  += cost_of(run_hazard(gs, scale, g, L, arr, latest, 0.5), g, brown_i)
        costs["hz7"]  += cost_of(run_hazard(gs, scale, g, L, arr, latest, 0.7), g, brown_i)
        costs["oracle"] += cost_of(run_clairvoyant(g, L, arr, latest), g, brown_i)
    BLINDS = ["drain", "react1", "react2", "clim", "hz3", "hz5", "hz7"]
    blind = min(costs[k] for k in BLINDS)
    voi = (blind - costs["oracle"]) / max(1e-9, blind)
    return voi, blind, costs["oracle"], min(BLINDS, key=lambda k: costs[k])

if __name__ == "__main__":
    print(f"{'stretch':>7} {'Lmin(s)':>7} {'slack':>11} {'brown':>5} | {'VoI%':>6} {'bestblind':>9}")
    results = []
    for stretch in [4, 6, 10]:
        for lmin in [0, 600, 1200]:
            for slack in [(600, 1200), (1200, 3000)]:
                for brown_i in [0.08, 0.35, 0.55]:
                    voi, b, o, bname = scan_one(stretch, lmin, slack[0], slack[1], brown_i)
                    tag = "  <-- PASS" if voi >= 0.30 else ""
                    print(f"{stretch:>7} {lmin:>7} {str(slack):>11} {brown_i:>5} | {voi*100:>5.1f} {bname:>9}{tag}", flush=True)
                    results.append((voi, stretch, lmin, slack, brown_i, bname))
    results.sort(reverse=True)
    print("\nTOP5:")
    for voi, st, lm, sl, bi, bn in results[:5]:
        print(f"  VoI={voi*100:.1f}%  stretch={st} Lmin={lm} slack={sl} brown={bi} (bestblind={bn})")
