#!/usr/bin/env python3
"""Analytic design scan for a temporal-gamble scenario where the forecast is
load-bearing. Pure numpy, no simulator, no training.

WHY THIS SHAPE (evidence-backed recipe, 2026-08-07):
  - forecast value = value of irreversible commitment under uncertainty;
  - spatial routing is hedgeable (128-way splitting) -> spatial lever dead;
  - the unhedgeable decision is WHEN: "run now on brown, or gamble that green
    returns before my deadline and lasts my whole runtime?"
  - that gamble needs: synchronized green (no spatial escape), troughs comparable
    to job length and slack (rwtight needed a 10x time stretch), tight slack,
    and scarcity. All four are grid axes here.

MODEL: one synchronized green pool g(t) (the 4 green DCs' turbines at the SAME
offset, stretched by F, scaled to a supply ratio rho = mean green / mean demand).
Jobs draw 1 power-unit for L steps once started; brown(t) = max(0, load - g).
Every policy starts each job no later than deadline - L (the deadline backstop),
so completion is 100% by construction and carbon is iso-completion by design.

POLICIES:
  drain     start at arrival (no forecast, no waiting)
  reactive  wait until green headroom exists, else forced at latest start
            (honest stand-in for a trained no-forecast policy)
  godeye    knows the true g(t); each job greedily picks the start time in
            [arrival, latest] minimising the brown it adds given load so far

PRE-REGISTERED CRITERION: a config is a PASS if mean godeye-vs-reactive carbon
saving >= 25% across 8 closed-book windows (and reactive vs drain is reported
so nobody mistakes drain-inflation for forecast value). No PASS anywhere in the
grid -> stop hunting, accept the frozen-paper framing.
"""
import csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
TURBINES = [7012, 7036, 7095, 7091, 7096, 7101, 7103]
T = 7200
NJOBS = 1000
WINDOWS = 8
STRIDE = 5  # candidate start-time stride for godeye (keeps it a lower bound)

def load_raw_green():
    tot = None
    for t in TURBINES:
        v = np.array([float(r["power_kw"]) for r in csv.DictReader(open(GATE / f"Turbine_{t}_2021.csv"))])
        tot = v if tot is None else tot + v
    return tot

def load_lp_steps(rng, n):
    tab = pq.read_table(REPO / "data/v2026_gpu/asi_opensource_job_execution_summary/part-000.parquet",
                        columns=["duration_hours", "priority_class"])
    d = tab.column("duration_hours").to_numpy(zero_copy_only=False).astype(float)
    pc = tab.column("priority_class").to_numpy(zero_copy_only=False)
    d = d[(pc == "LP") & np.isfinite(d) & (d > 0)] * 3600.0
    return np.clip(np.round(rng.choice(d, n) / 20.0), 1, 1000).astype(int)

def simulate(g, L, arr, latest, order_by, pick_start):
    """Generic fleet sim: jobs processed in `order_by` order; each job's start
    chosen by pick_start(job_idx, load) given the load committed so far."""
    load = np.zeros(T)
    for j in order_by:
        s = pick_start(j, load)
        load[s:s + L[j]] += 1.0
    return np.maximum(0.0, load - g).sum()

def run_config(raw, F, slack_lo, slack_hi, rho, rng):
    gs = np.repeat(raw, F)
    L = load_lp_steps(rng, NJOBS)
    demand_mean = L.sum() / T
    savings_r, savings_d = [], []
    for w in range(WINDOWS):
        off = int(rng.integers(0, max(1, len(gs) - T - 1)))
        g = gs[off:off + T].copy()
        g = g * (rho * demand_mean / max(1e-9, g.mean()))
        slack = rng.integers(slack_lo, slack_hi + 1, NJOBS)
        arr = np.array([rng.integers(0, max(1, T - int(l) - int(sl))) for l, sl in zip(L, slack)])
        latest = arr + slack  # latest allowed start (deadline = arr + L + slack)

        # drain
        c_drain = simulate(g, L, arr, latest, np.argsort(arr), lambda j, load: arr[j])

        # reactive: step through time, start waiting jobs when green headroom
        # exists, force at latest start
        load = np.zeros(T)
        pending = sorted(range(NJOBS), key=lambda j: arr[j])
        waiting, pi = [], 0
        for t in range(T):
            while pi < NJOBS and arr[pending[pi]] <= t:
                waiting.append(pending[pi]); pi += 1
            waiting.sort(key=lambda j: latest[j])
            still = []
            for j in waiting:
                if t >= latest[j] or load[t] + 1.0 <= g[t]:
                    load[t:t + L[j]] += 1.0
                else:
                    still.append(j)
            waiting = still
        c_react = np.maximum(0.0, load - g).sum()

        # godeye: greedy per job, true future known. Urgency order (earliest
        # latest-start first) plus ONE re-planning sweep to fix pile-ups: the
        # arrival-order single-pass version provably underestimates the oracle
        # (it scored BELOW reactive in some cells, impossible for a true oracle).
        def pick(j, load):
            f = np.minimum(1.0, np.maximum(0.0, load + 1.0 - g))  # brown added if run at t
            cs = np.concatenate([[0.0], np.cumsum(f)])
            lo, hi = int(arr[j]), int(latest[j])
            cands = list(range(lo, hi + 1, STRIDE))
            if cands[-1] != hi:
                cands.append(hi)
            costs = [cs[s + L[j]] - cs[s] for s in cands]
            return cands[int(np.argmin(costs))]
        order = np.argsort(latest)
        load = np.zeros(T)
        starts = {}
        for j in order:                      # pass 1: commit by urgency
            s = pick(j, load); starts[j] = s; load[s:s + L[j]] += 1.0
        for j in order:                      # pass 2: re-plan each job once
            s0 = starts[j]; load[s0:s0 + L[j]] -= 1.0
            s = pick(j, load); starts[j] = s; load[s:s + L[j]] += 1.0
        c_god = np.maximum(0.0, load - g).sum()

        savings_r.append((c_react - c_god) / max(1e-9, c_react))
        savings_d.append((c_drain - c_god) / max(1e-9, c_drain))
    return (float(np.mean(savings_r)), float(np.min(savings_r)),
            float(np.mean(savings_d)))

def main():
    import sys
    focus = "--focus" in sys.argv
    global WINDOWS
    raw = load_raw_green()
    rng = np.random.default_rng(11)
    if focus:
        # refinement round around the round-1 PASS cell (F=1, U[800,2000], rho=1.2).
        # FINAL pre-registered gate: mean >= 25% AND worst window >= +5%.
        # Hard verdict after this round, no further rounds.
        WINDOWS = 16
        grid_F = [1, 2]
        grid_slack = [(600, 1500), (800, 2000), (1200, 3000)]
        grid_rho = [1.0, 1.1, 1.2, 1.3]
    else:
        grid_F = [1, 5, 10]
        grid_slack = [(100, 400), (300, 900), (800, 2000)]
        grid_rho = [0.6, 0.9, 1.2]
    print(f"{'F':>3} {'slack':>10} {'rho':>4} | {'god-vs-REACTIVE':>16} {'(worst win)':>11} {'god-vs-drain':>12}  verdict")
    best = None
    for F in grid_F:
        for slack_lo, slack_hi in grid_slack:
            for rho in grid_rho:
                mr, worst, md = run_config(raw, F, slack_lo, slack_hi, rho, rng)
                stable = mr >= 0.25 and worst >= 0.05
                tag = "PASS-STABLE" if stable else ("pass-mean" if mr >= 0.25 else ("close" if mr >= 0.15 else ""))
                print(f"{F:>3} U[{slack_lo:>4},{slack_hi:>4}] {rho:>4.1f} | {100*mr:>15.1f}% {100*worst:>10.1f}% {100*md:>11.1f}%  {tag}", flush=True)
                if best is None or mr > best[0]:
                    best = (mr, worst, F, (slack_lo, slack_hi), rho)
    print(f"\nBEST: godeye-vs-reactive mean {100*best[0]:.1f}% (worst {100*best[1]:.1f}%) at stretch={best[2]}, slack=U{best[3]}, rho={best[4]}")
    print("FINAL gate: PASS-STABLE = mean >=25% AND worst window >=+5%. None -> stop hunting.")

if __name__ == "__main__":
    main()
