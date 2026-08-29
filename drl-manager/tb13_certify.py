"""Per-job, capacity-constrained certification of a candidate testbed.

The fluid screen in tb13_search.py lets movable load split across sites
instantly, which is exactly the relaxation the question is about, so its gap is
an upper bound and can only rule configurations out. This model commits whole
jobs to one site for their whole runtime, respects per-site capacity, and gives
both policies the same deadline and backstop machinery. The only difference
between them is what they may read.

  blind        at each step, for the jobs eligible now, chooses a site with
               green headroom, or waits if none has any and slack allows.
               Reads g_d(t') only for t' <= t.
  clairvoyant  the same admission rule, but scores a (start, site) pair by the
               brown energy the job would actually draw, using the whole future
               green trace.

Neither is optimal. Both are greedy, so the reported gap is a lower bound on
what perfect information is worth, which is the safe direction for a screen
that is looking for a large gap.

Anchors, per the frozen spec: the model must reproduce a known positive (TB12,
non-fluid, single site) and a known negative (C-regime as configured). If
either fails the model cannot be used to judge anything.
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "cloudsimplus-gateway/src/main/resources"
WIND = RES / "windProduction/simplified"
IDLE_W, PEAK_W, HOST_PES = 51.4, 214.0, 64
DYN_PER_PE = (PEAK_W - IDLE_W) / HOST_PES


def wind_series(turbines, offset, steps, scale, year=2021):
    acc = None
    for t in turbines:
        f = WIND / f"Turbine_{t}_{year}.csv"
        v = np.array([float(x["power_kw"] or 0) for x in csv.DictReader(open(f))])
        acc = v if acc is None else acc + v
    seg = acc[offset:offset + steps]
    if len(seg) < steps:
        seg = np.pad(seg, (0, steps - len(seg)))
    return seg * 1000.0 / scale


def load_jobs(trace, mips, util=0.5, limit=None):
    rows = list(csv.DictReader(open(RES / trace)))
    if limit:
        rows = rows[:limit]
    a = np.array([float(r["arrival_time"]) for r in rows])
    ln = np.array([float(r["length"]) for r in rows])
    pes = np.array([float(r["pes_required"]) for r in rows]).astype(int)
    ddl = np.array([float(r["deadline"]) for r in rows])
    run = np.maximum(1, np.round(ln / (pes * mips) / util)).astype(int)
    return a.astype(int), run, pes, ddl.astype(int)


def simulate(G, cap, intens, arr, run, pes, ddl, steps, clair, static_w=None, batch=None,
             summary=False):
    """batch: if set, every job admitted in the same step must go to the SAME
    site, which is what a 128-slot routing action with one shared advantage can
    express. Comparing batch=None against batch=1 separates 'the forecast has
    no value' from 'the action space cannot spend it'."""
    """Greedy admission. Returns total carbon in kg."""
    D = len(cap)
    load = np.zeros((D, steps + max(run) + 2))     # committed PEs
    order = np.argsort(arr)
    place = np.full(len(arr), -1)
    start = np.full(len(arr), -1)
    pending = []
    ai = 0
    for t in range(steps):
        while ai < len(order) and arr[order[ai]] <= t:
            pending.append(order[ai]); ai += 1
        if not pending:
            continue
        still = []
        forced = None      # under batching, the site chosen by the first job binds the rest
        for j in pending:
            r, p = run[j], pes[j]
            latest = ddl[j] - r
            feas = [d for d in range(D) if (load[d, t:t + r] + p <= cap[d]).all()]
            if batch and forced is not None:
                feas = [d for d in feas if d == forced]
            if not feas:
                (still if t < latest else still).append(j)
                continue
            if clair:
                # brown energy this job would draw at each feasible site now
                cost = []
                for d in feas:
                    if summary:
                        # Only the MEAN future green over the job window, which is
                        # what dc_future_long_mean carries. min(demand, green) is
                        # nonlinear in the profile, so a level statistic cannot
                        # express when green actually covers the draw.
                        gm = float(G[d, t:t + r].mean())
                        lm = float((load[d, t:t + r] * DYN_PER_PE).mean())
                        head = max(0.0, p * DYN_PER_PE - max(0.0, gm - lm)) * r
                        cost.append(head * intens[d])
                    else:
                        head = np.maximum(0.0, p * DYN_PER_PE
                                          - np.maximum(0.0, G[d, t:t + r] - load[d, t:t + r] * DYN_PER_PE))
                        cost.append(head.sum() * intens[d])
                d_best = feas[int(np.argmin(cost))]
                # would waiting one step be cheaper? only if slack allows
                if t < latest:
                    fut = []
                    for d in range(D):
                        s2 = min(t + 1, latest)
                        if (load[d, s2:s2 + r] + p <= cap[d]).all():
                            if summary:
                                gm = float(G[d, s2:s2 + r].mean())
                                lm = float((load[d, s2:s2 + r] * DYN_PER_PE).mean())
                                fut.append(max(0.0, p*DYN_PER_PE - max(0.0, gm-lm)) * r * intens[d])
                            else:
                                head = np.maximum(0.0, p * DYN_PER_PE
                                                  - np.maximum(0.0, G[d, s2:s2 + r] - load[d, s2:s2 + r] * DYN_PER_PE))
                                fut.append(head.sum() * intens[d])
                    if fut and min(fut) < min(cost) * 0.999:
                        still.append(j); continue
            else:
                # blind: current green headroom only, cheapest site by intensity
                head = [(max(0.0, G[d, t] - load[d, t] * DYN_PER_PE), -intens[d], d) for d in feas]
                head.sort(reverse=True)
                if head[0][0] <= 0.0 and t < latest:
                    still.append(j); continue
                d_best = head[0][2]
            load[d_best, t:t + r] += p
            place[j], start[j] = d_best, t
            if batch and forced is None:
                forced = d_best
        pending = still
    # settle anything never placed at its latest start
    for j in pending:
        r, p = run[j], pes[j]
        d = int(np.argmin(intens))
        s = min(max(arr[j], 0), steps - 1)
        load[d, s:s + r] += p
        place[j], start[j] = d, s
    # Idle hosts draw power whatever the policy does, and that draw takes green
    # before any job sees it. Leaving it out inflates the relative gap several
    # times over, because the unmovable brown it creates is what dilutes the
    # difference between two placements.
    carbon = 0.0
    for d in range(D):
        dem = load[d, :steps] * DYN_PER_PE + (static_w[d] if static_w is not None else 0.0)
        brown = np.maximum(0.0, dem - G[d])
        carbon += brown.sum() / 3600.0 / 1000.0 * intens[d]        # Wh -> kWh -> kg
    return carbon, place, start


def run_case(name, turbines, divisor, trace, cap, intens, offset, steps, mips, limit=None):
    G = np.array([wind_series(t, offset, steps, divisor) for t in turbines])
    arr, run, pes, ddl = load_jobs(trace, mips, limit=limit)
    # static draw split across sites in proportion to fleet size, calibrated so
    # the whole fleet matches the 332 W measured on C-regime
    tot_cap = sum(cap)
    stat = [332.0 * c / tot_cap for c in cap]
    cb, _, _ = simulate(G, cap, intens, arr, run, pes, ddl, steps, clair=False, static_w=stat)
    cc, _, _ = simulate(G, cap, intens, arr, run, pes, ddl, steps, clair=True, static_w=stat)
    ccS, _, _ = simulate(G, cap, intens, arr, run, pes, ddl, steps, clair=True, static_w=stat, summary=True)
    gS = 100*(cb-ccS)/cb if cb>0 else float("nan")
    gap = 100 * (cb - cc) / cb if cb > 0 else float("nan")
    print(f"{name:<24}{len(arr):>6}{cb:>10.5f}{cc:>10.5f}{gap:>9.2f}%{ccS:>12.5f}{gS:>9.2f}%")
    return gap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--turbines", default=None, help="e.g. 8,9,29")
    ap.add_argument("--divisor", type=float, default=10000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config_C.yml"))
    C = cfg["experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan"]
    caps = [sum(v for k, v in d.items() if k.startswith("host_count_")) * HOST_PES
            for d in C["datacenters"][:3]]
    intens = [d["brown_carbon_factor"] for d in C["datacenters"][:3]]
    print(f"{'case':<24}{'jobs':>6}{'盲态':>10}{'全知(精确)':>10}{'间隙':>9}{'全知(仅均值)':>12}{'间隙':>9}")
    if a.anchors:
        run_case("anchor 负例 C-regime", [[12, 36], [95, 91], [96]], 1500.0,
                 C["cloudlet_trace_file"], caps, intens, 19184, 7200, 40000)
    if a.turbines:
        ts = [[int(x)] for x in a.turbines.split(",")]
        run_case(f"候选 {a.turbines}", ts, a.divisor,
                 C["cloudlet_trace_file"], caps, intens, 19184, 7200, 40000)
