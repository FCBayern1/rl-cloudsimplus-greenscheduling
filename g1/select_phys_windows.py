"""Choose the three 2021 calibration windows on the physical time base.

Selection rule, frozen by Codex 2026-08-31 before any arm's carbon was seen:

    candidate quantity   three-site total green energy over the cell, each wind row
                         weighted by the seconds the episode actually spends in it
    quantiles            empirical p10 / p50 / p90
    pick                 the real candidate nearest each target quantile
    tie-break            smallest offset
    constraint           the per-turbine read intervals of the three windows are disjoint
    if disjointness forces a substitution
                         among all legal triples, minimise the sum of quantile-rank errors,
                         then break ties by the lexicographic order of the offsets

min and max are deliberately not used: they estimate annual extreme events rather than a
low or high wind regime, and the windless end is mechanism-neutral, so putting it in the
main verdict would dilute the direction gate without testing anything. The windless window
is registered separately as a negative control.
"""
import csv
import hashlib
import json
import os

import numpy as np
import yaml

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
WD = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified")
STRIDE, ROW_SECONDS, TERMINAL_STEPS, CLOCK0, YEAR = 1009, 600.0, 12000, 13.0, 2021
QUANTILES = (10, 50, 90)

blk = yaml.safe_load(open(os.path.join(REPO, "config_C.yml")))["experiment_g1eval_matchedvan"]
dcs = sorted(blk["datacenters"], key=lambda x: x["datacenter_id"])
tz = {d["datacenter_id"]: int(d.get("time_zone_offset_rows", 0)) for d in dcs}
turb = {d["datacenter_id"]: (d.get("turbine_ids") or []) for d in dcs}
divisor = float(blk.get("compressed_power_divisor") or 1500.0)
rng = int(blk["green_episode_offset_range"])

series = {}
for ts in turb.values():
    for t in ts:
        if t not in series:
            with open(f"{WD}/Turbine_{t}_{YEAR}.csv") as f:
                series[t] = np.array([float(r["power_kw"] or 0.0) for r in csv.DictReader(f)])
n_rows = min(len(v) for v in series.values())

steps = np.arange(TERMINAL_STEPS, dtype=np.float64)
rel_rows, rel_secs = np.unique(
    np.floor((CLOCK0 + steps) / ROW_SECONDS).astype(np.int64), return_counts=True)
span, max_tz = int(rel_rows[-1]) + 1, max(tz.values())
need = span + max_tz
usable = min(rng, n_rows - need - 1)
dc_acc = {d: (sum(series[t] for t in ts) if ts else None) for d, ts in turb.items()}


def green_wh(offset):
    total = 0.0
    for d, acc in dc_acc.items():
        if acc is None:
            continue
        rows = np.clip(offset + tz[d] + rel_rows, 0, len(acc) - 1)
        total += float(np.sum(acc[rows] * 1000.0 / divisor * rel_secs / 3600.0))
    return total


def read_interval(offset):
    """Lowest and highest CSV row any turbine of this cell will read."""
    los = [offset + tz[d] for d, a in dc_acc.items() if a is not None]
    return min(los), max(los) + span - 1


ks = sorted({k for k in range(1, 4000) if (STRIDE * k) % rng <= usable})
offs = np.array([(STRIDE * k) % rng for k in ks])
vals = np.array([green_wh(o) for o in offs])
ranks = np.argsort(np.argsort(vals)) / (len(vals) - 1) * 100.0
print(f"candidates {len(ks)}   rows per cell {len(rel_rows)}   window occupies {need} rows")
print(f"green Wh  min={vals.min():.1f}  p10={np.percentile(vals,10):.1f}  "
      f"p50={np.percentile(vals,50):.1f}  p90={np.percentile(vals,90):.1f}  max={vals.max():.1f}")


def nearest(q):
    target = np.percentile(vals, q)
    err = np.abs(vals - target)
    cand = np.flatnonzero(err == err.min())
    return int(cand[np.argmin(offs[cand])])          # tie-break: smallest offset


def disjoint(idxs):
    iv = sorted(read_interval(int(offs[i])) for i in idxs)
    return all(iv[j][1] < iv[j + 1][0] for j in range(len(iv) - 1))


first = [nearest(q) for q in QUANTILES]
if disjoint(first):
    chosen, note = first, "nearest to each quantile, disjoint as chosen"
else:
    best = None
    for a in range(len(ks)):
        for b in range(len(ks)):
            for c in range(len(ks)):
                trip = (a, b, c)
                if len({a, b, c}) < 3 or not disjoint(trip):
                    continue
                err = sum(abs(ranks[i] - q) for i, q in zip(trip, QUANTILES))
                key = (err, tuple(sorted(int(offs[i]) for i in trip)))
                if best is None or key < best[0]:
                    best = (key, trip)
    chosen, note = list(best[1]), "substituted: minimal quantile-rank error over legal triples"

names = ["low", "mid", "high"]
out = {"stride": STRIDE, "range": rng, "row_seconds": ROW_SECONDS, "clock0": CLOCK0,
       "terminal_steps": TERMINAL_STEPS, "rows_per_cell": int(len(rel_rows)),
       "row_span": span, "max_tz": max_tz, "quantiles": list(QUANTILES),
       "selection_note": note, "windows": {}}
print(f"\n{note}")
print(f"{'name':>5} {'k':>5} {'offset':>7} {'green Wh':>11} {'rank pct':>9} {'read rows':>16}")
for nm, i in zip(names, chosen):
    o = int(offs[i]); lo, hi = read_interval(o)
    out["windows"][nm] = {"k": int(ks[i]), "offset": o, "green_wh": float(vals[i]),
                          "rank_pct": float(ranks[i]), "read_rows": [lo, hi]}
    print(f"{nm:>5} {ks[i]:>5} {o:>7} {vals[i]:>11.2f} {ranks[i]:>9.1f} {str([lo,hi]):>16}")
print(f"disjoint: {disjoint(chosen)}")

zi = int(np.argmin(vals))
lo, hi = read_interval(int(offs[zi]))
out["negative_control"] = {"k": int(ks[zi]), "offset": int(offs[zi]),
                           "green_wh": float(vals[zi]), "read_rows": [lo, hi],
                           "role": "mechanism-neutral, excluded from the direction gate"}
print(f"negative control (windless): k={ks[zi]} offset={offs[zi]} green={vals[zi]:.2f}")
out["disjoint"] = bool(disjoint(chosen))
out["selection_hash"] = hashlib.sha256(
    json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]
print(f"selection hash: {out['selection_hash']}")
json.dump(out, open(os.path.join(REPO, "g1/phys_windows.json"), "w"), indent=2, sort_keys=True)
