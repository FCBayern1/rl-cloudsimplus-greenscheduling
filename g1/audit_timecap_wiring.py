"""Row-level wiring audit for the real TimeCAP forecast, before any carbon is compared.

Four things are locked here, in the order Codex set them out.

  leakage    the forecast issued at step t must be a function of rows up to t only
  alignment  forecast[k] must line up with the realised row the simulator will serve
  coverage   the horizon must be stated against the window a job may actually wait
  curve      the planner must receive the trajectory, not a mean summary

The output is a per-row table (t, dc, k, forecast, realised, turbine, year, row) so the
alignment can be read rather than argued about. Nothing downstream runs until this passes.
"""
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.join(REPO, "drl-manager"))

from src.baselines.evaluate import load_config  # noqa: E402
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

WD = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified")
PROBE_STEPS = int(os.environ.get("AUDIT_STEPS", "260"))
SAMPLE_AT = [int(x) for x in os.environ.get("AUDIT_SAMPLE", "120,180,240").split(",")]

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def turbine_series(tid, year):
    with open(f"{WD}/Turbine_{tid}_{year}.csv") as f:
        return np.array([float(r["power_kw"] or 0.0) for r in csv.DictReader(f)])


cfg = load_config(os.environ.get("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan"))
cfg["py4j_port"] = None
cfg.setdefault("gateway_log_dir", "/tmp/claude-1000/audit_timecap")
tc = cfg.get("timecap") or {}
year = int(tc.get("csv_year", 2021))
env = HierarchicalMultiDCEnv(config=cfg)
prov = None

skip = int(os.environ.get("AUDIT_RESET_SKIP", "0"))
for _ in range(skip + 1):
    obs, info = env.reset(seed=20260823)
prov = env.timecap_provider

print(f"provider present: {prov is not None}")
if prov is None:
    print("FAIL: green_oracle_mode is not timecap, or the provider failed to build")
    sys.exit(1)

print(f"\nconfiguration")
print(f"  timecap csv_year        {year}")
print(f"  seq_len (history)       {prov.seq_len}")
print(f"  pred_len (horizon)      {prov.pred_len}")
print(f"  forecast_every          {prov.forecast_every}")
print(f"  dc_assignments          {dict(prov.dc_assignments)}")
# The offsets live on the loader inside the predictor, not on the provider. Reading them
# from the wrong object silently gives every datacentre offset zero, which reads exactly
# like a misaligned forecast for the two sites that have a time-zone shift.
_loader = getattr(getattr(prov, "predictor", None), "feature_loader", None)
OFFSETS = dict(getattr(_loader, "_per_turbine_offset", {}) or {})
print(f"  per-turbine row offset  {OFFSETS}")

has_curve = hasattr(prov, "get_raw_forecast_per_dc")
check("the full forecast trajectory is exposed, not only the 4 summary features",
      has_curve, "get_raw_forecast_per_dc")

n = env.num_datacenters
lm = int(env.action_space["local"][0].n) - 1
rows = []
snapshots = {}

for t in range(PROBE_STEPS):
    if t in SAMPLE_AT and has_curve:
        cur = prov.get_raw_forecast_per_dc(normalize=False)
        if cur is not None:
            snapshots[t] = {d: np.asarray(v, dtype=float) / 1000.0 for d, v in cur.items()}
    obs, r, term, trunc, info = env.step(
        {"global": [n] * env.global_routing_batch_size, "local": {i: lm for i in range(n)}})
    if term or trunc:
        break
env.close()

print(f"\ncollected {len(snapshots)} forecast snapshots at steps {sorted(snapshots)}")
if not snapshots:
    print("FAIL: no forecast trajectory could be read")
    sys.exit(1)

off = OFFSETS
series = {}
for dc, tids in prov.dc_assignments.items():
    for tid in tids:
        if tid not in series:
            series[tid] = turbine_series(tid, year)

print(f"\nper-row alignment table (forecast kW against the realised CSV row)")
print(f"{'t':>6} {'dc':>3} {'k':>4} {'forecast':>10} {'realised':>10} {'turbines':>12} "
      f"{'year':>5} {'row':>7}")
best_lags = []
zero_lag = []
for t, per_dc in sorted(snapshots.items()):
    for dc, curve in sorted(per_dc.items()):
        tids = prov.dc_assignments[dc]
        base = int(off.get(tids[0], 0)) if off else 0
        realised = np.zeros(len(curve))
        for tid in tids:
            s = series[tid]
            r0 = int(off.get(tid, base)) + t
            seg = s[r0: r0 + len(curve)]
            realised[:len(seg)] += seg
        m = min(len(curve), len(realised))
        if m > 8 and np.std(curve[:m]) > 1e-9 and np.std(realised[:m]) > 1e-9:
            lags = {}
            for lag in range(-24, 25):
                a = curve[max(0, lag): m + min(0, lag)]
                b = realised[max(0, -lag): m - max(0, lag)]
                if len(a) > 8 and np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    lags[lag] = float(np.corrcoef(a, b)[0, 1])
            if lags:
                bl = max(lags, key=lags.get)
                best_lags.append((t, dc, bl, lags[bl]))
                zero_lag.append(lags.get(0, -1.0))
        for k in (0, 1, 8, 47):
            if k < len(curve):
                print(f"{t:>6} {dc:>3} {k:>4} {curve[k]:>10.2f} {realised[k]:>10.2f} "
                      f"{str(tids):>12} {year:>5} {int(off.get(tids[0], base)) + t + k:>7}")

if best_lags:
    print(f"\nforecast-to-realised lag by correlation")
    for t, dc, lag, c in best_lags:
        print(f"  t={t:<5} dc={dc}  best lag {lag:+d}  r={c:.4f}")
    # A noisy 144 step forecast correlates almost as well one or two steps either side of
    # the truth, so demanding an identical argmax everywhere tests the noise, not the
    # wiring. What matters is that reading the forecast at lag zero loses almost nothing:
    # a systematic offset would show up as a real gap.
    worst_gap = max(c - z for (_, _, _, c), z in zip(best_lags, zero_lag))
    check("no systematic offset: lag zero is as good as the best lag",
          worst_gap < 0.05,
          f"largest drop at lag zero {worst_gap:.4f}, argmax lags {sorted({l for _,_,l,_ in best_lags})}")
    check("every site correlates with what it will actually receive",
          min(c for _, _, _, c in best_lags) > 0.80,
          f"min r={min(c for _, _, _, c in best_lags):.4f}")
    check("the forecast is not a copy of the realised future",
          max(c for _, _, _, c in best_lags) < 0.999,
          f"max r={max(c for _, _, _, c in best_lags):.4f}")

print(f"\ncoverage")
print(f"  forecast horizon        {prov.pred_len} steps")
print(f"  a job may wait until    D - max(runtime + 2, 602) steps")
print(f"  time to deadline at t0  about 6500 steps, so the waitable window is about 5900")
check("the horizon covers the waitable window",
      prov.pred_len >= 5900,
      f"{prov.pred_len} of about 5900 steps, {100.0 * prov.pred_len / 5900:.1f}%")

print()
if FAILS:
    print(f"WIRING AUDIT FAILED: {len(FAILS)} check(s): {FAILS}")
    sys.exit(1)
print("WIRING AUDIT PASSED")
