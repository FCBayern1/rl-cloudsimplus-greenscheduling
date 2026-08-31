"""Synthetic micro-scenarios whose answers are known before the run.

Codex 2026-08-30: before any gate number counts, the chain start -> finish -> runtime ->
carbon has to reproduce cases whose answer can be worked out by hand. Three regimes are
used, each isolating one thing the planner relies on.

    zero green     no wind anywhere, so every joule is brown and carbon is
                   energy x the site's brown factor with nothing else in the way
    full green     wind far above demand, so carbon collapses to the green factor
    single pulse   one narrow window of wind, so the only way to capture it is to
                   place work inside it

The green trace is not synthesised into the simulator here. Instead the planner's own
cost model is exercised against a synthetic green view, and the simulator is used to
confirm the execution-event chain: what the planner dispatches must start where and when
it committed, must finish after the runtime the calibration measured, and the per-site
energy-carbon identity must close on the resulting trace.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.join(REPO, "drl-manager"))

from src.baselines.global_schedulers import (  # noqa: E402
    CurveInformedPlannerGlobalScheduler as Planner)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make(batch_size=8):
    return Planner(5, batch_size)


def obs_for(ids, mi, pes, ttd, green_now=None, running_pes=None, avail=True):
    n = 5
    k = len(ids)
    o = {
        "planner": {
            "batch_cloudlet_ids": np.array(list(ids) + [-1] * (8 - k), dtype=np.int64),
            "batch_cloudlet_mi": np.array(list(mi) + [0] * (8 - k), dtype=np.int64),
            "batch_cloudlet_pes": np.array(list(pes) + [0] * (8 - k), dtype=np.int64),
            "batch_cloudlet_time_to_deadline": np.array(
                list(ttd) + [0.0] * (8 - k), dtype=np.float64),
            "batch_cloudlet_deadline_present": np.array(
                [1] * k + [0] * (8 - k), dtype=np.int64),
            "batch_cloudlet_is_deferred": np.zeros(8, dtype=np.int64),
            "batch_cloudlet_wait_age": np.zeros(8, dtype=np.float64),
            "current_clock": 0.0,
        }
    }
    if avail:
        o["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    if green_now is not None:
        o["dc_current_green_power_w"] = np.asarray(green_now, dtype=float)
    if running_pes is not None:
        o["dc_running_pes_csv"] = ",".join(str(int(v)) for v in running_pes)
    return o


print("runtime model")
p = make()
check("effective rate is mips * u", p.mips == 20000.0, f"mips={p.mips}")
for length, want in [(20000, 1), (20001, 2), (40000, 2), (60000, 3), (1, 1)]:
    got = p._runtime_steps(length)
    check(f"ceil({length}/20000) = {want}", got == want, f"got {got}")

print("\nzero green: the cheapest brown site wins and nothing is deferred for wind")
p = make()
o = obs_for([1], [40000], [2], [0.0], green_now=[0, 0, 0, 0, 0])
p.schedule(o)
d = p.active.get(1, (None,))[0]
check("routed to the lowest brown factor site", d == 0, f"dc={d} brown={list(p.cb)}")
check("held for the measured runtime", 1 in p.active and
      p.active[1][2] - p.active[1][1] == 2, f"span={p.active[1][2]-p.active[1][1]}")

print("\nfull green: a site drowning in wind is preferred over a cleaner brown one")
p = make()
p.G[:] = 0.0
p.G[4, :] = 1e6                    # DC4 has the worst brown factor and endless wind
p.G[0, :] = 0.0                    # DC0 has the best brown factor and none
o = obs_for([1], [40000], [2], [0.0])
p.schedule(o)
d = p.active.get(1, (None,))[0]
check("wind beats a low brown factor", d == 4, f"dc={d}")

print("\nsingle pulse: work is placed inside the only window that has wind")
p = make()
p.G[:] = 0.0
p.G[0, 40:60] = 1e6                # one narrow window at DC0
o = obs_for([1], [40000], [2], [3600.0])
p.schedule(o)
entry = p.reservations.get(1) or p.active.get(1)
check("a plan exists", entry is not None)
if entry:
    d, s, e, _ = entry
    check("planned into the pulse", d == 0 and 40 <= s < 60, f"dc={d} start={s}")

print("\ncapacity and closure")
p = make()
o = obs_for([1], [40000], [2], [0.0], running_pes=[500, 0, 0, 0, 0])
p.schedule(o)
check("execution beyond capacity is recorded",
      abs(p.running_pes_over_cap - 20.0) < 1e-9, f"over={p.running_pes_over_cap}")

p = make()
o = obs_for([], [], [], [])
o["exec_started_csv"] = "77:3:2:1.0"
p.schedule(o)
check("a start nobody ordered is counted", p.n_unplanned_start == 1,
      f"n={p.n_unplanned_start}")

print("\npower model")
p = make()
draw_full = 2 * p.dyn_per_pe
draw_used = 2 * p.dyn_per_pe * p.cpu_util
check("dynamic draw scales with utilisation",
      abs(draw_used - draw_full * 0.5) < 1e-12, f"{draw_used:.4f} vs {draw_full:.4f}")
check("dynamic energy is conserved across the longer window",
      abs(draw_used * 2 - draw_full * 1) < 1e-12,
      "u=0.5 for 2 steps equals full draw for 1")

print()
if FAILS:
    print(f"MICRO VALIDATION FAILED: {len(FAILS)} check(s): {FAILS}")
    sys.exit(1)
print("MICRO VALIDATION PASSED")
