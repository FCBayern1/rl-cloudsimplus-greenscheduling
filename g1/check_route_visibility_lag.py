"""When does a route at step t show up in dc_available_pes: at t, or at t+1?

Codex 2026-08-30: the planner's active ledger has to be aligned to the simulator's real
visible timing, and the alignment must be fixed by a single-job experiment rather than
chosen after seeing a cell result. One cloudlet of known PES is routed to one site while
every other slot defers, and the per-step availability of that site is read either side
of the routing step.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "drl-manager"))

from src.baselines.evaluate import load_config  # noqa: E402
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

TARGET_DC = 0

cfg = load_config(os.environ.get("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan"))
cfg["py4j_port"] = None
cfg.setdefault("gateway_log_dir", "/tmp/claude-1000/route_lag")
env = HierarchicalMultiDCEnv(config=cfg)
obs, info = env.reset(seed=20260823)
n = env.num_datacenters
batch = env.global_routing_batch_size
# In dispatch_rate mode the local action is a release COUNT, and the drain baseline
# always releases the maximum. Passing 0, as an earlier version of this probe did,
# releases nothing and freezes the queue, which looks exactly like "routing has no
# effect" while in fact nothing was ever started.
local_max = int(env.action_space["local"][0].n) - 1 if hasattr(
    env.action_space["local"], "__getitem__") else int(env.max_vms)
print(f"local release action = {local_max} (drain semantics)")

routed_at = None
routed_pes = None
routed_mi = None
routed_pes_each = None
history = []

for step in range(int(os.environ.get("PROBE_STEPS", "30"))):
    p = info["planner"]
    ids = np.asarray(p["batch_cloudlet_ids"], dtype=np.int64)
    pes = np.asarray(p["batch_cloudlet_pes"], dtype=np.int64)
    g = obs["global"]
    avail = np.asarray(g["dc_available_pes"], dtype=float).ravel()
    q = np.asarray(g["dc_queue_sizes"], dtype=float).ravel()
    u = np.asarray(g["dc_utilizations"], dtype=float).ravel()
    history.append((step, avail.copy()))
    if step % 100 == 0 or step < 30:
        print(f"  [t={step:>4}] avail={avail.astype(int).tolist()} queue0={q[0]:>5.0f} "
              f"util0={u[0]:.4f} real_slots={int((ids >= 0).sum()):>4}")

    action = [n] * batch                      # everything defers by default
    if routed_at is None and step >= int(os.environ.get("ROUTE_AT", "5")) and (ids >= 0).any():
        slots = np.flatnonzero(ids >= 0)
        if os.environ.get("ROUTE_ONE") == "1":
            slots = slots[:1]
        for slot in slots:
            action[int(slot)] = TARGET_DC
        routed_at, routed_pes = step, int(pes[slots].sum())
        routed_mi = np.asarray(p["batch_cloudlet_mi"], dtype=float)[slots]
        routed_pes_each = pes[slots].astype(float)
        print(f"routing {len(slots)} cloudlets, {routed_pes} PEs total, "
              f"to DC{TARGET_DC} at step {step}")
        print(f"  mi={routed_mi.tolist()} pes={routed_pes_each.tolist()}")
        print(f"  r=mi/mips     = {(routed_mi / 40000.0).round(2).tolist()}")
        print(f"  r=mi/(pes*mips) = {(routed_mi / (40000.0 * routed_pes_each)).round(2).tolist()}")

    obs, rewards, terminated, truncated, info = env.step(
        {"global": action, "local": {i: local_max for i in range(n)}})
    if terminated or truncated:
        break
env.close()

base = history[0][1][TARGET_DC]
print(f"\nDC{TARGET_DC} available PEs, baseline {base:.0f}, one job of {routed_pes} PEs "
      f"routed during step {routed_at}")
for step, avail in history:
    delta = avail[TARGET_DC] - base
    mark = "  <-- routing step" if step == routed_at else ""
    print(f"  step {step:>3}  avail={avail[TARGET_DC]:>6.0f}  delta={delta:>+6.0f}{mark}")

# How long the PEs stay occupied, against what r = mi/mips predicts.
occupied = [s for s, a in history if a[TARGET_DC] < base]
if occupied:
    first_busy, last_busy = occupied[0], occupied[-1]
    print(f"\nPEs at DC{TARGET_DC} were below baseline from step {first_busy} to {last_busy} "
          f"({last_busy - first_busy + 1} steps); routed at {routed_at}")
    if routed_mi is not None:
        pred = routed_mi / 40000.0
        print(f"  predicted runtimes r=mi/mips: max {pred.max():.1f} steps, "
              f"mean {pred.mean():.1f}, sum-of-PE-steps {(pred * routed_pes_each).sum():.0f}")
        span = last_busy - (routed_at + 1) + 1
        print(f"  observed occupancy span {span} steps vs longest predicted "
              f"{pred.max():.1f} -> stretch {span / max(pred.max(), 1e-9):.2f}x")
changed = [s for s, a in history if a[TARGET_DC] != base]
if not changed:
    print("\nNO CHANGE observed. dc_available_pes did not react within the window.")
else:
    first = changed[0]
    print(f"\nfirst change at step {first}; routed during step {routed_at}; "
          f"lag = {first - routed_at} step(s)")
