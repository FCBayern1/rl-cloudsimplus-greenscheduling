"""Runtime calibration against the simulator's own execution events.

Codex 2026-08-30: which of the two candidate models describes how long a cloudlet really
occupies a site,

    r = length / mips
    r = length / (pes * mips)

is settled here by start and finish events reported by the simulator, not by reading
dc_available_pes. That field is a sum of Vm.getFreePesNumber() over created VMs, and
treating it as execution occupancy produced three wrong conclusions in a row.

A small batch is routed to one site and left alone. Every finish event carries the
cloudlet's length, its PES and its real start-to-finish time, so both models can be
scored directly.
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "drl-manager"))

from src.baselines.evaluate import load_config  # noqa: E402
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

TARGET_DC = int(os.environ.get("CAL_DC", "0"))
ROUTE_AT = int(os.environ.get("CAL_ROUTE_AT", "5"))
N_JOBS = int(os.environ.get("CAL_JOBS", "8"))
STEPS = int(os.environ.get("CAL_STEPS", "2000"))
MIPS = float(os.environ.get("CAL_MIPS", "40000"))


def parse(csv, fields):
    out = []
    if not csv:
        return out
    for rec in csv.split(";"):
        parts = rec.split(":")
        if len(parts) < fields:
            continue
        out.append(parts)
    return out


cfg = load_config(os.environ.get("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan"))
print("config cloudlet_cpu_utilization =", cfg.get("cloudlet_cpu_utilization"))
if os.environ.get("CAL_CPU_UTIL"):
    cfg["cloudlet_cpu_utilization"] = float(os.environ["CAL_CPU_UTIL"])
    print("overridden to", cfg["cloudlet_cpu_utilization"])
cfg["py4j_port"] = None
cfg.setdefault("gateway_log_dir", "/tmp/claude-1000/calibrate_runtime")
env = HierarchicalMultiDCEnv(config=cfg)
obs, info = env.reset(seed=20260823)
n = env.num_datacenters
batch = env.global_routing_batch_size
local_max = int(env.action_space["local"][0].n) - 1

# The trace rides the step info, not the reset info: at reset nothing is executing.
checked = False

routed = {}
starts = {}
finishes = []
free_series = []

for step in range(STEPS):
    if step == 1 and not checked:
        if "exec_finished_csv" not in info:
            print("FAIL: the gateway has no execution trace. Rebuild the jar.")
            sys.exit(1)
        checked = True
        print("execution trace present:",
              sorted(k for k in info if k.startswith("exec_") or k.startswith("dc_free")
                     or k.startswith("dc_running")))
    p = info["planner"]
    ids = np.asarray(p["batch_cloudlet_ids"], dtype=np.int64)
    pes = np.asarray(p["batch_cloudlet_pes"], dtype=np.int64)
    mi = np.asarray(p["batch_cloudlet_mi"], dtype=np.float64)

    for rec in parse(info.get("exec_started_csv", ""), 4):
        cid = int(rec[0])
        starts.setdefault(cid, (int(rec[1]), int(rec[2]), float(rec[3]), step))
    for rec in parse(info.get("exec_finished_csv", ""), 6):
        finishes.append(dict(id=int(rec[0]), dc=int(rec[1]), finish=float(rec[2]),
                             elapsed=float(rec[3]), length=float(rec[4]),
                             pes=float(rec[5]), step=step))
    free_series.append((step, info.get("dc_free_vm_pes_csv", ""),
                        info.get("dc_running_pes_csv", "")))

    if routed and (step % 200 == 0 or step - ROUTE_AT in (1, 2, 3, 5, 10, 50)):
        run_ids = {int(r[0]) for r in parse(info.get("exec_running_csv", ""), 3)}
        q_ids = {int(r[0]) for r in parse(info.get("exec_queued_csv", ""), 2)}
        mine = set(routed)
        print(f"  [t={step:>5}] mine running={len(mine & run_ids)} queued={len(mine & q_ids)} "
              f"elsewhere={len(mine - run_ids - q_ids)} | "
              f"all_running={len(run_ids)} all_queued={len(q_ids)} | "
              f"free={info.get('dc_free_vm_pes_csv','')} run_pes={info.get('dc_running_pes_csv','')}")

    action = [n] * batch
    if not routed and step >= ROUTE_AT and (ids >= 0).any():
        slots = np.flatnonzero(ids >= 0)[:N_JOBS]
        for slot in slots:
            action[int(slot)] = TARGET_DC
            routed[int(ids[slot])] = (float(mi[slot]), float(pes[slot]))
        print(f"routed {len(slots)} cloudlets to DC{TARGET_DC} at step {step}")
    obs, rewards, terminated, truncated, info = env.step(
        {"global": action, "local": {i: local_max for i in range(n)}})
    if terminated or truncated:
        break
env.close()

ours = [f for f in finishes if f["id"] in routed]
print(f"\nrouted {len(routed)}, of which finished within {STEPS} steps: {len(ours)}")
if not ours:
    print("none finished; cannot calibrate")
    sys.exit(2)

print(f"\n{'id':>6} {'len':>10} {'pes':>4} {'elapsed':>9} {'len/mips':>9} {'len/(pes*mips)':>15} "
      f"{'ratio_a':>8} {'ratio_b':>8}")
ra, rb = [], []
for f in sorted(ours, key=lambda x: x["id"]):
    a = f["length"] / MIPS
    b = f["length"] / (MIPS * max(f["pes"], 1.0))
    ra.append(f["elapsed"] / a)
    rb.append(f["elapsed"] / b)
    print(f"{f['id']:>6} {f['length']:>10.0f} {f['pes']:>4.0f} {f['elapsed']:>9.2f} "
          f"{a:>9.2f} {b:>15.2f} {ra[-1]:>8.3f} {rb[-1]:>8.3f}")

ra, rb = np.array(ra), np.array(rb)
print(f"\nelapsed / (length/mips)        mean {ra.mean():.4f}  sd {ra.std():.4f}  "
      f"min {ra.min():.4f}  max {ra.max():.4f}")
print(f"elapsed / (length/(pes*mips))  mean {rb.mean():.4f}  sd {rb.std():.4f}  "
      f"min {rb.min():.4f}  max {rb.max():.4f}")
winner = "length/mips" if abs(ra.mean() - 1) < abs(rb.mean() - 1) else "length/(pes*mips)"
print(f"\ncloser to 1.0: {winner}")

late = [s for s in free_series if s[0] > ROUTE_AT + 2][:3]
for step, free, run in late:
    print(f"  [t={step}] free_vm_pes={free}  running_pes={run}")
