"""End-to-end check that the evaluator-only planner channel reaches a scheduler.

Boots the real gateway, resets, takes a few steps and reports what the planner block
actually contains. The point is the two things the previous build could not see: a
stable id per slot, and a real time_to_deadline in seconds rather than the 1e9 that
_as_np_1d filled in when the key was missing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "drl-manager"))

from src.baselines.evaluate import _convert_global_obs_for_scheduler, load_config  # noqa: E402
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa: E402

exp = os.environ.get("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan")
cfg = load_config(exp)
cfg["py4j_port"] = None       # force a dedicated gateway rather than a shared one
cfg.setdefault("gateway_log_dir", os.environ.get(
    "CHECK_LOG_DIR", "/tmp/claude-1000/planner_channel_check"))
env = HierarchicalMultiDCEnv(config=cfg)
obs, info = env.reset(seed=20260823)

print("info keys:", sorted(k for k in info)[:12])
planner = info.get("planner")
if planner is None:
    print("FAIL: no planner block. The gateway is probably the frozen jar.")
    sys.exit(1)
print("planner keys:", sorted(planner))

conv = _convert_global_obs_for_scheduler(obs["global"], info)
print("forwarded to scheduler:", "planner" in conv)

for step in range(int(os.environ.get("CHECK_STEPS", "3"))):
    p = info.get("planner")
    ids = np.asarray(p["batch_cloudlet_ids"])
    ttd = np.asarray(p["batch_cloudlet_time_to_deadline"])
    present = np.asarray(p["batch_cloudlet_deadline_present"])
    real = ids >= 0
    print(f"[step {step}] clock={p['current_clock']:.0f} real_slots={int(real.sum())}/{ids.size} "
          f"ids[:6]={ids[:6].tolist()} ttd_real={np.round(ttd[real][:6], 1).tolist()} "
          f"deadline_present={int(present[real].sum())}/{int(real.sum())}")
    n = env.num_datacenters
    action = {"global": [n] * len(ids), "local": {i: 0 for i in range(n)}}
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
print("OK")
