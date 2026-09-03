"""Does the planner's green view G[d, t] equal the simulator's current green at t?

The planner rebuilds G from turbine CSVs (ORACLE_OFFSET_ROWS + time-zone rows + weather
clock) while the simulator reads its own rows; green_now comes from the observation each
step. If the two disagree, "godeye" was never a perfect forecast and every COMPRESSED
ladder result rests on a shifted view. This probe records both series over one episode
and reports per-DC agreement and the best cross-correlation lag (0 = aligned).
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "drl-manager"))
from src.baselines.evaluate import _convert_global_obs_for_scheduler, load_config  # noqa
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa
from src.baselines.global_schedulers import PerturbedOraclePlannerGlobalScheduler  # noqa

exp = os.environ["ORACLE_EXPERIMENT"]
k = int(os.environ.get("PROBE_RESET_SKIP", "2"))
steps_max = int(os.environ.get("PROBE_STEPS", "400"))
cfg = load_config(exp)
cfg["py4j_port"] = None
cfg.setdefault("gateway_log_dir", "/tmp/claude-1000/green_align_probe")
env = HierarchicalMultiDCEnv(config=cfg)
for _ in range(k):
    env.reset(seed=42)
obs, info = env.reset(seed=42)
n = env.num_datacenters
planner = PerturbedOraclePlannerGlobalScheduler(n, cfg.get("batch_size", 128))
now, view = [], []
for step in range(steps_max):
    conv = _convert_global_obs_for_scheduler(obs["global"], info)
    actions = planner.schedule(conv)
    t = planner.t - 1 if planner.t > 0 else 0
    now.append(np.array(planner.green_now[:3], dtype=float))
    view.append(np.array([planner.G[d, min(t, planner.T - 1)] for d in range(3)]))
    action = {"global": [int(a) for a in actions], "local": {i: 0 for i in range(n)}}
    obs, r, term, trunc, info = env.step(action)
    if term or trunc:
        break
env.close()
now = np.array(now); view = np.array(view)
out = {"steps": int(len(now)), "reset_skip": k}
for d in range(3):
    a, b = now[:, d], view[:, d]
    rel = float(np.median(np.abs(a - b) / np.maximum(np.abs(a), 1.0)))
    corr0 = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    best_lag, best_c = 0, -2.0
    for lag in range(-60, 61):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:lag], b[-lag:]
        if len(x) < 50 or x.std() == 0 or y.std() == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_c:
            best_c, best_lag = c, lag
    out[f"dc{d}"] = {"median_rel_err": rel, "corr_lag0": corr0,
                     "best_lag": best_lag, "corr_best": best_c,
                     "mean_now": float(a.mean()), "mean_view": float(b.mean())}
print(json.dumps(out, indent=2))
