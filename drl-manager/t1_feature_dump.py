"""T1 step 1: dump the global observation each step, at the three registered windows.

The regression that follows needs the forecast features and every blind-visible
variable side by side, at the same step, under the same simulator the campaign
used. Actions are scripted (round-robin over datacentres) because the exogenous
quantities the analysis depends on - green power, green ratio, the forecast -
do not depend on the action, and a scripted policy keeps the dump reproducible.
Queue and utilisation DO depend on the action, so the analysis reports the
exogenous subset separately from the full blind set.
"""
import csv
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("EVAL_CONFIG_PATH", str(ROOT / "config_C.yml"))

from src.baselines.evaluate import load_config            # noqa: E402
from gym_cloudsimplus.envs import HierarchicalMultiDCEnv  # noqa: E402

ART = json.loads((ROOT / "drl-manager/calib/p0c_green_windows.json").read_text())
EXP = "experiment_g1eval_matchedvan"
FCST = ["dc_future_short_mean", "dc_future_short_trend",
        "dc_future_long_mean", "dc_future_long_peak_timing"]
EXOG = ["dc_current_green_power_w", "dc_green_ratio", "dc_current_power_w",
        "dc_cumulative_wasted_green_wh"]
OUT = ROOT / "g1/t1"


def per_dc_keys(obs):
    """Observation keys that are one value per datacentre."""
    n = len(np.atleast_1d(obs["dc_green_ratio"]))
    return [k for k, v in obs.items()
            if isinstance(v, np.ndarray) and v.ndim == 1 and v.size == n]


def dump(win):
    k, offset = win["episode_index_k"], win["offset_rows"]
    cfg = load_config(EXP)
    cfg["py4j_port"] = None
    cfg.setdefault("gateway_log_dir", f"/tmp/t1_gw_k{k}")
    env = HierarchicalMultiDCEnv(config=cfg)
    path = OUT / f"obs_k{k}.csv"
    n_steps = 0
    try:
        for _ in range(k):
            env.reset(seed=20260823)
        obs, _ = env.reset(seed=20260823)
        g = obs.get("global", obs)
        keys = per_dc_keys(g)
        ndc = len(np.atleast_1d(g["dc_green_ratio"]))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "dc"] + keys)
            done = False
            while not done and n_steps < 7200:
                g = obs.get("global", obs)
                for i in range(ndc):
                    w.writerow([n_steps, i] + [float(np.atleast_1d(g[c])[i]) for c in keys])
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                done = term or trunc
                n_steps += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    print(f"k={k} offset={offset}: {n_steps} steps -> {path}")
    return n_steps


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for w in ART["windows"]:
        dump(w)
