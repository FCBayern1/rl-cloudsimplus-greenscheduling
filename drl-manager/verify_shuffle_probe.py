#!/usr/bin/env python3
"""Does FORECAST_PERTURB_MODE=shuffle actually corrupt the forecast the anti-phase
policy sees -- and is the corruption MEANINGFUL on the 8-DC layout?

shuffle == reverse the per-DC forecast array ([::-1]), designed for the 5-DC
clean->dirty ordering. Anti-phase is 8 DC with green at indices {0,1,2,5} and
non-green (no turbines) at {3,4,6,7}. Reversal maps 0<->7,1<->6,2<->5,3<->4, so
three of four green forecasts land on green-less DCs. This probe records the
first-step forecast obs under clean vs shuffle and reports (a) that the obs
changed at all, and (b) how much of the change lands on green vs non-green DCs.

Run standalone (spawns one gateway); ~60 steps, no training.
    python verify_shuffle_probe.py
"""
import os, sys
from pathlib import Path
import numpy as np
import yaml

REPO = Path("/home/joshua/rl-cloudsimplus-greenscheduling")
sys.path.insert(0, str(REPO / "drl-manager"))
os.chdir(REPO / "drl-manager")
os.environ["GATEWAY_LIBS"] = str(REPO / "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib")
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa

GREEN_DCS = {0, 1, 2, 5}          # Nordic, Germany, US_East, Nordic2
KEYS = ["dc_future_short_mean", "dc_future_short_trend", "dc_future_long_mean"]
N = 40


def collect(mode):
    os.environ["FORECAST_PERTURB_MODE"] = mode
    cfg = yaml.safe_load(open(REPO / "config_C.yml"))["experiment_dc8_antiphase_oracle"]
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/shufprobe"); cfg.setdefault("output_dir", "/tmp/shufprobe")
    os.makedirs("/tmp/shufprobe", exist_ok=True)
    env = HierarchicalMultiDCEnv(config=cfg)
    obs, _ = env.reset(seed=1)
    nd = env.num_datacenters; batch = env.global_routing_batch_size
    rows = {k: [] for k in KEYS}
    for _ in range(N):
        g = obs["global"]
        for k in KEYS:
            v = np.asarray(g.get(k, []), dtype=float).ravel()
            if v.size >= nd:
                rows[k].append(v[:nd].copy())
        obs, _, term, trunc, _ = env.step({"global": [0] * batch, "local": {d: 64 for d in range(nd)}})
        if term or trunc:
            break
    env.close()
    return {k: (np.array(v) if v else np.zeros((0, nd))) for k, v in rows.items()}, nd


def main():
    clean, nd = collect("none")
    shuf, _ = collect("shuffle")
    green = sorted(GREEN_DCS); nong = [i for i in range(nd) if i not in GREEN_DCS]
    print(f"\n=== shuffle-verify (anti-phase 8DC, {N} steps) ===")
    any_change = False
    for k in KEYS:
        c, s = clean[k], shuf[k]
        n = min(len(c), len(s))
        if n == 0:
            print(f"{k:22s}: NO OBS CAPTURED"); continue
        c, s = c[:n], s[:n]
        d = np.abs(c - s)
        changed = d.mean() > 1e-6
        any_change |= changed
        # how much of the total change lands on green vs non-green DCs
        gch = d[:, green].sum(); nch = d[:, nong].sum(); tot = d.sum() + 1e-12
        print(f"{k:22s}: changed={changed}  mean|Δ|={d.mean():.4f}  "
              f"green-share={100*gch/tot:4.0f}%  nongreen-share={100*nch/tot:4.0f}%")
    print(f"\nVERDICT: shuffle {'DOES' if any_change else 'DOES NOT'} alter the forecast obs.")
    print("If a large share of |Δ| lands on non-green DCs, the reversal wastes its"
          " corruption budget on green-less sites -> weak liability on 8-DC (use 'anti'/invert instead).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
