#!/usr/bin/env python3
"""Integration probe for the per-episode green-window shift (green_episode_offset_range).

Checks, on the real simulator:
  A. consecutive episodes see DIFFERENT green traces (no more memorisable replay);
  B. the offset follows the deterministic schedule (1009*k mod range) -- episode k's
     green matches the verified CSV equation green_i(t) = sum(CSV[WARM + tz_i + off_k + t])/DIV;
  C. cross-DC phase structure is preserved (same off_k added to every DC).

Run standalone (spawns one gateway), ~2 episodes x 60 steps, no training.
"""
import os, sys, csv
from pathlib import Path
import numpy as np
import yaml

REPO = Path("/home/joshua/rl-cloudsimplus-greenscheduling")
sys.path.insert(0, str(REPO / "drl-manager"))
os.chdir(REPO / "drl-manager")
os.environ["GATEWAY_LIBS"] = str(REPO / "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib")
os.environ["FORECAST_PERTURB_MODE"] = "none"
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa

GATE = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
WARM, DIV = 13, 1.5
GREEN = {0: ([7012, 7036], 0), 1: ([7095, 7091], 1000), 2: ([7096], 2000), 5: ([7101, 7103], 3000)}
N, RANGE = 60, 4800

def turb(tid):
    return np.array([float(r["power_kw"]) for r in csv.DictReader(open(GATE / f"Turbine_{tid}_2021.csv"))])
RAW = {i: sum(turb(t) for t in ids) for i, (ids, _) in GREEN.items()}

def analytic(dc, off_k, n):
    tz = GREEN[dc][1]
    s = WARM + tz + off_k
    return RAW[dc][s:s + n] / DIV

def main():
    cfg = yaml.safe_load(open(REPO / "config_C.yml"))["experiment_v2026_spread_oracle"]
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/epoffprobe"); cfg.setdefault("output_dir", "/tmp/epoffprobe")
    os.makedirs("/tmp/epoffprobe", exist_ok=True)
    env = HierarchicalMultiDCEnv(config=cfg)
    nd, batch = None, None
    traces = []
    for ep in range(2):
        obs, _ = env.reset(seed=1)
        nd = env.num_datacenters; batch = env.global_routing_batch_size
        rows = []
        for _ in range(N):
            v = np.asarray(obs["global"].get("dc_current_green_power_w", []), dtype=float).ravel()[:nd]
            rows.append(v.copy())
            obs, _, term, trunc, _ = env.step({"global": [0] * batch, "local": {d: 64 for d in range(nd)}})
            if term or trunc:
                break
        traces.append(np.array(rows))
    env.close()

    ok = True
    # A: episodes differ
    d01 = np.abs(traces[0] - traces[1]).mean()
    print(f"A. mean|ep0 - ep1| = {d01:.4f}  ->", "DIFFER ✓" if d01 > 1e-3 else "IDENTICAL ✗")
    ok &= d01 > 1e-3
    # B: each episode matches the analytic equation at its scheduled offset
    for ep, k in [(0, 0), (1, 1)]:
        off_k = (1009 * k) % RANGE
        for dc in GREEN:
            a = analytic(dc, off_k, len(traces[ep]))
            s = traces[ep][:, dc]
            r = np.corrcoef(a, s)[0, 1] if s.std() > 0 else (1.0 if a.std() == 0 else 0.0)
            mae = np.abs(a - s).mean()
            good = mae < 1e-3 or r > 0.9999
            print(f"B. ep{ep} (off={off_k:4d}) DC{dc}: corr={r:.6f} mae={mae:.2e} ->", "✓" if good else "✗")
            ok &= good
    print("\nVERDICT:", "EPISODE OFFSET WORKS — closed-book green confirmed" if ok else "FAILED — do not launch training")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
