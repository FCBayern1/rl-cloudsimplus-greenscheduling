#!/usr/bin/env python3
"""End-to-end sync probe: under closed-book windows, does the TimeCAP provider
read the SAME wind slice the simulator replays?

The provider is a second CSV reader. If its per-episode offset diverges from
Java's ((1009*k) mod range), it forecasts a different day and the timecap arm
silently measures garbage (npy-desync bug class). This probe runs 2 episodes of
experiment_v2026_gamble_timecap and checks, for turbine 7012, that the LAST row
pushed into the provider's history buffer equals the simplified CSV row at
    expected = WARM + tz(=0) + (1009*k mod 4800) + sim_step
for both k=0 and k=1. Feature index for raw power is auto-detected in episode 0
by value matching, then required to match in episode 1.
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

WARM, RANGE, TID = 13, 4800, 7012
STEPS = 8
CSV = REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified/Turbine_7012_2021.csv"
POWER = np.array([float(r["power_kw"]) for r in csv.DictReader(open(CSV))])

def main():
    cfg = yaml.safe_load(open(REPO / "config_C.yml"))["experiment_v2026_gamble_timecap"]
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/tcprobe"); cfg.setdefault("output_dir", "/tmp/tcprobe")
    os.makedirs("/tmp/tcprobe", exist_ok=True)
    env = HierarchicalMultiDCEnv(config=cfg)
    ok = True
    pow_idx = None
    for ep in range(2):
        obs, _ = env.reset(seed=1)
        nd, batch = env.num_datacenters, env.global_routing_batch_size
        for _ in range(STEPS):
            obs, _, term, trunc, _ = env.step({"global": [0] * batch, "local": {d: 64 for d in range(nd)}})
        prov = env.timecap_provider
        hist = prov.predictor._history[TID]
        last = np.asarray(hist[-1], dtype=float)
        off = (1009 * ep) % RANGE
        # the last pushed row corresponds to the provider's latest update step;
        # scan a small window of plausible steps to be robust to warmup pushes
        cands = [WARM + off + s for s in range(0, STEPS + prov.seq_len + 2)]
        if pow_idx is None:
            # detect which feature dimension carries raw power (kW): value match
            best = None
            for fi in range(last.shape[0]):
                d = np.min([abs(last[fi] - POWER[c]) for c in cands if c < len(POWER)])
                if best is None or d < best[1]:
                    best = (fi, d)
            pow_idx, d0 = best
            print(f"ep0: power feature index={pow_idx} (min |Δ|={d0:.4f} over candidate rows)")
            ok &= d0 < 1e-3
        else:
            d = np.min([abs(last[pow_idx] - POWER[c]) for c in cands if c < len(POWER)])
            in_old = np.min([abs(last[pow_idx] - POWER[WARM + 0 + s]) for s in range(0, STEPS + prov.seq_len + 2)])
            print(f"ep1 (off={off}): |Δ| vs shifted rows={d:.4f}  vs UNSHIFTED rows={in_old:.4f}")
            ok &= d < 1e-3
            if in_old < 1e-3 and d >= 1e-3:
                print("  -> provider still reading the OLD window: OFFSET NOT APPLIED ✗")
    env.close()
    print("\nVERDICT:", "TIMECAP OFFSET SYNC OK — provider follows the per-episode window" if ok
          else "SYNC FAILED — do NOT launch timecap arms")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
