"""Smoke test: does the new Alibaba workload experiment load and run in the sim?

Builds the real multi-DC env for `experiment_multi_10dc_carbon_v2_alibaba` (auto-launches
the Java gateway), resets it (which makes Java load the Alibaba CSV), and steps through
the episode with a random policy. Confirms: cloudlets get created from the trace, steps
execute without error, rewards/obs are finite. NOT a performance test.
"""
import os
import sys
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRL = os.path.dirname(HERE)
sys.path.insert(0, DRL)
sys.path.insert(0, os.path.join(DRL, "src"))

import yaml
from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(DRL, "..", "config.yml"))
    ap.add_argument("--experiment", default="experiment_multi_10dc_carbon_v2_alibaba")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))[args.experiment]
    print(f"[smoke] experiment={args.experiment}  trace={cfg.get('cloudlet_trace_file')}")

    env = HierarchicalMultiDCParallelEnv(config=cfg)
    print(f"[smoke] env built. agents={env.agents}")

    obs, info = env.reset(seed=42)
    print(f"[smoke] reset OK. obs agents={list(obs.keys())}")

    rng = np.random.default_rng(0)
    total_reward = 0.0
    last_info = {}
    nan_seen = False
    done = False
    n = 0
    for n in range(1, args.steps + 1):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        r = sum(rewards.values())
        total_reward += r
        for a, o in obs.items():
            arr = o["observation"] if isinstance(o, dict) and "observation" in o else o
            if isinstance(arr, np.ndarray) and not np.all(np.isfinite(arr)):
                nan_seen = True
        last_info = next(iter(infos.values())) if infos else {}
        if any(terms.values()) or any(truncs.values()):
            done = True
            break

    print(f"[smoke] ran {n} steps. done={done}  sum_reward={total_reward:.3f}  nan_in_obs={nan_seen}")
    # surface completion-related info keys if present
    keys = [k for k in last_info.keys() if any(s in k.lower() for s in
            ("cloudlet", "complet", "finish", "wait", "sla", "submit", "created"))]
    print("[smoke] info completion keys:", {k: last_info.get(k) for k in keys})
    env.close()
    ok = (n > 0) and (not nan_seen) and np.isfinite(total_reward)
    print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
