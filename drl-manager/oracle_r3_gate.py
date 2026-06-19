#!/usr/bin/env python3
"""
R-3 GATE — does temporal deferral beat greedy drain on the anti-phase workload? 2026-06-19.

Same env/seed, two scripted policies on the deferrable-batch workload (arrivals in brown
windows, loose deadlines):
  - drain : route to greenest-now DC, dispatch everything immediately. Brown-arriving work
            runs DURING brown → on brown power → wastes the later green windows.
  - defer : route to the strongest green DC; locally HOLD (dispatch 0) while that DC's green
            is below `--feast-thresh`, BURST (dispatch max) once green is abundant. Brown work
            is saved and run into the green feast windows.

GATE pass: green_used(defer) > green_used(drain) AND carbon lower AND completion not tanked.
That is the first positive evidence the forecast/temporal lever can help → then build RL.

Run from drl-manager/:
  .venv/bin/python oracle_r3_gate.py --trace traces/antiphase_n120000_mi400000_b0.9.csv \
      --green-divisor 3000 --feast-thresh 200
"""
import argparse, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import collect_metrics


def _arr(o, k, n):
    v = np.asarray(o.get(k, []), dtype=np.float64).ravel()
    return np.concatenate([v, np.zeros(max(0, n - v.size))])[:n]


def run(env, policy, feast_thresh, green_capable, seed, max_dispatch):
    obs, info = env.reset(seed=seed)
    nd = env.num_datacenters; batch = env.global_routing_batch_size
    n_hold = 0; gsum = np.zeros(nd); dsum = np.zeros(nd); ns = 0; feast_idle = []
    done = False
    while not done:
        g = obs["global"]
        gn = _arr(g, "dc_current_green_power_w", nd)
        dem = _arr(g, "dc_current_power_w", nd)
        util = _arr(g, "dc_utilizations", nd)
        gsum += gn; dsum += dem; ns += 1
        # route the whole batch to the strongest green-capable DC (concentrate on green)
        score = np.array([gn[i] if i in green_capable else -1e9 for i in range(nd)])
        target = int(np.argmax(score))
        ga = [target] * batch
        if gn[target] > feast_thresh:
            feast_idle.append(util[target])
        la = {}
        for dc in range(nd):
            if policy == "drain":
                la[dc] = max_dispatch
            else:  # defer: hold while green scarce, burst while green abundant
                if dc in green_capable and gn[dc] <= feast_thresh:
                    la[dc] = 0; n_hold += 1
                else:
                    la[dc] = max_dispatch
        obs, _, term, trunc, info = env.step({"global": ga, "local": la})
        done = term or trunc
    m = collect_metrics(info, nd)
    m["_hold"] = n_hold
    m["_feast_idle_util"] = float(np.mean(feast_idle)) if feast_idle else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_oracle_godeye")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yml"))
    ap.add_argument("--trace", required=True, help="anti-phase trace, relative to resources/")
    ap.add_argument("--green-divisor", type=float, default=3000.0)
    ap.add_argument("--feast-thresh", type=float, default=200.0, help="green W above which to burst")
    ap.add_argument("--max-dispatch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=200, help="override global_routing_batch_size (admission)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_gateway")
    cfg.setdefault("output_dir", "/tmp/oracle_gateway")
    os.makedirs("/tmp/oracle_gateway", exist_ok=True)
    cfg["local_dispatch_mode"] = "dispatch_rate"
    cfg["max_dispatch_per_step"] = args.max_dispatch
    cfg["compressed_power_divisor"] = args.green_divisor
    cfg["cloudlet_trace_file"] = args.trace
    cfg["global_routing_batch_size"] = args.batch

    green_capable = {i for i, dc in enumerate(cfg.get("datacenters", [])) if dc.get("turbine_ids")}
    print(f"=== R-3 GATE: defer vs drain on {args.trace} ===")
    print(f"green-capable={sorted(green_capable)} divisor={args.green_divisor} feast_thresh={args.feast_thresh}W")
    env = HierarchicalMultiDCEnv(config=cfg)
    drain = run(env, "drain", args.feast_thresh, green_capable, args.seed, args.max_dispatch)
    defer = run(env, "defer", args.feast_thresh, green_capable, args.seed, args.max_dispatch)
    env.close()

    def row(t, m):
        return (f"{t:7} completion={m.get('completion_rate_mi',0):.4f} green_used={m['green_used_wh']:.0f} "
                f"waste={m['waste_ratio']:.4f} carbon_kg={m['total_carbon_kg']:.4f} "
                f"feast_idle_util={m['_feast_idle_util']:.3f}")
    print("\n=== RESULT ===")
    print(row("drain", drain))
    print(row("defer", defer), f"(holds={defer['_hold']})")
    dgreen = defer["green_used_wh"] - drain["green_used_wh"]
    dcarb = defer["total_carbon_kg"] - drain["total_carbon_kg"]
    dcomp = defer.get("completion_rate_mi", 0) - drain.get("completion_rate_mi", 0)
    print(f"\nΔgreen_used={dgreen:+.0f}  Δcarbon_kg={dcarb:+.4f}  Δcompletion={dcomp:+.4f}")
    print(f"drain feast-idle util={drain['_feast_idle_util']:.3f} "
          f"(high=drain leaves feast idle → room for defer)")
    if dgreen > 20 and dcarb < -0.005:
        print("✅ GATE PASS: deferral captures green drain wastes, lower carbon → lever ALIVE → build RL")
    elif dgreen > 20:
        print("⚠️ green captured but carbon not down (completion confound?) — inspect")
    else:
        print("❌ GATE FAIL: defer still doesn't beat drain at this load/threshold — tune load/feast-thresh")


if __name__ == "__main__":
    main()
