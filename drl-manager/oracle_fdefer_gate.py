#!/usr/bin/env python3
"""
F-DEFER GATE (v2) — is the FORECAST indispensable for deferral under TIGHT deadlines? 2026-07-02.

Three scripted policies, same env/seed, on the tight-deadline deferrable trace
(per-job slack ~U[120,900]s straddles the time-to-next-feast distribution p75=79/p90=401/p95=758):

  - drain    : dispatch everything immediately (no temporal behavior).
  - reactive : hold while target-DC green <= feast-thresh, burst when green. NO future
               knowledge — the wait is unbounded ("wait and see"). Under tight deadlines,
               long troughs make it hold jobs past their slack → deadline-forced late brown
               bursts / misses / completion loss.
  - fcast    : godeye deferral — hold ONLY IF the true future (green_ts.npy) says the feast
               arrives within --horizon steps (~median slack); otherwise dispatch NOW (do not
               wait for a feast that won't come in time). This is the decision only a
               forecast can make.

GATE PASS = fcast beats reactive (higher completion / fewer deadline-forced, carbon not worse)
AND fcast still beats drain on carbon (temporal lever alive). Then forecast is LOAD-BEARING:
an RL policy must consult the forecast to schedule well → over-trust collapse has teeth.

Run from drl-manager/:
  .venv/bin/python oracle_fdefer_gate.py --trace traces/realwind_defer_tight.csv \
      --config ../config_C.yml --experiment experiment_multi_5dc_carbon_v2_deferrable_gdpd \
      --green-divisor 1500 --feast-thresh 200 --horizon 500
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


def _feast_eta(green_ts, thresh):
    """eta[t, dc] = steps from t until green_ts[t', dc] >= thresh (inf if never)."""
    T, nd = green_ts.shape
    eta = np.full((T, nd), np.inf)
    for d in range(nd):
        nxt = np.inf
        for t in range(T - 1, -1, -1):
            nxt = 0.0 if green_ts[t, d] >= thresh else (nxt + 1.0)
            eta[t, d] = nxt
    return eta


def _deadline_stats(info):
    """Best-effort: dig deadline_miss_rate / forced count out of the info dict."""
    out = {}
    def walk(d, depth=0):
        if not isinstance(d, dict) or depth > 4:
            return
        for k, v in d.items():
            lk = str(k).lower()
            if "deadline" in lk and isinstance(v, (int, float)):
                out[str(k)] = float(v)
            walk(v, depth + 1)
    walk(info)
    return out


def _route_batch(gn, green_capable, nd, batch, mode):
    """Global routing for one batch. argmax = historical behaviour (whole
    batch to the greenest DC). spread = proportional allocation across
    green-capable DCs by current green power (largest-remainder counts, evenly interleaved so
    any prefix stays proportional); equal split when no DC has green right now. Decouples
    "forecast value" from the single-DC pile-up flaw of argmax routing."""
    greens = sorted(green_capable)
    if mode == "argmax" or not greens:
        score = np.array([gn[i] if i in green_capable else -1e9 for i in range(nd)])
        return [int(np.argmax(score))] * batch
    w = np.array([max(float(gn[i]), 0.0) for i in greens])
    if w.sum() <= 0.0:
        w = np.ones(len(greens))
    shares = w / w.sum() * batch
    base = np.floor(shares).astype(int)
    rem = batch - int(base.sum())
    order = np.argsort(-(shares - base))
    for j in range(rem):
        base[order[j % len(greens)]] += 1
    # Smooth weighted round-robin (nginx-style): emit each DC base[k] times but
    # EVENLY interspersed, so ANY prefix of the list is proportional to green.
    # A blocked list ([0]*k0 + [1]*k1 + ...) breaks when the routing batch is
    # not full (arrivals < batch_size): the env consumes only the prefix, which
    # would be all DC0 -> single-DC pile-up. Interleaving fixes that.
    weights = [int(b) for b in base]
    total = sum(weights)
    ga, current = [], [0] * len(greens)
    for _ in range(total):
        for k in range(len(greens)):
            current[k] += weights[k]
        k = max(range(len(greens)), key=lambda k: current[k])
        current[k] -= total
        ga.append(int(greens[k]))
    return ga


def run(env, policy, feast_thresh, green_capable, seed, max_dispatch, eta, horizon, routing="argmax"):
    obs, info = env.reset(seed=seed)
    nd = env.num_datacenters; batch = env.global_routing_batch_size
    n_hold = 0; n_giveup = 0; t = 0
    done = False
    T = eta.shape[0]
    while not done:
        g = obs["global"]
        gn = _arr(g, "dc_current_green_power_w", nd)
        ga = _route_batch(gn, green_capable, nd, batch, routing)
        la = {}
        for dc in range(nd):
            if policy == "drain":
                la[dc] = max_dispatch
            elif policy == "reactive":
                if dc in green_capable and gn[dc] <= feast_thresh:
                    la[dc] = 0; n_hold += 1
                else:
                    la[dc] = max_dispatch
            else:  # fcast: hold only if the (true-future) feast arrives within horizon
                if dc in green_capable and gn[dc] <= feast_thresh:
                    e = eta[min(t, T - 1), dc]
                    if e <= horizon:
                        la[dc] = 0; n_hold += 1
                    else:
                        la[dc] = max_dispatch; n_giveup += 1
                else:
                    la[dc] = max_dispatch
        obs, _, term, trunc, info = env.step({"global": ga, "local": la})
        done = term or trunc; t += 1
    m = collect_metrics(info, nd)
    m["_hold"] = n_hold
    m["_giveup"] = n_giveup
    m["_deadline"] = _deadline_stats(info)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_deferrable_gdpd")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config_C.yml"))
    ap.add_argument("--trace", required=True)
    ap.add_argument("--green-ts", default="/tmp/oracle_gateway/green_ts.npy")
    ap.add_argument("--green-divisor", type=float, default=1500.0)
    ap.add_argument("--feast-thresh", type=float, default=200.0)
    ap.add_argument("--horizon", type=float, default=500.0, help="fcast hold horizon ~ median slack")
    ap.add_argument("--max-dispatch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--routing", choices=["argmax", "spread"], default="argmax",
                    help="global routing: argmax=whole batch to greenest DC (historical); spread=proportional to green power")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_gateway")
    cfg.setdefault("output_dir", "/tmp/oracle_gateway")
    os.makedirs("/tmp/oracle_gateway", exist_ok=True)
    cfg["use_flat_obs_protocol"] = False
    cfg["local_dispatch_mode"] = "dispatch_rate"
    cfg["max_dispatch_per_step"] = args.max_dispatch
    cfg["compressed_power_divisor"] = args.green_divisor
    cfg["cloudlet_trace_file"] = args.trace
    cfg["global_routing_batch_size"] = args.batch

    green_ts = np.load(args.green_ts)
    eta = _feast_eta(green_ts, args.feast_thresh)
    green_capable = {i for i, dc in enumerate(cfg.get("datacenters", [])) if dc.get("turbine_ids")}
    print(f"=== F-DEFER GATE: drain vs reactive vs fcast on {args.trace} ===")
    print(f"green-capable={sorted(green_capable)} divisor={args.green_divisor} "
          f"thresh={args.feast_thresh}W horizon={args.horizon}s")

    env = HierarchicalMultiDCEnv(config=cfg)
    res = {}
    for pol in ("drain", "reactive", "fcast"):
        res[pol] = run(env, pol, args.feast_thresh, green_capable, args.seed,
                       args.max_dispatch, eta, args.horizon, routing=args.routing)
    env.close()

    print("\n=== RESULT ===")
    for pol in ("drain", "reactive", "fcast"):
        m = res[pol]
        dl = m["_deadline"]
        dls = " ".join(f"{k}={v:.4g}" for k, v in sorted(dl.items())) or "no-deadline-keys"
        print(f"{pol:9} completion={m.get('completion_rate_mi',0):.4f} "
              f"green_used={m['green_used_wh']:.0f} carbon_kg={m['total_carbon_kg']:.4f} "
              f"holds={m['_hold']} giveups={m['_giveup']} | {dls}")

    dr, re_, fc = res["drain"], res["reactive"], res["fcast"]
    comp_gain = fc.get("completion_rate_mi", 0) - re_.get("completion_rate_mi", 0)
    carb_vs_re = fc["total_carbon_kg"] - re_["total_carbon_kg"]
    carb_vs_dr = fc["total_carbon_kg"] - dr["total_carbon_kg"]
    print(f"\nfcast-vs-reactive: Δcompletion={comp_gain:+.4f} Δcarbon={carb_vs_re:+.4f}")
    print(f"fcast-vs-drain   : Δcarbon={carb_vs_dr:+.4f}")
    if comp_gain > 0.03 and carb_vs_re < 0.005 and carb_vs_dr < -0.005:
        print("✅ GATE PASS: forecast-deferral beats reactive (completion) and drain (carbon) "
              "→ forecast is LOAD-BEARING → train RL arms on this regime")
    elif carb_vs_dr < -0.005 and comp_gain > 0:
        print("⚠️ PARTIAL: temporal lever alive and fcast edges reactive — check deadline stats, "
              "consider tighter slack or heavier load")
    else:
        print("❌ GATE FAIL: reactive suffices (or lever dead) — tighten slack / raise load")


if __name__ == "__main__":
    main()
