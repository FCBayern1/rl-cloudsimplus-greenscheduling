#!/usr/bin/env python3
"""
R-1 reshape diagnostic — does a lighter load create feast-time IDLE capacity? 2026-06-18.

The temporal deferral lever needs (C1): during a DC's feast (green>demand) there is
IDLE compute capacity that deferred work could fill. At load_scale=1 the cluster is
backlog-saturated (util≈1 everywhere) so feast windows are already full → no room.
This probe scales the workload MI down (preserving the diurnal arrival pattern) and
measures, per green-capable DC, the mean utilization DURING its feast windows.

  feast-util ≈ 1.0  → no idle at feast → lever still dead at this load
  feast-util < ~0.8 → idle capacity at feast → deferral has room → go to R-3 sweep

Also reports green⊥load alignment: corr(total_green(t), total_demand(t)). Strongly
positive (aligned) → feast already coincides with high natural demand → may need a
phase shift; ≤0 (anti-aligned) → feast lands on demand troughs → ideal for deferral.

Run from drl-manager/:
  .venv/bin/python oracle_reshape_diag.py --load-scale 0.3 --green-divisor 1200
"""
import argparse, os, sys, csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import collect_metrics


def _arr(o, k, n):
    v = np.asarray(o.get(k, []), dtype=np.float64).ravel()
    return np.concatenate([v, np.zeros(max(0, n - v.size))])[:n]


def scale_trace(src_abs, scale, out_dir, pes_cap=None):
    """Write a copy of the CSV trace with the MI (length) column scaled by `scale`
    and (optionally) every cloudlet's pes_required capped at `pes_cap` — removing the
    large-VM placement bottleneck so utilization tracks aggregate load cleanly."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"scaled_{scale:.3f}" + (f"_pes{pes_cap}" if pes_cap else "")
    dst = os.path.join(out_dir, f"{tag}_{os.path.basename(src_abs)}")
    rows = list(csv.reader(open(src_abs)))
    hdr = rows[0]
    # cols: cloudlet_id,arrival_time,length,pes_required,...
    out = [hdr]
    for r in rows[1:]:
        if not r:
            continue
        r = list(r)
        r[2] = str(max(1, int(round(float(r[2]) * scale))))
        if pes_cap is not None:
            r[3] = str(min(int(r[3]), pes_cap))
        out.append(r)
    with open(dst, "w", newline="") as f:
        csv.writer(f).writerows(out)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_oracle_godeye")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-scale", type=float, default=1.0)
    ap.add_argument("--pes-cap", type=int, default=None,
                    help="cap every cloudlet's pes_required (e.g. 2) to remove the "
                         "large-VM placement bottleneck so util tracks load cleanly")
    ap.add_argument("--green-divisor", type=float, default=1200.0)
    ap.add_argument("--max-dispatch", type=int, default=100)
    ap.add_argument("--host-mult", type=float, default=1.0,
                    help="scale host_count_* (<1 SHRINKS cluster → raises utilization "
                         "so demand rises toward green and famine windows appear)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    cfg.pop("py4j_port", None)
    if args.host_mult != 1.0:
        for dc in cfg.get("datacenters", []):
            for k in list(dc.keys()):
                if k.startswith("host_count"):
                    dc[k] = max(1, int(round(dc[k] * args.host_mult)))
        print(f"[capacity] host_count_* ×{args.host_mult} → "
              f"{[{k:v for k,v in dc.items() if k.startswith('host_count')} for dc in cfg.get('datacenters',[])]}")
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_gateway")
    cfg.setdefault("output_dir", "/tmp/oracle_gateway")
    os.makedirs("/tmp/oracle_gateway", exist_ok=True)
    cfg["local_dispatch_mode"] = "dispatch_rate"
    cfg["max_dispatch_per_step"] = args.max_dispatch
    cfg["compressed_power_divisor"] = args.green_divisor

    # Resolve + scale the trace.
    res_root = Path(__file__).resolve().parent.parent / "cloudsimplus-gateway/src/main/resources"
    src = str(res_root / cfg["cloudlet_trace_file"])
    if args.load_scale != 1.0 or args.pes_cap is not None:
        scaled = scale_trace(src, args.load_scale, "/tmp/oracle_gateway/traces", args.pes_cap)
        # express relative to resources root so the Java reader finds it; copy into tree
        import shutil
        rel = "traces/" + os.path.basename(scaled)
        shutil.copy(scaled, str(res_root / rel))
        cfg["cloudlet_trace_file"] = rel
        print(f"[load] MI ×{args.load_scale} pes_cap={args.pes_cap} → {rel}")

    green_capable = {i for i, dc in enumerate(cfg.get("datacenters", [])) if dc.get("turbine_ids")}
    print(f"=== R-1 reshape diag (load×{args.load_scale}, green/{args.green_divisor}, seed={args.seed}) ===")
    print(f"green-capable DCs: {sorted(green_capable)}")
    env = HierarchicalMultiDCEnv(config=cfg)
    obs, info = env.reset(seed=args.seed)
    nd = env.num_datacenters; batch = env.global_routing_batch_size
    G = [[] for _ in range(nd)]; D = [[] for _ in range(nd)]; U = [[] for _ in range(nd)]
    done = False
    while not done:
        g = obs["global"]
        gn = _arr(g, "dc_current_green_power_w", nd)
        dem = _arr(g, "dc_current_power_w", nd)
        util = _arr(g, "dc_utilizations", nd)
        for i in range(nd):
            G[i].append(gn[i]); D[i].append(dem[i]); U[i].append(util[i])
        ga = [int(np.argmax(gn))] * batch
        la = {dc: args.max_dispatch for dc in range(nd)}
        obs, _, term, trunc, info = env.step({"global": ga, "local": la})
        done = term or trunc
    m = collect_metrics(info, nd)
    env.close()

    Gt = np.array(G); Dt = np.array(D); Ut = np.array(U)
    print(f"\ncompletion={m.get('completion_rate_mi',0):.4f}  waste_ratio={m['waste_ratio']:.4f}  "
          f"carbon_kg={m['total_carbon_kg']:.4f}  green_used={m['green_used_wh']:.0f}")
    print(f"\n{'DC':>3} {'mean_util':>9} {'feast%':>7} {'util@feast':>11} {'idle@feast?':>11}")
    for i in range(nd):
        feast = Gt[i] > Dt[i]
        fu = Ut[i][feast].mean() if feast.any() else float('nan')
        flag = "—" if i not in green_capable else ("✅ idle" if fu < 0.8 else "❌ full")
        print(f"{i:>3} {Ut[i].mean():>9.3f} {100*feast.mean():>6.1f}% {fu:>11.3f} {flag:>11}")
    # green ⊥ load alignment
    tg = Gt.sum(0); td = Dt.sum(0)
    if tg.std() > 1e-6 and td.std() > 1e-6:
        corr = np.corrcoef(tg, td)[0, 1]
        align = "aligned (feast on demand PEAK → little room)" if corr > 0.2 else \
                ("anti-aligned (feast on demand TROUGH → ideal)" if corr < -0.2 else "uncorrelated (ok)")
        print(f"\ncorr(total_green, total_demand) = {corr:+.3f} → {align}")
    gc_feast_util = np.nanmean([Ut[i][Gt[i] > Dt[i]].mean() for i in green_capable
                                if (Gt[i] > Dt[i]).any()])
    print(f"\n=== VERDICT (load×{args.load_scale}) ===")
    if m.get('completion_rate_mi', 0) > 0.9 and gc_feast_util < 0.8:
        print(f"✅ completion {m.get('completion_rate_mi',0):.2f} + feast-util {gc_feast_util:.2f} "
              f"→ SLACK at feast → run R-3 deferral sweep at this load")
    elif gc_feast_util < 0.8:
        print(f"⚠️ feast-util {gc_feast_util:.2f} has idle but completion {m.get('completion_rate_mi',0):.2f} "
              f"still <0.9 (other bottleneck: admission batch / PES)")
    else:
        print(f"❌ feast-util {gc_feast_util:.2f} still saturated → lighter load needed (lower --load-scale)")


if __name__ == "__main__":
    main()
