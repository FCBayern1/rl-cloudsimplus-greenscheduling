#!/usr/bin/env python3
"""
Run the heuristic baselines on 5-DC v2 and emit a clean comparison table
using the SAME completion metric the RL trainer logs (completion_rate_mi),
so baseline vs RL c/c is apples-to-apples.

All baselines run with green_oracle_mode=godeye (no TimeCAP / no GPU) — the
heuristics only read current green_ratio / queue, never the forecast, so this
doesn't change their decisions or the carbon they produce, and it keeps the
GPU free for a concurrently-running RL sweep.

Usage (from drl-manager/):
    .venv/bin/python scripts/run_5dc_baselines.py \\
        --experiment experiment_multi_5dc_carbon_v2 \\
        --out /tmp/5dc_baseline_table.csv
"""
import argparse
import csv
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.WARNING)

# (global, local) heuristic combos to evaluate.
COMBOS = [
    ("round_robin", "first_fit"),
    ("green_aware", "first_fit"),
    ("min_queue", "first_fit"),
    ("green_queue_balanced", "first_fit"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/5dc_baseline_table.csv")
    args = ap.parse_args()

    from src.baselines.evaluate import run_evaluation, load_config, _apply_overrides

    cfg = load_config(args.experiment)
    cfg = _apply_overrides(cfg, ["green_oracle_mode=godeye"])
    # Strip the config-declared py4j_port (25333) so the env auto-launches its
    # own Java gateway on a free port instead of trying to connect to a
    # non-running one.  Same fix as the PPO env_creator / BC warmstart.
    cfg.pop("py4j_port", None)
    cfg["gateway_log_dir"] = str(Path(args.out).parent)

    rows = []
    for g, l in COMBOS:
        print(f"\n=== {g} + {l} ===", flush=True)
        results = run_evaluation(
            global_scheduler_name=g,
            local_scheduler_name=l,
            config=cfg,
            num_episodes=args.episodes,
            seed=args.seed,
            verbose=False,
        )
        # Average across episodes
        import statistics
        compl_mi = statistics.mean(r.get("completion_rate_mi", 0.0) for r in results)
        carbon = statistics.mean(r.get("total_carbon_kg", 0.0) for r in results)
        finished_rate = statistics.mean(r.get("finished_rate", 0.0) for r in results)
        cc = carbon / compl_mi if compl_mi > 0 else float("nan")
        rows.append({
            "global": g, "local": l,
            "completion_rate_mi": round(compl_mi, 4),
            "finished_rate_count": round(finished_rate, 4),
            "total_carbon_kg": round(carbon, 4),
            "c_per_completion_mi": round(cc, 4),
        })
        print(f"  completion_mi={compl_mi:.4f}  carbon={carbon:.4f}  c/c={cc:.4f}", flush=True)

    # Write table
    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*70}")
    print(f"{'scheduler':<26} {'compl_mi':>9} {'carbon':>8} {'c/c':>8}")
    print(f"{'-'*70}")
    for r in sorted(rows, key=lambda x: x["c_per_completion_mi"]):
        print(f"{r['global']:<26} {r['completion_rate_mi']:>9.4f} "
              f"{r['total_carbon_kg']:>8.4f} {r['c_per_completion_mi']:>8.4f}")
    print(f"{'='*70}")
    print(f"Table → {out}")


if __name__ == "__main__":
    main()
