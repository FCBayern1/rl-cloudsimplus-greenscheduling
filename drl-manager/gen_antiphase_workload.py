#!/usr/bin/env python3
"""
R-2' — generate a deferrable-batch workload ANTI-PHASE with green. 2026-06-19.

The temporal forecast lever only beats greedy drain when feast (high-green) windows
coincide with WORK SCARCITY: drain runs dry during feast → wastes green → deferral
fills those idle feast windows with work saved from brown windows. This generator
builds exactly that structure (the canonical carbon-aware deferrable-batch scenario):

  - arrivals concentrated in BROWN windows (low DC0+DC1 green), anti-phase with wind;
  - LOW average load so drain empties its queue and leaves feast windows idle;
  - loose per-job deadline (defer hours / across green windows);
  - pes=2 uniform (no large-VM placement bottleneck).

Reads the green(t) profile saved by the probe to /tmp/oracle_gateway/green_ts.npy.
Writes a trace CSV (cloudlet_id,arrival_time,length,pes_required,file_size,output_size,
deadline) into resources/traces/.

Run from drl-manager/:
  .venv/bin/python gen_antiphase_workload.py --n 40000 --mi 400000 --brown-frac 0.9
"""
import argparse, os
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--green-ts", default="/tmp/oracle_gateway/green_ts.npy")
    ap.add_argument("--n", type=int, default=40000, help="number of cloudlets (load knob)")
    ap.add_argument("--mi", type=int, default=400000, help="MI per cloudlet (load knob)")
    ap.add_argument("--pes", type=int, default=2)
    ap.add_argument("--brown-frac", type=float, default=0.9,
                    help="fraction of arrivals drawn from brown (low-green) steps")
    ap.add_argument("--arrival-window-frac", type=float, default=0.83,
                    help="only emit arrivals in the first frac of the episode (leave a drain tail)")
    ap.add_argument("--deadline-steps", type=int, default=3600,
                    help="per-job slack: deadline = arrival + this many steps (loose=hours)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    G = np.load(args.green_ts)                       # [T, nd]
    T = G.shape[0]
    g = G[:, 0] + G[:, 1]                            # the two real green DCs
    horizon = int(T * args.arrival_window_frac)
    g = g[:horizon]
    rng = np.random.default_rng(args.seed)

    # brown = below-median green steps; green = above. Sample arrivals mostly from brown.
    thresh = np.quantile(g[g > 0], 0.5) if (g > 0).any() else 0.0
    brown_steps = np.where(g <= thresh)[0]
    green_steps = np.where(g > thresh)[0]
    if brown_steps.size == 0:
        brown_steps = np.arange(horizon)
    if green_steps.size == 0:
        green_steps = brown_steps

    n_brown = int(round(args.n * args.brown_frac))
    n_green = args.n - n_brown
    arrivals = np.concatenate([
        rng.choice(brown_steps, size=n_brown, replace=True),
        rng.choice(green_steps, size=n_green, replace=True),
    ])
    arrivals.sort()

    res_root = Path(__file__).resolve().parent.parent / "cloudsimplus-gateway/src/main/resources"
    rel = args.out or f"traces/antiphase_n{args.n}_mi{args.mi}_b{args.brown_frac}.csv"
    dst = res_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        f.write("cloudlet_id,arrival_time,length,pes_required,file_size,output_size,deadline\n")
        for i, a in enumerate(arrivals):
            ddl = int(a) + args.deadline_steps
            f.write(f"{i},{int(a)},{args.mi},{args.pes},512,256,{ddl}\n")

    # report the anti-phase structure
    in_brown = np.isin(arrivals, brown_steps).mean()
    print(f"wrote {args.n} cloudlets → resources/{rel}")
    print(f"  episode T={T}, arrival horizon={horizon} steps, MI={args.mi}, pes={args.pes}, "
          f"deadline=+{args.deadline_steps} steps")
    print(f"  green threshold (median>0)={thresh:.0f}W; "
          f"{100*in_brown:.0f}% of arrivals land in brown steps (target {100*args.brown_frac:.0f}%)")
    print(f"  total work = {args.n*args.mi/1e9:.1f} G-MI  (tune --n/--mi for target util)")


if __name__ == "__main__":
    main()
