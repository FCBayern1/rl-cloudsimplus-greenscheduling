#!/usr/bin/env python3
"""
R-2' — generate a deferrable-batch workload ANTI-PHASE with green. 2026-06-19.
2026-06-23: cloudlet sizes made HETEROGENEOUS (Gaussian) — see note below.

The temporal forecast lever only beats greedy drain when feast (high-green) windows
coincide with WORK SCARCITY: drain runs dry during feast → wastes green → deferral
fills those idle feast windows with work saved from brown windows. This generator
builds exactly that structure (the canonical carbon-aware deferrable-batch scenario):

  - arrivals concentrated in BROWN windows (low DC0+DC1 green), anti-phase with wind;
  - LOW average load so drain empties its queue and leaves feast windows idle;
  - loose per-job deadline (defer hours / across green windows);
  - HETEROGENEOUS cloudlet sizes (Gaussian around --mi, mean preserved).

WHY heterogeneous sizes matter: the anti-phase structure comes purely from arrival
TIMING. Cloudlet size used to be a single constant, which made all 128 cloudlets in a
routing step identical → identical per-slot scores → a deterministic policy routes the
WHOLE batch to one DC (all-or-nothing collapse, overload). Drawing length/pes from a
distribution gives each slot a distinct score so a greedy policy can spread a batch
across DCs. Mean load is preserved (lengths are rescaled to exact mean --mi), so util
and the green-power calibration are unchanged vs the old uniform trace.

Reads the green(t) profile saved by the probe to /tmp/oracle_gateway/green_ts.npy.
Writes a trace CSV (cloudlet_id,arrival_time,length,pes_required,file_size,output_size,
deadline) into resources/traces/.

Run from drl-manager/:
  .venv/bin/python gen_antiphase_workload.py --n 40000 --mi 400000 --brown-frac 0.9
"""
import argparse
from pathlib import Path
import numpy as np


def build_cloudlets(
    green_ts: np.ndarray,
    n: int,
    mi: int,
    pes: int,
    brown_frac: float = 0.9,
    arrival_window_frac: float = 0.83,
    deadline_steps: int = 3600,
    mi_std_frac: float = 0.45,
    mi_min_frac: float = 0.2,
    mi_max_frac: float = 2.6,
    pes_max: int = 4,
    seed: int = 7,
):
    """Build the anti-phase, size-heterogeneous cloudlet table.

    Returns (rows, stats) where rows is a list of
    (id, arrival, length, pes, file_size, output_size, deadline) tuples and stats is a
    dict describing the realized structure. Pure function (no I/O) so it is unit-testable.
    """
    G = np.asarray(green_ts)
    T = G.shape[0]
    g = G[:, 0] + G[:, 1]                            # the two real green DCs
    horizon = max(1, int(T * arrival_window_frac))
    g = g[:horizon]
    rng = np.random.default_rng(seed)

    # brown = below-median green steps; green = above. Sample arrivals mostly from brown.
    thresh = np.quantile(g[g > 0], 0.5) if (g > 0).any() else 0.0
    brown_steps = np.where(g <= thresh)[0]
    green_steps = np.where(g > thresh)[0]
    if brown_steps.size == 0:
        brown_steps = np.arange(horizon)
    if green_steps.size == 0:
        green_steps = brown_steps

    n_brown = int(round(n * brown_frac))
    n_green = n - n_brown
    arrivals = np.concatenate([
        rng.choice(brown_steps, size=n_brown, replace=True),
        rng.choice(green_steps, size=n_green, replace=True),
    ])
    arrivals.sort()

    # --- heterogeneous cloudlet sizes (Gaussian, mean preserved) ---
    lengths = rng.normal(mi, mi * mi_std_frac, size=n)
    lengths = np.clip(lengths, mi * mi_min_frac, mi * mi_max_frac)
    # Rescale so the realized mean is exactly `mi` → total work == n*mi (load preserved).
    lengths *= mi / lengths.mean()
    lengths = np.maximum(1, lengths.round().astype(np.int64))

    pes_arr = rng.normal(pes, 1.0, size=n)
    pes_arr = np.clip(np.round(pes_arr), 1, pes_max).astype(np.int64)

    # file/output sizes scale loosely with length (also heterogeneous, not load-critical).
    file_sizes = np.maximum(64, (lengths // 1500)).astype(np.int64)
    out_sizes = np.maximum(32, (file_sizes // 2)).astype(np.int64)

    rows = []
    for i in range(n):
        a = int(arrivals[i])
        rows.append((i, a, int(lengths[i]), int(pes_arr[i]),
                     int(file_sizes[i]), int(out_sizes[i]), a + deadline_steps))

    in_brown = float(np.isin(arrivals, brown_steps).mean())
    stats = {
        "T": int(T), "horizon": horizon, "thresh": float(thresh),
        "in_brown": in_brown, "total_work_gmi": float(lengths.sum() / 1e9),
        "mean_len": float(lengths.mean()), "len_uniq": int(np.unique(lengths).size),
        "len_min": int(lengths.min()), "len_max": int(lengths.max()),
        "pes_uniq": int(np.unique(pes_arr).size), "mean_pes": float(pes_arr.mean()),
    }
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--green-ts", default="/tmp/oracle_gateway/green_ts.npy")
    ap.add_argument("--n", type=int, default=40000, help="number of cloudlets (load knob)")
    ap.add_argument("--mi", type=int, default=400000, help="MEAN MI per cloudlet (load knob); "
                    "per-cloudlet length is drawn Gaussian around this with the mean preserved")
    ap.add_argument("--mi-std-frac", type=float, default=0.45,
                    help="length heterogeneity: Gaussian std as a fraction of --mi. 0 ≈ old uniform trace.")
    ap.add_argument("--mi-min-frac", type=float, default=0.2, help="clip lower bound (fraction of --mi)")
    ap.add_argument("--mi-max-frac", type=float, default=2.6, help="clip upper bound (fraction of --mi)")
    ap.add_argument("--pes", type=int, default=2, help="MEAN pes; per-cloudlet pes drawn around this")
    ap.add_argument("--pes-max", type=int, default=4, help="clip upper bound for per-cloudlet pes")
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
    rows, st = build_cloudlets(
        G, n=args.n, mi=args.mi, pes=args.pes, brown_frac=args.brown_frac,
        arrival_window_frac=args.arrival_window_frac, deadline_steps=args.deadline_steps,
        mi_std_frac=args.mi_std_frac, mi_min_frac=args.mi_min_frac, mi_max_frac=args.mi_max_frac,
        pes_max=args.pes_max, seed=args.seed,
    )

    res_root = Path(__file__).resolve().parent.parent / "cloudsimplus-gateway/src/main/resources"
    rel = args.out or f"traces/antiphase_n{args.n}_mi{args.mi}_b{args.brown_frac}_het.csv"
    dst = res_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        f.write("cloudlet_id,arrival_time,length,pes_required,file_size,output_size,deadline\n")
        for (i, a, length, p, fs, os_, ddl) in rows:
            f.write(f"{i},{a},{length},{p},{fs},{os_},{ddl}\n")

    print(f"wrote {args.n} cloudlets → resources/{rel}")
    print(f"  episode T={st['T']}, arrival horizon={st['horizon']} steps, "
          f"deadline=+{args.deadline_steps} steps")
    print(f"  length: mean={st['mean_len']:.0f} (target {args.mi}), uniq={st['len_uniq']}, "
          f"range=[{st['len_min']},{st['len_max']}]")
    print(f"  pes: mean={st['mean_pes']:.2f}, uniq={st['pes_uniq']} (max {args.pes_max})")
    print(f"  green threshold (median>0)={st['thresh']:.0f}W; "
          f"{100*st['in_brown']:.0f}% of arrivals land in brown steps (target {100*args.brown_frac:.0f}%)")
    print(f"  total work = {st['total_work_gmi']:.1f} G-MI  (mean preserved → util unchanged)")


if __name__ == "__main__":
    main()
