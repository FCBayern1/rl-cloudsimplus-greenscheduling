#!/usr/bin/env python3
"""
TimeCAP GPU inference benchmark.

Measures the per-step latency that an RL env would pay if it called
TimeCAPGodEyeProvider.step_and_get(step) once every simulation step
(forecast_every=1).

Reports
-------
1. Single-turbine forward latency on GPU (mean / median / p90 / p99)
2. Sweep over N turbines per provider, using the current sequential
   per-turbine forward (the production code path)
3. Same sweep using a batched forward (proof-of-concept; useful to see how
   much we'd gain by batching all turbines through the model in one shot)

Run via the sbatch wrapper: scripts/timecap/benchmark_gpu.sh

Or stand-alone (assumes a CUDA device is visible):
    cd drl-manager
    python ../scripts/timecap/benchmark_gpu.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
_DRLMANAGER = _REPO / "drl-manager"
for _p in (str(_DRLMANAGER), str(_DRLMANAGER / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prediction.timecap_godeye_provider import TimeCAPGodEyeProvider     # noqa: E402
from timecap_prediction.predictor import TimeCAP_GreenPredictor          # noqa: E402


DEFAULT_CSV_DIR = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/split"
DEFAULT_CKPT = (
    _REPO / "drl-manager/timecap_prediction/TimeCAP/model"
    / "finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR,
                   help="Directory containing Turbine_<id>_2021.csv files")
    p.add_argument("--device", default="cuda",
                   help='"cuda", "cuda:0", or "cpu" (cpu is for sanity checks)')
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--bench-iters", type=int, default=100)
    p.add_argument("--turbine-counts", type=str, default="1,2,4,8,16,32,64,134",
                   help="Comma-separated turbine counts to sweep over")
    p.add_argument("--no-batched", action="store_true",
                   help="Skip the batched-forward proof-of-concept")
    return p.parse_args()


def discover_turbines(csv_dir: Path, want: int) -> List[int]:
    """Pick the first `want` available turbine ids from the CSV directory."""
    ids = []
    for f in sorted(csv_dir.glob("Turbine_*_2021.csv")):
        # Filename: Turbine_<id>_2021.csv
        try:
            tid = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        ids.append(tid)
        if len(ids) >= want:
            break
    if len(ids) < want:
        raise RuntimeError(f"Only found {len(ids)} turbine CSVs in {csv_dir}, need {want}")
    return ids


def percentiles(samples_ms: List[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)
    def at(q: float) -> float:
        return s[min(n - 1, int(q * n))]
    return {
        "mean":   statistics.mean(s),
        "median": statistics.median(s),
        "p90":    at(0.90),
        "p99":    at(0.99),
        "min":    min(s),
        "max":    max(s),
    }


def fmt_row(label: str, stats: dict, extra: str = "") -> str:
    return (f"  {label:<35s} "
            f"mean={stats['mean']:7.2f}  "
            f"median={stats['median']:7.2f}  "
            f"p90={stats['p90']:7.2f}  "
            f"p99={stats['p99']:7.2f}  "
            f"min={stats['min']:7.2f}  "
            f"max={stats['max']:7.2f}  ms"
            + (f"  {extra}" if extra else ""))


def cuda_sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def bench_provider_step(prov: TimeCAPGodEyeProvider,
                        warmup: int, iters: int,
                        device: torch.device) -> dict:
    """Benchmark prov.step_and_get(step) end-to-end (update + forecast + aggregate)."""
    prov.reset()
    prov.warmup(start_step=0)

    # Warm-up — first GPU forward is much slower (cudnn JIT, allocator warm-up)
    for s in range(96, 96 + warmup):
        _ = prov.step_and_get(s)
    cuda_sync(device)

    times_ms: List[float] = []
    base = 96 + warmup
    for s in range(base, base + iters):
        cuda_sync(device)
        t0 = time.perf_counter()
        _ = prov.step_and_get(s)
        cuda_sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return percentiles(times_ms)


def bench_batched_forward(predictor: TimeCAP_GreenPredictor,
                          turbine_ids: List[int],
                          warmup: int, iters: int,
                          device: torch.device) -> dict:
    """
    Proof-of-concept: stack all turbines' history buffers into a single
    (N, seq_len, num_features) tensor and run one forward. Doesn't go through
    the provider — measures pure forward time only.
    """
    seq_len = predictor.seq_len
    num_features = predictor.num_features

    # Construct one batched input from current buffers, zero-padded if buffers short
    def build_batch() -> torch.Tensor:
        rows = []
        for tid in turbine_ids:
            hist = list(predictor._history[tid])
            if len(hist) < seq_len:
                pad = [np.zeros(num_features, dtype=np.float32)] * (seq_len - len(hist))
                hist = pad + hist
            rows.append(np.stack(hist, axis=0))
        x = np.stack(rows, axis=0)                          # (N, seq_len, num_features)
        return torch.from_numpy(x).float().to(device)

    # Make sure the buffers have content
    predictor.reset()
    for s in range(seq_len):
        predictor.update(s)

    x_batch = build_batch()

    # Warm-up
    for _ in range(warmup):
        with torch.no_grad():
            _ = predictor.model(x_batch, activate_os_head=True)
    cuda_sync(device)

    times_ms: List[float] = []
    for _ in range(iters):
        cuda_sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = predictor.model(x_batch, activate_os_head=True)
        cuda_sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return percentiles(times_ms)


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    if not args.csv_dir.exists():
        sys.exit(f"csv directory not found: {args.csv_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("Requested CUDA but torch.cuda.is_available() is False.")

    # cudnn benchmark mode helps for fixed-shape inputs (our case)
    torch.backends.cudnn.benchmark = True

    print("=" * 86)
    print("TimeCAP GPU inference benchmark")
    print("=" * 86)
    print(f"  device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(device)}")
        props = torch.cuda.get_device_properties(device)
        print(f"  VRAM total   : {props.total_memory / 1e9:.1f} GB")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  warmup iters : {args.warmup_iters}")
    print(f"  bench iters  : {args.bench_iters}")
    print(f"  pid          : {os.getpid()}")

    counts = [int(x) for x in args.turbine_counts.split(",")]
    max_count = max(counts)

    # Discover turbines once (one big CSV map; we'll subset per benchmark)
    print(f"\nDiscovering turbines in {args.csv_dir} ...")
    all_tids = discover_turbines(args.csv_dir, max_count)
    print(f"  using {len(all_tids)} turbines: {all_tids[:8]}{' ...' if len(all_tids) > 8 else ''}")

    # ---------------------------------------------------------------------
    # Section 1 — provider step_and_get latency vs turbine count
    # (this is the actual production code path RL will pay each env step)
    # ---------------------------------------------------------------------
    print()
    print("─" * 86)
    print("Section 1 — TimeCAPGodEyeProvider.step_and_get()  [end-to-end, single DC]")
    print("─" * 86)

    section1_results = {}
    for n in counts:
        tids = all_tids[:n]
        csv_paths = {tid: str(args.csv_dir / f"Turbine_{tid}_2021.csv") for tid in tids}
        prov = TimeCAPGodEyeProvider(
            dc_assignments={0: tids},
            turbine_csv_paths=csv_paths,
            checkpoint_path=str(args.checkpoint),
            forecast_every=1,
            device=str(device),
        )
        stats = bench_provider_step(prov, args.warmup_iters, args.bench_iters, device)
        section1_results[n] = stats
        per_t = stats['median'] / max(n, 1)
        print(fmt_row(f"N={n:>3d} turbines/DC", stats, extra=f"({per_t:5.2f} ms / turbine)"))
        del prov  # free GPU memory between trials
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Section 2 — batched forward (proof-of-concept)
    # ---------------------------------------------------------------------
    if not args.no_batched:
        print()
        print("─" * 86)
        print("Section 2 — Batched forward (model only, all turbines in one batch)")
        print("─" * 86)
        print("  Tells you how much the per-turbine for-loop is costing vs. ideal batching.")
        print("  Excludes update()/aggregation overhead — pure model.forward().")

        # One predictor with the maximum turbine count, then we vary the batch we feed it
        tids_full = all_tids[:max_count]
        csv_paths = {tid: str(args.csv_dir / f"Turbine_{tid}_2021.csv") for tid in tids_full}
        big_prov = TimeCAPGodEyeProvider(
            dc_assignments={0: tids_full},
            turbine_csv_paths=csv_paths,
            checkpoint_path=str(args.checkpoint),
            forecast_every=1,
            device=str(device),
        )

        for n in counts:
            stats = bench_batched_forward(
                big_prov.predictor, tids_full[:n],
                args.warmup_iters, args.bench_iters, device,
            )
            per_t = stats['median'] / max(n, 1)
            print(fmt_row(f"N={n:>3d} turbines (batched)", stats, extra=f"({per_t:5.2f} ms / turbine)"))

        del big_prov
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Summary headline
    # ---------------------------------------------------------------------
    print()
    print("=" * 86)
    print("Summary")
    print("=" * 86)
    if 1 in section1_results:
        s1 = section1_results[1]
        print(f"  Single-turbine end-to-end forecast latency: {s1['median']:.2f} ms (median)")
        rl_step_budget_ms = 100  # typical
        slack = (rl_step_budget_ms - s1['median']) / rl_step_budget_ms * 100
        print(f"  → at forecast_every=1, leaves ~{slack:.0f}% of a typical {rl_step_budget_ms}-ms RL step idle")
    print("=" * 86)


if __name__ == "__main__":
    main()
