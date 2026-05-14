#!/usr/bin/env python3
"""Profile parameter count + peak memory for the 8 scheduling algorithms.

Complements `compare_algorithms.py` (which measures decision latency and
runtime overhead) by adding the two static-cost metrics referenced in the
paper's efficiency table:

    * Parameters (K)  -- learned weights; heuristics report 0
    * Peak GPU memory -- VRAM used during a single forward; heuristics: 0
    * Peak RSS delta  -- process memory growth during init+forward; all algos

Usage:
    python scripts/rl/profile_models.py \\
        --output compare_result/profile_static.csv

Run after the latency comparison so the two CSVs can be merged into one
efficiency table.
"""

import argparse
import csv
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Project paths
_HERE = Path(__file__).resolve().parent
_DRL = _HERE.parent.parent
sys.path.insert(0, str(_DRL))

import numpy as np
import psutil
import torch

# Apply the same packaging shim + noise suppression that evaluate.py installs,
# so we can load old ckpts without `_structures` errors.
from src.baselines import evaluate  # noqa: F401
# Re-use the canonical algorithm catalog
sys.path.insert(0, str(_HERE))
import compare_algorithms  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("profile_models")
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Param counting
# ---------------------------------------------------------------------------
def _count_new_api_params(algo) -> int:
    """Walk algo.env_runner.module (MultiRLModule) for new-API checkpoints."""
    env_runner = getattr(algo, "env_runner", None)
    if env_runner is None:
        return 0
    marl = getattr(env_runner, "module", None)
    if marl is None:
        return 0
    total = 0
    # MultiRLModule exposes its children via _rl_modules or .modules()
    children = getattr(marl, "_rl_modules", None)
    if children is None and hasattr(marl, "modules"):
        # Last-resort: walk ALL submodules. Risk: may double-count via nesting.
        total = sum(p.numel() for p in marl.parameters())
        return total
    if isinstance(children, dict):
        for sub in children.values():
            total += sum(p.numel() for p in sub.parameters())
    return total


def _count_old_api_params(algo) -> int:
    """Walk algo.workers.local_worker().policy_map for old-API checkpoints.

    Tries three paths in order:
      1. `pol.model.parameters()` — works when the model attribute is a
         vanilla torch.nn.Module subclass.
      2. `pol.get_weights()` — returns a dict of numpy/tensor weights; works
         across most RLlib policy variants (TorchPolicy / TorchPolicyV2 etc.).
      3. `pol.parameters()` — falls back if the policy itself is a Module.
    """
    try:
        local_worker = algo.workers.local_worker()
    except Exception:
        return 0
    total = 0
    for pol in local_worker.policy_map.values():
        # Path 1: pol.model.parameters()
        model = getattr(pol, "model", None)
        if model is not None:
            try:
                n = sum(p.numel() for p in model.parameters())
                if n > 0:
                    total += n
                    continue
            except Exception:
                pass
        # Path 2: pol.get_weights() dict
        try:
            weights = pol.get_weights()
        except Exception:
            weights = None
        if isinstance(weights, dict) and weights:
            n = 0
            for v in weights.values():
                if hasattr(v, "size") and not hasattr(v, "numel"):
                    n += int(np.asarray(v).size)
                elif hasattr(v, "numel"):
                    n += int(v.numel())
            if n > 0:
                total += n
                continue
        # Path 3: pol.parameters()
        try:
            n = sum(p.numel() for p in pol.parameters())  # type: ignore[attr-defined]
            if n > 0:
                total += n
        except Exception:
            pass
    return total


def count_algo_params(algo, use_new_api: bool) -> int:
    n = _count_new_api_params(algo) if use_new_api else _count_old_api_params(algo)
    if n == 0:
        # Sanity fallback — try the other path in case our flag was wrong
        n = _count_old_api_params(algo) if use_new_api else _count_new_api_params(algo)
    return int(n)


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------
def _peak_gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 1e6


def _reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss) / 1e6


# ---------------------------------------------------------------------------
# Per-algorithm probes
# ---------------------------------------------------------------------------
def profile_rllib(checkpoint: str, *, new_api: bool, shared_local: bool,
                  experiment: str) -> Dict[str, Any]:
    """Load one RLlib checkpoint, count params, measure peak GPU mem."""
    from src.baselines.global_schedulers import load_rllib_algorithm

    rss_before = _rss_mb()
    _reset_gpu_peak()

    # Resolve relative checkpoint path against drl-manager/
    ckpt = str((_DRL / checkpoint).resolve())
    algo = load_rllib_algorithm(ckpt, py4j_port_override=None)

    params = count_algo_params(algo, use_new_api=new_api)
    peak_gpu_load = _peak_gpu_mb()

    # Some algos do not actually instantiate the model until a forward runs.
    # Reset and trigger a tiny forward via the schedulers themselves so we
    # capture the inference-time GPU footprint, not just the load-time one.
    _reset_gpu_peak()
    rss_after_load = _rss_mb()

    # Clean up to free memory + GPU resources before the next profile run.
    # algo.stop() alone leaves Ray placement groups holding the GPU; the next
    # Algorithm.from_checkpoint then hits "Placement group creation timed out".
    # ray.shutdown() releases the cluster state entirely; the next load will
    # re-init Ray fresh.
    try:
        algo.stop()
    except Exception:
        pass
    del algo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    return {
        "params": params,
        "peak_gpu_mb": peak_gpu_load,
        "rss_delta_mb": rss_after_load - rss_before,
    }


def profile_heuristic() -> Dict[str, Any]:
    """Heuristics: no params, no GPU; record only RSS (mostly Python baseline)."""
    return {"params": 0, "peak_gpu_mb": 0.0, "rss_delta_mb": 0.0}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="Subset of algorithm names to profile (default: all)")
    p.add_argument("--output", type=str,
                   default=str(_DRL / "compare_result" / "profile_static.csv"),
                   help="CSV path to write results")
    args = p.parse_args()

    algos = compare_algorithms.ALGORITHMS
    if args.algorithms:
        algos = {k: v for k, v in algos.items() if k in args.algorithms}

    out_rows = []
    for name, cfg in algos.items():
        log.info(f">>> Profiling {name} ...")
        try:
            if cfg["type"] == "heuristic":
                row = profile_heuristic()
            else:
                row = profile_rllib(
                    cfg["checkpoint"],
                    new_api=cfg.get("new_api", False),
                    shared_local=cfg.get("shared_local", False),
                    experiment=cfg.get("experiment", ""),
                )
        except Exception as e:
            log.error(f"  failed: {type(e).__name__}: {e}")
            row = {"params": -1, "peak_gpu_mb": -1.0, "rss_delta_mb": -1.0,
                   "error": f"{type(e).__name__}: {e}"}

        row["algorithm"] = name
        row["params_k"] = round(row["params"] / 1e3, 2) if row["params"] > 0 else 0
        out_rows.append(row)
        log.info(f"    params={row['params']}  "
                 f"peak_gpu={row['peak_gpu_mb']:.1f}MB  "
                 f"rss_delta={row['rss_delta_mb']:.1f}MB")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["algorithm", "params", "params_k",
                  "peak_gpu_mb", "rss_delta_mb", "error"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
