#!/usr/bin/env python3
"""
Offline probe for the within-batch dispersion of the CRD responsibility share.

Restores a trained EU-CRD checkpoint, runs exactly ONE training iteration
(sampling one batch through the live simulator, then a learner update), and
reads back the dispersion diagnostics that ``crd_q_loss._log_crd_diagnostics``
emits — in particular ``crd/reweight_w_std`` = std of w = rho_t / mean(rho),
the direct measure of how strongly the mean-preserving reweighting
redistributes credit. If this is ~0 the reweighting is inert on this data
distribution; if it is substantial, the responsibility panel should be drawn
as a mean + p10-p90 band rather than a bare mean line.

The updated weights are DISCARDED (nothing is saved back to the checkpoint);
the one extra gradient step exists only to drive the logging path.

Usage::

    python drl-manager/scripts/probe_rho_dispersion.py \\
        --checkpoint logs/v3ht_knSb_s1/.../checkpoint_000010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# Keys we report, in print order. reweight_w_std is the headline.
REPORT_KEYS = (
    "crd/reweight_w_std",
    "crd/reweight_w_max",
    "crd/rho_routing_std",
    "crd/rho_routing_p10",
    "crd/rho_routing_p90",
    "crd/rho_routing_mean",
    "crd/rho_forecast_mean",
    "crd/c_t_mean",
)


def extract_crd_metrics(result: Dict[str, Any],
                        module_id: str = "global_policy") -> Dict[str, float]:
    """Pull the crd/* scalars for ``module_id`` out of an algo.train() result.

    RLlib nests learner stats either under ``learners`` (new API) or
    ``info/learner`` (older layouts); tolerate both and return {} when the
    module or metrics are absent — the caller treats that as a failed probe.
    """
    for path in (("learners",), ("info", "learner")):
        node: Any = result
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        stats = node.get(module_id, {}) if isinstance(node, dict) else {}
        crd = {k: float(v) for k, v in stats.items()
               if isinstance(k, str) and k.startswith("crd/")
               and isinstance(v, (int, float))}
        if crd:
            return crd
    return {}


def format_report(crd: Dict[str, float]) -> str:
    """One grep-able line: RHO_DISPERSION key=value ..."""
    parts = []
    for k in REPORT_KEYS:
        if k in crd:
            parts.append(f"{k.split('/')[-1]}={crd[k]:.4f}")
    return "RHO_DISPERSION " + " ".join(parts) if parts else "RHO_DISPERSION <empty>"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--module", default="global_policy")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[probe] checkpoint not found: {ckpt}")
        return 1

    # Heavy imports behind main() so the unit tests can import the helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.baselines.global_schedulers import load_rllib_algorithm

    print(f"[probe] restoring {ckpt}")
    algo = load_rllib_algorithm(str(ckpt))
    print("[probe] running one training iteration (weights are discarded)...")
    result = algo.train()

    crd = extract_crd_metrics(result, module_id=args.module)
    print(format_report(crd))
    if "crd/reweight_w_std" not in crd:
        print("[probe] WARNING: dispersion keys absent — checkpoint code path "
              "may predate the dispersion logging, or module_id is wrong.")
    try:
        algo.stop()
    except Exception:
        pass
    return 0 if crd else 1


if __name__ == "__main__":
    raise SystemExit(main())
