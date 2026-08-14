#!/usr/bin/env python3
"""Offline calibration of the per-action carbon normalisation (mu/sigma).

Produces the artifact consumed by per_action_carbon_norm: scale_only /
centered_zscore (per_action_carbon_mu / per_action_carbon_sigma in
config_C.yml). One artifact is produced per scenario and SHARED by every arm,
seed and worker — never per-run online statistics, which would give different
arms different reward functions.

What is sampled
---------------
marginalKg for ALL candidate (task, DC) pairs — green AND brown DCs — not the pairs a policy
actually picks, so the distribution is not skewed by any policy's choices —
at decision times drawn across the whole closed-book offset range. The formula
mirrors MultiDatacenterSimulationCore.computeDcCostFeatures under
window_carbon_source=persistence:

    greenRatio = min(1, greenW_now / demandW)
    marginalKg = (MI / miPerKg) * (greenRatio*greenF + (1-greenRatio)*brownF)

Defer is NOT included as carbon=0 (that would manufacture an always-defer
positive reward under centered_zscore).

Modelling assumption (recorded in the artifact)
-----------------------------------------------
demandW per DC is not observable offline (it depends on runtime utilisation
with idle_host_power_down), so it is swept over three utilisation levels and
the artifact reports mu/sigma per level plus the recommended middle level.
VALIDATION HOOK: after the first 100k smoke run compare the artifact mu with
epCarbonRawKgSum/epCarbonNormSampleCount from the gateway; if off by more than
2x, re-calibrate (docs/V31_PREREG.md §7).

Usage:
    .venv/bin/python calibrate_reward_norm.py --experiment experiment_v3_1_oracle \
        [--samples 2000] [--seed 20260813] [--out calib/<exp>_carbon_norm.json]
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources"

# Per-host demand (W) at the three swept utilisation levels, matching the
# spec_asus power models' rough scale after the compressed divisor is applied
# to the GREEN side only (demand is in real Watts at DC scale in the sim's
# routing-time estimate). These are assumptions; they are recorded verbatim.
UTIL_LEVELS = {"low_util": 0.3, "mid_util": 0.6, "high_util": 0.9}
WATT_PER_HOST_FULL = 300.0   # rough spec_asus full-load draw per host


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--samples", type=int, default=2000,
                    help="decision times sampled across the offset range")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO / "config_C.yml"))[args.experiment]
    rng = np.random.default_rng(args.seed)

    trace_path = GATE / "traces" / Path(cfg["cloudlet_trace_file"]).name
    rows = list(csv.DictReader(open(trace_path)))
    mi = np.array([float(r["length"]) for r in rows])
    mi_per_kg = max(1e3, float(cfg.get("mi_per_kg_factor") or 3.5e6))
    divisor = float(cfg["compressed_power_divisor"])

    # green power series per green DC (W, compressed scale — same as
    # getCurrentPowerW), plus per-DC host counts for the demand model
    # ALL datacenters are routing candidates — brown-only DCs (no turbines)
    # contribute marginalKg = (MI/miPerKg)·brown_f at greenRatio=0. Sampling
    # only green DCs (the first-draft bug) under-covered the right tail of the
    # distribution the normaliser will actually see.
    dcs, wind_files = [], []
    for d in cfg["datacenters"]:
        tot = None
        for t in d.get("turbine_ids") or []:
            p = GATE / f"windProduction/simplified/Turbine_{t}_2021.csv"
            wind_files.append(p)
            v = np.array([float(r["power_kw"]) for r in csv.DictReader(open(p))])
            tot = v if tot is None else tot + v
        hosts = sum(v for k, v in d.items() if k.startswith("host_count_") and isinstance(v, int))
        dcs.append({
            "name": d["name"],
            "green_w": (tot * 1000.0 / divisor) if tot is not None else None,
            "hosts": max(1, hosts),
            "green_f": float(d["green_carbon_factor"]),
            "brown_f": float(d["brown_carbon_factor"]),
        })

    offset_range = int(cfg.get("green_episode_offset_range", 0)) or 1
    horizon = 7200
    n_rows = min(len(d["green_w"]) for d in dcs if d["green_w"] is not None)
    times = rng.integers(0, offset_range + horizon, size=args.samples) % n_rows
    task_idx = rng.integers(0, len(mi), size=args.samples)

    per_level = {}
    per_dc_summary = {}
    for level, util in UTIL_LEVELS.items():
        kg = []
        by_dc = {d["name"]: [] for d in dcs}
        for t, j in zip(times, task_idx):
            for d in dcs:                       # ALL candidate DCs (green AND brown)
                demand_w = d["hosts"] * WATT_PER_HOST_FULL * util
                green_now = d["green_w"][t] if d["green_w"] is not None else 0.0
                ratio = min(1.0, green_now / max(1e-9, demand_w))
                m = (mi[j] / mi_per_kg) * (ratio * d["green_f"] + (1 - ratio) * d["brown_f"])
                kg.append(m)
                by_dc[d["name"]].append(m)
        kg = np.array(kg)
        mu, sigma = float(kg.mean()), float(kg.std())
        # V3.2 sigma_spatial: spread of candidate DIFFERENCES from the
        # per-decision candidate mean (the quantity the candidate-centered
        # spatial term divides by), NOT the pooled spread across decisions.
        n_dc_all = len(dcs)
        kg_mat = kg.reshape(-1, n_dc_all)
        centered = kg_mat - kg_mat.mean(axis=1, keepdims=True)
        sigma_spatial = float(centered.std())
        per_level[level] = {
            "mu": mu, "sigma": sigma, "sigma_spatial": sigma_spatial,
            "quantiles": {q: float(np.percentile(kg, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "clip_rate_at_pm5sigma": float(np.mean(np.abs((kg - mu) / max(1e-12, sigma)) > 5.0)),
            "n": int(kg.size),
        }
        per_dc_summary[level] = {n: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                                 for n, v in by_dc.items()}

    artifact = {
        "artifact": "per_action_carbon_norm calibration",
        "date": "2026-08-13",
        "experiment": args.experiment,
        "units": "kg per routing action (marginalKg as in computeDcCostFeatures)",
        "reference_policy": "all candidate (task, DC) pairs incl. brown DCs, uniform decision "
                            "times over offset range + horizon (policy-free)",
        "seed": args.seed,
        "samples_decision_times": args.samples,
        "trace_file": str(trace_path.name), "trace_sha": sha12(trace_path),
        "wind_files": sorted({p.name for p in wind_files}),
        "wind_sha": {p.name: sha12(p) for p in sorted(set(wind_files))},
        "config_keys": {k: cfg.get(k) for k in
                        ("compressed_power_divisor", "mi_per_kg_factor",
                         "green_episode_offset_range", "window_carbon_source")},
        "demand_model": {"watt_per_host_full": WATT_PER_HOST_FULL,
                         "util_levels": UTIL_LEVELS,
                         "note": "demandW unobservable offline; swept. RECOMMENDED "
                                 "level = mid_util. Validate vs epCarbonRawKgSum "
                                 "after the 100k smoke (PREREG §7)."},
        "recommended": {"per_action_carbon_mu": per_level["mid_util"]["mu"],
                        "per_action_carbon_sigma": per_level["mid_util"]["sigma"],
                        "per_action_spatial_sigma": per_level["mid_util"]["sigma_spatial"]},
        "per_level": per_level,
        "per_dc": per_dc_summary,
    }

    out = Path(args.out) if args.out else REPO / "drl-manager" / "calib" / f"{args.experiment}_carbon_norm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"artifact -> {out}")
    print(f"recommended mu={artifact['recommended']['per_action_carbon_mu']:.4g} "
          f"sigma={artifact['recommended']['per_action_carbon_sigma']:.4g}")
    for lvl, st in per_level.items():
        print(f"  {lvl:10s} mu={st['mu']:.4g} sigma={st['sigma']:.4g} "
              f"clip@5sigma={st['clip_rate_at_pm5sigma']*100:.2f}%")


if __name__ == "__main__":
    main()
