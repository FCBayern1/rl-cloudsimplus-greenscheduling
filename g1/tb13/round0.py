"""Round 0: the physical pre-screen. No solving, no workload, no seeds.

Codex 2026-09-01: the screen must not carry words that are not arithmetic. Every gate here
is a formula over the wind and the site constants, evaluated on physical keys only.

A physical key is what the wind and the sites make of a candidate, independent of how many
jobs arrive or how the delay budget is set:

    pes_per_job, concurrency, turbines_per_site, installed_divisor, horizon,
    triplet index, season index

`axes_grid()` in instance_gen also numbers 8,640, which is a coincidence of the axis sizes
and not the same set: that one carries n_jobs, wait_cap and budget_fraction and carries no
triplet or season. The two are built and tested separately for that reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402

YEAR = 2021
CORR_BAND = (0.70, 0.95)          # positive correlation only, never an absolute value
BEST_DC_CHANGE_MIN = 0.10
N_TRIPLETS = 6
N_SEASONS = 6
ANCHORS_PER_LAYER = 4


def discovery_pool():
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data_split.txt")).read()
    return [int(x) for x in txt.split("DISCOVERY [")[1].split("]")[0].split(",")]


def confirmation_pool():
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data_split.txt")).read()
    return [int(x) for x in txt.split("CONFIRMATION [")[1].split("]")[0].split(",")]


def round0_physical_keys():
    """Every physical unit, in a fixed order. No n_jobs, no budget, no wait cap, no seed."""
    pool = discovery_pool()
    keys = []
    for tps in ig.TURBINES_PER_SITE:
        triples = ig.turbine_triples(pool, tps, N_TRIPLETS)
        for T in ig.HORIZON:
            seasons = ig.offsets_for(YEAR, max(ig.HORIZON), N_SEASONS)
            for ti, triple in enumerate(triples):
                for si, off in enumerate(seasons):
                    for pes in ig.PES_PER_JOB:
                        for c in ig.CONCURRENCY:
                            for div in ig.INSTALLED_DIVISOR:
                                keys.append({
                                    "pes_per_job": pes, "concurrency": c,
                                    "turbines_per_site": tps, "installed_divisor": div,
                                    "horizon": T, "triplet_index": ti,
                                    "season_index": si, "triplet": triple,
                                    "season_offset": off, "year": YEAR,
                                })
    return keys


def anchor_sha(key):
    """Canonical SHA of a physical key, used to pick anchors without looking at results."""
    payload = {
        "grid_hash": ig.grid_hash(), "year": key["year"], "triplet": key["triplet"],
        "season_offset": key["season_offset"], "pes_per_job": key["pes_per_job"],
        "concurrency": key["concurrency"], "turbines_per_site": key["turbines_per_site"],
        "installed_divisor": key["installed_divisor"], "horizon": key["horizon"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def residual_green(key):
    """Per-site residual green over the window, in watts."""
    static = ig.HOST_IDLE_W * ig.HOSTS_PER_SITE
    T, off, div = key["horizon"], key["season_offset"], key["installed_divisor"]
    g = np.zeros((ig.N_DC, T))
    for d, ts in enumerate(key["triplet"]):
        acc = None
        for t in ts:
            v = ig._series(t, key["year"])[off:off + T]
            acc = v if acc is None else acc + v
        g[d] = acc * 1000.0 / div
    return np.maximum(g - static, 0.0), g


def physical_metrics(key):
    """Every gate quantity, all arithmetic, none of them thresholds in disguise."""
    gres, graw = residual_green(key)
    p_job = key["pes_per_job"] * ig.DYN_W_PER_PE
    cb = np.asarray(ig.BROWN_FACTORS, dtype=float)
    cg = np.asarray(ig.GREEN_FACTORS, dtype=float)

    # A time is poor when no site can cover one job of this size from residual green.
    poor = (gres.max(axis=0) < p_job)
    simultaneous_poor_fraction = float(poor.mean())

    # Marginal carbon of placing one such job at each site, at each epoch.
    marg = (cb.reshape(-1, 1) * np.maximum(p_job - gres, 0.0)
            + cg.reshape(-1, 1) * np.minimum(p_job, gres))
    best = np.argmin(marg, axis=0)               # ties go to the lower site index
    counts = np.bincount(best, minlength=ig.N_DC)
    best_dc_change_fraction = float(1.0 - counts.max() / len(best))

    with np.errstate(invalid="ignore"):
        C = np.corrcoef(gres)
    pair = [C[i, j] for i in range(ig.N_DC) for j in range(i + 1, ig.N_DC)]
    degenerate = any(np.isnan(x) for x in pair)

    demand = key["concurrency"] * key["pes_per_job"] * ig.DYN_W_PER_PE
    return {
        "rho_residual": float(demand / max(gres.mean(), 1e-9)),
        "pes_share": key["pes_per_job"] / ig.CAP_PES_PER_SITE,
        "pairwise_corr": [float(x) for x in pair],
        "simultaneous_poor_fraction": simultaneous_poor_fraction,
        "best_dc_change_fraction": best_dc_change_fraction,
        "mean_residual_green_w": float(gres.mean()),
        "corr_degenerate": bool(degenerate),
    }


def passes_physical_gate(m):
    """Correlation is positive and banded; the other two quantities are non-degenerate.

    The load ratio is recorded for every unit and is never a rejection criterion here: a
    band on it would have been chosen after seeing the design pilot, so the decision is
    left to the solved gates downstream.
    """
    if m["corr_degenerate"]:
        return False, "a site has no variation in residual green"
    if not all(CORR_BAND[0] <= r <= CORR_BAND[1] for r in m["pairwise_corr"]):
        return False, f"pairwise correlation {m['pairwise_corr']} outside {CORR_BAND}"
    f = m["simultaneous_poor_fraction"]
    if not (0.0 < f < 1.0):
        return False, f"simultaneous-poor fraction {f} is degenerate"
    if m["best_dc_change_fraction"] < BEST_DC_CHANGE_MIN:
        return False, (f"best site is fixed for "
                       f"{100 * (1 - m['best_dc_change_fraction']):.0f}% of the window")
    return True, ""


def neighbourhood(div):
    """Anchor's divisor plus its neighbours; at an edge, the nearest three consecutive."""
    order = list(ig.INSTALLED_DIVISOR)
    i = order.index(div)
    if i == 0:
        return order[:3]
    if i == len(order) - 1:
        return order[-3:]
    return order[i - 1:i + 2]


def select_anchors(passing):
    """Four smallest hashes per triplet-season layer, among physical-gate passers."""
    layers = {}
    for k in passing:
        layers.setdefault((k["triplet_index"], k["season_index"], k["turbines_per_site"]),
                          []).append(k)
    chosen, empty = [], []
    for lid in sorted(layers):
        ranked = sorted(layers[lid], key=lambda k: anchor_sha(k))
        if not ranked:
            empty.append(lid)
            continue
        chosen.extend(ranked[:ANCHORS_PER_LAYER])
    return chosen, empty
