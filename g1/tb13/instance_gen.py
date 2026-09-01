"""Reproducible TB13 instance generation. Every field is derived, none is hand-picked.

Codex 2026-09-01: a dimension count is not an instance set. A cell is only executable when
the arrivals, runtimes, deadlines, capacities, powers and turbine assignment can all be
regenerated from the frozen axes and a seed.
"""
from __future__ import annotations

import csv
import functools
import hashlib
import os

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__)) + "/../.."
WD = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified")

# Measured on the testbed. (214 - 51.4)/64 = 2.5406 W per PE at full load, and the
# cloudlets run at cloudlet_cpu_utilization = 0.5, so a PE actually draws half that.
DYN_W_PER_PE = (214.0 - 51.4) / 64.0 * 0.5          # 1.2703
HOST_IDLE_W = 51.4
EPOCH_SECONDS = 600.0

# Frozen axes.
# 2 PE is kept only as a fluid negative control: at 16 PE per site it occupies 12.5% and
# capacity never binds, which is the regime TB13 exists to avoid.
PES_PER_JOB = (2, 4, 8)
FLUID_CONTROL_PES = (2,)
MIN_PES_SHARE = 0.25            # a passing cell needs pes/cap at or above this
CONCURRENCY = (1, 2, 3, 5)
TURBINES_PER_SITE = (1, 2)
INSTALLED_DIVISOR = (1500, 3000, 6000, 12000, 24000)
N_JOBS = (8, 10, 12)
HORIZON = (36, 48)
BUDGET_FRACTION = (0.10, 0.20, 0.30, 0.40)
WAIT_CAP_ROWS = (6, 12, 24)                          # tier one only; 144 is tier two
SEEDS = (0, 1, 2)
N_DC = 3
HOSTS_PER_SITE = 1
# Two 8-PE VMs per site. The host carries 64 PEs but only what is placed in VMs is
# schedulable, and a 64-PE scheduling capacity makes an 8-PE job 12.5% of a site, which
# never binds. The simulator side must create exactly these two VMs.
VMS_PER_SITE = 2
PES_PER_VM = 8
CAP_PES_PER_SITE = VMS_PER_SITE * PES_PER_VM         # 16

# One epoch is 600 s, so 1-4 hours is 6-24 rows. Tier one takes the short end so a
# horizon of 36-48 rows still holds a runtime plus its wait; tier two spans the full
# registered range. Runtimes are drawn from the frozen set, never shortened to fit.
RUNTIME_ROWS_TIER1 = (6, 12)
RUNTIME_ROWS_TIER2 = (6, 12, 24)
BROWN_FACTORS = (0.30, 0.50, 0.70)
GREEN_FACTORS = (0.02, 0.02, 0.02)


@functools.lru_cache(maxsize=None)
def _series(turbine, year):
    """One turbine-year, parsed once. Callers must treat the array as read only.

    Round 0 visits 8,640 physical units over a handful of turbines; re-parsing a 52,559
    row CSV each time would dominate the run. Caching changes only the speed.
    """
    with open(f"{WD}/Turbine_{turbine}_{year}.csv") as f:
        a = np.array([float(r["power_kw"] or 0.0) for r in csv.DictReader(f)])
    a.setflags(write=False)
    return a


def turbine_triples(pool, per_site, n_triples, seed=20260901):
    """Deterministic site groupings drawn from a pool, disjoint within a triple."""
    rnd = np.random.default_rng(seed)
    need = N_DC * per_site
    out = []
    seen = set()
    for _ in range(n_triples * 40):
        pick = tuple(sorted(rnd.choice(pool, size=need, replace=False).tolist()))
        if pick in seen:
            continue
        seen.add(pick)
        out.append([list(pick[i * per_site:(i + 1) * per_site]) for i in range(N_DC)])
        if len(out) == n_triples:
            break
    return out


def offsets_for(year, horizon, n_offsets, seed=20260901):
    """Season-spread window starts, deterministic and disjoint."""
    length = len(_series(2, year))
    span = horizon + 4
    usable = length - span - 1
    step = usable // n_offsets
    return [int(i * step + step // 3) for i in range(n_offsets)]


def build_instance(axes, seed, year=2021):
    """Return (Scenario kwargs, provenance dict). Pure function of axes and seed."""
    from exact_oracle import Scenario

    rng = np.random.default_rng(seed * 1_000_003 + axes["offset"])
    T = axes["horizon"]
    sites = axes["turbines"]
    div = axes["installed_divisor"]
    green = np.zeros((N_DC, T))
    for d, ts in enumerate(sites):
        acc = None
        for t in ts:
            v = _series(t, year)[axes["offset"]:axes["offset"] + T]
            acc = v if acc is None else acc + v
        green[d] = acc * 1000.0 / div

    static = np.full(N_DC, HOST_IDLE_W * HOSTS_PER_SITE, dtype=float)
    n = axes["n_jobs"]
    pes = np.full(n, axes["pes_per_job"], dtype=int)
    r = rng.choice(axes.get("runtime_set", RUNTIME_ROWS_TIER1), size=n)
    # Arrivals are spread so that the mean number of jobs in service matches the target
    # concurrency: n jobs of mean runtime r_bar over an arrival span S give S = n*r_bar/c.
    span = max(1, int(round(n * float(r.mean()) / axes["concurrency"])))
    span = min(span, max(1, T - int(r.max()) - axes["wait_cap"] - 1))
    a = np.sort(rng.integers(0, span, n))
    wait_cap = axes["wait_cap"]
    dl = np.minimum(a + r + wait_cap, T)
    room = np.minimum(wait_cap, np.maximum(0, np.minimum(dl - r, T - r) - a))
    budget = int(round(axes["budget_fraction"] * float(room.sum())))

    kw = dict(green_w=green, static_w=static,
              brown_factor=list(BROWN_FACTORS), green_factor=list(GREEN_FACTORS),
              cap_pes=[CAP_PES_PER_SITE] * N_DC, arrival=a, runtime=r, pes=pes,
              deadline=dl, dyn_w_per_pe=DYN_W_PER_PE,
              per_job_wait_max=wait_cap, budget_total=budget)

    gres = np.maximum(green - static.reshape(-1, 1), 0.0)
    demand = axes["concurrency"] * axes["pes_per_job"] * DYN_W_PER_PE
    rho = demand / max(float(gres.mean()), 1e-9)
    prov = {"rho_residual": float(rho), "demand_w": float(demand),
            "pes_share": float(axes["pes_per_job"]) / CAP_PES_PER_SITE,
            "is_fluid_control": axes["pes_per_job"] in FLUID_CONTROL_PES,
            "mean_residual_green_w": float(gres.mean()), "budget_rows": budget,
            "arrival_span": int(span), "mean_runtime_rows": float(r.mean()),
            "clim_residual_green": _climatology(sites, axes["offset"], div, static, year)}
    return Scenario(**kw), prov


def _climatology(sites, offset, divisor, static, year):
    """Per-site RESIDUAL green from the history strictly before the window.

    The static draw is subtracted here and nowhere else. Consumers receive residual green
    and must not subtract it a second time.

    Every turbine of the site is summed, with the same divisor and the same static
    subtraction the live signal uses. Reading only the first turbine of a two-turbine site
    would understate the level by roughly half.
    """
    out = []
    for d, ts in enumerate(sites):
        if offset <= 0:
            out.append(0.0)
            continue
        acc = None
        for t in ts:
            v = _series(t, year)[:offset]
            acc = v if acc is None else acc + v
        out.append(float(max(acc.mean() * 1000.0 / divisor - static[d], 0.0)))
    return out


def axes_grid():
    """Every frozen combination, in a fixed order."""
    out = []
    for pes in PES_PER_JOB:
        for c in CONCURRENCY:
            for tps in TURBINES_PER_SITE:
                for div in INSTALLED_DIVISOR:
                    for n in N_JOBS:
                        for T in HORIZON:
                            for wc in WAIT_CAP_ROWS:
                                for bf in BUDGET_FRACTION:
                                    out.append(dict(pes_per_job=pes, concurrency=c,
                                                    turbines_per_site=tps,
                                                    installed_divisor=div, n_jobs=n,
                                                    horizon=T, wait_cap=wc,
                                                    budget_fraction=bf))
    return out


def grid_hash():
    payload = repr((PES_PER_JOB, CONCURRENCY, TURBINES_PER_SITE, INSTALLED_DIVISOR,
                    N_JOBS, HORIZON, BUDGET_FRACTION, WAIT_CAP_ROWS, SEEDS, N_DC,
                    HOSTS_PER_SITE, VMS_PER_SITE, PES_PER_VM, CAP_PES_PER_SITE,
                    RUNTIME_ROWS_TIER1, RUNTIME_ROWS_TIER2, FLUID_CONTROL_PES,
                    MIN_PES_SHARE, BROWN_FACTORS, GREEN_FACTORS, DYN_W_PER_PE,
                    HOST_IDLE_W, EPOCH_SECONDS))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
