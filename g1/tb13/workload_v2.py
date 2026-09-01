"""Round1-v2 workload generation: content decided by the key alone, then reused.

Codex 2026-09-01. A workload is a function of

    seed, horizon, pes_per_job, concurrency, n_jobs, wait_cap, runtime_set

and of nothing else. The wind trace, the turbine triple, the installed divisor, the
season offset and the budget fraction are all excluded from the random stream. In v1 the
season offset was mixed into the seed, so the same key resampled per season and 1,296
cells carried 272 distinct loads instead of the 99 the design intends.

Acceptance is schedule-only: whether a load can be run at all, under capacity, deadlines,
the per-job wait cap and the tightest delay budget. It never reads the weather or the emissions ledger, so the
set of accepted workloads cannot depend on the weather.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402

MAX_RETRIES = 64
STRICTEST_BUDGET_FRACTION = 0.10
CPSAT_RANDOM_SEED = 20260901
CPSAT_MAX_DETERMINISTIC_TIME = 30.0      # deterministic units, never wall clock


def workload_key(seed, horizon, pes_per_job, concurrency, n_jobs, wait_cap,
                 runtime_set=None):
    """The frozen key. runtime_set is a JSON list so the bytes never depend on the type."""
    return {"seed": int(seed), "horizon": int(horizon),
            "pes_per_job": int(pes_per_job), "concurrency": int(concurrency),
            "n_jobs": int(n_jobs), "wait_cap": int(wait_cap),
            "runtime_set": list(runtime_set or ig.RUNTIME_ROWS_TIER1)}


def frozen_seed(key, k):
    payload = json.dumps(key, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((payload + ":" + str(k)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def draw(key, k):
    """One candidate load. Pure function of the key and the retry index."""
    rng = np.random.default_rng(frozen_seed(key, k))
    n, T, wc = key["n_jobs"], key["horizon"], key["wait_cap"]
    r = rng.choice(key["runtime_set"], size=n)
    span = max(1, int(round(n * float(r.mean()) / key["concurrency"])))
    span = min(span, max(1, T - int(r.max()) - wc - 1))
    a = np.sort(rng.integers(0, span, n))
    pes = np.full(n, key["pes_per_job"], dtype=int)
    dl = np.minimum(a + r + wc, T)
    return {"arrival": a.astype(int), "runtime": r.astype(int), "pes": pes,
            "deadline": dl.astype(int), "horizon": T, "wait_cap": wc}


def content_hash(w):
    """Hash of what the jobs actually are. Two cells reusing a load must match here."""
    payload = {"arrival": [int(x) for x in w["arrival"]],
               "runtime": [int(x) for x in w["runtime"]],
               "pes": [int(x) for x in w["pes"]],
               "deadline": [int(x) for x in w["deadline"]],
               "horizon": int(w["horizon"])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def budget_for(w, budget_fraction):
    """Delay budget as the registered fraction of the room the contract actually allows."""
    a, r, dl, T = w["arrival"], w["runtime"], w["deadline"], w["horizon"]
    room = np.minimum(w["wait_cap"], np.maximum(0, np.minimum(dl - r, T - r) - a))
    return int(round(budget_fraction * float(room.sum())))


@functools.lru_cache(maxsize=None)
def _accepted_cached(payload):
    return _accept(json.loads(payload))


def accepted(key):
    """The first candidate that passes acceptance, or None when the retries run out."""
    return _accepted_cached(json.dumps(key, sort_keys=True, separators=(",", ":")))


def _accept(key):
    from schedule_feasibility import capacity_ok, reservation_edf
    for k in range(MAX_RETRIES):
        w = draw(key, k)
        b = budget_for(w, STRICTEST_BUDGET_FRACTION)
        status = capacity_ok(w, b)
        if status == "UNKNOWN":
            continue                     # not evidence of infeasibility; try the next seed
        if status != "FEASIBLE":
            continue
        if reservation_edf(w, b)[0] is None:
            continue
        return {"key": key, "retry": k, "workload": w,
                "content_hash": content_hash(w)}
    return None
