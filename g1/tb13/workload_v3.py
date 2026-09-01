"""TB13-v3 workload construction. Arrivals are never squeezed, two separated streams.

Codex 2026-09-01. In v2 the arrival span was squeezed to fit a target concurrency, a full
slack and a short horizon at once, so 84 of 99 keys had their arrivals squeezed onto a
handful of rows and the realised concurrency reached 120 against a registered 1-5. v3 removes the squeeze by
admitting only axis combinations that are compatible in the first place:

    runtime      exactly half the jobs at 6 rows and half at 12, order permuted
                 so sum(runtime) = 9 n and max(runtime) = 12 by construction
    S            = ceil(9 n / concurrency)
    compatible   horizon >= S + 12 + wait_cap
    arrival      [0, S) split into n consecutive intervals, one draw per interval
    deadline     arrival + runtime + wait_cap

The arrival and the runtime permutation draw from domain-separated seeds, so changing the
order in which the code calls them cannot change the load.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HORIZONS = (72, 96, 144)
N_JOBS = (8, 10, 12)
CONCURRENCY = (1, 2, 3, 5)
WAIT_CAPS = (6, 12, 24)
PES_PER_JOB = (2, 4, 8)
RUNTIME_HALVES = (6, 12)
MAX_RUNTIME = max(RUNTIME_HALVES)
MEAN_RUNTIME = sum(RUNTIME_HALVES) / 2.0          # 9 rows
MAX_RETRIES = 64
STRICTEST_BUDGET_FRACTION = 0.10


def service_span(n_jobs, concurrency):
    """Rows needed to offer the target concurrency: S = ceil(sum(runtime) / c)."""
    return math.ceil(MEAN_RUNTIME * n_jobs / concurrency)


def compatible(horizon, n_jobs, concurrency, wait_cap):
    return horizon >= service_span(n_jobs, concurrency) + MAX_RUNTIME + wait_cap


def compatible_axes():
    """Every (horizon, n_jobs, concurrency, wait_cap) that can hold its own contract."""
    return [(h, n, c, w) for h in HORIZONS for n in N_JOBS
            for c in CONCURRENCY for w in WAIT_CAPS if compatible(h, n, c, w)]


def workload_key(seed, horizon, pes_per_job, concurrency, n_jobs, wait_cap):
    return {"seed": int(seed), "horizon": int(horizon),
            "pes_per_job": int(pes_per_job), "concurrency": int(concurrency),
            "n_jobs": int(n_jobs), "wait_cap": int(wait_cap),
            "runtime_set": list(RUNTIME_HALVES)}


def _payload(key):
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def domain_seed(key, domain, k):
    digest = hashlib.sha256(f"{_payload(key)}:{domain}:{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def draw(key, k):
    """One candidate load, a pure function of the key, the domain and the retry index."""
    n, S = key["n_jobs"], service_span(key["n_jobs"], key["concurrency"])
    if n % 2:
        raise ValueError("n_jobs must be even so the runtime halves are exact")

    runtimes = np.array([RUNTIME_HALVES[0]] * (n // 2) + [RUNTIME_HALVES[1]] * (n // 2))
    rng_r = np.random.default_rng(domain_seed(key, "runtime", k))
    runtimes = runtimes[rng_r.permutation(n)]

    rng_a = np.random.default_rng(domain_seed(key, "arrival", k))
    arrivals = np.empty(n, dtype=int)
    for i in range(n):
        lo = (i * S) // n
        hi = ((i + 1) * S) // n
        arrivals[i] = lo if hi <= lo else int(rng_a.integers(lo, hi))
    arrivals.sort()

    deadline = arrivals + runtimes + key["wait_cap"]
    return {"arrival": arrivals, "runtime": runtimes,
            "pes": np.full(n, key["pes_per_job"], dtype=int),
            "deadline": deadline.astype(int), "horizon": key["horizon"],
            "wait_cap": key["wait_cap"], "service_span": S}


def content_hash(w):
    payload = {"arrival": [int(x) for x in w["arrival"]],
               "runtime": [int(x) for x in w["runtime"]],
               "pes": [int(x) for x in w["pes"]],
               "deadline": [int(x) for x in w["deadline"]],
               "horizon": int(w["horizon"])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def budget_for(w, fraction):
    a, r, dl, T = w["arrival"], w["runtime"], w["deadline"], w["horizon"]
    room = np.minimum(w["wait_cap"], np.maximum(0, np.minimum(dl - r, T - r) - a))
    return int(round(fraction * float(room.sum())))


def assertions(w, key):
    """The three per-cell checks the registration demands."""
    a, r = w["arrival"], w["runtime"]
    out = {
        "arrival_span_gt_1": int(a.max() - a.min() + 1) > 1,
        "deadline_within_horizon": bool(np.all(w["deadline"] <= w["horizon"])),
        "service_span_matches": w["service_span"] == math.ceil(
            float(r.sum()) / key["concurrency"]),
    }
    return out, all(out.values())


@functools.lru_cache(maxsize=None)
def _accepted_cached(payload):
    return _accept(json.loads(payload))


def accepted(key):
    return _accepted_cached(_payload(key))


def _accept(key):
    from schedule_feasibility import capacity_ok, reservation_edf
    for k in range(MAX_RETRIES):
        w = draw(key, k)
        _checks, ok = assertions(w, key)
        if not ok:
            continue
        b = budget_for(w, STRICTEST_BUDGET_FRACTION)
        status = capacity_ok(w, b)
        if status != "FEASIBLE":
            continue                       # UNKNOWN is not evidence of infeasibility
        if reservation_edf(w, b)[0] is None:
            continue
        return {"key": key, "retry": k, "workload": w, "content_hash": content_hash(w)}
    return None
