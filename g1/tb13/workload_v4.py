"""TB13-v4 workloads: v3's construction with the v4 job sizes and site capacity.

The axis rules, the two separated random streams, the partitioned arrivals, the runtime
halves, the content hash and the frozen retry sequence are v3's and are imported rather
than restated. Only two things differ, and both come from the registered physical map: a
job is 8, 16 or 32 PE instead of 2, 4 or 8, and a site holds 64 PE instead of 16.

Compatibility of an axis combination does not depend on either, so the 89 combinations and
the 267 workload keys carry over unchanged.
"""
from __future__ import annotations

import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants_v4 as c4                        # noqa: E402
import workload_v3 as w3                         # noqa: E402

HORIZONS = w3.HORIZONS
N_JOBS = w3.N_JOBS
CONCURRENCY = w3.CONCURRENCY
WAIT_CAPS = w3.WAIT_CAPS
RUNTIME_HALVES = w3.RUNTIME_HALVES
MAX_RETRIES = w3.MAX_RETRIES
STRICTEST_BUDGET_FRACTION = w3.STRICTEST_BUDGET_FRACTION

PES_PER_JOB = c4.PES_PER_JOB
CAP_PES_PER_SITE = c4.CAP_PES_PER_SITE

service_span = w3.service_span
compatible = w3.compatible
compatible_axes = w3.compatible_axes
workload_key = w3.workload_key
draw = w3.draw
content_hash = w3.content_hash
budget_for = w3.budget_for
assertions = w3.assertions
domain_seed = w3.domain_seed


@functools.lru_cache(maxsize=None)
def _accepted_cached(payload):
    return _accept(json.loads(payload))


def accepted(key):
    return _accepted_cached(json.dumps(key, sort_keys=True, separators=(",", ":")))


def _accept(key):
    """The v3 acceptance contract, evaluated against a 64-PE site."""
    from schedule_feasibility import capacity_ok, reservation_edf
    for k in range(MAX_RETRIES):
        w = draw(key, k)
        _checks, ok = assertions(w, key)
        if not ok:
            continue
        b = budget_for(w, STRICTEST_BUDGET_FRACTION)
        if capacity_ok(w, b, cap=CAP_PES_PER_SITE) != "FEASIBLE":
            continue                       # UNKNOWN is not evidence of infeasibility
        if reservation_edf(w, b, cap=CAP_PES_PER_SITE)[0] is None:
            continue
        return {"key": key, "retry": k, "workload": w, "content_hash": content_hash(w)}
    return None
