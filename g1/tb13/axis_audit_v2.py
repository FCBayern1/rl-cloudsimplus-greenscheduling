"""Zero-emissions audit of what the v2 axes actually produced. No wind, no ledger.

Codex 2026-09-01: three keys exhausting their retries is a symptom. The question is how
often the generator had to clip the arrival span to fit a target concurrency, a full slack
and a short horizon at once, because a clipped span turns a load into a single-epoch burst
whether or not it later passed the reservation gate.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402
import preflight_v2 as pf  # noqa: E402
import round1 as r1  # noqa: E402
import schedule_feasibility as sf  # noqa: E402
import workload_v2 as wv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def audit_key(key):
    rows = []
    for k in range(wv.MAX_RETRIES):
        rng = np.random.default_rng(wv.frozen_seed(key, k))
        n, T, wc = key["n_jobs"], key["horizon"], key["wait_cap"]
        r = rng.choice(key["runtime_set"], size=n)
        target_span = max(1, int(round(n * float(r.mean()) / key["concurrency"])))
        capacity_span = T - int(r.max()) - wc
        allowed = max(1, capacity_span - 1)
        span = min(target_span, allowed)
        w = wv.draw(key, k)
        a, rr, dl = w["arrival"], w["runtime"], w["deadline"]
        offered = float(sum(rr) / max(1, (a.max() - a.min() + 1)))
        slack = dl - a - rr
        rows.append({
            "retry": k, "target_span": target_span, "capacity_span": int(capacity_span),
            "allowed_span": int(allowed), "actual_span": int(span),
            "clipped": bool(target_span > allowed),
            "arrival_spread": int(a.max() - a.min() + 1),
            "offered_concurrency": round(offered, 3),
            "slack_min": int(slack.min()), "slack_med": float(np.median(slack)),
            "slack_max": int(slack.max()),
            "sum_runtime": int(rr.sum()), "max_runtime": int(rr.max()),
        })
        if k == 0:
            break                     # retry 0 characterises the key's geometry
    return rows[0]


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "axis_audit_v2_out")
    os.makedirs(out_dir, exist_ok=True)
    cells = pf.cell_plan(os.path.join(HERE, "round0_out"))
    keys = {json.dumps(c["key"], sort_keys=True, separators=(",", ":")): c["key"]
            for c in cells}
    rejected = set(json.load(open(os.path.join(
        HERE, "preflight_v2_out/preflight_v2_summary.json")))["rejected_keys"])
    rows = []
    for kj, key in keys.items():
        a = audit_key(key)
        a["key"] = key
        a["accepted"] = kj not in rejected
        rows.append(a)
    clipped = [r for r in rows if r["clipped"]]
    summary = {
        "keys": len(rows), "clipped": len(clipped),
        "clipped_and_accepted": sum(1 for r in clipped if r["accepted"]),
        "clipped_and_rejected": sum(1 for r in clipped if not r["accepted"]),
        "unclipped_and_rejected": sum(1 for r in rows
                                      if not r["clipped"] and not r["accepted"]),
        "capacity_span_le_zero": sum(1 for r in rows if r["capacity_span"] <= 0),
        "arrival_spread_is_one": sum(1 for r in rows if r["arrival_spread"] == 1),
        "offered_concurrency": {
            "min": min(r["offered_concurrency"] for r in rows),
            "median": float(np.median([r["offered_concurrency"] for r in rows])),
            "max": max(r["offered_concurrency"] for r in rows)},
        "by_horizon": {str(h): sum(1 for r in rows
                                   if r["key"]["horizon"] == h and r["clipped"])
                       for h in ig.HORIZON},
        "by_wait_cap": {str(w): sum(1 for r in rows
                                    if r["key"]["wait_cap"] == w and r["clipped"])
                        for w in ig.WAIT_CAP_ROWS},
    }
    r1._write(os.path.join(out_dir, "axis_audit_rows.jsonl"), rows, lines=True)
    r1._write(os.path.join(out_dir, "axis_audit_summary.json"), summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True, indent=2))
