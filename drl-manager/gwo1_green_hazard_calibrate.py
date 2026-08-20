#!/usr/bin/env python3
"""gwo1 green-window hazard freeze (5080, 2026-08-20).

The trough-side threshold q*=0.5 was calibrated on a decision set that
explicitly skips green arrivals (`sqt2_hazard_calibrate.decision_set`:
`if not in_trough: continue`). The gwo1 ladder reuses the same q for a rule
with a DIFFERENT question:

    trough side : "will this trough END inside my budget?"      P(end | age, B)
    green  side : "will the green run out BEFORE my job finishes?"
                                                                P(rem < runtime | age)

Sharing one threshold across two different posteriors is an unargued
assumption, so the green side gets its own pre-registered freeze on the same
candidate grid. Offline accuracy only - the carbon/SLA freeze stays with
sqt2_blind_freeze.py, exactly as on the trough side.

Label (the clairvoyant answer the blind arm is trying to guess):
    worthy = rem_green < runtime  AND  rem_green + trough_dur <= budget
i.e. the job does NOT fit in the remaining green, and the next onset is
reachable inside the budget.
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gen_sqt2 import ON_LO, ON_HI                                  # noqa: E402
from oracle_slack_planner import WARMUP_ROWS                       # noqa: E402
from teacher_reward_audit import episode_offset, effective_budget  # noqa: E402
from sqt2_prescreen import ANCHORS, MARGIN_S, MIPS, RELEASE_EPS_S, HORIZON_S  # noqa: E402

CANDIDATES = (0.25, 0.40, 0.50, 0.60)
REPO = pathlib.Path(__file__).resolve().parent


def green_p_ends_within(age: float, need: float,
                        lo: float = ON_LO, hi: float = ON_HI) -> float:
    """P(remaining ON < `need` | the ON window has already lasted `age`).

    ON ~ U[lo, hi]. Conditioned on age, the remaining time is uniform on
    [lo-age, hi-age] while age < lo, and on [0, hi-age] once age >= lo.
    """
    if need <= 0:
        return 0.0
    if age >= hi:
        return 1.0
    if age < lo:
        return float(np.clip((need - (lo - age)) / (hi - lo), 0.0, 1.0))
    return float(np.clip(need / (hi - age), 0.0, 1.0))


def green_intervals(troughs):
    """ON spans are the complement of the trough spans: (start, end, next_dur).

    The LEADING span (row 0 .. first trough) is a legitimate decision point -
    it has a following trough, so "is the next onset reachable?" is defined.
    Dropping it (the 2026-08-20 first version did, via zip(iv, iv[1:])) shrank
    the decision set by ~7% and made the two machines disagree on the
    denominator while agreeing on every ratio - exactly the signature the
    3060 reported. The TRAILING span has no following trough, so a job there
    can never reach a next onset; it is excluded on purpose, not by accident.
    """
    iv = sorted((t["start"], t["start"] + t["dur"], t["dur"]) for t in troughs)
    spans = []
    if iv and iv[0][0] > 0:
        spans.append((0.0, float(iv[0][0]), iv[0][2]))     # leading green span
    for (s0, e0, _), (s1, _, d1) in zip(iv, iv[1:]):
        if s1 > e0:
            spans.append((float(e0), float(s1), d1))
    return spans


def decision_set(rows, spans, off_range, anchors):
    """(worthy, green_age, runtime, budget, mi) for green arrivals with budget."""
    out = []
    for k in anchors:
        off = episode_offset(k, off_range)
        for r in rows:
            a, mi = float(r["arrival_time"]), float(r["length"])
            pes = max(1, int(r["pes_required"]))
            if mi <= 0:
                continue
            runtime = mi / (pes * MIPS)
            budget = effective_budget(float(r["deadline"]) - a, runtime,
                                      MARGIN_S, HORIZON_S - a)
            if budget <= RELEASE_EPS_S:
                continue
            row = int(WARMUP_ROWS + off + a)
            span = next(((s, e, d) for s, e, d in spans if s <= row < e), None)
            if span is None:
                continue                      # arrived in a trough: other rule
            s, e, dur = span
            rem, age = e - row, row - s
            worthy = (rem < runtime) and (rem + dur <= budget)
            out.append((worthy, float(age), runtime, budget, mi))
    return out


def accuracies(dset):
    worthy = np.array([d[0] for d in dset], dtype=bool)
    mi = np.array([d[4] for d in dset], dtype=float)
    p = np.array([green_p_ends_within(d[1], d[2]) for d in dset])
    res = {"always_wait": {"acc": float(worthy.mean()),
                           "acc_mi": float(mi[worthy].sum() / mi.sum())},
           "never_wait": {"acc": float((~worthy).mean()),
                          "acc_mi": float(mi[~worthy].sum() / mi.sum())}}
    for q in CANDIDATES:
        ok = (p >= q) == worthy
        res[f"green_hazard@{q:.2f}"] = {
            "acc": float(ok.mean()),
            "acc_mi": float(mi[ok].sum() / mi.sum())}
    return res


def freeze(res):
    q = max(CANDIDATES, key=lambda x: res[f"green_hazard@{x:.2f}"]["acc"])
    return q, res[f"green_hazard@{q:.2f}"]["acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="calib/gwo1_schedule.json")
    ap.add_argument("--trace", default="gwo1_n1200_x130.csv")
    ap.add_argument("--offset-range", type=int, default=180000)
    ap.add_argument("--out", default="calib/gwo1_green_hazard_freeze.json")
    a_ = ap.parse_args()
    art = json.loads((REPO / a_.schedule).read_text())
    spans = green_intervals(art["troughs"])
    rows = list(csv.DictReader(
        open(REPO.parent / "cloudsimplus-gateway/src/main/resources/traces" / a_.trace)))
    dset = decision_set(rows, spans, a_.offset_range, ANCHORS)
    res = accuracies(dset)
    q, acc = freeze(res)
    for k, v in res.items():
        print(f"  {k:20s} acc={v['acc']:.4f}  acc_mi={v['acc_mi']:.4f}")
    print(f"FROZEN green q*={q} (acc {acc:.4f}) over n={len(dset)} green-arrival decisions")
    (REPO / a_.out).write_text(json.dumps(
        {"protocol": "gwo1 green-window hazard freeze (calibration data only)",
         "candidates": list(CANDIDATES), "q_star_green": q,
         "n_decisions": len(dset), "accuracies": res,
         "schedule": a_.schedule, "trace": a_.trace,
         "on_law": [ON_LO, ON_HI], "anchors": list(ANCHORS)}, indent=1))
    print(f"-> {a_.out}")


if __name__ == "__main__":
    main()
