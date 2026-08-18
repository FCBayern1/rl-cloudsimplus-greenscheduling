#!/usr/bin/env python3
"""SQT2.2-Clean hazard freeze (Codex layer 2, 2026-08-18).

Selects the hazard posterior threshold q* and the frozen blind comparator on
the CALIBRATION schedule/trace ONLY, before the held-out verdict run ever
starts. Candidates are pre-registered: q in {0.25, 0.40, 0.50, 0.60}. The
formal prescreen may use exactly one hazard policy (q*) and exactly one
"strongest blind" comparator - never a per-window or post-hoc pick.

Decision set (pre-registered): every (episode offset k=0..178, job) pair
whose arrival row falls inside a trough, with mi>0 and executable budget
B_eff = min(deadline-arrival-runtime-120, 7200-arrival-runtime-120) > 0.
Label: worthy = (true residual trough time <= B_eff). Predictors:
    naive      defer always (inside the decision set)
    hazard@q   defer iff P(trough ends within B_eff | age) >= q
Freeze: q* = argmax unweighted accuracy; comparator = the better of
naive / hazard@q* by the same unweighted accuracy. MI-weighted accuracy is
reported for information only and plays no role in the freeze.
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_slack_planner import WARMUP_ROWS  # noqa: E402
from teacher_reward_audit import episode_offset, effective_budget  # noqa: E402
from sqt2_prescreen import hazard_p_end_within, TroughIndex, MIPS, MARGIN_S  # noqa: E402

CANDIDATES = (0.25, 0.40, 0.50, 0.60)
HORIZON_S = 7200.0
N_EPISODES = 179
REPO = pathlib.Path(__file__).resolve().parent
TRACE = REPO.parent / "cloudsimplus-gateway/src/main/resources/traces/sqt2_n1200_t60.csv"
SCHEDULE = REPO / "calib/sqt2_schedule.json"
OUT = REPO / "calib/sqt2_hazard_freeze.json"


def decision_set(rows, tindex, off_range):
    """Yield (worthy, age, budget, mi) for every in-trough discretionary job."""
    out = []
    for k in range(N_EPISODES):
        off = episode_offset(k, off_range)
        for r in rows:
            arrival = float(r["arrival_time"])
            mi = float(r["length"])
            pes = max(1, int(r["pes_required"]))
            if mi <= 0:
                continue
            runtime = mi / (pes * MIPS)
            ttd = float(r["deadline"]) - arrival
            budget = effective_budget(ttd, runtime, MARGIN_S, HORIZON_S - arrival)
            if budget <= 0:
                continue
            in_trough, age, residual = tindex.query(int(WARMUP_ROWS + off + arrival))
            if not in_trough:
                continue
            out.append((residual <= budget, age, budget, mi))
    return out


def accuracies(dset):
    """Unweighted + MI-weighted accuracy for naive and every hazard candidate."""
    worthy = np.array([d[0] for d in dset], dtype=bool)
    mi = np.array([d[3] for d in dset], dtype=float)
    p = np.array([hazard_p_end_within(d[1], d[2]) for d in dset], dtype=float)
    res = {"naive": {"acc": float(np.mean(worthy)),
                     "acc_mi": float(np.sum(mi[worthy]) / np.sum(mi))}}
    for q in CANDIDATES:
        ok = (p >= q) == worthy
        res[f"hazard@{q:.2f}"] = {"acc": float(np.mean(ok)),
                                  "acc_mi": float(np.sum(mi[ok]) / np.sum(mi))}
    return res


def freeze(res):
    """q* by unweighted accuracy; comparator = better of naive / hazard@q*."""
    q_star = max(CANDIDATES, key=lambda q: res[f"hazard@{q:.2f}"]["acc"])
    hz, nv = res[f"hazard@{q_star:.2f}"]["acc"], res["naive"]["acc"]
    return q_star, ("hazard" if hz >= nv else "naive")


def main():
    art = json.loads(SCHEDULE.read_text())
    assert art.get("variant", "cal") == "cal", "freeze runs on calibration data only"
    tindex = TroughIndex(art["troughs"])
    rows = list(csv.DictReader(open(TRACE)))
    off_range = 180000
    dset = decision_set(rows, tindex, off_range)
    res = accuracies(dset)
    q_star, comparator = freeze(res)
    out = {"protocol": "SQT2.2-Clean layer 2 (frozen before held-out run)",
           "candidates": list(CANDIDATES), "q_star": q_star,
           "comparator": comparator, "n_decisions": len(dset),
           "accuracies": res, "schedule_seed": art["seed"],
           "trace": TRACE.name, "episodes": N_EPISODES,
           "offset_range": off_range}
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in res.items():
        print(f"{k:13s} acc={v['acc']:.4f} acc_mi={v['acc_mi']:.4f}")
    print(f"FROZEN: q*={q_star} comparator={comparator} "
          f"(n={len(dset)}) -> {OUT.name}")


if __name__ == "__main__":
    main()
