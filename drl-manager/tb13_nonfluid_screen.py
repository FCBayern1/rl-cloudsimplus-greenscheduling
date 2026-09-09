#!/usr/bin/env python3
"""TB13 stage-1b: indivisible, capacity-constrained multi-DC screen.

This is deliberately *not* a simulator certification.  It closes the largest
loophole in ``tb13_search.py``: that script compares ``max_d G_d`` with
``sum_d G_d`` at the same instant, so its number is a spatial pooling/fluidity
gap, not a forecast-information gap.

Here every job is committed to exactly one DC and one contiguous start time.
The clairvoyant and every blind arm see the same jobs, capacity, deadlines and
already committed demand.  Clairvoyant may inspect future wind; blind arms may
inspect only current/past wind.  A blind threshold is selected on the
calibration year and then frozen for the evaluation year.

The default turbine triple (8,9,29) was selected after inspecting 2021 and is
therefore DIAGNOSTIC ONLY.  A formal TB13 candidate must be selected on 2020
and evaluated on untouched 2021 data.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import pathlib
from typing import Iterable, Sequence

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent.parent
WIND = ROOT / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
TB12_ARTIFACT = pathlib.Path(__file__).resolve().parent / "calib/tb12_v2.json"
ROW_S = 600.0
C_BROWN, C_GREEN = 0.55, 0.01
P_JOB_W = 28.75
EP_ROWS = 288
BLIND_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, float("inf"))


@dataclasses.dataclass(frozen=True)
class Job:
    arrival: int
    latest: int
    duration: int
    power_w: float
    job_id: int


def _as_groups(turbines: Sequence[int | Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple((int(x),) if isinstance(x, (int, np.integer))
                 else tuple(int(y) for y in x) for x in turbines)


def load_wind(turbines: Sequence[int | Sequence[int]], year: int) -> np.ndarray:
    """One output row per DC; a DC may aggregate multiple real turbines."""
    rows = []
    for group in _as_groups(turbines):
        members = []
        for tid in group:
            path = WIND / f"Turbine_{tid}_{year}.csv"
            with path.open(newline="") as fh:
                members.append(np.asarray(
                    [float(r["power_kw"] or 0.0) * 1000.0
                     for r in csv.DictReader(fh)], dtype=float))
        n_group = min(map(len, members))
        rows.append(sum(x[:n_group] for x in members))
    n = min(map(len, rows))
    return np.stack([x[:n] for x in rows])


def tb12_jobs() -> list[Job]:
    art = json.loads(TB12_ARTIFACT.read_text())
    duration = int(round(float(art["rt_h"]) * 3600.0 / ROW_S))
    slack = int(round(float(art["slack_h"]) * 3600.0 / ROW_S))
    out = []
    for raw in art["jobs"]:
        jid, arrival_s = int(raw[0]), float(raw[1])
        arrival = int(np.ceil(arrival_s / ROW_S))
        out.append(Job(arrival, arrival + slack, duration, P_JOB_W, jid))
    return out


def split_jobs(jobs: Sequence[Job], factor: int) -> list[Job]:
    """Approach the fluid limit in both power and time, preserving work.

    Each job becomes ``factor**2`` pieces, each with 1/factor power and
    1/factor duration.  Splitting power alone leaves a long non-preemptive
    commitment and is not the fluid negative control described by the TB12
    relaxation theorem.
    """
    if factor < 1:
        raise ValueError("split factor must be positive")
    if any(j.duration % factor for j in jobs):
        raise ValueError("split factor must divide every job duration")
    out = []
    for job in jobs:
        for part in range(factor * factor):
            out.append(Job(job.arrival, job.latest, job.duration // factor,
                           job.power_w / factor,
                           job.job_id * factor * factor + part))
    return out


def seasonal_offsets(n_rows: int, count: int = 10) -> list[int]:
    if n_rows < EP_ROWS:
        raise ValueError("wind series shorter than one episode")
    return [int(x) for x in np.linspace(0, n_rows - EP_ROWS, count)]


def split_capacity(total_slots: int, n_dc: int, p_job: float = P_JOB_W) -> np.ndarray:
    """Keep aggregate service capacity fixed while changing DC count."""
    if n_dc < 1 or total_slots < n_dc:
        raise ValueError("need at least one aggregate slot per DC")
    slots = np.full(n_dc, total_slots // n_dc, dtype=int)
    slots[: total_slots % n_dc] += 1
    return slots.astype(float) * p_job


def calibrate_scale(raw: np.ndarray, offsets: Sequence[int], jobs: Sequence[Job],
                    kappa: float) -> float:
    """Freeze an integrated green/job-energy scale on calibration episodes."""
    green = sum(float(raw[:, off:off + EP_ROWS].sum()) for off in offsets)
    work = len(offsets) * sum(j.power_w * j.duration for j in jobs)
    if green <= 0 or work <= 0:
        raise ValueError("non-positive calibration energy")
    return float(kappa * work / green)


def _feasible(demand: np.ndarray, capacity: np.ndarray, dc: int, start: int,
              job: Job) -> bool:
    stop = start + job.duration
    return (stop <= demand.shape[1]
            and np.all(demand[dc, start:stop] + job.power_w
                       <= capacity[dc] + 1e-12))


def _incremental_brown(demand: np.ndarray, green: np.ndarray, dc: int,
                       start: int, job: Job) -> float:
    stop = start + job.duration
    old = demand[dc, start:stop]
    supply = green[dc, start:stop]
    return float((np.maximum(0.0, old + job.power_w - supply)
                  - np.maximum(0.0, old - supply)).sum())


def _commit(demand: np.ndarray, dc: int, start: int, job: Job) -> None:
    demand[dc, start:start + job.duration] += job.power_w


def schedule_clairvoyant(jobs: Sequence[Job], green: np.ndarray,
                         capacity: np.ndarray) -> list[tuple[int, int]]:
    """Online-arrival clairvoyant: future wind yes, future jobs no.

    Jobs are processed in arrival order and commitments are never rearranged.
    Thus a later job cannot change an earlier commitment.
    """
    demand = np.zeros_like(green, dtype=float)
    out = []
    for job in sorted(jobs, key=lambda j: (j.arrival, j.job_id)):
        best = None
        last = min(job.latest, green.shape[1] - job.duration)
        for start in range(job.arrival, last + 1):
            for dc in range(green.shape[0]):
                if not _feasible(demand, capacity, dc, start, job):
                    continue
                key = (_incremental_brown(demand, green, dc, start, job),
                       start, dc)
                if best is None or key < best:
                    best = key
        if best is None:
            raise RuntimeError(f"no feasible clairvoyant placement for job {job.job_id}")
        _, start, dc = best
        _commit(demand, dc, start, job)
        out.append((dc, start))
    return out


def schedule_blind(jobs: Sequence[Job], green: np.ndarray, capacity: np.ndarray,
                   threshold: float) -> list[tuple[int, int]]:
    """Joint causal threshold blind: decide both when and where.

    At row t it reads only current green and already committed demand.  It
    releases when the best feasible DC's *current* residual-green ratio reaches
    the frozen threshold; latest-start is the common backstop.  The family
    includes nowait (0) and always-wait (inf).
    """
    demand = np.zeros_like(green, dtype=float)
    ordered = sorted(jobs, key=lambda j: (j.arrival, j.job_id))
    pending: list[Job] = []
    placements: dict[int, tuple[int, int]] = {}
    incoming = 0
    for now in range(green.shape[1]):
        while incoming < len(ordered) and ordered[incoming].arrival <= now:
            pending.append(ordered[incoming])
            incoming += 1
        # All decisions at ``now`` use only G[:now+1].  EDF is deterministic;
        # a release updates committed demand before the next pending job.
        keep = []
        for job in sorted(pending, key=lambda j: (j.latest, j.job_id)):
            feasible = [dc for dc in range(green.shape[0])
                        if _feasible(demand, capacity, dc, now, job)]
            if not feasible:
                keep.append(job)
                continue
            ranked = sorted(
                feasible,
                key=lambda dc: (-(green[dc, now] - demand[dc, now]), dc),
            )
            dc = ranked[0]
            ratio = (green[dc, now] - demand[dc, now]) / job.power_w
            forced = now >= min(job.latest, green.shape[1] - job.duration)
            if ratio >= threshold or forced:
                placements[job.job_id] = (dc, now)
                _commit(demand, dc, now, job)
            else:
                keep.append(job)
        pending = keep
        if incoming == len(ordered) and not pending:
            break
    if len(placements) != len(jobs):
        missing = sorted(j.job_id for j in jobs if j.job_id not in placements)
        raise RuntimeError(f"no feasible blind placement for jobs {missing}")
    return [placements[j.job_id] for j in ordered]


def score(jobs: Sequence[Job], placements: Sequence[tuple[int, int]],
          green: np.ndarray) -> dict[str, float]:
    demand = np.zeros_like(green, dtype=float)
    by_id = {j.job_id: j for j in jobs}
    ordered = sorted(jobs, key=lambda j: (j.arrival, j.job_id))
    for job, (dc, start) in zip(ordered, placements):
        _commit(demand, dc, start, by_id[job.job_id])
    used_green = np.minimum(demand, green).sum()
    brown = np.maximum(0.0, demand - green).sum()
    carbon = C_GREEN * used_green + C_BROWN * brown
    return {"carbon": float(carbon), "green": float(used_green),
            "brown": float(brown), "work": float(demand.sum())}


def _run_year(raw: np.ndarray, offsets: Sequence[int], scale: float,
              jobs: Sequence[Job], capacity: np.ndarray,
              threshold: float | None) -> list[dict[str, float]]:
    rows = []
    for off in offsets:
        green = raw[:, off:off + EP_ROWS] * scale
        if threshold is None:
            placements = schedule_clairvoyant(jobs, green, capacity)
        else:
            placements = schedule_blind(jobs, green, capacity, threshold)
        rows.append(score(jobs, placements, green))
    return rows


def run(turbines: Sequence[int | Sequence[int]], kappa: float = 0.8,
        cal_year: int = 2020, eval_year: int = 2021,
        n_jobs: int = 5, split_factor: int = 1,
        slack_rows: int | None = None) -> dict:
    all_jobs = tb12_jobs()
    if not 1 <= n_jobs <= len(all_jobs):
        raise ValueError(f"n_jobs must be in [1,{len(all_jobs)}]")
    if n_jobs < len(turbines):
        raise ValueError("n_jobs must be >= number of DCs so each DC retains capacity")
    selected = all_jobs[:n_jobs]
    if slack_rows is not None:
        selected = [Job(j.arrival, j.arrival + int(slack_rows), j.duration,
                        j.power_w, j.job_id) for j in selected]
    jobs = split_jobs(selected, split_factor)
    groups = _as_groups(turbines)
    raw_cal = load_wind(groups, cal_year)
    raw_eval = load_wind(groups, eval_year)
    cal_offsets = seasonal_offsets(raw_cal.shape[1])
    eval_offsets = seasonal_offsets(raw_eval.shape[1])
    scale = calibrate_scale(raw_cal, cal_offsets, jobs, kappa)
    # Five original job slots worth of aggregate service capacity at every
    # split factor: (base_jobs * factor) slots x (P/factor) watts.
    capacity = split_capacity(n_jobs * split_factor, len(groups),
                              P_JOB_W / split_factor)

    cal_by_q = {}
    for q in BLIND_THRESHOLDS:
        recs = _run_year(raw_cal, cal_offsets, scale, jobs, capacity, q)
        cal_by_q[q] = sum(x["carbon"] for x in recs)
    q_star = min(BLIND_THRESHOLDS, key=lambda q: (cal_by_q[q], q))

    eval_by_q = {q: _run_year(raw_eval, eval_offsets, scale, jobs, capacity, q)
                 for q in BLIND_THRESHOLDS}
    blind = eval_by_q[q_star]
    clair = _run_year(raw_eval, eval_offsets, scale, jobs, capacity, None)
    paired = [(c["carbon"] - b["carbon"]) / b["carbon"]
              for b, c in zip(blind, clair)]
    pooled = ((sum(x["carbon"] for x in clair)
               - sum(x["carbon"] for x in blind))
              / sum(x["carbon"] for x in blind))
    # Deliberately impossible stress comparator: after each evaluation episode,
    # choose the best threshold using that episode's outcome.  It is not a
    # deployable blind policy and never replaces q_star; surviving it shows the
    # result is not merely a calibration-year threshold mismatch.
    oracle_q_carbon = [min(eval_by_q[q][i]["carbon"]
                           for q in BLIND_THRESHOLDS)
                       for i in range(len(eval_offsets))]
    paired_oracle_q = [(c["carbon"] - b) / b
                       for b, c in zip(oracle_q_carbon, clair)]
    pooled_oracle_q = ((sum(x["carbon"] for x in clair)
                        - sum(oracle_q_carbon)) / sum(oracle_q_carbon))
    return {
        "status": "diagnostic_only" if groups == ((8,), (9,), (29,)) else "screen",
        "warning": ("T8/T9/T29 was selected after inspecting 2021; do not use "
                    "this run as held-out evidence") if groups == ((8,), (9,), (29,)) else None,
        "turbine_groups": [list(x) for x in groups], "base_jobs": n_jobs,
        "split_factor": split_factor, "n_jobs": len(jobs),
        "n_eff": float(len(jobs)), "slack_rows": selected[0].latest - selected[0].arrival,
        "cal_year": cal_year,
        "eval_year": eval_year, "kappa_target": kappa,
        "scale": scale, "capacity_w": capacity.tolist(),
        "blind_family": [str(x) for x in BLIND_THRESHOLDS],
        "q_star": str(q_star), "cal_carbon_by_q": {
            str(k): v for k, v in cal_by_q.items()},
        "eval_offsets": eval_offsets, "paired_effects": paired,
        "wins": int(sum(x < 0 for x in paired)),
        "pooled_effect": pooled, "median_effect": float(np.median(paired)),
        "oracle_threshold_stress": {
            "noncausal": True,
            "description": "best threshold selected separately after each eval episode",
            "paired_effects": paired_oracle_q,
            "wins": int(sum(x < 0 for x in paired_oracle_q)),
            "pooled_effect": pooled_oracle_q,
            "median_effect": float(np.median(paired_oracle_q)),
        },
        "blind": blind, "clair": clair,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbines", default="8,9,29")
    ap.add_argument("--kappa", type=float, default=0.8)
    ap.add_argument("--cal-year", type=int, default=2020)
    ap.add_argument("--eval-year", type=int, default=2021)
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--split-factor", type=int, default=1)
    ap.add_argument("--slack-h", type=float)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    groups = tuple(tuple(int(y) for y in x.split("+"))
                   for x in args.turbines.split(","))
    result = run(groups, args.kappa,
                 args.cal_year, args.eval_year, args.n_jobs,
                 args.split_factor,
                 None if args.slack_h is None else int(round(args.slack_h * 6)))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"blind", "clair", "cal_carbon_by_q"}},
                     indent=2))
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
