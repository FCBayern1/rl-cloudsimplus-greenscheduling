"""Causal rolling expert (reports/CAUSAL_EXPERT_PREREG.md §2): the pure decision core plans
only the newly visible jobs on the committed load, on the masked candidate starts, to proven
optimality, and emits starts the every-step executor reaches exactly."""
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "g1" / "compressed_timecap_s2"))
from src.baselines.global_schedulers import causal_decide  # noqa: E402
from ladder_planner import Job, site_from_profile  # noqa: E402

GRID = list(range(73))


def _sites():
    return [site_from_profile("a", "SPEC_ASUS_RS500A_DYN", hosts=2, vms=4),
            site_from_profile("b", "SPEC_ASUS_RS700A_DYN", hosts=2, vms=4)]


def test_expert_picks_the_green_window_it_can_reach_and_respects_committed_load():
    sites = _sites(); P = sites[0].job_power_mw(32)
    H = 200
    curve = np.zeros((2, H)); curve[0, 30:36] = 70.0; curve[1, 60:66] = 70.0      # one job's worth of green, twice
    committed_draw = np.zeros((2, H), dtype=np.int64); committed_occ = np.zeros((2, H + 1), dtype=np.int64)
    committed_draw[0, 30:36] = P; committed_occ[0, 30:37] = 1                    # site a's window already taken
    job = Job(id=7, arrival=10, runtime=6, pes=32, deadline=150)
    sched, res = causal_decide(10, {7: job}, {7: None}, GRID, sites, curve, committed_draw, committed_occ)
    assert res["status"] == "OPTIMAL" and sched == {7: (1, 60)}                  # the free green window, on site b
    # kappa = start - t - 1 = 49 is on the grid; the start is >= t + 2
    assert 60 - 10 - 1 in GRID


def test_expert_honours_the_legality_mask_and_the_window():
    sites = _sites(); n, K = 2, len(GRID)
    H = 200; curve = np.zeros((2, H)); curve[1, 60:66] = 70.0
    zero = np.zeros((2, H), dtype=np.int64); occ = np.zeros((2, H + 1), dtype=np.int64)
    job = Job(id=3, arrival=10, runtime=6, pes=32, deadline=150)
    mask = np.ones(n * K); mask[1 * K + (60 - 10 - 1)] = 0.0                     # (site b, kappa 49) forbidden
    sched, _ = causal_decide(10, {3: job}, {3: mask}, GRID, sites, curve, zero, occ)
    assert sched[3] != (1, 60)
    # a tight deadline: latest = deadline - runtime - 2
    tight = Job(id=4, arrival=10, runtime=6, pes=32, deadline=20)
    sched2, _ = causal_decide(10, {4: tight}, {4: None}, GRID, sites, curve, zero, occ)
    assert sched2[4][1] <= 20 - 6 - 2 and sched2[4][1] >= 12


def test_two_new_jobs_are_planned_jointly_without_sharing_a_host_row():
    sites = _sites(); H = 120
    curve = np.zeros((2, H)); curve[0, 20:26] = 140.0                            # two jobs' green on site a, rows 20..25
    zero = np.zeros((2, H), dtype=np.int64); occ = np.zeros((2, H + 1), dtype=np.int64)
    jobs = {1: Job(id=1, arrival=5, runtime=6, pes=32, deadline=100), 2: Job(id=2, arrival=5, runtime=6, pes=32, deadline=100)}
    sched, res = causal_decide(5, jobs, {1: None, 2: None}, GRID, sites, curve, zero, occ)
    assert res["status"] == "OPTIMAL" and sched[1] == (0, 20) and sched[2] == (0, 20)   # both fit: two hosts on site a
