import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settle_probe import DYN, decompose, job_counts, site_busy_at_start  # noqa: E402


def test_decompose_recovers_jobs_and_hosts_from_a_power_sample():
    assert decompose(0.0) == (0, 0, 0.0)
    assert decompose(65.64)[:2] == (1, 1)            # one job, its own host
    assert decompose(130.28)[:2] == (2, 1)           # two jobs packed on one host (model)
    assert decompose(131.28)[:2] == (2, 2)           # two jobs on two hosts (simulator)
    assert decompose(195.92)[:2] == (3, 2)
    assert decompose(196.92)[:2] == (3, 3)
    j, h, res = decompose(65.6)                       # DC2 RS700A: 0.04 W short of the 64-PE model
    assert (j, h) == (1, 1) and abs(res - 0.04) < 1e-6


def test_job_counts_over_a_grid():
    S = np.array([[0.0, 65.64, 131.28], [130.28, 0.0, 0.0]])
    js, hs = job_counts(S)
    assert js.tolist() == [[0, 1, 2], [2, 0, 0]] and hs.tolist() == [[0, 1, 2], [1, 0, 0]]
    assert abs(js.sum() * DYN + hs.sum() - S.sum()) < 1e-9


def test_site_busy_at_start_rule():
    sched = {0: (0, 20), 1: (0, 20), 2: (0, 30), 3: (0, 68), 4: (1, 30), 5: (0, 120)}
    assert not site_busy_at_start(sched, 0) and not site_busy_at_start(sched, 1)   # simultaneous starts on an idle site
    assert site_busy_at_start(sched, 2)                                             # job 0 started earlier and still runs
    assert site_busy_at_start(sched, 3)                                             # jobs 0/1 finish at 68 = its start
    assert not site_busy_at_start(sched, 4)                                         # other site
    assert not site_busy_at_start(sched, 5)                                         # job 3 finished at 116; site idle at 120

