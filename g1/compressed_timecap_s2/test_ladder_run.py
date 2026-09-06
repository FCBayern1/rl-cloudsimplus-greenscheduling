import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_run import closure_check, gate_l1, gate_l2, jobs_from_dump, rung_curve  # noqa: E402


def test_jobs_from_dump_take_first_sighting_and_deadline_step():
    rows = [{"cloudlet_id": "3", "step": "9", "pes": "32", "mi": "1920000", "time_to_deadline": "0.017", "ttd_sec": "120.0", "deadline_present": "1"},
            {"cloudlet_id": "3", "step": "10", "pes": "32", "mi": "1920000", "time_to_deadline": "0.016", "ttd_sec": "119.0", "deadline_present": "1"},
            {"cloudlet_id": "-1", "step": "10", "pes": "0", "mi": "0", "time_to_deadline": "0", "ttd_sec": "0", "deadline_present": "0"}]
    jobs = jobs_from_dump(rows, 40000.0, 1.0)
    assert len(jobs) == 1 and jobs[0].arrival == 9 and jobs[0].runtime == 48 and jobs[0].deadline == 129
    assert jobs[0].latest == 129 - 48 - 2


def test_rung_curves():
    G = np.array([[0.0, 10.0, 20.0, 30.0]])
    assert np.array_equal(rung_curve(G, "truth", [15.0], "k"), G)
    s = rung_curve(G, "shrink_0.5", [15.0], "k")
    assert np.allclose(s, [[7.5, 12.5, 17.5, 22.5]])
    assert np.allclose(rung_curve(G, "shrink_0", [15.0], "k"), [[15.0] * 4])
    a = rung_curve(G, "anti", [15.0], "k")
    assert np.array_equal(a, [[30.0, 20.0, 10.0, 0.0]])
    sh1, sh2 = rung_curve(G, "shuffle", [15.0], "k"), rung_curve(G, "shuffle", [15.0], "k")
    assert np.array_equal(sh1, sh2) and sorted(sh1[0].tolist()) == [0.0, 10.0, 20.0, 30.0]


def test_closure_check_per_job_and_counters():
    sched = {1: (0, 10), 2: (1, 20)}
    led = [{"id": "1", "dc": "0", "t_s": "10.0", "stale": "False"}, {"id": "2", "dc": "1", "t_s": "20.5", "stale": "False"}]
    ok = closure_check(1.0, 1.02, led, sched, {"deadline_forced_count": 0})
    assert ok["pass"] is True and abs(ok["rel_err"] - 0.02) < 1e-12
    bad = closure_check(1.0, 1.05, led, sched, {"deadline_forced_count": 0})
    assert bad["pass"] is False and any("carbon" in v for v in bad["violations"])
    wrong = [{"id": "1", "dc": "2", "t_s": "10.0", "stale": "False"}, {"id": "2", "dc": "1", "t_s": "23.0", "stale": "False"}]
    v = closure_check(1.0, 1.0, wrong, sched, {"deadline_forced_count": 1})["violations"]
    assert any("site" in x for x in v) and any("started" in x for x in v) and any("forced" in x for x in v)
    dup = led + [led[0]]
    assert any("appears 2" in x for x in closure_check(1.0, 1.0, dup, sched, {})["violations"])


def test_gates_l1_and_l2():
    c_truth = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    c_flat = {0: 1.5, 1: 1.5, 2: 1.05, 3: 1.5}                 # window 2 has no headroom
    l1 = gate_l1(c_flat, c_truth)
    assert [l1[k]["valid"] for k in range(4)] == [True, True, False, True]
    losses = {"shrink_0.5": {0: 0.1, 1: 0.1, 2: -0.5, 3: 0.1},   # harmful on all valid windows, 10 % pooled
              "shrink_0.75": {0: 0.01, 1: 0.0, 2: 0.0, 3: 0.0}}  # 0.3 % pooled, one window
    l2 = gate_l2(losses, l1, c_truth)
    assert l2["shrink_0.5"]["load_bearing"] is True and l2["shrink_0.75"]["load_bearing"] is False
    assert abs(l2["shrink_0.5"]["harm_headroom_share"] - 1.0) < 1e-12


def test_atomic_json_writes_complete_records(tmp_path):
    from ladder_run import atomic_json
    p = str(tmp_path / "r.json")
    atomic_json(p, {"status": "FEASIBLE", "wall_s": 3600.0})
    import json
    assert json.load(open(p))["status"] == "FEASIBLE" and not os.path.exists(p + ".tmp")
