import os
import sys

import numpy as np
import pytest

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


def test_judge_treats_windows_never_solved_as_unresolved(tmp_path, monkeypatch):
    # Addendum E: the solve stops at the first unproven cell, so later windows never enter the
    # summary; the judge must report them as unresolved instead of failing on the missing key.
    import json
    import ladder_run
    monkeypatch.setattr(ladder_run, "LAD", str(tmp_path))
    monkeypatch.setattr(ladder_run, "_dev", lambda: [16477, 4240, 9154])
    os.makedirs(tmp_path / "solve"); os.makedirs(tmp_path / "replay")
    summary = {"0": {"quantisation_ok": True, "rungs": {"truth": {"status": "OPTIMAL", "C_model_truth_kg": 1.0}}},
               "1": {"quantisation_ok": True, "rungs": {"truth": {"status": "FEASIBLE"}}},
               "environment": {}}
    json.dump(summary, open(tmp_path / "solve_summary.json", "w"))
    ladder_run.cmd_judge(("truth",))
    res = json.load(open(tmp_path / "truth_closure.json"))
    assert res["verdict"] == "STOP_SOLVER_RUNG_UNRESOLVED"
    assert res["unresolved"] == ["k1:truth", "k2:truth (not solved: after the stop)"]


def test_replay_uses_the_every_step_offset_grid():
    # prereg §2.2 / Addendum A: settlement path = every-step (DC, offset) executor; the replay
    # arm indexes actions on the 73-value grid, so the simulator must be told to use it too.
    from ladder_run import replay_env
    env = replay_env("/x/k0_truth.json")
    assert env["OFFSET_GRID_DENSE"] == "1" and env["SCHEDULE_JSON"] == "/x/k0_truth.json"


def _wind_dir(tmp_path, n_rows=100):
    d = tmp_path / "wind"; d.mkdir()
    for tid in (7, 8):
        lines = ["timestamp,power_kw"] + [f"2021-01-01 00:{i:02d}:00,{(tid * 1000 + i):.2f}" for i in range(n_rows)]
        (d / f"Turbine_{tid}_2021.csv").write_text("\n".join(lines) + "\n")
    return str(d)


def _blk():
    return {"compressed_power_divisor": 3000.0, "wind_csv_year": 2021, "min_time_between_events": 1.0,
            "datacenters": [{"datacenter_id": 0, "time_zone_offset_rows": 0, "turbine_ids": [7, 8], "green_energy_enabled": True,
                             "host_count_spec_asus_rs500a_dyn": 10, "initial_s_vm_count": 20, "small_vm_pes": 32, "vm_pe_mips": 40000},
                            {"datacenter_id": 1, "time_zone_offset_rows": 5, "turbine_ids": [7], "green_energy_enabled": True,
                             "host_count_spec_asus_rs700a_dyn": 5, "initial_s_vm_count": 20, "small_vm_pes": 32, "vm_pe_mips": 40000},
                            {"datacenter_id": 2, "time_zone_offset_rows": 9, "turbine_ids": [], "green_energy_enabled": False,
                             "host_count_spec_asus_rs500a_dyn": 3, "initial_s_vm_count": 6, "small_vm_pes": 32, "vm_pe_mips": 40000}]}


def test_truth_curve_reads_the_rows_the_simulator_reads(tmp_path):
    # diagnostic C: row(t) = offset + tz + t + OBS_CLOCK_LAG(1) + SPLINE_SKIP_ROWS(12)
    from ladder_run import truth_curve, LadderStop, OBS_CLOCK_LAG, SPLINE_SKIP_ROWS
    wd = _wind_dir(tmp_path)
    G, meta = truth_curve(_blk(), offset=10, T=4, wind_dir=wd)
    assert G.shape == (3, 4) and (OBS_CLOCK_LAG, SPLINE_SKIP_ROWS) == (1, 12)
    row0 = 10 + 0 + 1 + 12
    assert abs(G[0, 0] - ((7000 + row0) + (8000 + row0)) * 1000 / 3000) < 1e-9
    assert abs(G[1, 2] - (7000 + 10 + 5 + 2 + 13) * 1000 / 3000) < 1e-9       # tz 5, step 2
    assert G[2].tolist() == [0.0] * 4                                            # no green on site 2
    assert meta["sites"][1]["row_start"] == 10 + 5 + 13 and meta["sites"][1]["row_end"] == 10 + 5 + 13 + 4
    assert len(meta["signature"]) == 16
    with pytest.raises(LadderStop):                                              # past the file: STOP, no wrap
        truth_curve(_blk(), offset=80, T=10, wind_dir=wd)


def test_curve_rows_match_and_signature():
    from ladder_run import curve_rows_match, curve_signature
    P = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    ok, n, m, bad = curve_rows_match(P, P[:, :3])
    assert ok and n == 3 and m == 0.0 and bad == -1
    O = P[:, :3].copy(); O[0, 1] += 0.5
    ok, n, m, bad = curve_rows_match(P, O)
    assert not ok and bad == 1 and abs(m - 0.5) < 1e-12
    assert not curve_rows_match(P, np.zeros((2, 5)))[0]                          # replay longer than the curve
    assert curve_signature(P) == curve_signature(P + 1e-7) and curve_signature(P) != curve_signature(P + 0.01)


def test_sites_from_config_uses_each_sites_host_profile():
    from ladder_run import sites_from_config
    s = sites_from_config({"common": {"host_pe_mips": 50000}}, _blk())
    assert [x.profile for x in s] == ["SPEC_ASUS_RS500A_DYN", "SPEC_ASUS_RS700A_DYN", "SPEC_ASUS_RS500A_DYN"]
    assert [x.hosts for x in s] == [10, 5, 3] and [x.cap for x in s] == [640, 640, 192]
    assert s[0].job_power_mw(32) == 65640 and s[1].job_power_mw(32) == 65600          # RS500A vs RS700A (diagnostic D)
    assert s[1].host_pes == 128


def test_atomic_json_writes_complete_records(tmp_path):
    from ladder_run import atomic_json
    p = str(tmp_path / "r.json")
    atomic_json(p, {"status": "FEASIBLE", "wall_s": 3600.0})
    import json
    assert json.load(open(p))["status"] == "FEASIBLE" and not os.path.exists(p + ".tmp")
