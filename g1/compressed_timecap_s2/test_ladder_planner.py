import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_planner import (  # noqa: E402
    HOST_FLOOR_MW, KG_PER_UNIT, MW_PER_PE, Job, build_instance, preflight_factors,
    quantisation_bound_kg, runtime_steps, settle, solve)


def test_exact_objective_units_and_quantisation_bound():
    # 1 mW*s of green at 0.01 kg/kWh = 1e-3 W * 1 s / 3.6e6 * 0.01 kg = 2.78e-12 kg
    assert abs(KG_PER_UNIT - 1e-3 / 3.6e6 * 0.01) < 1e-20
    assert abs(quantisation_bound_kg(5, 600) - 2.04e-7) < 2e-9          # C3 of the preregistration
    assert quantisation_bound_kg(5, 600) < 0.001 * 0.01897
    preflight_factors([{"brown_carbon_factor": 0.5, "green_carbon_factor": 0.01}])
    with pytest.raises(RuntimeError):
        preflight_factors([{"brown_carbon_factor": 0.6, "green_carbon_factor": 0.01}])
    assert runtime_steps(1920000, 40000, 1.0) == 48 and runtime_steps(1920000, 40000, 0.5) == 96


def test_settlement_charges_mips_utilisation_and_the_host_floor():
    jobs = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=10)]
    G = np.zeros((1, 6))                                     # no green: everything brown
    inst = build_instance(jobs, cap=[64], curves_w=G)
    r = settle(inst, {1: (0, 1)})
    # steps 1,2: 32 PEs -> 64640 mW + one host floor 1000 mW = 65640 mW = 65.64 W
    assert r["draw_mw"][0, 1] == 32 * MW_PER_PE + HOST_FLOOR_MW == 65640
    assert r["brown_mw"][0, 1] == 65640 and r["green_mw"][0, 1] == 0
    assert r["J_int"] == 50 * 65640 * 2
    # with plenty of green the same schedule is all green: J = draw
    inst2 = build_instance(jobs, cap=[64], curves_w=np.full((1, 6), 100.0))
    assert settle(inst2, {1: (0, 1)})["J_int"] == 65640 * 2


def test_solver_finds_the_green_window_and_respects_deadline_and_capacity():
    # green only on steps 4-5; job can wait until start 4 (deadline 8, r 2, eps 2 -> latest 4)
    G = np.zeros((1, 10)); G[0, 4:6] = 100.0
    jobs = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=8)]
    inst = build_instance(jobs, cap=[64], curves_w=G)
    out = solve(inst, time_limit_s=30)
    assert out["status"] == "OPTIMAL" and out["schedule"] == {1: (0, 4)}
    assert out["J_int"] == 65640 * 2                                  # all green
    # two 32-PE jobs on a 32-PE site cannot overlap: one must take the brown steps
    jobs2 = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=8), Job(id=2, arrival=0, runtime=2, pes=32, deadline=8)]
    inst2 = build_instance(jobs2, cap=[32], curves_w=G)
    out2 = solve(inst2, time_limit_s=30)
    assert out2["status"] == "OPTIMAL"
    s = out2["schedule"]
    assert abs(s[1][1] - s[2][1]) >= 2
    assert out2["J_int"] == settle(inst2, s)["J_int"]                # solver objective == settlement


def test_dominance_truth_schedule_never_beaten_in_model_settlement():
    rng = np.random.default_rng(0)
    G = rng.uniform(0, 120, size=(2, 30))
    jobs = [Job(id=i, arrival=i * 2, runtime=3, pes=32, deadline=25) for i in range(4)]
    inst = build_instance(jobs, cap=[64, 32], curves_w=G)
    truth = solve(inst, time_limit_s=60)
    assert truth["status"] == "OPTIMAL"
    j_truth = settle(inst, truth["schedule"])["J_int"]
    # schedules from wrong curves, settled on truth, are never better than the truth optimum
    for seed in range(3):
        wrong = np.maximum(0.0, G + rng.normal(0, 40, size=G.shape))
        alt = solve(build_instance(jobs, cap=[64, 32], curves_w=wrong), time_limit_s=60)
        assert alt["status"] == "OPTIMAL"
        assert settle(inst, alt["schedule"])["J_int"] >= j_truth


def test_milp_agrees_with_cpsat_on_small_instances():
    from ladder_planner import solve_milp
    G = np.zeros((1, 10)); G[0, 4:6] = 100.0
    jobs = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=8)]
    inst = build_instance(jobs, cap=[64], curves_w=G)
    r = solve_milp(inst, time_limit_s=30)
    assert r["status"] == "OPTIMAL" and r["schedule"] == {1: (0, 4)} and r["J_int"] == 65640 * 2
    rng = np.random.default_rng(1)
    G2 = rng.uniform(0, 120, size=(2, 30))
    jobs2 = [Job(id=i, arrival=i * 2, runtime=3, pes=32, deadline=25) for i in range(4)]
    inst2 = build_instance(jobs2, cap=[64, 32], curves_w=G2)
    a, b = solve(inst2, time_limit_s=60), solve_milp(inst2, time_limit_s=60)
    assert a["status"] == "OPTIMAL" and b["status"] == "OPTIMAL"
    assert settle(inst2, a["schedule"])["J_int"] == settle(inst2, b["schedule"])["J_int"] == b["J_int"]


def test_optimal_is_a_compound_condition_and_the_verifier_catches_violations():
    from ladder_planner import solve_milp, verify_schedule, schedule_hash
    G = np.zeros((1, 10)); G[0, 4:6] = 100.0
    jobs = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=8), Job(id=2, arrival=0, runtime=2, pes=32, deadline=8)]
    inst = build_instance(jobs, cap=[32], curves_w=G)
    r = solve_milp(inst, time_limit_s=30)
    assert r["status"] == "OPTIMAL" and all(r["checks"].values())
    assert r["mip_dual_bound"] is not None and r["J_int"] - r["mip_dual_bound"] < 1.0
    assert abs(r["fun"] - r["J_int"]) < 0.5 and r["verify_violations"] == []
    assert len(r["schedule_hash"]) == 16 and r["schedule_hash"] == schedule_hash(r["schedule"])
    assert r["mip_node_count"] is not None
    # the verifier: overlap on a 32-PE site, a start before arrival + lag, a start past latest, a missing job
    assert any("capacity" in v for v in verify_schedule(inst, {1: (0, 4), 2: (0, 5)}))
    assert any("before arrival" in v for v in verify_schedule(inst, {1: (0, 0), 2: (0, 4)}))
    assert any("after latest" in v for v in verify_schedule(inst, {1: (0, 6), 2: (0, 1)}))
    assert any("missing" in v for v in verify_schedule(inst, {1: (0, 4)}))
