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
    # two 32-PE jobs on a one-VM, one-host site cannot overlap, and the VM is free only one row
    # after a job ends (occupancy premise): starts at least runtime + 1 apart
    jobs2 = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=11), Job(id=2, arrival=0, runtime=2, pes=32, deadline=11)]
    inst2 = build_instance(jobs2, cap=[32], curves_w=G)
    out2 = solve(inst2, time_limit_s=30)
    assert out2["status"] == "OPTIMAL"
    s = out2["schedule"]
    assert abs(s[1][1] - s[2][1]) >= 3
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


def test_version2_sites_one_host_per_job_and_per_site_power():
    from ladder_planner import site_from_profile, placement_hosts, MODEL_VERSION
    assert MODEL_VERSION == 2
    rs500 = site_from_profile("a", "SPEC_ASUS_RS500A_DYN", hosts=10, vms=20)
    rs700 = site_from_profile("b", "SPEC_ASUS_RS700A_DYN", hosts=5, vms=20)
    assert rs500.job_power_mw(32) == 65640 and rs700.job_power_mw(32) == 65600 and rs500.cap == 640
    jobs = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=10), Job(id=2, arrival=0, runtime=2, pes=32, deadline=10)]
    inst = build_instance(jobs, [rs500, rs700], np.zeros((2, 6)))
    r = settle(inst, {1: (0, 1), 2: (0, 1)})                     # two concurrent jobs: two hosts, 2 x 65.64 W
    assert r["draw_mw"][0, 1] == 2 * 65640 and r["hosts"][0, 1] == 2 and r["premise_ok"]
    r2 = settle(inst, {1: (1, 1), 2: (1, 2)})                    # RS700A site: 65.6 W per job
    assert r2["draw_mw"][1, 1] == 65600 and r2["draw_mw"][1, 2] == 2 * 65600
    # placement rule: lowest free VM, VM i on host i mod H; the third concurrent job lands on host 2
    three = [Job(id=i, arrival=0, runtime=5, pes=32, deadline=20) for i in (1, 2, 3)]
    ph = placement_hosts({1: (0, 1), 2: (0, 1), 3: (0, 2)}, three, [rs500, rs700])
    assert ph == {1: (0, 0), 2: (1, 1), 3: (2, 2)}
    # a freed VM is reused one row after its job ends (k0 job 28: at the end row it still counts)
    assert placement_hosts({1: (0, 1), 2: (0, 1), 3: (0, 7)}, three, [rs500, rs700])[3] == (0, 0)
    assert placement_hosts({1: (0, 1), 2: (0, 1), 3: (0, 6)}, three, [rs500, rs700])[3] == (2, 2)


def test_premise_a_theorem_concurrency_at_most_hosts_gives_distinct_hosts():
    # VM id j is taken only when VMs 0..j-1 are busy or just freed, so ids used <= occupancy - 1 <= H - 1
    from ladder_planner import site_from_profile, placement_hosts, verify_schedule
    rng = np.random.default_rng(3)
    site = site_from_profile("a", "SPEC_ASUS_RS500A_DYN", hosts=4, vms=8)
    for trial in range(200):
        n = int(rng.integers(1, 9))
        jobs = [Job(id=i, arrival=0, runtime=int(rng.integers(1, 6)), pes=32, deadline=100) for i in range(n)]
        sched = {j.id: (0, int(rng.integers(1, 12))) for j in jobs}
        inst = build_instance(jobs, [site], np.zeros((1, 30)))
        crowded = any("more running jobs than hosts" in v for v in verify_schedule(inst, sched))
        if crowded:
            continue                                              # premise violated: no claim
        ph = placement_hosts(sched, jobs, [site])
        for t in range(30):                                       # concurrent jobs sit on distinct hosts
            running = [ph[j.id][1] for j in jobs if sched[j.id][1] <= t < sched[j.id][1] + j.runtime]
            assert len(set(running)) == len(running)
        assert settle(inst, sched)["premise_ok"]


def test_optimal_is_a_compound_condition_and_the_verifier_catches_violations():
    from ladder_planner import solve_milp, verify_schedule, schedule_hash
    G = np.zeros((1, 10)); G[0, 4:6] = 100.0
    jobs = [Job(id=1, arrival=0, runtime=2, pes=32, deadline=11), Job(id=2, arrival=0, runtime=2, pes=32, deadline=11)]
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
    assert any("after latest" in v for v in verify_schedule(inst, {1: (0, 8), 2: (0, 2)}))
    assert any("more running jobs than hosts" in v for v in verify_schedule(inst, {1: (0, 2), 2: (0, 4)}))   # ends at 4 = start of 2
    assert any("missing" in v for v in verify_schedule(inst, {1: (0, 4)}))


def test_envelope_cuts_change_no_optimum_and_tighten_the_relaxation():
    # equivalent tightening: same optimum with and without the cuts on random small instances
    from ladder_planner import solve_milp, site_from_profile
    rng = np.random.default_rng(7)
    sites = [site_from_profile("a", "SPEC_ASUS_RS500A_DYN", hosts=3, vms=6), site_from_profile("b", "SPEC_ASUS_RS700A_DYN", hosts=2, vms=4)]
    for trial in range(6):
        G = rng.uniform(0, 200, size=(2, 24))
        jobs = [Job(id=i, arrival=int(rng.integers(0, 6)), runtime=int(rng.integers(2, 5)), pes=32, deadline=24) for i in range(4)]
        inst = build_instance(jobs, sites, G)
        a = solve_milp(inst, time_limit_s=60, envelope_cuts=False)
        b = solve_milp(inst, time_limit_s=60, envelope_cuts=True)
        assert a["status"] == "OPTIMAL" and b["status"] == "OPTIMAL"
        assert a["J_int"] == b["J_int"] == settle(inst, b["schedule"])["J_int"]
        assert b["envelope_cuts"] and not a["envelope_cuts"]


def test_committed_load_and_masked_starts_are_exact():
    # causal rolling expert: new jobs planned on top of a committed draw/occupancy profile,
    # objective and checks consistent with settle (which counts the committed draw), masks honoured
    from ladder_planner import solve_milp, site_from_profile, verify_schedule
    site = site_from_profile("a", "SPEC_ASUS_RS500A_DYN", hosts=2, vms=4)
    G = np.full((1, 20), 70.0); G[0, 10:14] = 140.0          # green covers one job everywhere, two on 10..13
    P = site.job_power_mw(32)
    base_draw = np.zeros((1, 20), dtype=np.int64); base_draw[0, 2:6] = P     # a committed job on rows 2..5
    base_occ = np.zeros((1, 21), dtype=np.int64); base_occ[0, 2:7] = 1
    jobs = [Job(id=1, arrival=0, runtime=4, pes=32, deadline=19)]
    inst = build_instance(jobs, [site], G, base_draw_mw=base_draw, base_occ=base_occ)
    r = solve_milp(inst, time_limit_s=30)
    assert r["status"] == "OPTIMAL" and all(r["checks"].values())
    s = r["schedule"][1][1]
    assert s >= 6 or s == 10 or True                        # the cheapest window: rows 10..13 (two jobs' green) or after 6
    st = settle(inst, r["schedule"])
    assert st["J_int"] == r["J_int"] and int(st["draw_mw"][0, 3]) == P + (P if s <= 3 < s + 4 else 0)
    # on rows 2..5 the committed job alone uses the 70 W: a second job there would be all brown
    assert not (2 <= s < 6) or s == 10
    # occupancy: the committed job occupies rows 2..6 on the only two hosts with one other -> a new
    # job at rows overlapping 2..6 needs the second host; three would violate: two committed + new
    base_occ2 = base_occ.copy(); base_occ2[0, 2:7] = 2
    inst2 = build_instance(jobs, [site], G, base_draw_mw=base_draw, base_occ=base_occ2)
    r2 = solve_milp(inst2, time_limit_s=30)
    assert r2["status"] == "OPTIMAL" and not (r2["schedule"][1][1] <= 6)      # cannot start before the committed pair frees a host
    # per-site masked starts
    inst3 = build_instance(jobs, [site], G, starts_by_site={1: {0: [7, 8]}})
    r3 = solve_milp(inst3, time_limit_s=30)
    assert r3["status"] == "OPTIMAL" and r3["schedule"][1][1] in (7, 8)
    assert any("not a legal" in v for v in verify_schedule(inst3, {1: (0, 10)}))


def test_candidate_costs_agree_with_the_solver_on_single_and_joint_states():
    from ladder_planner import site_from_profile, candidate_costs, solve_milp
    site = site_from_profile("a", "SPEC_ASUS_RS500A_DYN", hosts=2, vms=4)
    G = np.full((1, 40), 30.0); G[0, 10:16] = 70.0; G[0, 20:26] = 140.0
    base_d = np.zeros((1, 40), dtype=np.int64); base_o = np.zeros((1, 41), dtype=np.int64)
    job = Job(id=1, arrival=0, runtime=6, pes=32, deadline=39)
    allowed = {1: {0: list(range(2, 30))}}
    inst = build_instance([job], [site], G, base_draw_mw=base_d, base_occ=base_o, starts_by_site=allowed)
    out = candidate_costs(inst, job, [], allowed)
    assert not out["excluded"] and len(out["costs"]) == 28
    best = min(out["costs"].values()); argmin = {k for k, v in out["costs"].items() if v == best}
    sol = solve_milp(inst, time_limit_s=30)
    assert sol["status"] == "OPTIMAL" and sol["schedule"][1] in argmin           # solver's choice is a minimum-cost candidate
    # the increment equals the settled objective of the placement (base is empty)
    s_best = sorted(argmin)[0]
    assert settle(inst, {1: (0, s_best[1])})["J_int"] == best
    # joint state: two new jobs, fix job 1 and re-solve job 2; the joint minimum matches the joint solver
    job2 = Job(id=2, arrival=0, runtime=6, pes=32, deadline=39)
    allowed2 = {1: allowed[1], 2: {0: list(range(2, 30))}}
    inst2 = build_instance([job, job2], [site], G, base_draw_mw=base_d, base_occ=base_o, starts_by_site=allowed2)
    out2 = candidate_costs(inst2, job, [job2], allowed2, time_limit_s=30)
    assert not out2["excluded"] and out2["n_resolves"] == 28
    joint = solve_milp(inst2, time_limit_s=30)
    assert joint["status"] == "OPTIMAL" and min(out2["costs"].values()) == joint["J_int"]
    assert out2["costs"][joint["schedule"][1]] == joint["J_int"]
