"""v2 workloads must be a function of the key alone, and acceptance must not see the wind.

The v1 generator mixed the season offset into the seed, so one key resampled per season
and 1,296 cells carried 272 distinct loads instead of 99. Reuse is what makes a cell's
budget and weather the only things that vary, so it is asserted rather than assumed.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402
import schedule_feasibility as sf  # noqa: E402
import workload_v2 as wv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = wv.workload_key(0, 36, 8, 3, 10, 12)


def test_neither_module_can_read_the_weather_or_the_ledger():
    for name in ("workload_v2", "schedule_feasibility"):
        src = open(os.path.join(HERE, f"{name}.py")).read()
        for banned in ("green", "cb[", "cg[", "climatology", "carbon", "_series"):
            assert banned not in src, f"{name} references {banned}"


def test_the_seed_is_byte_exact_and_reproducible():
    import hashlib
    payload = json.dumps(K, sort_keys=True, separators=(",", ":"))
    for k in (0, 1, 7):
        d = hashlib.sha256((payload + ":" + str(k)).encode()).digest()
        assert wv.frozen_seed(K, k) == int.from_bytes(d[:8], "big") % 2**31


def test_the_runtime_set_is_serialised_as_a_list():
    from_tuple = wv.workload_key(0, 36, 8, 3, 10, 12, runtime_set=(6, 12))
    from_list = wv.workload_key(0, 36, 8, 3, 10, 12, runtime_set=[6, 12])
    assert from_tuple == from_list
    assert isinstance(from_tuple["runtime_set"], list)
    assert wv.frozen_seed(from_tuple, 0) == wv.frozen_seed(from_list, 0)


def test_the_content_is_a_function_of_the_key_alone():
    a = wv.draw(K, 0)
    b = wv.draw(dict(K), 0)
    assert wv.content_hash(a) == wv.content_hash(b)
    assert np.array_equal(a["arrival"], b["arrival"])


def test_budget_and_weather_never_enter_the_key():
    forbidden = {"budget_fraction", "installed_divisor", "offset", "season_offset",
                 "triplet", "turbines", "turbines_per_site"}
    assert not (forbidden & set(K))


def test_the_budget_scales_with_the_fraction_but_not_the_content():
    w = wv.draw(K, 0)
    h = wv.content_hash(w)
    budgets = [wv.budget_for(w, f) for f in ig.BUDGET_FRACTION]
    assert budgets == sorted(budgets) and budgets[0] < budgets[-1]
    assert wv.content_hash(w) == h, "computing a budget changed the load"


def test_acceptance_returns_the_first_passing_retry():
    acc = wv.accepted(K)
    assert acc is not None
    assert acc["content_hash"] == wv.content_hash(wv.draw(K, acc["retry"]))
    for k in range(acc["retry"]):
        w = wv.draw(K, k)
        b = wv.budget_for(w, wv.STRICTEST_BUDGET_FRACTION)
        ok = (sf.capacity_ok(w, b) == "FEASIBLE"
              and sf.reservation_edf(w, b)[0] is not None)
        assert not ok, f"retry {k} would have passed but was skipped"


def test_reservation_edf_respects_capacity_deadlines_and_budget():
    w = wv.draw(K, 0)
    b = wv.budget_for(w, 0.10)
    assign, spent = sf.reservation_edf(w, b)
    assert assign is not None
    used = np.zeros((sf.N_DC, w["horizon"]), dtype=int)
    for i, (d, s) in assign.items():
        assert s >= w["arrival"][i]
        assert s + w["runtime"][i] <= w["deadline"][i]
        assert s - w["arrival"][i] <= w["wait_cap"]
        used[d, s:s + w["runtime"][i]] += w["pes"][i]
    assert used.max() <= sf.CAP
    assert spent <= b


def test_reservation_edf_is_deterministic_and_prefers_the_earliest_then_lowest_site():
    w = {"arrival": np.array([0, 0]), "runtime": np.array([2, 2]),
         "pes": np.array([16, 16]), "deadline": np.array([8, 8]),
         "horizon": 8, "wait_cap": 6}
    a1, _ = sf.reservation_edf(w, 100)
    a2, _ = sf.reservation_edf(w, 100)
    assert a1 == a2
    # A full site cannot take both, so the second must move to the next site at the same
    # epoch rather than wait.
    assert {a1[0], a1[1]} == {(0, 0), (1, 0)}


def test_reservation_edf_fails_rather_than_backtracking():
    """Three jobs each filling a whole site, with only two sites free at that epoch."""
    w = {"arrival": np.array([0, 0, 0, 0]), "runtime": np.array([8, 8, 8, 8]),
         "pes": np.array([16, 16, 16, 16]), "deadline": np.array([8, 8, 8, 8]),
         "horizon": 8, "wait_cap": 0}
    assign, spent = sf.reservation_edf(w, 0)
    assert assign is None and spent is None


def test_feasibility_uses_deterministic_time_not_wall_clock():
    src = open(os.path.join(HERE, "schedule_feasibility.py")).read()
    assert "max_deterministic_time" in src
    assert "max_time_in_seconds" not in src
    assert "num_search_workers = 1" in src
    assert "random_seed" in src


def test_unknown_is_not_treated_as_infeasible():
    src = open(os.path.join(HERE, "workload_v2.py")).read()
    body = src[src.index("def _accept"):]
    assert 'status == "UNKNOWN"' in body and "continue" in body
    assert "INFEASIBLE" not in body.split('status == "UNKNOWN"')[0].split("for k in")[1]
