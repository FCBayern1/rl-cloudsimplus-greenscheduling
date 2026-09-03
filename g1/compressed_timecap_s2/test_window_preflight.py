import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from window_preflight import (OFFSET_STEP, adjacent_k_overlap, footprint_len,  # noqa: E402
                              interval, overlaps, plan)


def test_codex_finding_adjacent_1009_windows_overlap_under_the_real_footprint():
    assert adjacent_k_overlap(2518) is True
    assert footprint_len(2518) > OFFSET_STEP


def test_plan_is_pairwise_disjoint_and_never_reuses_a_read_window():
    p = plan(2518, 671, n_eval=6, n_train=8, min_eval=6, min_train=4)
    assert p["pairwise_clashes"] == []
    read = [tuple(w["rows"]) for w in p["read_windows"]]
    for w in p["eval_windows"] + p["train_windows"]:
        assert all(not overlaps(tuple(w["rows"]), r) for r in read)
        assert w["rows"][1] <= 52559 and w["rows"][0] >= 0


def test_plan_is_deterministic():
    a = plan(2518, 671, 6, 8, 6, 4)
    b = plan(2518, 671, 6, 8, 6, 4)
    assert a == b


def test_insufficient_room_is_a_stop_not_a_fallback():
    # a workload whose deadline makes the footprint bigger than any free gap
    p = plan(20000, 20000, n_eval=6, n_train=8, min_eval=6, min_train=4)
    assert p["status"] == "STOP_WINDOW_SPLIT"
    assert p["eval_windows"] == [] or len(p["eval_windows"]) < 6


def test_footprint_includes_every_registered_component():
    assert footprint_len(0) == 108 + 0 + 48 + 144 + 4 + 100
    assert interval(1000, 10) == (996, 1000 + footprint_len(10))
