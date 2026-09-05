import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from option_bc import MIN_HELD_JOBS, MIN_PER_CLASS, corpus_valid, first_decisions  # noqa: E402


def test_first_decisions_keeps_one_row_per_job_and_names_branch_and_site():
    n = 5
    rows = [{"cloudlet_id": "7", "step": "3", "slot": "1", "action": "8"},     # HOLD(3)
            {"cloudlet_id": "7", "step": "9", "slot": "0", "action": "2"},     # later sighting ignored
            {"cloudlet_id": "8", "step": "3", "slot": "2", "action": "2"},     # ROUTE_NOW(2)
            {"cloudlet_id": "-1", "step": "3", "slot": "3", "action": "0"}]    # padding
    d = first_decisions(rows, n)
    assert set(d) == {7, 8}
    assert d[7] == {"id": 7, "step": 3, "slot": 1, "action": 8, "is_hold": 1, "site": 3}
    assert d[8] == {"id": 8, "step": 3, "slot": 2, "action": 2, "is_hold": 0, "site": 2}


def test_corpus_validity_rule_a7():
    assert corpus_valid([1] * 30 + [0] * 30)["valid"] is True
    assert corpus_valid([1] * 50 + [0] * 9)["valid"] is False          # n < 60
    assert corpus_valid([1] * 70 + [0] * 10)["valid"] is False         # a class < 15
    assert MIN_HELD_JOBS == 60 and MIN_PER_CLASS == 15


def test_option_module_builds_from_the_option_block_without_a_jvm():
    from option_bc import build_module, load_block
    cfg = load_block()
    mod, obs_space, act_space = build_module(cfg, seed=1)
    nvec = [int(x) for x in act_space.nvec]
    assert nvec[0] == 2 * len(cfg["datacenters"]) and len(nvec) == int(cfg["global_routing_batch_size"])
    assert getattr(mod, "option_mode", False) and mod.has_hold_mask


def test_offset_mode_decisions_and_delay_columns():
    from option_bc import delay_columns
    grid = [0, 1, 2, 4]
    n = 2
    rows = [{"cloudlet_id": "3", "step": "2", "slot": "0", "action": str(1 * 4 + 2)},   # site 1, κ=2
            {"cloudlet_id": "4", "step": "2", "slot": "1", "action": "0"}]               # site 0, κ=0
    d = first_decisions(rows, n, grid)
    assert d[3] == {"id": 3, "step": 2, "slot": 0, "action": 6, "is_hold": 1, "site": 1, "kappa": 2}
    assert d[4]["is_hold"] == 0 and d[4]["kappa"] == 0
    assert delay_columns(8, n, grid) == [1, 2, 3, 5, 6, 7]
    assert delay_columns(4, n, None) == [2, 3]
