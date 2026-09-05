"""Stage D' evaluation additions: per-slot decision rows (Q4 corpus) and the discounted
return column. Pure parts only; the loops are exercised by the existing evaluation smoke."""
import numpy as np

from src.baselines.evaluate import decision_rows


def _obs():
    return {
        "batch_cloudlet_mi": np.array([100.0, 200.0, 0.0]),
        "batch_cloudlet_pes": np.array([2.0, 4.0, 0.0]),
        "batch_cloudlet_time_to_deadline": np.array([0.5, 1.0, 0.0]),
        "batch_cloudlet_deadline_present": np.array([1.0, 1.0, 0.0]),
        "batch_cloudlet_wait_age": np.array([0.1, 0.0, 0.0]),
        "batch_cloudlet_is_deferred": np.array([1.0, 0.0, 0.0]),
    }


def test_rows_skip_padding_and_flag_defer_using_planner_ids():
    rows = decision_rows(1, 7, _obs(), [5, 2, 0], planner_ids=[11, 12, -1], num_dcs=5)
    assert [r["slot"] for r in rows] == [0, 1]
    assert rows[0]["is_defer"] == 1 and rows[0]["action"] == 5 and rows[0]["cloudlet_id"] == 11
    assert rows[1]["is_defer"] == 0 and rows[1]["time_to_deadline"] == 1.0 and rows[1]["wait_age"] == 0.0
    assert rows[0]["is_deferred"] == 1.0 and rows[0]["defer_allowed"] is None      # key absent -> None


def test_rows_without_planner_ids_use_mi_to_drop_padding():
    rows = decision_rows(2, 3, _obs(), [1, 5, 5], planner_ids=None, num_dcs=5)
    assert [r["slot"] for r in rows] == [0, 1] and rows[1]["is_defer"] == 1
    assert all(r["cloudlet_id"] == -1 for r in rows)


def test_no_action_gives_no_rows():
    assert decision_rows(1, 1, _obs(), None, None, 5) == []


def test_option_mode_rows_name_the_hold_site_and_carry_the_legality_row():
    import numpy as np
    obs = _obs()
    obs["batch_cloudlet_hold_allowed"] = np.array([[1, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    rows = decision_rows(1, 7, obs, [7, 2, 5], planner_ids=[11, 12, -1], num_dcs=5, option_mode=True)
    assert [r["cloudlet_id"] for r in rows] == [11, 12]
    assert rows[0]["is_defer"] == 1 and rows[0]["hold_dc"] == 2 and rows[0]["hold_allowed"] == "1;0;1;1;1"
    assert rows[1]["is_defer"] == 0 and rows[1]["hold_dc"] == -1 and rows[1]["hold_allowed"] == "0;0;0;0;0"
    legacy = decision_rows(1, 7, obs, [5, 2, 5], planner_ids=[11, 12, -1], num_dcs=5)
    assert legacy[0]["is_defer"] == 1 and legacy[0]["hold_dc"] == -1 and legacy[0]["hold_allowed"] == ""


def test_offset_mode_rows_name_site_and_kappa():
    grid = [0, 1, 2, 4, 8]
    rows = decision_rows(1, 7, _obs(), [1 * 5 + 3, 0, 4], planner_ids=[11, 12, -1], num_dcs=5, offset_grid=grid)
    assert rows[0]["site"] == 1 and rows[0]["kappa"] == 4 and rows[0]["is_defer"] == 1
    assert rows[1]["site"] == 0 and rows[1]["kappa"] == 0 and rows[1]["is_defer"] == 0
