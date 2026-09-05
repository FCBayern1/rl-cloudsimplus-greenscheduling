import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from margin_probe import margin_from_delays  # noqa: E402


def test_margin_is_ceil_of_worst_delay_plus_one_step():
    out = margin_from_delays([3.2, 7.9, 0.0], timestep_sec=1.0)
    assert out["worst_delay_sec"] == 7.9 and out["margin_steps"] == 9 and out["margin_sec"] == 9.0


def test_zero_delay_still_leaves_one_step():
    out = margin_from_delays([], timestep_sec=2.0)
    assert out["margin_steps"] == 1 and out["margin_sec"] == 2.0


def test_exact_multiple_rounds_up_by_one_only():
    assert margin_from_delays([4.0], 2.0)["margin_steps"] == 3
