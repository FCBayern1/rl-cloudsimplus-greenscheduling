"""Deadline-safe DEFER mask (STAGE_D_PRIME_DESIGN §3): the env-side pure function."""
import numpy as np

from gym_cloudsimplus.envs.hierarchical_multidc_env import defer_allowed_from

MIPS, U, STEP = 40000.0, 1.0, 1.0


def test_padding_slots_are_never_allowed_to_wait():
    out = defer_allowed_from([100.0, 100.0], [1, 1], [0.0, 1.0], [1, 1], MIPS, U, 0.0, STEP)
    assert out.tolist() == [0.0, 1.0]


def test_no_deadline_slot_may_always_wait():
    out = defer_allowed_from([-5.0], [0], [1e6], [1], MIPS, U, 0.0, STEP)
    assert out.tolist() == [1.0]


def test_tight_slot_is_masked_one_step_before_the_backstop_would_fire():
    mi = 1.92e6                      # 48 rows at 40000 MIPS
    runtime = mi / MIPS              # 48 s
    # exactly one step of slack after the wait: still allowed
    assert defer_allowed_from([runtime + STEP + 0.5], [1], [mi], [32], MIPS, U, 0.0, STEP).tolist() == [1.0]
    # no slack left after one more step: must route now
    assert defer_allowed_from([runtime + STEP], [1], [mi], [32], MIPS, U, 0.0, STEP).tolist() == [0.0]


def test_margin_and_utilization_both_tighten_the_mask():
    mi = 1.92e6
    ttd = mi / MIPS + STEP + 10.0
    assert defer_allowed_from([ttd], [1], [mi], [32], MIPS, U, 0.0, STEP).tolist() == [1.0]
    assert defer_allowed_from([ttd], [1], [mi], [32], MIPS, U, 20.0, STEP).tolist() == [0.0]     # margin
    assert defer_allowed_from([ttd], [1], [mi], [32], MIPS, 0.5, 0.0, STEP).tolist() == [0.0]    # u halves the rate


def test_shapes_pad_and_truncate_to_the_deadline_array():
    out = defer_allowed_from([100.0, 100.0, 100.0], [1], [1.0, 1.0], [1], MIPS, U, 0.0, STEP)
    assert out.shape == (3,) and out.dtype == np.float32
    assert out.tolist() == [1.0, 1.0, 0.0]      # third slot: no mi -> padding
