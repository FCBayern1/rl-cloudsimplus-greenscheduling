"""The evaluation config must be the training config, plus exactly one key.

This guard exists because the pre-existing evaluation block for the Vanilla arm
(`experiment_p0cprobe_van`) was not a copy of the matched Vanilla at all. It was
a stale pre-v5 block differing in twelve keys, eight of them objective-level:
carbon_penalty_mode, per_action_carbon_weight, per_action_completion_weight,
global_reward_beta, global_completion_rate_mi_coef, carbon_normalization_fixed_max,
and two model hyperparameters. Those eight are the same family that invalidated
the original Vanilla comparator, so the error P0-B was built to catch was sitting
unguarded in the evaluation path.

The reward configuration does not steer an argmax policy, but it does drive the
reported carbon accounting, and a stale block is evidence the copy was made by
hand rather than derived. Derive it, and assert the difference set.
"""
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT.parent / "config_C.yml"

PAIRS = {
    "experiment_g1eval_knSV3b":
        "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b",
    "experiment_g1eval_matchedvan":
        "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan",
}
ALLOWED = {"green_episode_offset_range"}
_MISSING = object()


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG.read_text())


def typed_ne(a, b):
    """Same comparison discipline as verify_matched_vanilla.py: a missing key is
    a difference, None is not 0, and True is not 1."""
    if a is _MISSING or b is _MISSING:
        return True
    if (a is None) != (b is None):
        return True
    if isinstance(a, bool) != isinstance(b, bool):
        return True
    return a != b


@pytest.mark.parametrize("evalkey,trainkey", sorted(PAIRS.items()))
def test_eval_block_differs_by_exactly_one_key(cfg, evalkey, trainkey):
    assert evalkey in cfg, f"{evalkey} missing from config_C.yml"
    a, b = cfg[evalkey], cfg[trainkey]
    observed = {k for k in set(a) | set(b)
                if typed_ne(a.get(k, _MISSING), b.get(k, _MISSING))}
    assert observed == ALLOWED, sorted(observed)


@pytest.mark.parametrize("evalkey", sorted(PAIRS))
def test_eval_block_can_reach_the_registered_windows(cfg, evalkey):
    """reset_skip only moves the window if the range is set; the training blocks
    deliberately leave it unset so training is open-book on k=0."""
    assert cfg[evalkey]["green_episode_offset_range"] == 44950


@pytest.mark.parametrize("trainkey", sorted(PAIRS.values()))
def test_training_blocks_stay_open_book(cfg, trainkey):
    assert "green_episode_offset_range" not in cfg[trainkey]


def test_the_two_eval_blocks_differ_only_in_crd_and_identity(cfg):
    """The matched-arm property must survive the copy."""
    a = cfg["experiment_g1eval_knSV3b"]
    b = cfg["experiment_g1eval_matchedvan"]
    observed = {k for k in set(a) | set(b)
                if typed_ne(a.get(k, _MISSING), b.get(k, _MISSING))}
    assert observed == {"crd", "experiment_name", "simulation_name"}, sorted(observed)


def test_stale_probe_blocks_are_not_used_for_g1():
    """experiment_p0cprobe_van is the stale block. It must not appear in any G1
    runner."""
    for script in (ROOT.parent / "g1").glob("*.sh"):
        assert "p0cprobe" not in script.read_text(), script.name
