"""The matched-Vanilla arm must differ from knSV3b by EXACTLY three keys.

Standing guard, not a one-off check (Codex P0-B, 2026-08-23). The previous
Vanilla arm was an invalid comparator because it differed from knSV3b in eight
objective-level parameters; if anyone later tunes one arm's reward, gamma,
entropy or trace and forgets the other, the main-arm comparison silently
becomes unfair again.
"""
import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from verify_matched_vanilla import (ALLOWED_DIFFS, CFG, EU, RUN_MANIFEST_KEYS,  # noqa: E402
                                    VAN, diff_keys, flat, typed_ne)


@pytest.fixture(scope="module")
def arms():
    cfg = yaml.safe_load(open(CFG))
    assert EU in cfg and VAN in cfg, "both experiment keys must exist in config_C.yml"
    return flat(cfg[EU]), flat(cfg[VAN])


def test_difference_set_is_exactly_the_admissible_set(arms):
    eu, van = arms
    assert diff_keys(eu, van) == ALLOWED_DIFFS


def test_crd_switch_is_a_boolean_flip(arms):
    eu, van = arms
    assert eu.get("crd.enabled") is True
    assert van.get("crd.enabled") is False


def test_vanilla_keeps_the_full_crd_subtree_inert(arms):
    eu, van = arms
    assert {k for k in eu if k.startswith("crd.")} == {k for k in van if k.startswith("crd.")}


@pytest.mark.parametrize("key", [
    # the eight that made the OLD Vanilla an invalid comparator - pinned explicitly
    "carbon_penalty_mode", "carbon_normalization_fixed_max",
    "global_completion_rate_mi_coef", "global_model.ent_coef", "global_model.gamma",
    "global_reward_beta", "per_action_carbon_weight", "per_action_completion_weight",
])
def test_objective_parameters_match(arms, key):
    eu, van = arms
    assert not typed_ne(eu.get(key), van.get(key)), f"{key} differs between arms"


def test_same_trace_and_green_scaling(arms):
    eu, van = arms
    for key in ("cloudlet_trace_file", "compressed_power_divisor"):
        assert not typed_ne(eu.get(key), van.get(key))


@pytest.mark.parametrize("key", sorted(RUN_MANIFEST_KEYS))
def test_run_manifest_keys_stay_out_of_semantic_config(arms, key):
    eu, van = arms
    assert key not in eu and key not in van


def test_typed_comparison_rejects_string_number_confusion():
    """A prior version compared str(value); "1" and 1 must not look equal."""
    assert typed_ne(1, "1")
    assert typed_ne(None, "None")
    assert typed_ne(True, 1)
    assert not typed_ne(1, 1.0)

