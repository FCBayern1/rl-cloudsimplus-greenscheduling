"""Control arms differ from their parent by exactly one leaf, plus identity.

Two hand-copied configs failed on 2026-08-23. The evaluation block for the
Vanilla arm was a stale pre-v5 copy differing in twelve keys, eight of them
objective-level. And every comparator arm carried
`per_action_carbon_weight = 0.5` against the main arms' 0.25, so each was
optimising a different objective from the pair it was compared with, which is
the same defect that invalidated the original Vanilla comparator. These tests
assert the property that failed, not the process that produced it.
"""
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT.parent / "config_C.yml"
CONTROLS = ROOT.parent / "config_controls.yml"

VAN = "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan"
EU = "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b"
IDENTITY = {"experiment_name", "simulation_name", "wandb"}
_MISSING = object()

# arm -> (parent, the single top-level key it is allowed to change beyond identity)
EXPECTED = {
    "experiment_ctrl_risk_cvar":  (VAN, "crd"),
    "experiment_ctrl_risk_mv":    (VAN, "crd"),
    "experiment_ctrl_risk_rs":    (VAN, "crd"),
    "experiment_ctrl_risk_dcvar": (VAN, "crd"),
    "experiment_ctrl_noforecast": (VAN, "forecast_mode"),
    "experiment_ctrl_ablG":       (EU,  "crd"),
    "experiment_ctrl_ablW":       (EU,  "crd"),
}
# arm -> the single crd subtree it is allowed to change
CRD_SUBTREE = {
    "experiment_ctrl_risk_cvar": "risk",
    "experiment_ctrl_risk_mv": "risk",
    "experiment_ctrl_risk_rs": "risk",
    "experiment_ctrl_risk_dcvar": "risk",
    "experiment_ctrl_ablG": "blender",
    "experiment_ctrl_ablW": "responsibility",
}


def typed_ne(a, b):
    if a is _MISSING or b is _MISSING:
        return True
    if (a is None) != (b is None):
        return True
    if isinstance(a, bool) != isinstance(b, bool):
        return True
    return a != b


def diff(a, b):
    return {k for k in set(a) | set(b) if typed_ne(a.get(k, _MISSING), b.get(k, _MISSING))}


@pytest.fixture(scope="module")
def cfg():
    base = yaml.safe_load(CONFIG.read_text())
    assert CONTROLS.is_file(), "run drl-manager/make_control_arms.py"
    base.update(yaml.safe_load(CONTROLS.read_text()))
    return base


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_arm_differs_from_parent_by_one_key_plus_identity(cfg, arm):
    parent, allowed = EXPECTED[arm]
    observed = diff(cfg[arm], cfg[parent])
    assert observed <= IDENTITY | {allowed}, sorted(observed - IDENTITY - {allowed})
    assert allowed in observed, f"{arm} does not actually change {allowed}"


@pytest.mark.parametrize("arm", sorted(CRD_SUBTREE))
def test_only_one_crd_subtree_moves(cfg, arm):
    parent, _ = EXPECTED[arm]
    observed = diff(cfg[arm]["crd"], cfg[parent]["crd"])
    assert observed == {CRD_SUBTREE[arm]}, sorted(observed)


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_carbon_weight_matches_the_parent(cfg, arm):
    """The regression this suite exists for: the old comparator arms ran at
    per_action_carbon_weight 0.5 against the main arms' 0.25."""
    parent, _ = EXPECTED[arm]
    assert cfg[arm]["per_action_carbon_weight"] == cfg[parent]["per_action_carbon_weight"]
    assert cfg[arm]["per_action_carbon_weight"] == 0.25


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_objective_family_is_untouched(cfg, arm):
    """The eight parameters that invalidated the original Vanilla comparator."""
    parent, _ = EXPECTED[arm]
    for key in ("carbon_penalty_mode", "carbon_normalization_mode",
                "carbon_normalization_fixed_max", "global_reward_alpha",
                "global_reward_beta", "global_reward_gamma",
                "global_completion_rate_mi_coef", "per_action_completion_weight"):
        assert not typed_ne(cfg[arm].get(key, _MISSING), cfg[parent].get(key, _MISSING)), key


def test_ablations_descend_from_eucrd_not_vanilla(cfg):
    """An ablation removes a component, so its parent must be the arm that has
    the component."""
    for arm in ("experiment_ctrl_ablG", "experiment_ctrl_ablW"):
        assert cfg[arm]["crd"]["enabled"] is True, arm
    assert cfg["experiment_ctrl_ablG"]["crd"]["blender"]["fixed_c"] == 1.0
    assert cfg["experiment_ctrl_ablW"]["crd"]["responsibility"]["reweight_advantages"] is False


def test_risk_and_noforecast_descend_from_vanilla(cfg):
    for arm in ("experiment_ctrl_risk_cvar", "experiment_ctrl_risk_mv",
                "experiment_ctrl_risk_rs", "experiment_ctrl_risk_dcvar",
                "experiment_ctrl_noforecast"):
        assert cfg[arm]["crd"]["enabled"] is False, arm


def test_risk_kinds_are_the_four_declared_objectives(cfg):
    got = {a: cfg[a]["crd"]["risk"]["kind"] for a in sorted(CRD_SUBTREE) if "risk" in a}
    assert set(got.values()) == {"cvar", "mean_variance", "risk_sensitive", "dist_cvar"}


def test_noforecast_changes_only_the_forecast_channel(cfg):
    """Codex's C2 ruling: only forecast_mode and identity. The old arm also
    switched green_oracle_mode to godeye, a second uncontrolled difference."""
    a, b = cfg["experiment_ctrl_noforecast"], cfg[VAN]
    assert a["forecast_mode"] == "none"
    assert a.get("green_oracle_mode") == b.get("green_oracle_mode")


def test_controls_are_not_in_the_frozen_config():
    """G1 re-reads config_C.yml on every training and the manifest pins its
    hash, so the control arms stay out until G1 is done."""
    frozen = yaml.safe_load(CONFIG.read_text())
    for arm in EXPECTED:
        assert arm not in frozen, arm


def test_training_arms_stay_open_book(cfg):
    for arm in EXPECTED:
        assert "green_episode_offset_range" not in cfg[arm], arm
