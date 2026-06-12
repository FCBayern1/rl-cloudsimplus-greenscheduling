"""
P1 wiring: `normalized_critic.enabled: true` in the experiment config must
(a) swap the learner class to NormalizedCriticPPOTorchLearner,
(b) inject the gate into the GLOBAL module's model_config,
(c) leave the LOCAL module ungated by default (healthy critic = smoke
    reference), and
(d) carry the vf_coef/vf_clip_param companion changes to the global module.

Uses spaces_only mode — no JVM, no Ray init.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_normalized_critic_wiring.py -v
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.learners.crd_q_loss import CRDPPOTorchLearner
from src.learners.normalized_critic_loss import NormalizedCriticPPOTorchLearner
from src.training.train_rlmodule_gtrxl import create_rlmodule_config, load_config

CONFIG_YML = REPO_ROOT.parent / "config.yml"
EXPERIMENT = "experiment_multi_5dc_carbon_v2"


@pytest.fixture(scope="module")
def exp_config():
    return load_config(str(CONFIG_YML))[EXPERIMENT]


def _build(exp, env_overrides=None):
    env_config = dict(exp)
    if env_overrides:
        env_config.update(env_overrides)
    return create_rlmodule_config(
        env_config,
        exp.get("global_model", {}),
        exp.get("local_model", {}),
        exp.get("training", {}),
    )


@pytest.fixture(scope="module")
def ppo_config(exp_config):
    return _build(exp_config)


def test_v2_experiment_has_p1_settings(exp_config):
    assert exp_config["normalized_critic"]["enabled"] is True
    # vf_coef stays 10: it compensates the tiny shared lr (6e-5) for the
    # critic via dual-trunk isolation. Smoke 20260612_001518 showed 1.0
    # under-trains (vf_loss flat at ~2.7σ², constant-predictor signature).
    assert exp_config["global_model"]["vf_coef"] == 10.0
    assert exp_config["global_model"]["vf_clip_param"] == 10.0  # σ² units now
    assert exp_config["lagrangian"]["enabled"] is False        # P2 debug freeze


def test_learner_class_swapped(ppo_config):
    assert ppo_config.learner_class is NormalizedCriticPPOTorchLearner


def test_gate_injected_into_global_module_only(ppo_config):
    specs = ppo_config.rl_module_spec.rl_module_specs
    assert specs["global_policy"].model_config["normalized_critic"]["enabled"] is True
    assert "normalized_critic" not in specs["shared_local_policy"].model_config


def test_global_overrides_carry_vf_companions(ppo_config):
    ov = ppo_config.algorithm_config_overrides_per_module["global_policy"]
    assert ov["vf_loss_coeff"] == 10.0
    assert ov["vf_clip_param"] == 10.0


def test_disabled_block_keeps_vanilla_learner(exp_config):
    cfg = _build(exp_config, {"normalized_critic": {"enabled": False}})
    # RLlib's NotProvided sentinel resolves to the vanilla PPO learner —
    # neither of our custom classes must be selected.
    assert cfg.learner_class is not NormalizedCriticPPOTorchLearner
    assert cfg.learner_class is not CRDPPOTorchLearner
    specs = cfg.rl_module_spec.rl_module_specs
    assert "normalized_critic" not in specs["global_policy"].model_config


def test_crd_learner_inherits_normalized_base():
    # The CRD learner must sit on top of the normalized critic so EU-CRD
    # runs honor the same per-module gate without a learner swap.
    assert issubclass(CRDPPOTorchLearner, NormalizedCriticPPOTorchLearner)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
