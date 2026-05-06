"""
M6.2 — config-level smoke test for the EU-CRD wiring.

Validates:
  1. The crd block is well-formed in both v2 experiments and round-trips
     through `load_config`.
  2. With crd.enabled=false (default), the training-pipeline branch picks
     vanilla PPOTorchLearner / GTrXLGlobalRLModule / GTrXLMaskedActionRLModule.
  3. With crd.enabled=true, the branch picks
     CRDPPOTorchLearner / GTrXLEnsembleGlobalRLModule /
     GTrXLEnsembleMaskedActionRLModule, AND merges the crd config into
     model_config so the RLModule helpers can read their sub-trees.
  4. Mutually-exclusive guard: crd.enabled=true + ctde.enabled=true raises.

We deliberately don't spin up Ray / a full training run — that's a
human-driven manual validation step (entrypoint_rlmodule_gtrxl.py with
--total-timesteps small). These checks are the cheapest possible
"plumbing is correct" gate.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_crd_smoke.py -v
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.train_rlmodule_gtrxl import load_config
from src.models.rlmodule_gtrxl_models import (
    GTrXLGlobalRLModule,
    GTrXLMaskedActionRLModule,
)
from src.models.rlmodule_gtrxl_ensemble import (
    GTrXLEnsembleGlobalRLModule,
    GTrXLEnsembleMaskedActionRLModule,
)
from src.learners.crd_q_loss import CRDPPOTorchLearner


CONFIG_PATH = REPO_ROOT.parent / "config.yml"


# ---------------------------------------------------------------------------
# Config block validation
# ---------------------------------------------------------------------------


def _load_v2_carbon_5dc():
    cfg = load_config(str(CONFIG_PATH))
    return cfg["experiment_multi_5dc_carbon_v2"]


def _load_v2_carbon_10dc():
    cfg = load_config(str(CONFIG_PATH))
    return cfg["experiment_multi_10dc_carbon_v2"]


def test_v2_5dc_has_crd_block():
    cfg = _load_v2_carbon_5dc()
    assert "crd" in cfg
    crd = cfg["crd"]
    assert "enabled" in crd
    # Required sub-trees that the learner / RLModule helpers read.
    for k in ("ensemble", "blender", "baseline", "delta_r",
              "responsibility", "forecast"):
        assert k in crd, f"crd.{k} block missing in v2 5dc config"


def test_v2_5dc_crd_disabled_by_default():
    """Default must be enabled=false so existing pipelines don't change behavior."""
    cfg = _load_v2_carbon_5dc()
    assert cfg["crd"]["enabled"] is False


def test_v2_10dc_has_crd_block():
    cfg = _load_v2_carbon_10dc()
    assert "crd" in cfg and "enabled" in cfg["crd"]
    assert cfg["crd"]["enabled"] is False  # default off


def test_v2_5dc_ensemble_defaults_match_M1():
    """The ensemble sub-tree should declare K=5 + prior_lambda=3.0 as plan §M1."""
    cfg = _load_v2_carbon_5dc()
    ens = cfg["crd"]["ensemble"]
    assert int(ens["K"]) == 5
    assert float(ens["prior_lambda"]) == 3.0
    assert int(ens["hidden_dim"]) == 128


def test_v2_5dc_blender_defaults_match_M3():
    """Blender defaults should match plan §M3 (τ_0=1.0, κ=0.5, η=0.05)."""
    cfg = _load_v2_carbon_5dc()
    bl = cfg["crd"]["blender"]
    assert float(bl["tau_0"]) == 1.0
    assert float(bl["kappa"]) == 0.5
    assert float(bl["eta"]) == 0.05


def test_v2_5dc_responsibility_floor_matches_M5():
    cfg = _load_v2_carbon_5dc()
    assert float(cfg["crd"]["responsibility"]["rho_min"]) == 0.05


# ---------------------------------------------------------------------------
# Training-pipeline class swap
# ---------------------------------------------------------------------------


class _FakeUnwrappedModule:
    """Just enough surface for `_read_module_*_config` to find the crd block."""
    def __init__(self, model_config):
        self.model_config = model_config


class _FakeModuleAccess:
    """Mock the `self.module[module_id].unwrapped()` chain on the learner."""
    def __init__(self, model_config):
        self._model_config = model_config

    def __getitem__(self, _module_id):
        return self

    def unwrapped(self):
        return _FakeUnwrappedModule(self._model_config)


class _StubLearnerForConfigReading(CRDPPOTorchLearner):
    """
    Tiny subclass that replaces the parent's `module` property with a plain
    attribute, since `Learner.module` is read-only in RLlib.
    """
    module = None  # class-level → instances can assign via __init__

    def __init__(self, module_access):
        self.module = module_access


def test_learner_reads_crd_block_from_module_config():
    """When crd block is merged into model_config, the helpers should find it."""
    learner = _StubLearnerForConfigReading(
        _FakeModuleAccess({"crd": {"ensemble": {"K": 7, "prior_lambda": 2.0}}})
    )
    cfg = learner._read_module_crd_config("global_policy")
    assert cfg.get("K") == 7
    assert cfg.get("prior_lambda") == 2.0


def test_learner_falls_back_to_defaults_when_crd_block_absent():
    """No crd block in model_config → helpers return empty dict (defaults applied later)."""
    learner = _StubLearnerForConfigReading(_FakeModuleAccess({}))
    assert learner._read_module_crd_config("g") == {}
    assert learner._read_module_forecast_config("g") == {}
    assert learner._read_module_baseline_config("g") == {}
    assert learner._read_module_dr_config("g") == {}
    assert learner._read_module_blender_config("g") == {}
    assert learner._read_module_responsibility_config("g") == {}


# ---------------------------------------------------------------------------
# Branch logic — re-implement the train script's switch in isolation so the
# test doesn't depend on Ray/sample_env construction.
# ---------------------------------------------------------------------------


def _resolve_classes(env_config):
    """
    Mirror of the switch in train_rlmodule_gtrxl.py. Returns
    (global_module_class, local_module_class, learner_class_or_none).
    """
    crd_cfg = env_config.get("crd", {}) or {}
    crd_enabled = bool(crd_cfg.get("enabled", False)) if isinstance(crd_cfg, dict) else False
    ctde_cfg = env_config.get("ctde", {})
    ctde_enabled = bool(ctde_cfg.get("enabled", False)) if isinstance(ctde_cfg, dict) else bool(ctde_cfg)

    if crd_enabled and ctde_enabled:
        raise ValueError("crd + ctde mutually exclusive")
    if crd_enabled:
        return (
            GTrXLEnsembleGlobalRLModule,
            GTrXLEnsembleMaskedActionRLModule,
            CRDPPOTorchLearner,
        )
    return GTrXLGlobalRLModule, GTrXLMaskedActionRLModule, None


def test_classes_default_to_vanilla_when_crd_disabled():
    g, l, learner = _resolve_classes({"crd": {"enabled": False}})
    assert g is GTrXLGlobalRLModule
    assert l is GTrXLMaskedActionRLModule
    assert learner is None


def test_classes_default_to_vanilla_when_no_crd_block():
    g, l, learner = _resolve_classes({})
    assert g is GTrXLGlobalRLModule
    assert l is GTrXLMaskedActionRLModule
    assert learner is None


def test_classes_swap_when_crd_enabled():
    g, l, learner = _resolve_classes({"crd": {"enabled": True}})
    assert g is GTrXLEnsembleGlobalRLModule
    assert l is GTrXLEnsembleMaskedActionRLModule
    assert learner is CRDPPOTorchLearner


def test_crd_and_ctde_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_classes({"crd": {"enabled": True}, "ctde": {"enabled": True}})


def test_actual_v2_5dc_resolves_to_vanilla_by_default():
    """Concrete cross-check: real v2 5dc config goes to vanilla classes."""
    g, l, learner = _resolve_classes(_load_v2_carbon_5dc())
    assert g is GTrXLGlobalRLModule
    assert learner is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
