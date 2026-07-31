"""crd.local_cf_enabled — ablation gate that switches the LOCAL counterfactual off.

Motivation (2026-07-18): method text v5 claims the local scheduler is trained
conventionally (it never observes the forecast), but the code ran the full M4
pipeline (BestFit CF + R_scheduling reweight + local Q-loss) in every v5.2 arm.
This knob makes "no local CF" a real, config-gated arm so the claim can be
tested as an ablation instead of silently diverging from the code.

Semantics:
  - default (knob absent / true): bit-identical to v5.2 — M4 fully active;
  - local_cf_enabled=false + Discrete-action module: the ENTIRE local CF
    pipeline is skipped (no forecast CF, no baseline, no R_scheduling, no
    advantage reweight) and the module's Q-ensemble TD loss is skipped too —
    the local layer trains as conventional PPO;
  - the global router (MultiDiscrete) is NEVER gated, whatever the knob says.

Run from drl-manager:  .venv/bin/python -m pytest tests/test_crd_local_cf_gate.py -v
"""
import sys
from pathlib import Path

import torch
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from src.learners.crd_q_loss import (
    COL_CRD_FORECAST,
    CRDPPOTorchLearner,
)
from tests.test_crd_compute_loss import (
    _StubLearner,
    _ensemble_fwd_out,
    _make_crd_info,
)


class _FakeModule:
    def __init__(self, model_config, action_space):
        self.model_config = model_config
        self.action_space = action_space

    def unwrapped(self):
        return self


def _make_learner(local_cf_enabled=None, action_space=None):
    """StubLearner + a fake self.module so the REAL config/action-space
    readers run (the base stub overrides them away)."""
    learner = _StubLearner()
    crd_cfg = {"forecast": {"beta": 0.5, "gamma": 0.3}}
    if local_cf_enabled is not None:
        crd_cfg["local_cf_enabled"] = local_cf_enabled
    learner._module = {
        "m": _FakeModule(
            {"crd": crd_cfg},
            action_space if action_space is not None else spaces.Discrete(5),
        )
    }
    # Un-stub the readers under test (base stub hardcodes them).
    learner._read_crd_local_cf_enabled = (
        lambda mid: CRDPPOTorchLearner._read_crd_local_cf_enabled(learner, mid)
    )
    return learner


def _forecast_batch():
    infos = [_make_crd_info(pred=[1.0, 1.0, 1.0]) for _ in range(4)]
    return {
        Columns.INFOS: infos,
        Postprocessing.ADVANTAGES: torch.full((1, 4), 4.0),
    }


# ---------------------------------------------------------------------------
# predicate
# ---------------------------------------------------------------------------


def test_predicate_fires_only_for_disabled_local_module():
    learner = _make_learner(local_cf_enabled=False, action_space=spaces.Discrete(5))
    assert learner._skip_local_cf("m") is True


def test_predicate_false_by_default():
    learner = _make_learner(local_cf_enabled=None, action_space=spaces.Discrete(5))
    assert learner._skip_local_cf("m") is False


def test_predicate_false_for_global_module_even_when_disabled():
    learner = _make_learner(
        local_cf_enabled=False, action_space=spaces.MultiDiscrete([6] * 10)
    )
    assert learner._skip_local_cf("m") is False


def test_predicate_false_when_module_uninspectable():
    """Missing module → gate must fail closed (never skip)."""
    learner = _StubLearner()
    learner._module = {}
    learner._read_crd_local_cf_enabled = lambda mid: False  # knob off ...
    # ... but the module can't be identified as local → no skip.
    assert learner._skip_local_cf("ghost") is False


# ---------------------------------------------------------------------------
# _compute_crd_terms gating
# ---------------------------------------------------------------------------


def test_disabled_local_module_skips_entire_cf_pipeline():
    learner = _make_learner(local_cf_enabled=False)
    batch = _forecast_batch()
    learner._compute_crd_terms(module_id="m", batch=batch, fwd_out=_ensemble_fwd_out())
    # No CF column written, advantages untouched.
    assert COL_CRD_FORECAST not in batch
    assert torch.equal(
        batch[Postprocessing.ADVANTAGES], torch.full((1, 4), 4.0)
    )


def test_default_keeps_pipeline_bit_identical():
    learner = _make_learner(local_cf_enabled=None)
    batch = _forecast_batch()
    learner._compute_crd_terms(module_id="m", batch=batch, fwd_out=_ensemble_fwd_out())
    # Forecast CF runs exactly as before the knob existed.
    assert COL_CRD_FORECAST in batch


def test_explicit_true_keeps_pipeline():
    learner = _make_learner(local_cf_enabled=True)
    batch = _forecast_batch()
    learner._compute_crd_terms(module_id="m", batch=batch, fwd_out=_ensemble_fwd_out())
    assert COL_CRD_FORECAST in batch


def test_disabled_gate_logs_once(caplog):
    import logging

    learner = _make_learner(local_cf_enabled=False)
    with caplog.at_level(logging.INFO, logger="src.learners.crd_q_loss"):
        for _ in range(3):
            learner._compute_crd_terms(
                module_id="m", batch=_forecast_batch(), fwd_out=_ensemble_fwd_out()
            )
    gate_logs = [r for r in caplog.records if "local_cf_enabled=false" in r.message]
    assert len(gate_logs) == 1


# ---------------------------------------------------------------------------
# Q-loss skip (compute_loss_for_module path)
# ---------------------------------------------------------------------------


def test_q_loss_skipped_for_disabled_local_module():
    """The gated module's loss must be exactly the base PPO loss (no q_loss)."""

    class _LossProbe(_StubLearner):
        # Real compute_loss_for_module without the heavy PPO super(): emulate
        # its post-super tail exactly the way the production code sequences it.
        def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
            self._compute_crd_terms(module_id=module_id, batch=batch, fwd_out=fwd_out)
            base_loss = torch.tensor(1.0)
            q_ensemble = fwd_out.get("crd_q_ensemble")
            if q_ensemble is None:
                return base_loss
            if self._skip_local_cf(module_id):
                return base_loss
            return base_loss + torch.tensor(100.0)  # stands in for coef*q_loss

    def _probe(enabled):
        learner = _LossProbe()
        crd_cfg = {"forecast": {"beta": 0.5, "gamma": 0.3}}
        if enabled is not None:
            crd_cfg["local_cf_enabled"] = enabled
        learner._module = {
            "m": _FakeModule({"crd": crd_cfg}, spaces.Discrete(5))
        }
        learner._read_crd_local_cf_enabled = (
            lambda mid: CRDPPOTorchLearner._read_crd_local_cf_enabled(learner, mid)
        )
        return learner.compute_loss_for_module(
            module_id="m", config=None, batch=_forecast_batch(),
            fwd_out=_ensemble_fwd_out(),
        )

    assert _probe(enabled=False).item() == 1.0  # gated → base loss only
    assert _probe(enabled=None).item() == 101.0  # default → q_loss added
    assert _probe(enabled=True).item() == 101.0


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


def test_reader_defaults_true_on_missing_module():
    learner = _StubLearner()
    learner._module = {}
    assert (
        CRDPPOTorchLearner._read_crd_local_cf_enabled(learner, "nope") is True
    )
