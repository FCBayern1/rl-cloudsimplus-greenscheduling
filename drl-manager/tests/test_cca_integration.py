"""Integration test: the learner-side `_apply_cca` hook rewrites ADVANTAGES
in place using the hindsight baseline, and is a strict no-op when disabled.

We bind the real `_apply_cca` + `_extract_actual_green_bt_d` onto a light stand-in
so no Ray Learner / RLModule bring-up is needed."""
import sys
from pathlib import Path

import torch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.core.columns import Columns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.learners.crd_q_loss import CRDPPOTorchLearner


class _FakeLearner:
    _extract_actual_green_bt_d = staticmethod(
        CRDPPOTorchLearner._extract_actual_green_bt_d
    )
    _apply_cca = CRDPPOTorchLearner._apply_cca

    def __init__(self, cfg):
        self._cfg = cfg
        self._cca_nets = {}
        self._cca_opts = {}

    def _read_module_cca_config(self, module_id):
        return self._cfg


def _make_batch(B=8, T=16, D=3):
    torch.manual_seed(0)
    green = torch.rand(B, T, D) * 5.0
    # Return = small controllable effect + large exogenous future-green signal.
    future_green_sum = torch.zeros(B, T)
    for t in range(T):
        nxt = green[:, t + 1 : min(t + 13, T), :]
        if nxt.numel():
            future_green_sum[:, t] = nxt.mean(dim=(1, 2))
    action_effect = torch.randn(B, T) * 0.3
    returns = action_effect + 4.0 * future_green_sum
    adv = returns - returns.mean()  # stand-in GAE advantage (carries green noise)
    return {
        Postprocessing.ADVANTAGES: adv.clone(),
        Postprocessing.VALUE_TARGETS: returns,
        Columns.VF_PREDS: torch.zeros(B, T),
        Columns.OBS: {"crd_aux": {"crd_actual_green_w": green}},
    }, adv


def test_apply_cca_disabled_is_noop():
    learner = _FakeLearner({"enabled": False})
    batch, adv0 = _make_batch()
    learner._apply_cca(module_id="default_policy", batch=batch)
    assert torch.equal(batch[Postprocessing.ADVANTAGES], adv0)


def test_apply_cca_absent_green_is_noop():
    learner = _FakeLearner({"enabled": True, "horizon": 12})
    batch, adv0 = _make_batch()
    del batch[Columns.OBS]["crd_aux"]  # no green snapshot → leave advantage alone
    learner._apply_cca(module_id="default_policy", batch=batch)
    assert torch.equal(batch[Postprocessing.ADVANTAGES], adv0)


def test_apply_cca_enabled_reduces_advantage_variance():
    learner = _FakeLearner({"enabled": True, "horizon": 12, "lr": 1e-2, "hidden": 64})
    batch, adv0 = _make_batch()
    # Train the hindsight net over several minibatch passes (as PPO epochs would).
    for _ in range(200):
        b = {
            Postprocessing.ADVANTAGES: adv0.clone(),
            Postprocessing.VALUE_TARGETS: batch[Postprocessing.VALUE_TARGETS],
            Columns.VF_PREDS: batch[Columns.VF_PREDS],
            Columns.OBS: batch[Columns.OBS],
        }
        learner._apply_cca(module_id="default_policy", batch=b)
        new_adv = b[Postprocessing.ADVANTAGES]
    assert new_adv.shape == adv0.shape
    # The hindsight baseline strips the exogenous green variance.
    assert new_adv.var().item() < 0.5 * adv0.var().item(), (
        f"cca var {new_adv.var().item():.3f} not << raw {adv0.var().item():.3f}"
    )
    # Per-module net/optimizer were created and cached.
    assert "default_policy" in learner._cca_nets
    assert "default_policy" in learner._cca_opts


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
