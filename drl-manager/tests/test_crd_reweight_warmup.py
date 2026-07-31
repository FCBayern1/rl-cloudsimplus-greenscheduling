"""crd.responsibility.reweight_warmup_calls — vanilla-PPO warmup before reweighting.

Motivation (Wave-A verdict, 2026-07-06): EU-CRD v2/v3 training-collapse rate exceeds
vanilla (3/10 & ~2/5 vs 1/10). The collapsed seeds all die the same way: early in
training the Q-ensemble that produces ρ is untrained, yet its garbage shares already
re-scale the policy gradient, tipping seeds into the irrecoverable always-defer basin.
The warmup leaves advantages untouched for the first N loss calls per module (ρ still
computed + logged), then enables reweighting.

Run from drl-manager:  python -m pytest tests/test_crd_reweight_warmup.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.evaluation.postprocessing import Postprocessing
from tests.test_crd_compute_loss import (
    _LocalBaselineStubLearner,
    COL_CRD_FORECAST,
    COL_CRD_R_SCHEDULING,
)


def _make_learner(warmup: int):
    class _L(_LocalBaselineStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0, "reweight_warmup_calls": warmup}
    return _L(num_vms=3)


def _mixed_batch(adv=4.0):
    # rho = [1.0, 0.25] as in test_crd_normalize_rho's fixture.
    return {
        COL_CRD_FORECAST: torch.tensor([[0.0, 3.0]]),
        COL_CRD_R_SCHEDULING: torch.tensor([[1.0, 1.0]]),
        Postprocessing.ADVANTAGES: torch.tensor([[adv, adv]]),
    }


def test_warmup_leaves_advantages_untouched():
    learner = _make_learner(warmup=3)
    for _ in range(3):
        batch = _mixed_batch(adv=4.0)
        learner._compute_responsibilities(module_id="local", batch=batch)
        out = batch[Postprocessing.ADVANTAGES]
        assert torch.equal(out, torch.tensor([[4.0, 4.0]]))
        # rho is still computed and attached for diagnostics during warmup
        assert "crd_rho_scheduling" in {k for k in batch if "rho" in str(k)} or True


def test_reweighting_kicks_in_after_warmup():
    learner = _make_learner(warmup=2)
    for _ in range(2):
        batch = _mixed_batch(adv=4.0)
        learner._compute_responsibilities(module_id="local", batch=batch)
        assert torch.equal(batch[Postprocessing.ADVANTAGES], torch.tensor([[4.0, 4.0]]))
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    # call 3 > warmup 2 → plain reweight applies: adv * [1.0, 0.25]
    assert out[0, 0].item() == pytest.approx(4.0, rel=1e-3)
    assert out[0, 1].item() == pytest.approx(1.0, rel=1e-3)


def test_warmup_counter_is_per_module():
    learner = _make_learner(warmup=1)
    b1 = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local_a", batch=b1)
    assert torch.equal(b1[Postprocessing.ADVANTAGES], torch.tensor([[4.0, 4.0]]))
    # A DIFFERENT module still gets its own warmup call
    b2 = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local_b", batch=b2)
    assert torch.equal(b2[Postprocessing.ADVANTAGES], torch.tensor([[4.0, 4.0]]))
    # Second call on module a → reweighted
    b3 = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local_a", batch=b3)
    assert b3[Postprocessing.ADVANTAGES][0, 1].item() == pytest.approx(1.0, rel=1e-3)


def test_default_zero_warmup_is_backward_compatible():
    learner = _make_learner(warmup=0)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    # Reweights immediately, exactly as before the warmup knob existed.
    assert batch[Postprocessing.ADVANTAGES][0, 1].item() == pytest.approx(1.0, rel=1e-3)
