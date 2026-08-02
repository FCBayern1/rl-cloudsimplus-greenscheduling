"""crd.responsibility.reweight_advantages=false — ablation bypass.

The ablation arm "-reweight" keeps the full CRD pipeline (rho computed and
logged) but must NOT touch the advantages: with the switch off the policy
gradient is byte-identical to vanilla PPO. This test pins that contract so
the ablation arm measures only the reweighting, not an accidental side effect.

Run from drl-manager:  python -m pytest tests/test_crd_reweight_advantages_off.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.evaluation.postprocessing import Postprocessing
from tests.test_crd_compute_loss import (  # reuse the stub learner fixture
    _LocalBaselineStubLearner,
    COL_CRD_FORECAST,
    COL_CRD_R_SCHEDULING,
)


def _make_learner(reweight: bool):
    class _L(_LocalBaselineStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {
                "rho_min": 0.0,
                "normalize_rho": True,
                "reweight_advantages": reweight,
            }
    return _L(num_vms=3)


def _mixed_batch(adv=4.0):
    # One own-responsibility transition (rho=1), one forecast-dominated
    # (rho=0.25) — with reweighting ON these advantages would diverge.
    return {
        COL_CRD_FORECAST: torch.tensor([[0.0, 3.0]]),
        COL_CRD_R_SCHEDULING: torch.tensor([[1.0, 1.0]]),
        Postprocessing.ADVANTAGES: torch.tensor([[adv, adv]]),
    }


def test_reweight_off_leaves_advantages_untouched():
    learner = _make_learner(reweight=False)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    assert out[0, 0].item() == pytest.approx(4.0, rel=1e-6)
    assert out[0, 1].item() == pytest.approx(4.0, rel=1e-6)


def test_reweight_on_changes_advantages():
    # Sanity inverse: same batch with the switch on must NOT be identity,
    # otherwise the test above would pass vacuously.
    learner = _make_learner(reweight=True)
    batch = _mixed_batch(adv=4.0)
    learner._compute_responsibilities(module_id="local", batch=batch)
    out = batch[Postprocessing.ADVANTAGES]
    assert out[0, 0].item() != pytest.approx(out[0, 1].item(), rel=1e-3)
