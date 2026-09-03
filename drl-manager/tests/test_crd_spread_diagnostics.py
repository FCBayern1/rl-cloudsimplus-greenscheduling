"""crd/dr_std, crd/dq_std and their p10/p90 are logged (Stage D health gate liveness)."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("ray")
from src.learners import crd_q_loss as cq  # noqa: E402


class _Metrics:
    def __init__(self):
        self.logged = {}

    def log_dict(self, d, key=None, window=None):
        self.logged.update(d)


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {
        cq.COL_CRD_DR: torch.randn(n, generator=g) * 0.5,
        cq.COL_CRD_DQ: torch.randn(n, generator=g) * 0.2,
        cq.COL_CRD_RHO_ROUTING: torch.rand(n, generator=g),
        cq.COL_CRD_RHO_FORECAST: torch.rand(n, generator=g) * 0.3,
        cq.COL_CRD_R_ROUTING: torch.randn(n, generator=g),
        cq.COL_CRD_SIGMA2: torch.rand(n, generator=g),
        cq.COL_CRD_C_T: torch.rand(n, generator=g),
    }
    return b


def test_spread_of_dr_and_dq_is_logged():
    stub = SimpleNamespace(metrics=_Metrics())
    batch = _batch()
    cq.CRDPPOTorchLearner._log_crd_diagnostics(stub, module_id="global_agent", batch=batch)
    got = stub.metrics.logged
    for key in ("crd/dr_mean", "crd/dr_std", "crd/dr_p10", "crd/dr_p90",
                "crd/dq_mean", "crd/dq_std", "crd/dq_p10", "crd/dq_p90"):
        assert key in got, key
    assert abs(got["crd/dr_std"] - batch[cq.COL_CRD_DR].std(unbiased=False).item()) < 1e-6
    assert got["crd/dr_p10"] < got["crd/dr_mean"] < got["crd/dr_p90"]
    # the rho reweight keys are the rho column's, not overwritten by dr/dq
    assert "crd/reweight_w_std" in got


def test_constant_dr_has_zero_spread():
    stub = SimpleNamespace(metrics=_Metrics())
    batch = _batch()
    batch[cq.COL_CRD_DR] = torch.full((64,), 0.25)
    cq.CRDPPOTorchLearner._log_crd_diagnostics(stub, module_id="global_agent", batch=batch)
    assert stub.metrics.logged["crd/dr_std"] == 0.0
