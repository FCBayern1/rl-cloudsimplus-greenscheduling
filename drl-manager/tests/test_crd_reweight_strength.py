"""Tests for the CRD reweight-strength diagnostics (rho dispersion logging)."""
import torch
import pytest
from src.learners.crd_q_loss import (
    COL_CRD_RHO_ROUTING, COL_CRD_R_ROUTING, COL_CRD_RHO_SCHEDULING,
)


class _Metrics:
    def __init__(self): self.logged = {}
    def log_dict(self, d, key=None, window=None): self.logged.update(d)


class _Learner:
    """Minimal stand-in exposing only what _log_crd_diagnostics touches."""
    _crd_diag_warned = False
    def __init__(self): self.metrics = _Metrics()


def _run(batch):
    from src.learners.crd_q_loss import CRDPPOTorchLearner as M
    lrn = _Learner()
    M._log_crd_diagnostics(lrn, module_id="global_policy", batch=batch)
    return lrn.metrics.logged


def test_uniform_rho_reports_zero_reweight_strength():
    """All-equal rho => w = rho/mean(rho) == 1 => the mechanism is a no-op."""
    rho = torch.full((64,), 0.9)
    got = _run({COL_CRD_RHO_ROUTING: rho, COL_CRD_R_ROUTING: torch.zeros(64)})
    assert got["crd/reweight_w_std"] == pytest.approx(0.0, abs=1e-6)
    assert got["crd/rho_routing_std"] == pytest.approx(0.0, abs=1e-6)


def test_dispersed_rho_reports_nonzero_reweight_strength():
    rho = torch.cat([torch.full((32,), 0.6), torch.full((32,), 1.0)])
    got = _run({COL_CRD_RHO_ROUTING: rho, COL_CRD_R_ROUTING: torch.zeros(64)})
    assert got["crd/reweight_w_std"] > 0.2
    assert got["crd/rho_routing_p10"] < got["crd/rho_routing_p90"]


def test_percentiles_bracket_the_values():
    rho = torch.linspace(0.05, 1.0, 100)
    got = _run({COL_CRD_RHO_ROUTING: rho, COL_CRD_R_ROUTING: torch.zeros(100)})
    assert 0.05 <= got["crd/rho_routing_p10"] < got["crd/rho_routing_p90"] <= 1.0
    assert got["crd/reweight_w_max"] > 1.0


def test_scheduling_module_uses_scheduling_share():
    """Without R_routing the module is a local scheduler -> use rho_scheduling."""
    got = _run({COL_CRD_RHO_SCHEDULING: torch.cat(
        [torch.full((16,), 0.2), torch.full((16,), 0.8)])})
    assert "crd/reweight_w_std" in got and got["crd/reweight_w_std"] > 0.1


def test_single_element_column_is_skipped():
    got = _run({COL_CRD_RHO_ROUTING: torch.tensor([0.9]),
                COL_CRD_R_ROUTING: torch.zeros(1)})
    assert "crd/reweight_w_std" not in got


def test_diagnostics_survive_a_broken_rho_column():
    """A column that trips the stats path must not stop training."""
    got = _run({COL_CRD_RHO_ROUTING: torch.tensor([float("nan")] * 8),
                COL_CRD_R_ROUTING: torch.zeros(8)})
    # no exception; the mean-based metrics may be absent but the call returned
    assert isinstance(got, dict)
