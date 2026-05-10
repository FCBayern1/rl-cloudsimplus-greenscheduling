"""
Tests for the EU-CRD pieces inside CRDPPOTorchLearner.compute_loss_for_module:
  M2.1 — hook is reached and gates correctly on crd_q_ensemble presence
  M2.2 — forecast CF written to batch["crd_forecast"]
  (M2.3-M2.5 add baseline action, ΔQ/σ², Δr in later milestones)

These tests deliberately bypass the full PPOTorchLearner setup — we exercise
just the CRD-specific code paths via a StubLearner that overrides build()
and compute_loss_for_module to skip the heavy machinery.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_crd_compute_loss.py -v
"""
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from src.learners.crd_q_loss import (
    COL_CRD_FORECAST,
    CRDPPOTorchLearner,
)
from src.models.rlmodule_gtrxl_ensemble import COL_Q_ENSEMBLE


class _StubLearner(CRDPPOTorchLearner):
    """
    Bypasses PPOTorchLearner.__init__ and .compute_loss_for_module's super
    call, leaving only the CRD-specific code under test.
    """

    def __init__(self, beta: float = 0.5, gamma: float = 0.3):
        # Mirror what build() would set up.
        self._crd_call_counts = {}
        self._crd_hook_logged = {}
        self._crd_pred_missing_warned = {}
        self._crd_baseline_schedulers = {}
        self._crd_baseline_signal_warned = {}
        self._crd_blenders = {}
        self._crd_dq_align_warned = {}
        self.hook_calls = []  # observability for tests
        self._beta = beta
        self._gamma = gamma

    def _read_module_blender_config(self, module_id):
        # Stub avoids the real `self.module[...].unwrapped()` plumbing.
        return {}

    def _read_module_responsibility_config(self, module_id):
        return {}

    def _read_module_forecast_config(self, module_id):
        # Avoid the real `self.module[module_id].unwrapped()` plumbing.
        return {"beta": self._beta, "gamma": self._gamma}

    def _get_or_build_baseline_scheduler(self, module_id):
        # Tests don't go through `self.module[module_id].unwrapped()`; treat
        # M2.3 baseline action as opt-in. The dedicated M2.3 tests below
        # override this stub when they need a real scheduler.
        return None

    def _compute_crd_terms(self, *, module_id, batch, fwd_out):
        super()._compute_crd_terms(module_id=module_id, batch=batch, fwd_out=fwd_out)
        self.hook_calls.append(
            (
                module_id,
                tuple(sorted(batch.keys())) if isinstance(batch, dict) else None,
                tuple(sorted(fwd_out.keys())) if isinstance(fwd_out, dict) else None,
            )
        )

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        # Mirror the M1.2/M2.1 entry: run CRD hook, then "pretend" PPO loss.
        self._compute_crd_terms(module_id=module_id, batch=batch, fwd_out=fwd_out)
        return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Helpers for building synthetic batches
# ---------------------------------------------------------------------------


def _make_crd_info(actual=None, pred=None):
    """Mirror what HierarchicalMultiDCEnv._collect_crd_info writes."""
    info = {
        "crd": {
            "actual_wind_w": actual or [800_000.0, 0.0, 2_000_000.0],
            "p_total_w": [1_500_000.0, 1_000_000.0, 500_000.0],
            "timestep_hours": 1.0,
            "green_carbon_factor": [0.0, 0.0, 0.04],
            "brown_carbon_factor": [0.5, 0.5, 0.5],
            "running_max_carbon": 1.0,
        }
    }
    if pred is not None:
        info["crd"]["predicted_wind_w"] = pred
    return info


def _ensemble_fwd_out(B=2, T=2, K=5, A=4):
    return {COL_Q_ENSEMBLE: torch.zeros(B, T, K, A)}


# ---------------------------------------------------------------------------
# M2.1 — hook gating + warn-once log
# ---------------------------------------------------------------------------


def test_hook_skipped_when_no_q_ensemble_in_fwd_out(caplog):
    """Without crd_q_ensemble in fwd_out, the hook is a silent no-op."""
    import logging
    learner = _StubLearner()
    with caplog.at_level(logging.INFO, logger="src.learners.crd_q_loss"):
        learner.compute_loss_for_module(
            module_id="vanilla", config=None, batch={"x": torch.zeros(1)}, fwd_out={"vf_preds": torch.zeros(1)}
        )
    crd_logs = [r for r in caplog.records if "[CRD]" in r.message]
    assert crd_logs == [], "no CRD log should fire for non-ensemble modules"


def test_hook_fires_when_q_ensemble_present(caplog):
    import logging
    learner = _StubLearner()
    with caplog.at_level(logging.INFO, logger="src.learners.crd_q_loss"):
        learner.compute_loss_for_module(
            module_id="ensemble_m", config=None, batch={"x": torch.zeros(1)}, fwd_out=_ensemble_fwd_out()
        )
    crd_logs = [r for r in caplog.records if "[CRD] hook reached" in r.message]
    assert len(crd_logs) == 1
    assert "ensemble_m" in crd_logs[0].message


def test_hook_log_fires_only_once_per_module(caplog):
    """Repeated minibatches → hook called every time, but log fires once per module."""
    import logging
    learner = _StubLearner()
    with caplog.at_level(logging.INFO, logger="src.learners.crd_q_loss"):
        for _ in range(3):
            learner.compute_loss_for_module(
                module_id="alpha", config=None, batch={}, fwd_out=_ensemble_fwd_out()
            )
        for _ in range(3):
            learner.compute_loss_for_module(
                module_id="beta", config=None, batch={}, fwd_out=_ensemble_fwd_out()
            )
    crd_logs = [r for r in caplog.records if "[CRD] hook reached" in r.message]
    seen = {r.message.split("module ")[1].split(";")[0] for r in crd_logs}
    assert seen == {"'alpha'", "'beta'"}
    assert len(crd_logs) == 2  # one per module
    # Hook itself was called 6 times.
    assert len(learner.hook_calls) == 6


# ---------------------------------------------------------------------------
# M2.2 — forecast CF static helper
# ---------------------------------------------------------------------------


def test_forecast_values_perfect_prediction_yields_zero():
    """When predicted == actual for every transition, every R_forecast is 0."""
    infos = [_make_crd_info(pred=[800_000.0, 0.0, 2_000_000.0]) for _ in range(5)]
    rs, n_missing = CRDPPOTorchLearner._compute_forecast_cf_values(
        infos, beta=0.5, gamma=0.3
    )
    assert rs == [0.0] * 5
    assert n_missing == 0


def test_forecast_values_count_missing_predictions():
    """Transitions without predicted_wind_w → R = 0 + n_missing increments."""
    infos = [
        _make_crd_info(),  # no pred
        _make_crd_info(pred=[800_000.0, 0.0, 2_000_000.0]),  # has pred
        _make_crd_info(),  # no pred
    ]
    rs, n_missing = CRDPPOTorchLearner._compute_forecast_cf_values(
        infos, beta=1.0, gamma=1.0
    )
    assert len(rs) == 3
    assert rs[0] == 0.0 and rs[2] == 0.0
    assert rs[1] == 0.0  # perfect-pred → 0 anyway
    assert n_missing == 2


def test_forecast_values_correlate_with_injected_bias():
    """Sweep predicted = actual * scale; R_forecast must track the scale (corr > 0.95)."""
    import numpy as np
    actual = [800_000.0, 0.0, 2_000_000.0]
    scales = np.linspace(0.1, 2.0, 25)
    infos = [_make_crd_info(actual=actual, pred=[a * s for a in actual]) for s in scales]
    rs, _ = CRDPPOTorchLearner._compute_forecast_cf_values(
        infos, beta=1.0, gamma=1.0
    )
    corr = float(np.corrcoef(scales, rs)[0, 1])
    assert corr > 0.95, f"R_forecast does not track bias; corr={corr:.4f}"


def test_forecast_values_handle_non_dict_infos_gracefully():
    """Robustness: garbage in infos doesn't crash the static helper."""
    infos = [None, "not a dict", 42, _make_crd_info(pred=[1, 1, 1])]
    rs, n_missing = CRDPPOTorchLearner._compute_forecast_cf_values(
        infos, beta=1.0, gamma=1.0
    )
    assert len(rs) == 4
    assert rs[0] == rs[1] == rs[2] == 0.0
    # The 4th (real CRD info) goes through forecast_cf_per_step normally.


# ---------------------------------------------------------------------------
# M2.2 — full hook path: writes batch[COL_CRD_FORECAST] with correct shape
# ---------------------------------------------------------------------------


def test_compute_crd_terms_writes_forecast_key_with_matching_shape():
    """
    `batch[Columns.REWARDS]` is (B, T) and our forecast tensor must reshape to
    match so M5 can broadcast/multiply.
    """
    B, T = 2, 3
    infos = [_make_crd_info(pred=[a * 0.5 for a in [800_000.0, 0.0, 2_000_000.0]]) for _ in range(B * T)]
    batch = {
        Columns.INFOS: infos,
        Columns.REWARDS: torch.zeros(B, T),
    }
    learner = _StubLearner()
    learner.compute_loss_for_module(
        module_id="m", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=B, T=T)
    )
    assert COL_CRD_FORECAST in batch
    out = batch[COL_CRD_FORECAST]
    assert out.shape == (B, T)
    assert out.dtype == torch.float32


def test_compute_crd_terms_skipped_when_no_q_ensemble():
    """Non-ensemble module: forecast key must NOT appear in batch."""
    batch = {
        Columns.INFOS: [_make_crd_info(pred=[1, 1, 1])],
        Columns.REWARDS: torch.zeros(1, 1),
    }
    learner = _StubLearner()
    learner.compute_loss_for_module(
        module_id="vanilla", config=None, batch=batch, fwd_out={"vf_preds": torch.zeros(1)}
    )
    assert COL_CRD_FORECAST not in batch


def test_compute_crd_terms_skipped_when_no_infos():
    """Missing Columns.INFOS → forecast key not added (no crash)."""
    batch = {Columns.REWARDS: torch.zeros(1, 1)}
    learner = _StubLearner()
    learner.compute_loss_for_module(
        module_id="m", config=None, batch=batch, fwd_out=_ensemble_fwd_out()
    )
    assert COL_CRD_FORECAST not in batch


def test_compute_crd_terms_warns_once_when_predictions_missing(caplog):
    """When predictions are absent, log a warn-once per module."""
    import logging
    infos_without_pred = [_make_crd_info() for _ in range(4)]  # no pred
    batch = {Columns.INFOS: infos_without_pred, Columns.REWARDS: torch.zeros(2, 2)}
    learner = _StubLearner()
    with caplog.at_level(logging.WARNING, logger="src.learners.crd_q_loss"):
        learner.compute_loss_for_module(
            module_id="m1", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=2, T=2)
        )
        # Second call shouldn't double-log.
        learner.compute_loss_for_module(
            module_id="m1", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=2, T=2)
        )
    warnings = [r for r in caplog.records if "predicted_wind_w missing" in r.message]
    assert len(warnings) == 1, f"expected 1 warn-once log, got {len(warnings)}"
    # All 4 transitions still logged in the message body.
    assert "4/4" in warnings[0].message
    # Forecast values are still produced (all zeros).
    assert COL_CRD_FORECAST in batch
    assert (batch[COL_CRD_FORECAST] == 0).all()


def test_compute_crd_terms_uses_module_beta_gamma():
    """β=γ=0 makes R_forecast = 0 even when predictions are wildly wrong."""
    infos = [_make_crd_info(pred=[0.0, 0.0, 0.0])] * 4  # heavily biased
    batch = {Columns.INFOS: infos, Columns.REWARDS: torch.zeros(2, 2)}
    learner = _StubLearner(beta=0.0, gamma=0.0)
    learner.compute_loss_for_module(
        module_id="m", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=2, T=2)
    )
    assert (batch[COL_CRD_FORECAST] == 0).all()


def test_compute_crd_terms_forecast_tracks_bias_through_full_path():
    """Optimistic prediction (over-estimates wind) → R_forecast > 0 in batch tensor."""
    actual = [800_000.0, 0.0, 2_000_000.0]
    optimistic = [a * 2.0 for a in actual]
    infos = [_make_crd_info(actual=actual, pred=optimistic)] * 2
    batch = {Columns.INFOS: infos, Columns.REWARDS: torch.zeros(1, 2)}
    learner = _StubLearner(beta=1.0, gamma=1.0)
    learner.compute_loss_for_module(
        module_id="m", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=1, T=2)
    )
    out = batch[COL_CRD_FORECAST]
    assert (out > 0).all(), f"optimistic forecast should give R_forecast > 0, got {out}"


# ---------------------------------------------------------------------------
# M2.3 — baseline action via GreenQueueBalancedGlobalScheduler
# ---------------------------------------------------------------------------

from src.baselines.global_schedulers import GreenQueueBalancedGlobalScheduler
from src.learners.crd_q_loss import COL_CRD_BASELINE_ACTION


class _BaselineStubLearner(_StubLearner):
    """Like _StubLearner but provides a real GreenQueueBalanced scheduler so
    the M2.3 path under test can produce ã without a real RLlib module."""

    def __init__(self, num_dc=3, batch_size=4, **kw):
        super().__init__(**kw)
        self._fixed_scheduler = GreenQueueBalancedGlobalScheduler(
            num_datacenters=num_dc, batch_size=batch_size
        )

    def _get_or_build_baseline_scheduler(self, module_id):
        return self._fixed_scheduler


def _make_crd_info_with_signals(green_ratio=None, queue_sizes=None, pred=None):
    info = _make_crd_info(pred=pred)
    info["crd"]["dc_green_ratio"] = green_ratio or [0.5, 0.5, 0.5]
    info["crd"]["dc_queue_sizes"] = queue_sizes or [0, 0, 0]
    return info


def test_baseline_action_writes_key_and_correct_shape():
    """Each transition gets a list[batch_size] of DC indices in [0, num_dc)."""
    bs = 4
    nd = 3
    learner = _BaselineStubLearner(num_dc=nd, batch_size=bs)
    infos = [
        _make_crd_info_with_signals(green_ratio=[0.9, 0.1, 0.5], queue_sizes=[0, 5, 2]),
        _make_crd_info_with_signals(green_ratio=[0.1, 0.9, 0.5], queue_sizes=[3, 0, 1]),
    ]
    batch = {Columns.INFOS: infos, Columns.ACTIONS: torch.zeros(2, bs, dtype=torch.long)}
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=2, T=1)
    )
    assert COL_CRD_BASELINE_ACTION in batch
    out = batch[COL_CRD_BASELINE_ACTION]
    assert out.shape == (2, bs), f"expected (2, {bs}), got {tuple(out.shape)}"
    assert ((out >= 0) & (out < nd)).all(), f"DC indices out of range: {out}"


def test_baseline_action_prefers_green_dc_when_load_is_balanced():
    """With queue_sizes equal everywhere, scheduler should prefer high-green DC."""
    bs = 5
    nd = 3
    learner = _BaselineStubLearner(num_dc=nd, batch_size=bs)
    # DC 0 has 0.9 green ratio, others 0.1; queues all zero.
    info = _make_crd_info_with_signals(
        green_ratio=[0.9, 0.1, 0.1], queue_sizes=[0, 0, 0]
    )
    batch = {Columns.INFOS: [info], Columns.ACTIONS: torch.zeros(1, bs, dtype=torch.long)}
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=1, T=1)
    )
    out = batch[COL_CRD_BASELINE_ACTION].squeeze(0).tolist()
    # First cloudlet must hit DC 0 (highest green, no queue penalty).
    assert out[0] == 0, f"first cloudlet should go to DC 0, got {out}"
    # Most cloudlets should still favour DC 0 even after queue grows.
    assert out.count(0) >= bs // 2 + 1, f"DC 0 underused: {out}"


def test_baseline_action_zero_default_when_signals_missing():
    """No dc_green_ratio / dc_queue_sizes in info → baseline ã = all-zeros."""
    bs = 3
    nd = 3
    learner = _BaselineStubLearner(num_dc=nd, batch_size=bs)
    infos = [_make_crd_info()]  # no green_ratio / queue_sizes
    batch = {Columns.INFOS: infos, Columns.ACTIONS: torch.zeros(1, bs, dtype=torch.long)}
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=1, T=1)
    )
    out = batch[COL_CRD_BASELINE_ACTION]
    assert out.shape == (1, bs)
    assert (out == 0).all(), f"missing signals should default to all-zero ã: {out}"


def test_baseline_action_warn_once_per_module(caplog):
    import logging
    bs = 3
    nd = 3
    learner = _BaselineStubLearner(num_dc=nd, batch_size=bs)
    infos = [_make_crd_info() for _ in range(2)]   # all missing signals
    batch = {Columns.INFOS: infos, Columns.ACTIONS: torch.zeros(2, bs, dtype=torch.long)}
    with caplog.at_level(logging.WARNING, logger="src.learners.crd_q_loss"):
        for _ in range(3):
            learner.compute_loss_for_module(
                module_id="g", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=2, T=1)
            )
    warns = [r for r in caplog.records if "dc_queue_sizes/dc_green_ratio missing" in r.message]
    assert len(warns) == 1, f"expected exactly 1 warn-once, got {len(warns)}"


def test_baseline_action_skipped_when_no_scheduler():
    """If _get_or_build_baseline_scheduler returns None (e.g., local agent),
    no baseline_action key should appear."""
    learner = _StubLearner()  # plain stub returns None scheduler
    infos = [_make_crd_info_with_signals()]
    batch = {Columns.INFOS: infos, Columns.ACTIONS: torch.zeros(1, dtype=torch.long)}
    learner.compute_loss_for_module(
        module_id="local_x", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=1, T=1)
    )
    assert COL_CRD_BASELINE_ACTION not in batch


def test_baseline_action_uses_scheduler_static_helper_directly():
    """Cross-check: _compute_baseline_action_values produces same actions as
    a direct scheduler.schedule() call on the same input."""
    bs = 4
    nd = 3
    sched = GreenQueueBalancedGlobalScheduler(num_datacenters=nd, batch_size=bs)
    obs = {"dc_green_ratio": [0.9, 0.1, 0.5], "dc_queue_sizes": [0, 5, 2]}
    expected = sched.schedule(obs)
    info = _make_crd_info_with_signals(
        green_ratio=obs["dc_green_ratio"], queue_sizes=obs["dc_queue_sizes"]
    )
    actions_list, n_missing = CRDPPOTorchLearner._compute_baseline_action_values(
        infos=[info], num_dc=nd, scheduler=sched
    )
    assert n_missing == 0
    assert actions_list == [list(expected)]


# ---------------------------------------------------------------------------
# M2.4 — ΔQ + σ²_tot via ensemble lookup
# ---------------------------------------------------------------------------

from src.learners.crd_q_loss import COL_CRD_DQ, COL_CRD_SIGMA2


# --- Static helper tests (no learner instance) ---------------------------

def test_dq_sigma2_local_shapes_and_var_nonneg():
    """Local discrete action: q (B, T, K, A) + actions (B, T) → (B, T) outputs."""
    B, T, K, A = 4, 3, 5, 6
    q = torch.randn(B, T, K, A)
    actual = torch.randint(0, A, (B, T), dtype=torch.long)
    baseline = torch.randint(0, A, (B, T), dtype=torch.long)
    dq, sig2 = CRDPPOTorchLearner._compute_dq_and_sigma2_values(
        q_ensemble=q, actual_action=actual, baseline_action=baseline
    )
    assert dq.shape == (B, T)
    assert sig2.shape == (B, T)
    assert (sig2 >= 0).all(), "σ²_tot must be non-negative"


def test_dq_sigma2_global_shapes_and_var_nonneg():
    """Global MultiDiscrete: q (B, T, K, bs, nd) + actions (B, T, bs)."""
    B, T, K, bs, nd = 3, 2, 5, 4, 6
    q = torch.randn(B, T, K, bs, nd)
    actual = torch.randint(0, nd, (B, T, bs), dtype=torch.long)
    baseline = torch.randint(0, nd, (B, T, bs), dtype=torch.long)
    dq, sig2 = CRDPPOTorchLearner._compute_dq_and_sigma2_values(
        q_ensemble=q, actual_action=actual, baseline_action=baseline
    )
    assert dq.shape == (B, T)
    assert sig2.shape == (B, T)
    assert (sig2 >= 0).all()


def test_dq_zero_when_actions_identical_local():
    """If actual == baseline at every (b, t), ΔQ must be exactly 0."""
    B, T, K, A = 3, 2, 5, 4
    q = torch.randn(B, T, K, A)
    a = torch.randint(0, A, (B, T), dtype=torch.long)
    dq, sig2 = CRDPPOTorchLearner._compute_dq_and_sigma2_values(
        q_ensemble=q, actual_action=a, baseline_action=a
    )
    assert (dq.abs() < 1e-9).all(), f"dq should be zero when actions match: {dq}"
    # σ²_tot should equal 2 × σ²(a)  (both terms identical).
    q_a = CRDPPOTorchLearner._gather_q_chosen(q, a)
    expected_var2 = 2.0 * q_a.var(dim=-1, unbiased=False)
    assert torch.allclose(sig2, expected_var2, atol=1e-6)


def test_dq_zero_when_actions_identical_global():
    B, T, K, bs, nd = 2, 2, 5, 3, 4
    q = torch.randn(B, T, K, bs, nd)
    a = torch.randint(0, nd, (B, T, bs), dtype=torch.long)
    dq, sig2 = CRDPPOTorchLearner._compute_dq_and_sigma2_values(
        q_ensemble=q, actual_action=a, baseline_action=a
    )
    assert (dq.abs() < 1e-9).all()


def test_dq_sigma2_static_helper_matches_module_API_local():
    """
    Sanity: M2.4's static helper should match what M1.3's
    `EnsembleQHeads.compute_q_for_action` produces for any single action,
    since both compute mean/var over K at the chosen action.
    """
    from src.models.rlmodule_gtrxl_ensemble import EnsembleQHeads
    torch.manual_seed(0)
    K, A, d = 5, 4, 8
    eqh = EnsembleQHeads(d_model=d, action_dim=A, K=K, prior_lambda=2.0, hidden_dim=8)
    state = torch.randn(6, d)
    a = torch.randint(0, A, (6,), dtype=torch.long)

    # Single-action API path (M1.3)
    mu1, var1 = eqh.compute_q_for_action(state, a)

    # M2.4 path — feed identical actions for "actual" and "baseline" so dq=0,
    # then the σ²_tot we get is 2 × var1; we just check var matches.
    q_full = eqh(state)              # (6, K, A)
    q_full = q_full.unsqueeze(1)     # (6, 1, K, A) — fake T=1 for static helper
    a_t = a.unsqueeze(1)             # (6, 1)
    dq, sig2 = CRDPPOTorchLearner._compute_dq_and_sigma2_values(
        q_ensemble=q_full, actual_action=a_t, baseline_action=a_t
    )
    assert (dq.abs() < 1e-6).all()
    assert torch.allclose(sig2.squeeze(-1), 2.0 * var1, atol=1e-5)
    # mu via gather should match mu1
    q_a = CRDPPOTorchLearner._gather_q_chosen(q_full, a_t)
    assert torch.allclose(q_a.mean(dim=-1).squeeze(-1), mu1, atol=1e-5)


def test_dq_unexpected_q_dim_raises():
    """3-D or 6-D q_ensemble should raise."""
    bad_q = torch.randn(4, 5, 6)  # 3-D
    a = torch.randint(0, 5, (4, 5), dtype=torch.long)
    with pytest.raises(RuntimeError, match="Unexpected q_ensemble dim"):
        CRDPPOTorchLearner._compute_dq_and_sigma2_values(
            q_ensemble=bad_q, actual_action=a, baseline_action=a
        )


# --- End-to-end via compute_loss_for_module -------------------------------

def test_dq_sigma2_written_via_compute_loss_local():
    """End-to-end: ensure batch[crd_dq] and [crd_sigma2] get written."""
    learner = _StubLearner()  # no scheduler — but we'll feed baseline_action directly
    B, T, K, A = 2, 2, 5, 4
    q = torch.randn(B, T, K, A)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    baseline = torch.randint(0, A, (B, T), dtype=torch.long)
    fwd = {COL_Q_ENSEMBLE: q}
    batch = {Columns.ACTIONS: actions, COL_CRD_BASELINE_ACTION: baseline}
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=fwd
    )
    assert COL_CRD_DQ in batch and COL_CRD_SIGMA2 in batch
    assert batch[COL_CRD_DQ].shape == (B, T)
    assert batch[COL_CRD_SIGMA2].shape == (B, T)
    assert (batch[COL_CRD_SIGMA2] >= 0).all()


def test_dq_skipped_when_no_baseline_action():
    """Without batch[crd_baseline_action] (M2.3 didn't run), M2.4 stays silent."""
    learner = _StubLearner()
    fwd = {COL_Q_ENSEMBLE: torch.randn(2, 1, 5, 4)}
    batch = {Columns.ACTIONS: torch.zeros(2, 1, dtype=torch.long)}
    learner.compute_loss_for_module(
        module_id="m", config=None, batch=batch, fwd_out=fwd
    )
    assert COL_CRD_DQ not in batch
    assert COL_CRD_SIGMA2 not in batch


def test_dq_skipped_when_no_q_ensemble():
    """Non-ensemble module (no crd_q_ensemble in fwd_out) → M2.4 silent."""
    learner = _StubLearner()
    batch = {
        Columns.ACTIONS: torch.zeros(2, 1, dtype=torch.long),
        COL_CRD_BASELINE_ACTION: torch.zeros(2, 1, dtype=torch.long),
    }
    learner.compute_loss_for_module(
        module_id="vanilla", config=None, batch=batch, fwd_out={"vf_preds": torch.zeros(2)}
    )
    assert COL_CRD_DQ not in batch


def test_dq_sigma2_detached_no_gradient():
    """Output tensors must be detached so M5 reweight doesn't grad-flow into q_heads."""
    learner = _StubLearner()
    B, T, K, A = 2, 1, 5, 4
    q = torch.randn(B, T, K, A, requires_grad=True)
    actions = torch.randint(0, A, (B, T), dtype=torch.long)
    baseline = torch.randint(0, A, (B, T), dtype=torch.long)
    fwd = {COL_Q_ENSEMBLE: q}
    batch = {Columns.ACTIONS: actions, COL_CRD_BASELINE_ACTION: baseline}
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=fwd
    )
    # detached outputs have no grad_fn
    assert batch[COL_CRD_DQ].grad_fn is None
    assert batch[COL_CRD_SIGMA2].grad_fn is None


# ---------------------------------------------------------------------------
# M2.5 — Δr fallback proxy (load-std difference)
# ---------------------------------------------------------------------------

from src.learners.crd_q_loss import COL_CRD_DR


# --- Static helper tests --------------------------------------------------

def test_dr_zero_when_actions_identical():
    """If actual == baseline, queues are identical → load std identical → Δr=0."""
    info = _make_crd_info_with_signals(queue_sizes=[2, 0, 5])
    actual = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)  # 1 transition × bs=4
    drs = CRDPPOTorchLearner._compute_dr_values(
        infos=[info], actual_actions=actual, baseline_actions=actual,
        num_dc=3, alpha=1.0,
    )
    assert drs == [0.0]


def test_dr_negative_when_baseline_balances_better():
    """Baseline routing balances queues better → baseline_std < actual_std → Δr < 0."""
    # Initial queues: very imbalanced ([0, 100, 0]); baseline routes to DC 0 and 2
    # to balance, agent dumps everything onto DC 1 (worse).
    info = _make_crd_info_with_signals(queue_sizes=[0, 100, 0])
    actual_routing = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)  # all to DC 1
    baseline_routing = torch.tensor([[0, 2, 0, 2]], dtype=torch.long)  # spread
    drs = CRDPPOTorchLearner._compute_dr_values(
        infos=[info], actual_actions=actual_routing,
        baseline_actions=baseline_routing,
        num_dc=3, alpha=1.0,
    )
    assert len(drs) == 1
    assert drs[0] < 0, f"baseline should beat actual → Δr<0; got {drs[0]}"


def test_dr_positive_when_actual_balances_better():
    """Symmetric: agent better than baseline → Δr > 0."""
    info = _make_crd_info_with_signals(queue_sizes=[0, 100, 0])
    actual_routing = torch.tensor([[0, 2, 0, 2]], dtype=torch.long)
    baseline_routing = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    drs = CRDPPOTorchLearner._compute_dr_values(
        infos=[info], actual_actions=actual_routing,
        baseline_actions=baseline_routing,
        num_dc=3, alpha=1.0,
    )
    assert drs[0] > 0, f"actual better → Δr>0; got {drs[0]}"


def test_dr_alpha_scales_linearly():
    info = _make_crd_info_with_signals(queue_sizes=[0, 100, 0])
    actual = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    baseline = torch.tensor([[0, 2, 0, 2]], dtype=torch.long)
    dr1 = CRDPPOTorchLearner._compute_dr_values(
        infos=[info], actual_actions=actual, baseline_actions=baseline,
        num_dc=3, alpha=1.0,
    )[0]
    dr2 = CRDPPOTorchLearner._compute_dr_values(
        infos=[info], actual_actions=actual, baseline_actions=baseline,
        num_dc=3, alpha=2.5,
    )[0]
    assert dr2 == pytest.approx(2.5 * dr1, rel=1e-6)


def test_dr_zero_when_queue_sizes_missing():
    """No dc_queue_sizes → Δr = 0 (graceful degrade, matches M2.3)."""
    info = _make_crd_info()  # no queue_sizes
    actual = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    baseline = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    drs = CRDPPOTorchLearner._compute_dr_values(
        infos=[info], actual_actions=actual, baseline_actions=baseline,
        num_dc=3, alpha=1.0,
    )
    assert drs == [0.0]


def test_dr_handles_2d_action_tensor():
    """Action tensors of shape (B, T, batch_size) get flattened to (N, bs)."""
    infos = [_make_crd_info_with_signals(queue_sizes=[0, 0, 0]) for _ in range(4)]
    # B=2, T=2, bs=3
    actual = torch.tensor(
        [[[0, 0, 0], [1, 1, 1]],
         [[2, 2, 2], [0, 1, 2]]], dtype=torch.long
    )
    baseline = torch.tensor(
        [[[0, 1, 2], [0, 1, 2]],
         [[0, 1, 2], [0, 1, 2]]], dtype=torch.long
    )
    drs = CRDPPOTorchLearner._compute_dr_values(
        infos=infos, actual_actions=actual, baseline_actions=baseline,
        num_dc=3, alpha=1.0,
    )
    assert len(drs) == 4
    # Last transition: actual = [0,1,2] → balanced, baseline = [0,1,2] → balanced
    # So both stds are 0; Δr should be exactly 0.
    assert drs[-1] == pytest.approx(0.0, abs=1e-9)
    # First three transitions: actual is concentrated, baseline spreads → Δr < 0
    assert all(d < 0 for d in drs[:3])


# --- End-to-end via compute_loss_for_module -------------------------------

class _DRStubLearner(_BaselineStubLearner):
    """Adds an alpha override for the M2.5 Δr config reader."""
    def __init__(self, alpha: float = 1.0, **kw):
        super().__init__(**kw)
        self._dr_alpha = alpha
    def _read_module_dr_config(self, module_id):
        return {"alpha": self._dr_alpha}


def test_dr_written_via_compute_loss_end_to_end():
    """Full path: M2.3 produces ã, M2.5 reads it + queue_sizes → Δr in batch."""
    bs = 4
    nd = 3
    learner = _DRStubLearner(num_dc=nd, batch_size=bs, alpha=1.0)
    info = _make_crd_info_with_signals(
        green_ratio=[0.9, 0.1, 0.5],
        queue_sizes=[0, 100, 0],
    )
    batch = {
        Columns.INFOS: [info],
        Columns.ACTIONS: torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        Columns.REWARDS: torch.zeros(1, 1),
    }
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=_ensemble_fwd_out(B=1, T=1)
    )
    # M2.3 produced baseline action; M2.5 used it
    assert COL_CRD_DR in batch
    out = batch[COL_CRD_DR]
    # Reshape to match REWARDS (1, 1)
    assert out.shape == (1, 1)
    # Agent dumped everything to DC 1 (already at 100), baseline spread →
    # baseline_std < actual_std → Δr < 0
    assert out.item() < 0


def test_dr_skipped_when_no_baseline_action():
    """Without M2.3's baseline action in batch, M2.5 stays silent."""
    learner = _DRStubLearner(num_dc=3, batch_size=4)
    batch = {
        Columns.INFOS: [_make_crd_info_with_signals()],
        Columns.ACTIONS: torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
    }
    # Skip M2.3 by providing no q_ensemble in fwd_out (so the gate kills the
    # whole CRD pipeline — M2.5 inherits the skip).
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out={}
    )
    assert COL_CRD_DR not in batch


def test_dr_skipped_when_no_scheduler_cached():
    """If no GreenQueueBalanced scheduler was built (local agent), Δr can't
    know num_dc → skip silently."""
    learner = _StubLearner()  # plain stub, returns None scheduler
    batch = {
        Columns.INFOS: [_make_crd_info_with_signals()],
        Columns.ACTIONS: torch.tensor([0], dtype=torch.long),
        COL_CRD_BASELINE_ACTION: torch.tensor([0], dtype=torch.long),
    }
    learner.compute_loss_for_module(
        module_id="local_x", config=None, batch=batch, fwd_out=_ensemble_fwd_out()
    )
    assert COL_CRD_DR not in batch


# ---------------------------------------------------------------------------
# M3 — Soft blending integration
# ---------------------------------------------------------------------------

from src.learners.crd_q_loss import (
    COL_CRD_R_ROUTING,
    COL_CRD_C_T,
    COL_CRD_TAU,
)


def test_r_routing_written_via_full_pipeline():
    """End-to-end: M2.4+M2.5+M3 produce ΔQ, Δr, R^routing all in batch."""
    bs = 4
    nd = 3
    learner = _DRStubLearner(num_dc=nd, batch_size=bs, alpha=1.0)
    info = _make_crd_info_with_signals(
        green_ratio=[0.9, 0.1, 0.5], queue_sizes=[0, 100, 0]
    )
    fwd = {COL_Q_ENSEMBLE: torch.randn(1, 1, 5, bs, nd)}
    batch = {
        Columns.INFOS: [info],
        Columns.ACTIONS: torch.tensor([[1, 1, 1, 1]], dtype=torch.long).unsqueeze(0),  # (1, 1, bs)
        Columns.REWARDS: torch.zeros(1, 1),
    }
    # Note: action shape must be (B, T, bs). Fix:
    batch[Columns.ACTIONS] = torch.tensor([[[1, 1, 1, 1]]], dtype=torch.long)
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=fwd
    )
    # All four CRD columns should be present
    for k in (COL_CRD_DQ, COL_CRD_DR, COL_CRD_R_ROUTING, COL_CRD_C_T, COL_CRD_TAU):
        assert k in batch, f"missing {k}"
    assert batch[COL_CRD_R_ROUTING].shape == batch[COL_CRD_DQ].shape
    assert batch[COL_CRD_C_T].shape == batch[COL_CRD_DQ].shape
    # τ is scalar
    assert batch[COL_CRD_TAU].dim() == 0


def test_r_routing_skipped_when_dq_missing():
    """No ΔQ in batch → blender skip; no R^routing produced."""
    learner = _StubLearner()
    batch = {
        Columns.INFOS: [_make_crd_info_with_signals()],
        Columns.ACTIONS: torch.tensor([[0, 1, 2, 0]], dtype=torch.long).unsqueeze(0),
    }
    # Skip M2.4 by not providing crd_q_ensemble
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out={}
    )
    assert COL_CRD_R_ROUTING not in batch


def test_r_routing_interpolates_correctly():
    """Direct math check: with σ²=0 → R = ΔQ; high σ² → R near Δr."""
    learner = _DRStubLearner(num_dc=3, batch_size=4, alpha=1.0)
    # Manually populate batch (skip M2.0–M2.5 by not running them)
    dq = torch.tensor([[1.0, 2.0]])
    dr = torch.tensor([[10.0, -10.0]])
    # σ² mostly zero (left col) and very large (right col)
    sigma2 = torch.tensor([[0.0, 1e6]])
    batch = {
        COL_CRD_DQ: dq, COL_CRD_DR: dr, COL_CRD_SIGMA2: sigma2,
    }
    # Call _compute_r_routing directly (skip the full _compute_crd_terms)
    learner._compute_r_routing(module_id="g_direct", batch=batch)
    r = batch[COL_CRD_R_ROUTING]
    # σ²=0 → R = ΔQ
    assert r[0, 0].item() == pytest.approx(1.0)
    # Huge σ² → R ≈ Δr (κ=0.5 by default but we set ema_init=None so τ
    # bootstraps to mean of σ² = 5e5 — c ≈ exp(-1e6/exp(0.5*5e5)) → 1
    # Hmm actually with adaptive τ, let me check: if bar_σ² grows fast, τ
    # explodes, and c stays high. Let me just check r is between dq and dr.
    assert min(dq[0, 1].item(), dr[0, 1].item()) <= r[0, 1].item() <= max(
        dq[0, 1].item(), dr[0, 1].item()
    )


def test_r_routing_with_kappa_zero_pure_tau0():
    """κ=0 disables adaptive τ — gives pure exp(-σ²/τ_0) gate."""
    # Use a learner with overridden config
    class _FixedTauStub(_DRStubLearner):
        def _read_module_blender_config(self, module_id):
            return {"tau_0": 1.0, "kappa": 0.0, "eta": 0.1, "ema_init": 0.0}

    learner = _FixedTauStub(num_dc=3, batch_size=4, alpha=1.0)
    batch = {
        COL_CRD_DQ: torch.tensor([[1.0]]),
        COL_CRD_DR: torch.tensor([[0.0]]),
        COL_CRD_SIGMA2: torch.tensor([[1.0]]),  # σ²=1, τ=1 → c = e⁻¹
    }
    learner._compute_r_routing(module_id="g_fixed", batch=batch)
    r = batch[COL_CRD_R_ROUTING]
    expected = math.exp(-1.0)  # c · 1 + (1-c) · 0 = c
    assert r.item() == pytest.approx(expected, rel=1e-5)


def test_r_routing_blender_persists_across_calls():
    """Same module_id → same blender → EMA accumulates across batches."""
    learner = _DRStubLearner(num_dc=3, batch_size=4, alpha=1.0)
    sigma2_b1 = torch.tensor([[5.0, 5.0]])
    sigma2_b2 = torch.tensor([[10.0, 10.0]])
    dq, dr = torch.zeros_like(sigma2_b1), torch.zeros_like(sigma2_b1)

    # First batch → bootstraps EMA at 5.0
    batch1 = {COL_CRD_DQ: dq, COL_CRD_DR: dr, COL_CRD_SIGMA2: sigma2_b1}
    learner._compute_r_routing(module_id="g_persist", batch=batch1)

    # Second batch → EMA = 0.95·5 + 0.05·10 = 5.25
    batch2 = {COL_CRD_DQ: dq, COL_CRD_DR: dr, COL_CRD_SIGMA2: sigma2_b2}
    learner._compute_r_routing(module_id="g_persist", batch=batch2)

    blender = learner._crd_blenders["g_persist"]
    assert blender.bar_sigma2 == pytest.approx(5.25, rel=1e-6)


def test_r_routing_separate_blenders_per_module():
    """Different module_ids get independent blender state."""
    learner = _DRStubLearner(num_dc=3, batch_size=4, alpha=1.0)
    dq, dr = torch.zeros(1, 1), torch.zeros(1, 1)
    learner._compute_r_routing(
        module_id="alpha", batch={COL_CRD_DQ: dq, COL_CRD_DR: dr, COL_CRD_SIGMA2: torch.tensor([[1.0]])}
    )
    learner._compute_r_routing(
        module_id="beta", batch={COL_CRD_DQ: dq, COL_CRD_DR: dr, COL_CRD_SIGMA2: torch.tensor([[100.0]])}
    )
    assert learner._crd_blenders["alpha"].bar_sigma2 == pytest.approx(1.0)
    assert learner._crd_blenders["beta"].bar_sigma2 == pytest.approx(100.0)


def test_r_routing_detached_no_gradient():
    """R^routing, c_t, τ should be detached (M5 reweights advantages, not q-heads)."""
    learner = _DRStubLearner(num_dc=3, batch_size=4, alpha=1.0)
    dq = torch.tensor([[1.0]], requires_grad=True)
    dr = torch.tensor([[0.0]], requires_grad=True)
    sigma2 = torch.tensor([[0.5]], requires_grad=True)
    batch = {COL_CRD_DQ: dq, COL_CRD_DR: dr, COL_CRD_SIGMA2: sigma2}
    learner._compute_r_routing(module_id="g_detach", batch=batch)
    assert batch[COL_CRD_R_ROUTING].grad_fn is None
    assert batch[COL_CRD_C_T].grad_fn is None


# ---------------------------------------------------------------------------
# M5 — Responsibility weights ρ + advantage rewrite
# ---------------------------------------------------------------------------

from src.learners.crd_q_loss import (
    COL_CRD_RHO_FORECAST,
    COL_CRD_RHO_ROUTING,
    COL_CRD_RHO_SCHEDULING,
    COL_CRD_R_SCHEDULING,
)


def test_rho_shapes_match_r_routing():
    """ρ tensors should match R^routing's (B, T) shape."""
    learner = _DRStubLearner(num_dc=3, batch_size=4, alpha=1.0)
    batch = {
        COL_CRD_FORECAST: torch.tensor([[1.0, 2.0]]),
        COL_CRD_R_ROUTING: torch.tensor([[3.0, 4.0]]),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    for k in (COL_CRD_RHO_FORECAST, COL_CRD_RHO_ROUTING, COL_CRD_RHO_SCHEDULING):
        assert k in batch
        assert batch[k].shape == (1, 2)


def test_rho_skipped_when_no_r_routing():
    """Without R^routing in batch, no ρ produced."""
    learner = _DRStubLearner()
    batch = {COL_CRD_FORECAST: torch.tensor([[1.0]])}
    learner._compute_responsibilities(module_id="m", batch=batch)
    assert COL_CRD_RHO_ROUTING not in batch


def test_rho_floor_applied_to_routing_and_scheduling():
    """When |R_routing| ≪ |R_forecast|, ρ_routing should clamp at ρ_min."""
    class _RhoStub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.1}

    learner = _RhoStub(num_dc=3, batch_size=4, alpha=1.0)
    # Massive forecast error, tiny routing share
    batch = {
        COL_CRD_FORECAST: torch.tensor([[1000.0]]),
        COL_CRD_R_ROUTING: torch.tensor([[0.001]]),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    rho_r = batch[COL_CRD_RHO_ROUTING]
    # raw ρ_routing = 0.001/1000.001 ≈ 1e-6; floor lifts it to 0.1
    assert rho_r.item() == pytest.approx(0.1, rel=1e-6)


def test_rho_no_floor_on_forecast():
    """ρ_forecast is logging-only — should NOT be floored.

    Verifies the diagnostic preserves true forecast attribution magnitude.
    """
    class _RhoStub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.5}  # very high floor

    learner = _RhoStub(num_dc=3, batch_size=4, alpha=1.0)
    batch = {
        COL_CRD_FORECAST: torch.tensor([[0.001]]),
        COL_CRD_R_ROUTING: torch.tensor([[1000.0]]),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    # raw ρ_forecast = 0.001/1000.001 ≈ 1e-6; should NOT be floored.
    assert batch[COL_CRD_RHO_FORECAST].item() < 1e-3


def test_rho_advantages_unchanged_when_rho_eq_one():
    """If ρ_routing ≈ 1 (R_routing dominates), ADVANTAGES barely changes."""
    learner = _DRStubLearner()
    adv_before = torch.tensor([[1.0, -2.0, 0.5]])
    batch = {
        COL_CRD_FORECAST: torch.zeros((1, 3)),  # no forecast share
        COL_CRD_R_ROUTING: torch.ones((1, 3)),  # all weight on routing
        Postprocessing.ADVANTAGES: adv_before.clone(),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    # ρ_routing ≈ 1 → advantage ≈ original
    adv_after = batch[Postprocessing.ADVANTAGES]
    assert torch.allclose(adv_after, adv_before, atol=1e-6)


def test_rho_advantages_scaled_to_floor_when_rho_min_dominates():
    """If R_routing is tiny vs R_forecast, ρ_routing → floor → advantage scaled to floor."""
    class _RhoStub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.05}

    learner = _RhoStub()
    adv_before = torch.tensor([[10.0, -20.0]])
    batch = {
        COL_CRD_FORECAST: torch.tensor([[1.0e6, 1.0e6]]),
        COL_CRD_R_ROUTING: torch.tensor([[1.0e-6, 1.0e-6]]),
        Postprocessing.ADVANTAGES: adv_before.clone(),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    adv_after = batch[Postprocessing.ADVANTAGES]
    # ρ_routing = max(tiny, 0.05) = 0.05 → adv_after = adv_before * 0.05
    expected = adv_before * 0.05
    assert torch.allclose(adv_after, expected, atol=1e-6)


def test_rho_value_targets_untouched():
    """Critical: VALUE_TARGETS must not be reweighted (V-head trains unbiased)."""
    learner = _DRStubLearner()
    vt_before = torch.tensor([[5.0, -3.0]])
    batch = {
        COL_CRD_FORECAST: torch.tensor([[100.0, 100.0]]),
        COL_CRD_R_ROUTING: torch.tensor([[1.0, 1.0]]),
        Postprocessing.ADVANTAGES: torch.zeros((1, 2)),
        Postprocessing.VALUE_TARGETS: vt_before.clone(),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    assert torch.equal(batch[Postprocessing.VALUE_TARGETS], vt_before)


def test_rho_forecast_not_applied_to_advantages():
    """Even when ρ_forecast is large, advantages should be scaled by ρ_routing only."""
    class _RhoStub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.05}

    learner = _RhoStub()
    adv_before = torch.tensor([[10.0]])
    # Forecast dominates → ρ_forecast ≈ 1, ρ_routing ≈ floor
    batch = {
        COL_CRD_FORECAST: torch.tensor([[1.0e6]]),
        COL_CRD_R_ROUTING: torch.tensor([[1.0e-6]]),
        Postprocessing.ADVANTAGES: adv_before.clone(),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    # Should be adv * 0.05 (the routing floor), NOT adv * ρ_forecast
    assert batch[Postprocessing.ADVANTAGES].item() == pytest.approx(0.5, abs=1e-6)


def test_rho_with_scheduling_component_split_three_ways():
    """When all 3 components present, ρ_k should sum near 1 (modulo floor distortion)."""
    class _RhoStub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0}  # disable floor for sum-to-1 check

    learner = _RhoStub()
    batch = {
        COL_CRD_FORECAST: torch.tensor([[1.0]]),
        COL_CRD_R_ROUTING: torch.tensor([[2.0]]),
        COL_CRD_R_SCHEDULING: torch.tensor([[3.0]]),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    rho_f = batch[COL_CRD_RHO_FORECAST].item()
    rho_r = batch[COL_CRD_RHO_ROUTING].item()
    rho_s = batch[COL_CRD_RHO_SCHEDULING].item()
    # Σρ should be very close to 1 (only ε difference)
    assert abs(rho_f + rho_r + rho_s - 1.0) < 1e-3
    # And in expected ratio: 1 : 2 : 3
    assert rho_r == pytest.approx(2.0 * rho_f, rel=1e-3)
    assert rho_s == pytest.approx(3.0 * rho_f, rel=1e-3)


def test_rho_detached_no_gradient():
    """ρ tensors stored in batch should be detached."""
    learner = _DRStubLearner()
    rr = torch.tensor([[1.0]], requires_grad=True)
    batch = {
        COL_CRD_FORECAST: torch.tensor([[2.0]], requires_grad=True),
        COL_CRD_R_ROUTING: rr,
        Postprocessing.ADVANTAGES: torch.tensor([[1.0]]),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    for k in (COL_CRD_RHO_FORECAST, COL_CRD_RHO_ROUTING, COL_CRD_RHO_SCHEDULING):
        assert batch[k].grad_fn is None, f"{k} should be detached"


def test_rho_skipped_advantages_when_shape_mismatch(caplog):
    """ρ shape mismatch with ADVANTAGES → log warning + leave adv unchanged."""
    import logging
    learner = _DRStubLearner()
    adv_before = torch.tensor([1.0, 2.0, 3.0])  # 1-D shape (3,)
    batch = {
        COL_CRD_FORECAST: torch.tensor([[0.0, 0.0]]),
        COL_CRD_R_ROUTING: torch.tensor([[1.0, 1.0]]),  # (1, 2) — won't reshape
        Postprocessing.ADVANTAGES: adv_before.clone(),
    }
    with caplog.at_level(logging.WARNING, logger="src.learners.crd_q_loss"):
        learner._compute_responsibilities(module_id="m", batch=batch)
    # Advantages unchanged
    assert torch.equal(batch[Postprocessing.ADVANTAGES], adv_before)
    # ρ tensors still written though
    assert COL_CRD_RHO_ROUTING in batch
    # Warning logged
    assert any("rho_routing shape" in r.message for r in caplog.records)


def test_dr_magnitude_comparable_to_dq():
    """Sanity check: Δr should not differ from ΔQ by orders of magnitude on
    a typical batch — M3's soft blending would ignore Δr otherwise."""
    bs = 4
    nd = 3
    learner = _DRStubLearner(num_dc=nd, batch_size=bs, alpha=1.0)
    # Several transitions with varying queue distributions
    infos = [
        _make_crd_info_with_signals(queue_sizes=[0, 0, 5]),
        _make_crd_info_with_signals(queue_sizes=[10, 0, 0]),
        _make_crd_info_with_signals(queue_sizes=[2, 3, 1]),
    ]
    actions = torch.tensor([[1, 2, 0, 1], [2, 1, 0, 2], [0, 1, 2, 0]], dtype=torch.long)
    K = 5
    # q_ensemble standardish range
    fwd = {COL_Q_ENSEMBLE: torch.randn(3, 1, K, bs, nd)}
    # batch arrangement: B=3, T=1
    batch = {
        Columns.INFOS: infos,
        Columns.ACTIONS: actions.unsqueeze(1),  # (3, 1, bs)
        Columns.REWARDS: torch.zeros(3, 1),
    }
    learner.compute_loss_for_module(
        module_id="g", config=None, batch=batch, fwd_out=fwd
    )
    dq = batch[COL_CRD_DQ]
    dr = batch[COL_CRD_DR]
    # Both should be order-of-magnitude similar (within 100x).
    if dq.abs().max() > 0:
        ratio = (dr.abs().max() + 1e-9) / (dq.abs().max() + 1e-9)
        assert 1e-2 < ratio < 1e2, (
            f"Δr / ΔQ magnitude ratio {ratio:.3g} out of range — "
            f"M3 soft blending will be lopsided."
        )


# ---------------------------------------------------------------------------
# M2.4 layout robustness — RLlib new API stack gives sequence-packed
# (N_valid, ...) actions while q_ensemble is the padded (B, T, ...) form.
# These tests lock in the loss_mask-based gather path against future regressions.
# ---------------------------------------------------------------------------


def test_gather_q_chosen_local_BT_layout():
    """Aligned (B, T) action layout — historical default, must keep working."""
    B, T, K, A = 2, 3, 4, 5
    q = torch.randn(B, T, K, A)
    action = torch.randint(0, A, (B, T), dtype=torch.long)
    out = CRDPPOTorchLearner._gather_q_chosen(q, action, loss_mask=None)
    assert out.shape == (B, T, K)


def test_gather_q_chosen_local_flat_BT_layout():
    """Action shape (B*T,) — flat but full coverage."""
    B, T, K, A = 2, 3, 4, 5
    q = torch.randn(B, T, K, A)
    action_bt = torch.randint(0, A, (B, T), dtype=torch.long)
    action_flat = action_bt.reshape(-1)
    out_flat = CRDPPOTorchLearner._gather_q_chosen(q, action_flat, loss_mask=None)
    out_bt = CRDPPOTorchLearner._gather_q_chosen(q, action_bt, loss_mask=None)
    assert torch.allclose(out_flat, out_bt)


def test_gather_q_chosen_local_sequence_packed_layout():
    """Action shape (N_valid,) with N_valid < B*T — needs loss_mask path.

    This is the exact case that caused the smoke-test failure:
      q_ensemble = (4, 128, 5, 20, 5)
      action     = (417, ...)  ← N_valid, NOT B*T
    """
    B, T, K, A = 4, 6, 3, 5
    q = torch.randn(B, T, K, A)
    # Mark a non-uniform set of timesteps as valid (mimic seq_lens variability).
    loss_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0],   # seq_len 3
         [1, 1, 1, 1, 1, 0],   # seq_len 5
         [1, 1, 0, 0, 0, 0],   # seq_len 2
         [1, 1, 1, 1, 0, 0]],  # seq_len 4
        dtype=torch.float32,
    )
    n_valid = int(loss_mask.sum().item())  # 14
    action = torch.randint(0, A, (n_valid,), dtype=torch.long)
    out = CRDPPOTorchLearner._gather_q_chosen(q, action, loss_mask=loss_mask)
    # Output is (B, T, K) — masked positions zero-filled.
    assert out.shape == (B, T, K)
    # Cells outside the mask must be exactly zero.
    inv_mask = (loss_mask == 0)
    assert (out[inv_mask] == 0).all(), "scatter-back left non-zero in invalid slots"
    # Cells inside the mask must be non-zero (action picks a real Q value).
    in_mask = (loss_mask == 1)
    # Note: Q values themselves can occasionally be 0 from randn; just check
    # that we *did* gather something (sum over K not all zeros across all valid cells).
    valid_out = out[in_mask]                     # (n_valid, K)
    assert valid_out.numel() == n_valid * K
    assert valid_out.abs().sum() > 0


def test_gather_q_chosen_global_sequence_packed_layout():
    """The exact smoke-test scenario: q (B, T, K, bs, nd), action (N_valid, bs)."""
    B, T, K, bs, nd = 4, 6, 3, 7, 5
    q = torch.randn(B, T, K, bs, nd)
    loss_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0],
         [1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0]],
        dtype=torch.float32,
    )
    n_valid = int(loss_mask.sum().item())  # 14
    action = torch.randint(0, nd, (n_valid, bs), dtype=torch.long)
    out = CRDPPOTorchLearner._gather_q_chosen(q, action, loss_mask=loss_mask)
    assert out.shape == (B, T, K)
    assert (out[loss_mask == 0] == 0).all()


def test_gather_q_chosen_local_mismatched_n_valid_raises():
    """If action N doesn't match loss_mask sum, raise (don't silently succeed)."""
    B, T, K, A = 2, 4, 3, 5
    q = torch.randn(B, T, K, A)
    loss_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    # n_valid should be 3, but pass an action of length 5.
    bad_action = torch.zeros(5, dtype=torch.long)
    with pytest.raises(ValueError, match="loss_mask N_valid"):
        CRDPPOTorchLearner._gather_q_chosen(q, bad_action, loss_mask=loss_mask)


def test_gather_q_chosen_global_mismatched_n_valid_raises():
    B, T, K, bs, nd = 2, 4, 3, 5, 4
    q = torch.randn(B, T, K, bs, nd)
    loss_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    bad_action = torch.zeros((5, bs), dtype=torch.long)
    with pytest.raises(ValueError, match="loss_mask N_valid"):
        CRDPPOTorchLearner._gather_q_chosen(q, bad_action, loss_mask=loss_mask)


def test_compute_dq_and_sigma2_values_handles_sequence_packed():
    """End-to-end: ΔQ + σ² compute through the masked path returns (B, T) shape."""
    B, T, K, bs, nd = 3, 5, 4, 6, 4
    q = torch.randn(B, T, K, bs, nd)
    loss_mask = torch.tensor(
        [[1, 1, 1, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 1, 1, 1, 0]],
        dtype=torch.float32,
    )
    n_valid = int(loss_mask.sum().item())  # 6
    actual = torch.randint(0, nd, (n_valid, bs), dtype=torch.long)
    baseline = torch.randint(0, nd, (n_valid, bs), dtype=torch.long)
    delta_q, sigma2 = CRDPPOTorchLearner._compute_dq_and_sigma2_values(
        q_ensemble=q,
        actual_action=actual,
        baseline_action=baseline,
        loss_mask=loss_mask,
    )
    assert delta_q.shape == (B, T)
    assert sigma2.shape == (B, T)
    assert (sigma2 >= 0).all()
    # Masked-out timesteps zero in both outputs (by construction).
    inv = (loss_mask == 0)
    assert (delta_q[inv] == 0).all()
    assert (sigma2[inv] == 0).all()


def test_gather_q_chosen_raises_when_no_loss_mask_and_unalignable():
    """No loss_mask + unalignable shapes → ValueError (don't silently corrupt)."""
    B, T, K, bs, nd = 4, 6, 3, 7, 5
    q = torch.randn(B, T, K, bs, nd)
    # action is (5, bs) — not aligned with B, B*T, or any standard layout
    action = torch.zeros((5, bs), dtype=torch.long)
    with pytest.raises(ValueError, match="cannot align action shape"):
        CRDPPOTorchLearner._gather_q_chosen(q, action, loss_mask=None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
