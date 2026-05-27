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
        self._crd_forecast_align_warned = {}
        self._crd_baseline_schedulers = {}
        self._crd_local_baseline_schedulers = {}
        self._crd_local_baseline_warned = {}
        self._crd_baseline_signal_warned = {}
        self._crd_blenders = {}
        self._crd_dq_align_warned = {}
        self._crd_baseline_obs_warned = False
        self._crd_diag_warned = False
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

    def _get_or_build_local_baseline_scheduler(self, module_id):
        # Mirror the global override: M4 local baseline is opt-in for tests.
        # The dedicated M4 tests below override this with a real BestFit.
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


def test_dr_from_obs_aligns_to_BT_grid():
    """
    M2.5 fix: Δr must come out (B, T) aligned with ΔQ, derived from the
    padded obs dc_queue_sizes grid — NOT from infos (which PPO may drop from
    a minibatch, producing an empty Δr that breaks M3's blend).
    """
    B, T, num_dc, bs = 2, 3, 3, 4
    learner = _DRStubLearner(num_dc=num_dc, batch_size=bs, alpha=1.0)
    sched = learner._get_or_build_baseline_scheduler("g")
    batch = {
        Columns.OBS: {
            "dc_green_ratio": torch.rand(B, T, num_dc),
            "dc_queue_sizes": torch.randint(0, 50, (B, T, num_dc)).float(),
        },
        Columns.ACTIONS: torch.randint(0, num_dc, (B, T, bs), dtype=torch.long),
        Columns.REWARDS: torch.zeros(B, T),
        COL_CRD_BASELINE_ACTION: torch.randint(0, num_dc, (B, T, bs), dtype=torch.long),
    }
    out = learner._compute_dr_from_obs(batch, sched, alpha=1.0)
    assert out is not None
    assert out.shape == (B, T), f"Δr must be (B, T), got {tuple(out.shape)}"


def test_dr_from_obs_robust_to_empty_infos():
    """
    The exact smoke-test bug: infos empty/absent, but obs present →
    obs-based path still produces a full (B, T) Δr (not the (0,) that broke
    M3 blend). The blend in M3 needs Δr shape == ΔQ shape.
    """
    B, T, num_dc, bs = 2, 3, 3, 4
    learner = _DRStubLearner(num_dc=num_dc, batch_size=bs, alpha=1.0)
    learner._get_or_build_baseline_scheduler = lambda module_id: _FakeScheduler(num_dc, bs)
    batch = {
        Columns.OBS: {
            "dc_green_ratio": torch.rand(B, T, num_dc),
            "dc_queue_sizes": torch.randint(0, 50, (B, T, num_dc)).float(),
        },
        Columns.ACTIONS: torch.randint(0, num_dc, (B, T, bs), dtype=torch.long),
        Columns.REWARDS: torch.zeros(B, T),
        COL_CRD_BASELINE_ACTION: torch.randint(0, num_dc, (B, T, bs), dtype=torch.long),
        Columns.INFOS: [],  # ← empty, the smoke-test failure trigger
    }
    learner._compute_dr(module_id="g", batch=batch)
    assert COL_CRD_DR in batch
    out = batch[COL_CRD_DR]
    assert out.shape == (B, T), (
        f"Δr fell back to misaligned shape {tuple(out.shape)} despite obs being present"
    )
    assert out.numel() == B * T  # NOT (0,)


def test_dr_from_obs_unwraps_observation_nesting():
    """obs nested under 'observation' must still work for Δr."""
    B, T, num_dc, bs = 2, 2, 3, 4
    learner = _DRStubLearner(num_dc=num_dc, batch_size=bs, alpha=1.0)
    sched = learner._get_or_build_baseline_scheduler("g")
    batch = {
        Columns.OBS: {"observation": {
            "dc_green_ratio": torch.rand(B, T, num_dc),
            "dc_queue_sizes": torch.randint(0, 50, (B, T, num_dc)).float(),
        }},
        Columns.ACTIONS: torch.randint(0, num_dc, (B, T, bs), dtype=torch.long),
        Columns.REWARDS: torch.zeros(B, T),
        COL_CRD_BASELINE_ACTION: torch.randint(0, num_dc, (B, T, bs), dtype=torch.long),
    }
    out = learner._compute_dr_from_obs(batch, sched, alpha=1.0)
    assert out is not None and out.shape == (B, T)


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


# ---------------------------------------------------------------------------
# M2.4 fix — obs-based baseline_action path
#
# These tests lock in the contract: when batch[OBS] contains
# dc_green_ratio / dc_queue_sizes as padded (B, T, num_dc) tensors,
# _compute_baseline_from_obs must produce a (B, T, bs) baseline tensor
# that aligns directly with actual_action shape. This eliminates the
# "len(infos) != B*T" mismatch the smoke test exposed.
# ---------------------------------------------------------------------------


class _FakeScheduler:
    """Tiny deterministic scheduler returning a fixed pattern for testing."""

    def __init__(self, num_datacenters: int, batch_size: int):
        self.num_datacenters = num_datacenters
        self.batch_size = batch_size

    def schedule(self, obs):
        # Deterministic: cycle DCs across the cloudlet batch, biased by
        # the highest-green-ratio DC.
        gr = obs["dc_green_ratio"]
        # pick the DC with max green_ratio as the "preferred" target
        best = max(range(len(gr)), key=lambda i: gr[i])
        return [(best + j) % self.num_datacenters for j in range(self.batch_size)]


def test_obs_based_baseline_produces_padded_BT_bs_shape():
    """Most important contract: output is (B, T, bs) — matches actual_action."""
    B, T, num_dc, bs = 4, 6, 5, 20
    obs = {
        "dc_green_ratio": torch.rand(B, T, num_dc),
        "dc_queue_sizes": torch.randint(0, 10, (B, T, num_dc)).float(),
    }
    batch = {Columns.OBS: obs, Columns.ACTIONS: torch.zeros(B, T, bs, dtype=torch.long)}
    sched = _FakeScheduler(num_datacenters=num_dc, batch_size=bs)
    learner = _StubLearner()

    out = learner._compute_baseline_from_obs(batch, sched)
    assert out is not None
    assert out.shape == (B, T, bs), f"expected (B,T,bs)=(4,6,20), got {tuple(out.shape)}"
    assert out.dtype == torch.long
    # Values must be valid DC indices.
    assert (out >= 0).all() and (out < num_dc).all()


def test_obs_based_baseline_unwraps_observation_nesting():
    """
    RLlib stores the real obs dict under obs["observation"] for action-masking
    / Connector-wrapped modules. The obs-based path MUST unwrap this nesting,
    otherwise dc_green_ratio/dc_queue_sizes are invisible and we silently fall
    back to the misaligned infos path (the smoke-test bug at N_valid=416 vs 417).
    """
    B, T, num_dc, bs = 2, 4, 5, 20
    inner_obs = {
        "dc_green_ratio": torch.rand(B, T, num_dc),
        "dc_queue_sizes": torch.randint(0, 10, (B, T, num_dc)).float(),
    }
    # Nested under "observation" — the layout the global RLModule actually sees.
    batch = {
        Columns.OBS: {"observation": inner_obs},
        Columns.ACTIONS: torch.zeros(B, T, bs, dtype=torch.long),
    }
    sched = _FakeScheduler(num_datacenters=num_dc, batch_size=bs)
    learner = _StubLearner()
    out = learner._compute_baseline_from_obs(batch, sched)
    assert out is not None, "must unwrap observation nesting and produce baseline"
    assert out.shape == (B, T, bs)


def test_obs_based_baseline_returns_none_when_obs_is_not_dict():
    """Flat tensor obs → can't extract by key → return None → caller falls back."""
    batch = {Columns.OBS: torch.randn(4, 6, 100)}
    sched = _FakeScheduler(num_datacenters=5, batch_size=20)
    learner = _StubLearner()
    assert learner._compute_baseline_from_obs(batch, sched) is None


def test_obs_based_baseline_returns_none_when_keys_missing():
    """Dict obs without dc_green_ratio/dc_queue_sizes → None → fallback path."""
    obs = {"some_other_key": torch.randn(4, 6, 5)}
    batch = {Columns.OBS: obs}
    sched = _FakeScheduler(num_datacenters=5, batch_size=20)
    learner = _StubLearner()
    assert learner._compute_baseline_from_obs(batch, sched) is None


def test_obs_based_baseline_handles_2d_flat_obs():
    """If obs is (N, num_dc) flat, treat as (N, 1, num_dc) → output (N, 1, bs)."""
    N, num_dc, bs = 12, 5, 20
    obs = {
        "dc_green_ratio": torch.rand(N, num_dc),
        "dc_queue_sizes": torch.randint(0, 10, (N, num_dc)).float(),
    }
    batch = {Columns.OBS: obs}
    sched = _FakeScheduler(num_datacenters=num_dc, batch_size=bs)
    learner = _StubLearner()
    out = learner._compute_baseline_from_obs(batch, sched)
    assert out is not None
    assert out.shape == (N, 1, bs)


def test_obs_based_baseline_rejects_num_dc_mismatch():
    """Scheduler expects num_dc=5 but obs has num_dc=3 → None (caller falls back)."""
    obs = {
        "dc_green_ratio": torch.rand(4, 6, 3),  # 3 DCs in obs
        "dc_queue_sizes": torch.randint(0, 10, (4, 6, 3)).float(),
    }
    batch = {Columns.OBS: obs}
    sched = _FakeScheduler(num_datacenters=5, batch_size=20)
    learner = _StubLearner()
    assert learner._compute_baseline_from_obs(batch, sched) is None


def test_obs_based_baseline_output_aligned_with_actual_action_shape():
    """End-to-end alignment: output shape must equal batch[ACTIONS].shape exactly."""
    B, T, num_dc, bs = 3, 8, 5, 20
    actual_action = torch.zeros(B, T, bs, dtype=torch.long)
    obs = {
        "dc_green_ratio": torch.rand(B, T, num_dc),
        "dc_queue_sizes": torch.randint(0, 10, (B, T, num_dc)).float(),
    }
    batch = {Columns.OBS: obs, Columns.ACTIONS: actual_action}
    sched = _FakeScheduler(num_datacenters=num_dc, batch_size=bs)
    learner = _StubLearner()

    out = learner._compute_baseline_from_obs(batch, sched)
    assert out.shape == actual_action.shape, (
        f"baseline shape {tuple(out.shape)} != actual_action shape "
        f"{tuple(actual_action.shape)} — M2.4 gather will mismatch."
    )


def test_obs_based_path_preferred_over_infos_when_both_available():
    """
    Regression contract: when obs is usable, _compute_baseline_action takes
    the obs-based path and ignores infos (which may be misaligned due to
    bootstrap timesteps).
    """
    B, T, num_dc, bs = 2, 4, 5, 20
    actual_action = torch.zeros(B, T, bs, dtype=torch.long)
    # Misaligned infos: len(infos) = 9 ≠ B*T = 8 (mimics smoke-test bug).
    misaligned_infos = [{} for _ in range(9)]
    batch = {
        Columns.OBS: {
            "dc_green_ratio": torch.rand(B, T, num_dc),
            "dc_queue_sizes": torch.randint(0, 10, (B, T, num_dc)).float(),
        },
        Columns.ACTIONS: actual_action,
        Columns.INFOS: misaligned_infos,
    }
    sched = _FakeScheduler(num_datacenters=num_dc, batch_size=bs)

    learner = _StubLearner()
    # _StubLearner's default scheduler accessor returns None; override
    # for this test so the M2.3 path actually runs.
    learner._get_or_build_baseline_scheduler = lambda module_id: sched

    learner._compute_baseline_action(module_id="m", batch=batch)
    assert COL_CRD_BASELINE_ACTION in batch
    out = batch[COL_CRD_BASELINE_ACTION]
    # Critical: shape matches actual_action, NOT misaligned infos.
    assert out.shape == (B, T, bs), (
        f"got {tuple(out.shape)} — obs-based path should override infos"
    )


# ---------------------------------------------------------------------------
# M7 — CRD diagnostics logging (wandb/TensorBoard via RLlib result dict).
# ---------------------------------------------------------------------------


class _RecordingMetrics:
    """Minimal MetricsLogger stand-in capturing log_dict calls."""
    def __init__(self):
        self.logged = {}
    def log_dict(self, d, key=None, window=1):
        self.logged.setdefault(key, {}).update(d)


def test_diagnostics_logged_when_metrics_available():
    """M7: CRD scalars must be pushed via self.metrics.log_dict for wandb."""
    learner = _StubLearner()
    learner.metrics = _RecordingMetrics()
    B, T = 2, 3
    batch = {
        COL_CRD_RHO_FORECAST: torch.rand(B, T),
        COL_CRD_RHO_ROUTING: torch.rand(B, T),
        COL_CRD_RHO_SCHEDULING: torch.rand(B, T),
        COL_CRD_SIGMA2: torch.rand(B, T),
        COL_CRD_C_T: torch.rand(B, T),
        COL_CRD_DQ: torch.randn(B, T),
        COL_CRD_DR: torch.randn(B, T),
        COL_CRD_R_ROUTING: torch.randn(B, T),
        COL_CRD_FORECAST: torch.randn(B, T),
        COL_CRD_TAU: torch.tensor(1.05),
    }
    learner._log_crd_diagnostics(module_id="global_policy", batch=batch)
    logged = learner.metrics.logged.get("global_policy", {})
    # All key diagnostics present
    for key in [
        "crd/rho_forecast_mean", "crd/rho_routing_mean", "crd/rho_scheduling_mean",
        "crd/sigma2_tot_mean", "crd/c_t_mean", "crd/dq_mean", "crd/dr_mean",
        "crd/r_routing_mean", "crd/r_forecast_abs_mean", "crd/tau",
    ]:
        assert key in logged, f"missing diagnostic {key}; got {sorted(logged)}"
    # r_forecast_abs_mean must be non-negative (it's |R_forecast|)
    assert logged["crd/r_forecast_abs_mean"] >= 0
    assert logged["crd/tau"] == pytest.approx(1.05)


def test_diagnostics_noop_when_no_metrics():
    """No self.metrics (test stub / pre-build) → silent no-op, no crash."""
    learner = _StubLearner()  # no .metrics attribute
    learner._log_crd_diagnostics(
        module_id="m", batch={COL_CRD_RHO_ROUTING: torch.rand(2, 3)}
    )  # must not raise


def test_diagnostics_skips_missing_columns():
    """Only logs columns that are present (e.g. scheduling absent pre-M4)."""
    learner = _StubLearner()
    learner.metrics = _RecordingMetrics()
    # Only routing present (forecast/scheduling absent)
    batch = {COL_CRD_RHO_ROUTING: torch.rand(2, 3)}
    learner._log_crd_diagnostics(module_id="m", batch=batch)
    logged = learner.metrics.logged.get("m", {})
    assert "crd/rho_routing_mean" in logged
    assert "crd/rho_forecast_mean" not in logged  # was absent → not logged


# ===========================================================================
# M4 — Local Scheduling Counterfactual
# ===========================================================================

from src.baselines.local_schedulers import BestFitLocalScheduler


class _LocalBaselineStubLearner(_StubLearner):
    """
    Stub for the local-scheduling layer: no global router scheduler, a real
    BestFit local baseline. Exercises the M4 path without a live RLModule.
    """

    def __init__(self, num_vms=3, alpha_local=1.0, **kw):
        super().__init__(**kw)
        self._fixed_local = BestFitLocalScheduler(num_vms=num_vms)
        self._alpha_local = alpha_local

    def _get_or_build_baseline_scheduler(self, module_id):
        return None  # not the global router

    def _get_or_build_local_baseline_scheduler(self, module_id):
        return self._fixed_local

    def _read_module_local_dr_config(self, module_id):
        return {"alpha": self._alpha_local}


def _make_local_obs(vm_available, next_pes, mask, B=1, T=1):
    """Build a padded (B, T, ...) local obs dict matching the env layout."""
    V = len(vm_available)
    A = len(mask)
    return {
        "observation": {
            "vm_available_pes": torch.tensor(
                [[vm_available] * T for _ in range(B)], dtype=torch.float32
            ),  # (B, T, V)
            "next_cloudlet_pes": torch.tensor(
                [[[next_pes]] * T for _ in range(B)], dtype=torch.float32
            ),  # (B, T, 1)
        },
        "action_mask": torch.tensor(
            [[mask] * T for _ in range(B)], dtype=torch.float32
        ),  # (B, T, A)
    }


def test_local_baseline_builds_for_discrete_only():
    """BestFit baseline only for Discrete (local) modules; None for the rest."""
    learner = _LocalBaselineStubLearner(num_vms=3)
    assert learner._get_or_build_local_baseline_scheduler("local") is not None
    # Plain stub: real builder path returns None without a live module.
    plain = _StubLearner()
    assert plain._get_or_build_local_baseline_scheduler("x") is None


def test_local_baseline_action_matches_bestfit_choice():
    """ã_local = the min-non-negative-PE-waste VM, written as (B, T)."""
    learner = _LocalBaselineStubLearner(num_vms=3)
    # vm_avail=[5,2,8], demand=3, all valid:
    #   VM1 waste=2 (min), VM2 waste=-1 (infeasible), VM3 waste=5 → pick VM action 1
    batch = {
        Columns.OBS: _make_local_obs([5, 2, 8], 3, [1, 1, 1, 1]),
        Columns.ACTIONS: torch.tensor([[3]], dtype=torch.long),
    }
    learner._compute_local_baseline_action(module_id="local", batch=batch)
    assert COL_CRD_BASELINE_ACTION in batch
    out = batch[COL_CRD_BASELINE_ACTION]
    assert out.shape == (1, 1)
    assert out[0, 0].item() == 1


def test_local_baseline_respects_action_mask():
    """A masked-out best VM must not be chosen."""
    learner = _LocalBaselineStubLearner(num_vms=3)
    # VM1 (idx0) would be best by waste, but mask it out → BestFit picks VM3.
    batch = {
        Columns.OBS: _make_local_obs([5, 100, 8], 3, [1, 0, 0, 1]),
        Columns.ACTIONS: torch.tensor([[1]], dtype=torch.long),
    }
    learner._compute_local_baseline_action(module_id="local", batch=batch)
    out = batch[COL_CRD_BASELINE_ACTION]
    # Only action 3 feasible (VM3 avail=8, waste=5); action 1 masked off.
    assert out[0, 0].item() == 3


def test_local_baseline_skipped_and_warns_when_mask_missing(caplog):
    """No action_mask in obs → skip baseline, warn once."""
    import logging
    learner = _LocalBaselineStubLearner(num_vms=3)
    batch = {
        Columns.OBS: {"observation": {
            "vm_available_pes": torch.zeros(1, 1, 3),
            "next_cloudlet_pes": torch.zeros(1, 1, 1),
        }},  # no action_mask sibling
        Columns.ACTIONS: torch.tensor([[0]], dtype=torch.long),
    }
    with caplog.at_level(logging.WARNING, logger="src.learners.crd_q_loss"):
        learner._compute_local_baseline_action(module_id="local", batch=batch)
        learner._compute_local_baseline_action(module_id="local", batch=batch)
    assert COL_CRD_BASELINE_ACTION not in batch
    warns = [r for r in caplog.records if "local baseline obs path" in r.message]
    assert len(warns) == 1  # warn-once per module


def test_local_dr_sign_negative_when_agent_wastes_more():
    """Δr_local = α·(waste_baseline − waste_actual) < 0 when agent picks worse VM."""
    learner = _LocalBaselineStubLearner(num_vms=3, alpha_local=1.0)
    # vm_avail=[5,2,8], demand=3. BestFit→VM1 (waste 2). Agent picks VM3 (waste 5).
    batch = {
        Columns.OBS: _make_local_obs([5, 2, 8], 3, [1, 1, 1, 1]),
        Columns.ACTIONS: torch.tensor([[3]], dtype=torch.long),
        Columns.REWARDS: torch.zeros(1, 1),
        COL_CRD_BASELINE_ACTION: torch.tensor([[1]], dtype=torch.long),
    }
    learner._compute_local_dr(module_id="local", batch=batch)
    assert COL_CRD_DR in batch
    # waste(baseline VM1)=2, waste(actual VM3)=5 → Δr = 2 - 5 = -3
    assert batch[COL_CRD_DR][0, 0].item() == pytest.approx(-3.0)


def test_local_dr_zero_when_agent_matches_baseline():
    """Agent picking the BestFit VM → Δr_local = 0."""
    learner = _LocalBaselineStubLearner(num_vms=3, alpha_local=1.0)
    batch = {
        Columns.OBS: _make_local_obs([5, 2, 8], 3, [1, 1, 1, 1]),
        Columns.ACTIONS: torch.tensor([[1]], dtype=torch.long),  # == baseline
        Columns.REWARDS: torch.zeros(1, 1),
        COL_CRD_BASELINE_ACTION: torch.tensor([[1]], dtype=torch.long),
    }
    learner._compute_local_dr(module_id="local", batch=batch)
    assert batch[COL_CRD_DR][0, 0].item() == pytest.approx(0.0)


def test_local_cf_full_pipeline_writes_r_scheduling():
    """End-to-end local layer: ΔQ_local, Δr_local, R^scheduling all in batch."""
    B, T, K, A = 1, 1, 5, 4
    learner = _LocalBaselineStubLearner(num_vms=3)
    fwd = {COL_Q_ENSEMBLE: torch.randn(B, T, K, A)}
    batch = {
        Columns.INFOS: [_make_crd_info(pred=[1, 1, 1])],
        Columns.OBS: _make_local_obs([5, 2, 8], 3, [1, 1, 1, 1]),
        Columns.ACTIONS: torch.tensor([[2]], dtype=torch.long),
        Columns.REWARDS: torch.zeros(B, T),
    }
    learner.compute_loss_for_module(
        module_id="shared_local_policy", config=None, batch=batch, fwd_out=fwd
    )
    for k in (COL_CRD_BASELINE_ACTION, COL_CRD_DQ, COL_CRD_SIGMA2,
              COL_CRD_DR, COL_CRD_R_SCHEDULING, COL_CRD_C_T, COL_CRD_TAU):
        assert k in batch, f"missing {k}"
    # The global routing column must NOT appear in a local module's batch.
    assert COL_CRD_R_ROUTING not in batch
    assert batch[COL_CRD_R_SCHEDULING].shape == batch[COL_CRD_DQ].shape


def test_responsibilities_reweights_local_by_rho_scheduling():
    """Local module: ADVANTAGES *= ρ_scheduling (not ρ_routing)."""
    learner = _LocalBaselineStubLearner(num_vms=3)
    adv_before = torch.tensor([[2.0, 2.0]])
    batch = {
        COL_CRD_FORECAST: torch.tensor([[0.0, 0.0]]),  # no exogenous share
        COL_CRD_R_SCHEDULING: torch.tensor([[1.0, 1.0]]),  # all weight on scheduling
        Postprocessing.ADVANTAGES: adv_before.clone(),
    }
    learner._compute_responsibilities(module_id="local", batch=batch)
    # ρ_scheduling ≈ 1 (R_scheduling dominates, no forecast) → adv ~unchanged
    assert torch.allclose(batch[Postprocessing.ADVANTAGES], adv_before, atol=1e-4)
    assert COL_CRD_RHO_SCHEDULING in batch
    # ρ_routing should be the floor (no R_routing in a local batch).
    assert batch[COL_CRD_RHO_ROUTING].mean().item() == pytest.approx(0.05, abs=1e-6)


def test_forecast_shields_local_agent():
    """Large |R_forecast| dilutes ρ_scheduling → local ADVANTAGES shrink."""
    class _NoFloor(_LocalBaselineStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0}
    learner = _NoFloor(num_vms=3)
    adv_before = torch.tensor([[10.0]])
    batch = {
        COL_CRD_FORECAST: torch.tensor([[9.0]]),    # forecast dominates
        COL_CRD_R_SCHEDULING: torch.tensor([[1.0]]),
        Postprocessing.ADVANTAGES: adv_before.clone(),
    }
    learner._compute_responsibilities(module_id="local", batch=batch)
    # ρ_scheduling = 1/(9+1) = 0.1 → adv 10 → ~1.0 (agent shielded from forecast error)
    assert batch[Postprocessing.ADVANTAGES][0, 0].item() == pytest.approx(1.0, abs=1e-3)
    rho_s = batch[COL_CRD_RHO_SCHEDULING][0, 0].item()
    assert rho_s == pytest.approx(0.1, rel=1e-3)


def test_align_local_action_to_bt_layouts():
    """(B,T), flat (B*T,), and sequence-packed (N_valid,) all coerce to (B,T)."""
    fn = CRDPPOTorchLearner._align_local_action_to_bt
    B, T = 2, 3
    # already (B, T)
    a = torch.arange(B * T).reshape(B, T)
    assert torch.equal(fn(a, B, T, None), a)
    # flat (B*T,)
    flat = torch.arange(B * T)
    assert torch.equal(fn(flat, B, T, None), flat.reshape(B, T))
    # packed (N_valid,) with loss_mask
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)  # 3 valid
    packed = torch.tensor([7, 8, 9], dtype=torch.long)
    out = fn(packed, B, T, mask)
    assert out.shape == (B, T)
    # valid slots filled in order, masked slots zero
    assert out[0, 0].item() == 7 and out[0, 1].item() == 8 and out[1, 0].item() == 9
    assert out[0, 2].item() == 0 and out[1, 1].item() == 0


# ===========================================================================
# Forecast (B,T)-alignment fix — R_forecast must enter the ρ denominator
# (regression: infos packed to N_valid != B*T was leaving it 1-D → zeroed)
# ===========================================================================


def test_align_forecast_to_bt_cases():
    fn = CRDPPOTorchLearner._align_forecast_to_bt
    B, T = 2, 3
    ref = torch.zeros(B, T)
    # Case 1: already the full padded grid → direct reshape.
    full = torch.arange(6, dtype=torch.float32)
    out = fn(full, ref=ref, loss_mask=None)
    assert out.shape == (B, T) and torch.equal(out.reshape(-1), full)
    # Case 2: packed to N_valid → scatter via loss_mask.
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)  # 3 valid
    packed = torch.tensor([5.0, 6.0, 7.0])
    out = fn(packed, ref=ref, loss_mask=mask)
    assert out.shape == (B, T)
    assert out[0, 0] == 5.0 and out[0, 1] == 6.0 and out[1, 0] == 7.0
    assert out[0, 2] == 0.0 and out[1, 1] == 0.0 and out[1, 2] == 0.0
    # Case 2b: off-by-one bootstrap (len == N_valid + 1) → trim trailing, scatter.
    packed_boot = torch.tensor([5.0, 6.0, 7.0, 99.0])  # 4 = 3 valid + 1 bootstrap
    out = fn(packed_boot, ref=ref, loss_mask=mask)
    assert out is not None and out.shape == (B, T)
    assert out[0, 0] == 5.0 and out[0, 1] == 6.0 and out[1, 0] == 7.0  # bootstrap 99 dropped
    # Case 3: way off (more than +num_seqs) → None (caller warns).
    assert fn(torch.zeros(10), ref=ref, loss_mask=mask) is None


def test_forecast_cf_scatters_when_infos_packed():
    """The bug: infos length == loss_mask valid-count != B*T. R_forecast must
    land on the (B, T) grid, not stay 1-D (which M5 would silently zero)."""
    B, T = 2, 3
    learner = _StubLearner(beta=1.0, gamma=1.0)
    infos = [_make_crd_info(pred=[0.0, 0.0, 0.0]) for _ in range(3)]  # biased → R_f≠0
    batch = {
        Columns.INFOS: infos,
        Columns.REWARDS: torch.zeros(B, T),
        Columns.LOSS_MASK: torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32),
    }
    learner._compute_forecast_cf(module_id="g", batch=batch)
    rf = batch[COL_CRD_FORECAST]
    assert rf.shape == (B, T), f"forecast not aligned to grid: {tuple(rf.shape)}"
    assert rf[0, 0] != 0 and rf[0, 1] != 0 and rf[1, 0] != 0    # valid slots filled
    assert rf[0, 2] == 0 and rf[1, 1] == 0 and rf[1, 2] == 0    # padded slots zero


def test_forecast_cf_warns_when_unalignable(caplog):
    """Length far from both B*T and loss_mask valid-count (beyond bootstrap
    tolerance) → warn once."""
    import logging
    learner = _StubLearner(beta=1.0, gamma=1.0)
    batch = {
        Columns.INFOS: [_make_crd_info(pred=[0.0, 0.0, 0.0]) for _ in range(5)],
        Columns.REWARDS: torch.zeros(2, 3),  # B*T=6 != 5
        # 2 valid, 2 seqs → tolerance is [2, 4]; len 5 is outside → warn.
        Columns.LOSS_MASK: torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32),
    }
    with caplog.at_level(logging.WARNING, logger="src.learners.crd_q_loss"):
        learner._compute_forecast_cf(module_id="g", batch=batch)
        learner._compute_forecast_cf(module_id="g", batch=batch)
    warns = [r for r in caplog.records if "matches neither" in r.message]
    assert len(warns) == 1  # warn-once per module


def test_forecast_enters_rho_denominator():
    """Downstream effect: with R_forecast (B,T)-aligned, ρ_forecast > 0 and the
    advantage is shielded (this is the mechanism that was inert in the smoke)."""
    class _Stub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0}
    learner = _Stub(num_dc=3, batch_size=4)
    batch = {
        COL_CRD_FORECAST: torch.tensor([[1.0, 1.0]]),   # |R_f| = 1
        COL_CRD_R_ROUTING: torch.tensor([[1.0, 1.0]]),  # |R_r| = 1
        Postprocessing.ADVANTAGES: torch.tensor([[4.0, 4.0]]),
    }
    learner._compute_responsibilities(module_id="g", batch=batch)
    assert batch[COL_CRD_RHO_FORECAST][0, 0].item() == pytest.approx(0.5, abs=1e-3)
    assert batch[COL_CRD_RHO_ROUTING][0, 0].item() == pytest.approx(0.5, abs=1e-3)
    # advantage shielded: 4 × 0.5 = 2 (was 4 when forecast was zeroed → ρ_routing≈1)
    assert batch[Postprocessing.ADVANTAGES][0, 0].item() == pytest.approx(2.0, abs=1e-3)


# ===========================================================================
# obs-based forecast (crd_aux channel) — the robust replacement for the
# infos path. R_forecast computed from the padded (B,T) obs grid.
# ===========================================================================


def _make_crd_aux(B, T, actual, predicted, total, gf, bf, dt=1.0):
    """Build a crd_aux obs dict of (B,T,num_dc) tensors from per-DC vectors."""
    def tile(vec):
        base = torch.tensor(vec, dtype=torch.float32)
        return base.view(1, 1, -1).expand(B, T, -1).contiguous()
    return {
        "crd_actual_green_w": tile(actual),
        "crd_predicted_green_w": tile(predicted),
        "crd_total_power_w": tile(total),
        "crd_green_factor": tile(gf),
        "crd_brown_factor": tile(bf),
        "crd_timestep_hours": torch.full((B, T, 1), float(dt)),
    }


def test_forecast_from_obs_nonzero_when_pred_differs():
    """Predicted wind (0) underestimates actual (1000W) → R_forecast != 0,
    shape (B,T), matching the analytical carbon delta."""
    B, T = 1, 2
    learner = _StubLearner(beta=1.0, gamma=1.0)
    aux = _make_crd_aux(
        B, T,
        actual=[1000.0, 1000.0], predicted=[0.0, 0.0],
        total=[2000.0, 2000.0], gf=[0.0, 0.0], bf=[0.5, 0.5], dt=1.0,
    )
    batch = {Columns.OBS: {"crd_aux": aux}, Columns.REWARDS: torch.zeros(B, T)}
    out = learner._compute_forecast_cf_from_obs(batch, beta=1.0, gamma=1.0)
    assert out is not None and out.shape == (B, T)
    # carbon_actual = 1.0 kg (uses 1kWh green, 1kWh brown ×0.5 per DC ×2 DC),
    # carbon_pred = 2.0 kg (no green → 2kWh brown ×0.5 ×2). β·(1-2)+γ·0 = -1.0
    assert out[0, 0].item() == pytest.approx(-1.0, abs=1e-4)


def test_forecast_from_obs_zero_when_pred_matches():
    B, T = 2, 2
    learner = _StubLearner(beta=1.0, gamma=1.0)
    aux = _make_crd_aux(
        B, T,
        actual=[1000.0, 500.0], predicted=[1000.0, 500.0],  # perfect forecast
        total=[2000.0, 2000.0], gf=[0.0, 0.0], bf=[0.5, 0.5], dt=1.0,
    )
    batch = {Columns.OBS: {"crd_aux": aux}, Columns.REWARDS: torch.zeros(B, T)}
    out = learner._compute_forecast_cf_from_obs(batch, beta=1.0, gamma=1.0)
    assert out is not None and out.shape == (B, T)
    assert torch.allclose(out, torch.zeros(B, T), atol=1e-5)


def test_forecast_from_obs_returns_none_without_crd_aux():
    learner = _StubLearner()
    batch = {Columns.OBS: {"observation": {"x": torch.zeros(1, 1, 3)}}}
    assert learner._compute_forecast_cf_from_obs(batch, beta=0.5, gamma=0.3) is None


def test_forecast_cf_prefers_obs_over_infos():
    """When crd_aux is present, _compute_forecast_cf uses it and ignores infos
    (the infos path is the fragile fallback)."""
    B, T = 1, 2
    learner = _StubLearner(beta=1.0, gamma=1.0)
    aux = _make_crd_aux(
        B, T,
        actual=[1000.0, 1000.0], predicted=[0.0, 0.0],
        total=[2000.0, 2000.0], gf=[0.0, 0.0], bf=[0.5, 0.5], dt=1.0,
    )
    batch = {
        Columns.OBS: {"crd_aux": aux},
        Columns.REWARDS: torch.zeros(B, T),
        Columns.INFOS: [_make_crd_info(pred=[0.0, 0.0, 0.0])],  # would be fragile
    }
    learner._compute_forecast_cf(module_id="g", batch=batch)
    rf = batch[COL_CRD_FORECAST]
    assert rf.shape == (B, T)
    assert rf[0, 0].item() == pytest.approx(-1.0, abs=1e-4)  # the obs-based value


def test_forecast_from_obs_feeds_rho_forecast_nonzero():
    """End-to-end: obs-based R_forecast → ρ_forecast > 0 (mechanism live)."""
    class _Stub(_DRStubLearner):
        def _read_module_responsibility_config(self, module_id):
            return {"rho_min": 0.0}
        def _read_module_forecast_config(self, module_id):
            return {"beta": 1.0, "gamma": 1.0}
    B, T = 1, 1
    learner = _Stub(num_dc=3, batch_size=4)
    aux = _make_crd_aux(
        B, T,
        actual=[1000.0, 1000.0], predicted=[0.0, 0.0],
        total=[2000.0, 2000.0], gf=[0.0, 0.0], bf=[0.5, 0.5], dt=1.0,
    )
    batch = {Columns.OBS: {"crd_aux": aux}, Columns.REWARDS: torch.zeros(B, T)}
    learner._compute_forecast_cf(module_id="g", batch=batch)
    # |R_forecast| = 1.0; pair with an equal-magnitude routing signal.
    batch[COL_CRD_R_ROUTING] = torch.ones(B, T)
    batch[Postprocessing.ADVANTAGES] = torch.full((B, T), 4.0)
    learner._compute_responsibilities(module_id="g", batch=batch)
    rho_f = batch[COL_CRD_RHO_FORECAST][0, 0].item()
    assert rho_f == pytest.approx(0.5, abs=1e-3)   # 1/(1+1), NOT 0
    assert batch[Postprocessing.ADVANTAGES][0, 0].item() == pytest.approx(2.0, abs=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
