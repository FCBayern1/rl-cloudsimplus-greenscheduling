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
        self.hook_calls = []  # observability for tests
        self._beta = beta
        self._gamma = gamma

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
