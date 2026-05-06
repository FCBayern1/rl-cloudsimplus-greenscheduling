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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
