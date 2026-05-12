"""
Regression test for the 2026-05-12 Fix 4 to LagrangianCallback.

Background: the dual update used `result["env_runners"]["c_ep_mean"]`, which
RLlib computes via `MetricsLogger.log_value(..., reduce="mean")` and is a
windowed mean over the last `metrics_num_episodes_for_smoothing` episodes
(default 100).  At our training scale (~1 episode/iter) that's effectively
a lifetime average — λ ramps linearly even after recent episodes have
improved (see run 20260509_011407: λ 0.017 → 1.247 over 76 iter while
completion plateaued).

Fix: callback also buffers c_ep values seen via `on_episode_end` and uses
the buffer's per-iter mean in `on_train_result` (clearing afterwards).
Falls back to the windowed metric when no episode finished this iter.
"""
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def callback():
    """Build a LagrangianCallback with hyperparams already 'loaded' and Lagrangian enabled."""
    from src.callbacks.lagrangian_callback import LagrangianCallback

    cb = LagrangianCallback(log_dir=None)
    cb._hyperparams_loaded = True
    cb._enabled = True
    cb._lambda = 0.0
    cb._lambda_init = 0.0
    cb._lambda_lr = 0.5
    cb._lambda_max = 20.0
    cb._c_ep_tolerance = 0.02
    return cb


def test_buffers_start_empty(callback):
    """Fresh callback must have empty per-iter buffers."""
    assert callback._iter_c_ep_values == []
    assert callback._iter_completion_values == []
    assert callback._iter_c_step_values == []
    assert callback._iter_pending_values == []


def test_per_iter_mean_used_when_buffer_populated(callback):
    """When episodes occurred this iter, dual update uses THEIR mean, not the windowed mean."""
    # Simulate three episodes this iter, with c_ep values 0.01, 0.02, 0.03.
    # The legacy windowed mean (mocked at result["env_runners"]["c_ep_mean"])
    # is set to a different value (0.10) — Fix 4 must prefer the buffer.
    callback._iter_c_ep_values = [0.01, 0.02, 0.03]
    callback._iter_completion_values = [0.85, 0.86, 0.87]
    callback._iter_c_step_values = [0.05, 0.05, 0.05]
    callback._iter_pending_values = [0.15, 0.14, 0.13]

    result = {
        "training_iteration": 1,
        "env_runners": {
            "num_episodes": 3,
            "c_ep_mean": 0.10,        # legacy windowed mean — should be IGNORED
            "completion_rate_mi": 0.0,
            "c_step_mean": 0.0,
            "sla_pending_ratio": 0.0,
        },
        "num_env_steps_sampled_lifetime": 8000,
    }

    # Mock _foreach_env_safely so we don't have to fabricate a full algorithm.
    with patch("src.callbacks.lagrangian_callback._foreach_env_safely"):
        callback.on_train_result(algorithm=MagicMock(), result=result)

    # Expected: λ_new = max(0, 0 + 0.5·(0.02 − 0.02)) = 0.  Since c_ep = 0.02
    # exactly equals tolerance, violation = 0 → decay branch fires → λ stays 0.
    # (We're testing that the per-iter mean WAS used; the value at exactly
    # tolerance is the discriminating one — windowed 0.10 would have driven λ up.)
    assert callback._lambda == pytest.approx(0.0, abs=1e-12), (
        f"λ should stay 0 with per-iter c_ep ≈ tol; got {callback._lambda} "
        "(if you see 0.04, the legacy windowed value was used instead)"
    )

    # Buffers MUST be cleared after consumption.
    assert callback._iter_c_ep_values == []
    assert callback._iter_completion_values == []
    assert callback._iter_c_step_values == []
    assert callback._iter_pending_values == []


def test_falls_back_to_windowed_when_no_episode_this_iter(callback):
    """If no episode finished, use the legacy windowed metric (graceful fallback)."""
    # Buffers empty — should consult result["env_runners"]["c_ep_mean"].
    result = {
        "training_iteration": 5,
        "env_runners": {
            "num_episodes": 0,
            "c_ep_mean": 0.08,        # legacy windowed mean
            "completion_rate_mi": 0.79,
            "c_step_mean": 0.18,
            "sla_pending_ratio": 0.21,
        },
        "num_env_steps_sampled_lifetime": 40000,
    }

    with patch("src.callbacks.lagrangian_callback._foreach_env_safely"):
        callback.on_train_result(algorithm=MagicMock(), result=result)

    # With ep_count=0 the dual update is a no-op (lam_new = lam_prev = 0).
    # Test that NO error was raised and λ is unchanged.
    assert callback._lambda == pytest.approx(0.0, abs=1e-12)


def test_per_iter_buffer_clears_between_iters(callback):
    """Two consecutive iters must NOT leak c_ep values across each other."""
    # Iter 1: three bad episodes.
    callback._iter_c_ep_values = [0.10, 0.10, 0.10]
    callback._iter_completion_values = [0.75, 0.75, 0.75]
    callback._iter_c_step_values = [0.18, 0.18, 0.18]
    callback._iter_pending_values = [0.25, 0.25, 0.25]
    result_iter1 = {
        "training_iteration": 1,
        "env_runners": {"num_episodes": 3, "c_ep_mean": 0.0,
                        "completion_rate_mi": 0.0, "c_step_mean": 0.0,
                        "sla_pending_ratio": 0.0},
        "num_env_steps_sampled_lifetime": 8000,
    }
    with patch("src.callbacks.lagrangian_callback._foreach_env_safely"):
        callback.on_train_result(algorithm=MagicMock(), result=result_iter1)

    lambda_after_iter1 = callback._lambda
    assert lambda_after_iter1 > 0.0, "Bad iter should ramp λ"

    # Now buffers should be empty.  Push three GOOD episodes for iter 2.
    callback._iter_c_ep_values = [0.0, 0.0, 0.0]
    callback._iter_completion_values = [0.92, 0.92, 0.92]
    callback._iter_c_step_values = [0.04, 0.04, 0.04]
    callback._iter_pending_values = [0.08, 0.08, 0.08]
    result_iter2 = {
        "training_iteration": 2,
        "env_runners": {"num_episodes": 3, "c_ep_mean": 0.10,
                        "completion_rate_mi": 0.0, "c_step_mean": 0.0,
                        "sla_pending_ratio": 0.0},
        "num_env_steps_sampled_lifetime": 16000,
    }
    with patch("src.callbacks.lagrangian_callback._foreach_env_safely"):
        callback.on_train_result(algorithm=MagicMock(), result=result_iter2)

    # If iter 1's bad values had leaked, mean would still be > tol and λ would
    # keep ramping.  With Fix 4 (per-iter buffer + clear) the good iter 2
    # should trigger the decay branch (violation ≤ 0 → λ ← 0.95·λ_prev).
    assert callback._lambda < lambda_after_iter1, (
        f"Good iter should decay λ from {lambda_after_iter1:.4f}, "
        f"got {callback._lambda:.4f} — likely buffer didn't clear"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
