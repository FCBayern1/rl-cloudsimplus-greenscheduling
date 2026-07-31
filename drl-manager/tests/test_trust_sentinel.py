"""EU-CRD deployment-time trust sentinel (src/baselines/trust_sentinel.py).

The sentinel reuses the trained Q-head ensemble as an epistemic-uncertainty
sensor at inference and (in gate mode) suppresses the global DEFER logit when
disagreement exceeds a calibrated threshold. These tests cover the load path
(learner-side module_state, shape inference, vanilla rejection) and the gate
semantics (above-threshold suppresses ONLY the defer column; below-threshold
and log-mode leave logits bitwise unchanged).

Run from drl-manager:  python -m pytest tests/test_trust_sentinel.py -v
"""
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.trust_sentinel import (
    TrustSentinel,
    load_q_heads_from_checkpoint,
)
from src.models.rlmodule_gtrxl_ensemble import EnsembleQHeads

D_MODEL = 16
HIDDEN = 8
K = 3
NUM_SLOTS = 4
NUM_CHOICES = 6  # 5 DCs + DEFER
ACTION_DIM = NUM_SLOTS * NUM_CHOICES


def _write_fake_checkpoint(tmp_path: Path, include_q_heads: bool = True) -> tuple:
    """Create <ckpt>/learner_group/learner/rl_module/global_policy/module_state.pt
    mimicking Ray's pickled-OrderedDict layout. Returns (ckpt_dir, heads)."""
    heads = EnsembleQHeads(
        d_model=D_MODEL, action_dim=ACTION_DIM, K=K,
        prior_lambda=3.0, hidden_dim=HIDDEN,
    )
    state = OrderedDict()
    state["policy_head.weight"] = torch.randn(4, D_MODEL)  # unrelated key
    if include_q_heads:
        for k, v in heads.state_dict().items():
            state[f"q_heads.{k}"] = v
    mod_dir = tmp_path / "learner_group" / "learner" / "rl_module" / "global_policy"
    mod_dir.mkdir(parents=True)
    with open(mod_dir / "module_state.pt", "wb") as f:
        pickle.dump(state, f)
    return tmp_path, heads


def test_load_reconstructs_trained_weights(tmp_path):
    ckpt, heads = _write_fake_checkpoint(tmp_path)
    loaded = load_q_heads_from_checkpoint(str(ckpt))
    assert loaded.K == K
    assert loaded.action_dim == ACTION_DIM
    x = torch.randn(2, D_MODEL)
    torch.testing.assert_close(loaded(x), heads(x))


def test_vanilla_checkpoint_rejected(tmp_path):
    ckpt, _ = _write_fake_checkpoint(tmp_path, include_q_heads=False)
    with pytest.raises(ValueError, match="vanilla"):
        load_q_heads_from_checkpoint(str(ckpt))


def test_gate_requires_threshold(tmp_path):
    ckpt, _ = _write_fake_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="TRUST_GATE_THRESH"):
        TrustSentinel(str(ckpt), num_slots=NUM_SLOTS, mode="gate", threshold=None)


def _sentinel(ckpt, **kw):
    defaults = dict(num_slots=NUM_SLOTS, mode="gate", threshold=0.0,
                    signal="defer", log_path=None)
    defaults.update(kw)
    return TrustSentinel(str(ckpt), **defaults)


def test_above_threshold_suppresses_only_defer_column(tmp_path):
    ckpt, _ = _write_fake_checkpoint(tmp_path)
    s = _sentinel(ckpt, threshold=-1.0)  # any real sigma² > -1 → always gates
    feats = torch.randn(1, 5, D_MODEL)   # (B, T, d_model)
    sigma = s.measure(feats)
    assert sigma >= 0.0
    logits = torch.randn(1, NUM_SLOTS, NUM_CHOICES)
    out = s.maybe_gate(logits.clone())
    assert torch.all(out[..., -1] == torch.finfo(out.dtype).min)
    torch.testing.assert_close(out[..., :-1], logits[..., :-1])
    assert "gated=1" in s.summary()


def test_below_threshold_leaves_logits_unchanged(tmp_path):
    ckpt, _ = _write_fake_checkpoint(tmp_path)
    s = _sentinel(ckpt, threshold=1e12)
    s.measure(torch.randn(1, 5, D_MODEL))
    logits = torch.randn(1, NUM_SLOTS, NUM_CHOICES)
    out = s.maybe_gate(logits)
    torch.testing.assert_close(out, logits)
    assert "gated=0" in s.summary()


def test_log_mode_never_gates_and_writes_csv(tmp_path):
    ckpt, _ = _write_fake_checkpoint(tmp_path)
    log = tmp_path / "sigma.csv"
    s = _sentinel(ckpt, mode="log", threshold=None, log_path=str(log))
    logits = torch.randn(1, NUM_SLOTS, NUM_CHOICES)
    for _ in range(3):
        s.measure(torch.randn(1, 5, D_MODEL))
        out = s.maybe_gate(logits)
        torch.testing.assert_close(out, logits)
    s.close()
    lines = log.read_text().strip().splitlines()
    assert lines[0] == "step,sigma2_all,sigma2_defer,gated"
    assert len(lines) == 4
    assert all(row.endswith(",0") for row in lines[1:])


def test_measure_without_features_raises(tmp_path):
    ckpt, _ = _write_fake_checkpoint(tmp_path)
    s = _sentinel(ckpt)
    with pytest.raises(RuntimeError, match="not captured"):
        s.measure(None)


def _feed(mon, forecast_of_green, steps=200, num_dc=5):
    """Drive the monitor with a synthetic sinusoid green signal on DC0-2
    (DC3-4 brown, constant 0) and forecast = forecast_of_green(green)."""
    import numpy as np
    logits = torch.randn(1, NUM_SLOTS, NUM_CHOICES)
    out = logits
    for t in range(steps):
        g = np.zeros(num_dc)
        g[:3] = 1000.0 * (1.0 + np.sin(0.1 * t + np.arange(3)))
        f = np.full(num_dc, 0.5)
        f[:3] = forecast_of_green(g[:3] / 2000.0)
        mon.measure_obs({
            "dc_future_short_mean": f,
            "dc_current_green_power_w": g,
        })
        out = mon.maybe_gate(logits)
    return logits, out


from src.baselines.trust_sentinel import ForecastResidualMonitor


def test_resid_truthful_forecast_no_gate():
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="gate",
                                  threshold=0.2, window=100, warmup=20)
    logits, out = _feed(mon, lambda x: x)
    assert mon._last_corr > 0.9
    torch.testing.assert_close(out, logits)
    assert "gated=0" not in mon.summary() or mon._n_gated == 0


def test_resid_anti_forecast_gates():
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="gate",
                                  threshold=0.2, window=100, warmup=20)
    logits, out = _feed(mon, lambda x: 1.0 - x)   # inverted = anti
    assert mon._last_corr < -0.9
    assert torch.all(out[..., -1] == torch.finfo(out.dtype).min)
    torch.testing.assert_close(out[..., :-1], logits[..., :-1])


def test_resid_frozen_forecast_counts_as_zero_information():
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="gate",
                                  threshold=0.2, window=100, warmup=20)
    _feed(mon, lambda x: x * 0 + 0.5)             # blend-to-neutral
    assert abs(mon._last_corr) < 1e-6
    assert mon._n_gated > 0


def test_resid_warmup_never_gates():
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="gate",
                                  threshold=0.2, window=100, warmup=50)
    _feed(mon, lambda x: 1.0 - x, steps=30)       # anti, but < warmup samples
    assert mon._n_gated == 0
    assert mon._last_corr != mon._last_corr       # NaN during warmup


def test_resid_log_mode_writes_and_never_gates(tmp_path):
    log = tmp_path / "corr.csv"
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="log",
                                  log_path=str(log), window=100, warmup=20)
    logits, out = _feed(mon, lambda x: 1.0 - x)
    torch.testing.assert_close(out, logits)
    mon.close()
    lines = log.read_text().strip().splitlines()
    assert lines[0] == "step,corr,n_samples,gated"
    assert len(lines) == 201
    assert all(l.endswith(",0") for l in lines[1:])


def _feed_measure_only(mon, forecast_of_green, steps=200, num_dc=5):
    import numpy as np
    obs = None
    for t in range(steps):
        g = np.zeros(num_dc)
        g[:3] = 1000.0 * (1.0 + np.sin(0.1 * t + np.arange(3)))
        f = np.full(num_dc, 0.5)
        f[:3] = forecast_of_green(g[:3] / 2000.0)
        obs = {
            "dc_future_short_mean": f,
            "dc_future_short_trend": np.full(num_dc, 0.3),
            "dc_future_long_mean": f.copy(),
            "dc_future_long_peak_timing": f.copy(),
            "dc_current_green_power_w": g,
        }
        mon.measure_obs(obs)
    return obs


def test_repair_inverts_lying_dcs_only():
    import numpy as np
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="repair",
                                  window=100, warmup=20)
    obs = _feed_measure_only(mon, lambda x: 1.0 - x)   # DCs 0-2 lie; 3-4 brown
    out = mon.repair(obs)
    np.testing.assert_allclose(
        out["dc_future_short_mean"][:3], 1.0 - obs["dc_future_short_mean"][:3])
    np.testing.assert_allclose(out["dc_future_short_trend"][:3], -0.3)
    # brown DCs (no green variance → corr NaN) untouched
    np.testing.assert_allclose(out["dc_future_short_mean"][3:],
                               obs["dc_future_short_mean"][3:])
    assert mon._n_repaired == 1
    assert "repaired=1" in mon.summary()


def test_repair_noop_on_truthful_forecast():
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="repair",
                                  window=100, warmup=20)
    obs = _feed_measure_only(mon, lambda x: x)
    out = mon.repair(obs)
    assert out is obs                     # identity, not even a copy
    assert mon._n_repaired == 0


def test_repair_noop_during_warmup_and_in_other_modes():
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="repair",
                                  window=100, warmup=50)
    obs = _feed_measure_only(mon, lambda x: 1.0 - x, steps=30)  # < warmup
    assert mon.repair(obs) is obs
    mon2 = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="log",
                                   log_path="/dev/null", window=100, warmup=20)
    obs2 = _feed_measure_only(mon2, lambda x: 1.0 - x)
    assert mon2.repair(obs2) is obs2      # log mode never repairs


def test_repair_buffer_keeps_raw_stream_no_oscillation():
    """Even after repair starts, the monitor keeps seeing the RAW (lying)
    forecast, so corr stays negative and repair stays ON (no flip-flop)."""
    mon = ForecastResidualMonitor(num_slots=NUM_SLOTS, mode="repair",
                                  window=100, warmup=20)
    _feed_measure_only(mon, lambda x: 1.0 - x, steps=100)
    assert mon._last_corr < -0.9
    obs = _feed_measure_only(mon, lambda x: 1.0 - x, steps=100)
    assert mon._last_corr < -0.9          # still negative after 100 more
    assert mon.repair(obs) is not obs     # still repairing


def test_anti_ood_features_raise_disagreement(tmp_path):
    """Sanity: features far off the (random-init) manifold produce larger
    ensemble variance than near-zero features — the OOD-detection premise."""
    ckpt, _ = _write_fake_checkpoint(tmp_path)
    s = _sentinel(ckpt, signal="all")
    lo = s.measure(torch.zeros(1, 3, D_MODEL))
    hi = s.measure(torch.full((1, 3, D_MODEL), 50.0))
    assert hi > lo
