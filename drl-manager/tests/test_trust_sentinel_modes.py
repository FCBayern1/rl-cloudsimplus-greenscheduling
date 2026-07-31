"""Refined auditor intervention modes: soft / maskdc / persist.

Design (2026-07-23): the hard DEFER ban (gate) proved self-harmful and the
inversion gamble (repair) systematically harmful; these modes intervene with
graded strength (soft), per-DC information removal (maskdc), or per-DC honest
substitution (persist). Detection path (rolling per-DC residual correlation)
is unchanged.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.baselines.trust_sentinel import ForecastResidualMonitor


def _monitor(mode, thresh=0.0, **kw):
    return ForecastResidualMonitor(num_slots=8, mode=mode, threshold=thresh, **kw)


def _obs(nd=3):
    return {
        "dc_future_short_mean": np.array([0.1, 0.8, 0.4], dtype=np.float32)[:nd],
        "dc_future_short_trend": np.array([0.2, -0.1, 0.3], dtype=np.float32)[:nd],
        "dc_future_long_mean": np.array([0.6, 0.7, 0.2], dtype=np.float32)[:nd],
        "dc_future_long_peak_timing": np.array([0.3, 0.9, 0.5], dtype=np.float32)[:nd],
        "dc_current_green_power_w": np.array([10.0, 50.0, 5.0], dtype=np.float32)[:nd],
    }


# ---------- soft ----------

def test_soft_shaves_defer_proportionally():
    m = _monitor("soft", thresh=0.0)
    m.soft_lambda = 6.0
    m._last_corr = -0.5           # severity = 0.25 -> penalty 1.5
    logits = torch.zeros(1, 4, 6)
    out = m.maybe_gate(logits)
    assert torch.allclose(out[..., -1], torch.full((1, 4), -1.5))
    assert torch.allclose(out[..., :-1], logits[..., :-1])


def test_soft_no_penalty_when_trusted_or_warmup():
    m = _monitor("soft", thresh=0.0)
    logits = torch.zeros(1, 4, 6)
    m._last_corr = 0.6            # trusted
    assert torch.equal(m.maybe_gate(logits.clone()), logits)
    m._last_corr = float("nan")   # warmup
    assert torch.equal(m.maybe_gate(logits.clone()), logits)


def test_soft_penalty_saturates():
    m = _monitor("soft", thresh=1.0)
    m.soft_lambda = 6.0
    m._last_corr = -1.0           # thresh - corr = 2.0 -> severity capped at 1
    out = m.maybe_gate(torch.zeros(1, 2, 6))
    assert torch.allclose(out[..., -1], torch.full((1, 2), -6.0))


# ---------- maskdc ----------

def test_maskdc_neutralizes_only_failing_dc():
    m = _monitor("maskdc", thresh=0.0)
    m._last_corrs = np.array([-0.8, 0.9, 0.7])   # only DC0 lies
    out = m.repair(_obs())
    assert out["dc_future_short_mean"][0] == pytest.approx(0.5)
    assert out["dc_future_short_trend"][0] == pytest.approx(0.0)
    assert out["dc_future_long_peak_timing"][0] == pytest.approx(0.5)
    # honest DCs untouched
    assert out["dc_future_short_mean"][1] == pytest.approx(0.8)
    assert out["dc_future_short_mean"][2] == pytest.approx(0.4)


def test_maskdc_noop_when_all_honest():
    m = _monitor("maskdc", thresh=0.0)
    m._last_corrs = np.array([0.5, 0.9, 0.7])
    obs = _obs()
    out = m.repair(obs)
    np.testing.assert_array_equal(out["dc_future_short_mean"], obs["dc_future_short_mean"])


# ---------- persist ----------

def test_persist_substitutes_normalized_realized_green():
    m = _monitor("persist", thresh=0.0)
    m._last_corrs = np.array([-0.9, 0.8, 0.8])
    # ring buffer: DC0 realized green history peak 100, latest 25 -> gnorm 0.25
    m._fore = np.zeros((m.window, 3))
    m._real = np.zeros((m.window, 3))
    m._real[0] = [100.0, 1.0, 1.0]
    m._real[1] = [25.0, 1.0, 1.0]
    m._n = 2
    out = m.repair(_obs())
    assert out["dc_future_short_mean"][0] == pytest.approx(0.25)
    assert out["dc_future_short_trend"][0] == pytest.approx(0.0)
    assert out["dc_future_short_mean"][1] == pytest.approx(0.8)  # honest DC kept


def test_persist_dead_green_maps_to_zero():
    m = _monitor("persist", thresh=0.0)
    m._last_corrs = np.array([-0.9, 0.8, 0.8])
    m._fore = np.zeros((m.window, 3))
    m._real = np.zeros((m.window, 3))   # DC0 never had green
    m._n = 1
    out = m.repair(_obs())
    assert out["dc_future_short_mean"][0] == pytest.approx(0.0)


# ---------- guards ----------

def test_warmup_noop_all_modes():
    for mode in ("maskdc", "persist"):
        m = _monitor(mode, thresh=0.0)
        m._last_corrs = None
        obs = _obs()
        assert m.repair(obs) is obs


def test_repair_mode_unchanged_semantics():
    m = _monitor("repair", thresh=None)
    m._last_corrs = np.array([-0.9, 0.2, 0.2])   # < -0.5 default
    out = m.repair(_obs())
    assert out["dc_future_short_mean"][0] == pytest.approx(1.0 - 0.1)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="TRUST_GATE_MODE"):
        _monitor("hard")
