"""
Synthetic forecast-shift injection (conservative-collapse experiment).

Verifies TimeCAPGodEyeProvider._resolve_shift_cfg / _apply_forecast_shift:
the shift must (a) modify the raw per-turbine forecast inside the configured
in-episode window, (b) leave it untouched outside / when disabled, (c) clip
at zero, and (d) reach BOTH consumers — the DC feature aggregation and the
CRD predicted_wind_w accessor — identically, because both read the same
shifted `_last_per_t_pred`.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_timecap_forecast_shift.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.timecap_godeye_provider import TimeCAPGodEyeProvider


def _pred(vals):
    return np.asarray(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# _resolve_shift_cfg
# ---------------------------------------------------------------------------


def test_resolve_none_and_disabled():
    assert TimeCAPGodEyeProvider._resolve_shift_cfg(None) is None
    assert TimeCAPGodEyeProvider._resolve_shift_cfg({}) is None
    assert TimeCAPGodEyeProvider._resolve_shift_cfg({"enabled": False, "factor": 9}) is None


def test_resolve_defaults_and_fields():
    cfg = TimeCAPGodEyeProvider._resolve_shift_cfg({"enabled": True, "factor": 2.5})
    assert cfg == {
        "mode": "scale", "factor": 2.5, "bias_kw": 0.0,
        "start_step": 0, "end_step": -1,
    }


def test_resolve_rejects_bad_mode():
    with pytest.raises(ValueError):
        TimeCAPGodEyeProvider._resolve_shift_cfg({"enabled": True, "mode": "nope"})


# ---------------------------------------------------------------------------
# _apply_forecast_shift
# ---------------------------------------------------------------------------


def test_scale_mode_multiplies():
    cfg = TimeCAPGodEyeProvider._resolve_shift_cfg(
        {"enabled": True, "mode": "scale", "factor": 2.0}
    )
    out = TimeCAPGodEyeProvider._apply_forecast_shift(
        {1: _pred([1.0, 2.0]), 2: _pred([0.5, 0.0])}, simulation_step=10, cfg=cfg
    )
    np.testing.assert_allclose(out[1], [2.0, 4.0])
    np.testing.assert_allclose(out[2], [1.0, 0.0])


def test_bias_mode_adds_and_clips_at_zero():
    cfg = TimeCAPGodEyeProvider._resolve_shift_cfg(
        {"enabled": True, "mode": "bias_kw", "bias_kw": -3.0}
    )
    out = TimeCAPGodEyeProvider._apply_forecast_shift(
        {1: _pred([5.0, 2.0, 0.0])}, simulation_step=0, cfg=cfg
    )
    np.testing.assert_allclose(out[1], [2.0, 0.0, 0.0])  # 2-3 and 0-3 clip to 0


def test_window_gates_application():
    cfg = TimeCAPGodEyeProvider._resolve_shift_cfg(
        {"enabled": True, "factor": 10.0, "start_step": 100, "end_step": 200}
    )
    base = {1: _pred([1.0])}
    before = TimeCAPGodEyeProvider._apply_forecast_shift(base, 99, cfg)
    inside = TimeCAPGodEyeProvider._apply_forecast_shift(base, 150, cfg)
    after = TimeCAPGodEyeProvider._apply_forecast_shift(base, 201, cfg)
    assert before[1][0] == pytest.approx(1.0)
    assert inside[1][0] == pytest.approx(10.0)
    assert after[1][0] == pytest.approx(1.0)


def test_end_step_minus_one_means_forever():
    cfg = TimeCAPGodEyeProvider._resolve_shift_cfg({"enabled": True, "factor": 3.0})
    out = TimeCAPGodEyeProvider._apply_forecast_shift({1: _pred([1.0])}, 10**9, cfg)
    assert out[1][0] == pytest.approx(3.0)


def test_none_cfg_is_identity_same_object():
    base = {1: _pred([1.0, 2.0])}
    out = TimeCAPGodEyeProvider._apply_forecast_shift(base, 5, None)
    assert out is base  # no copy, zero overhead when disabled


def test_input_not_mutated():
    cfg = TimeCAPGodEyeProvider._resolve_shift_cfg({"enabled": True, "factor": 2.0})
    base = {1: _pred([1.0])}
    TimeCAPGodEyeProvider._apply_forecast_shift(base, 0, cfg)
    assert base[1][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Consumer consistency: agent features and CRD predicted_wind_w must see the
# SAME shifted forecast (the experiment's core invariant).
# ---------------------------------------------------------------------------


class _StubPredictor:
    def __init__(self, per_t_pred):
        self._per_t_pred = per_t_pred

    def predict_per_turbine(self):
        return {k: v.copy() for k, v in self._per_t_pred.items()}


def _make_provider_with_stub(per_t_pred, shift_cfg, dc_assignments=None):
    p = TimeCAPGodEyeProvider.__new__(TimeCAPGodEyeProvider)
    p.dc_assignments = dc_assignments or {0: [1]}
    p.dc_ids = sorted(p.dc_assignments.keys())
    p.short_term_steps = 2
    p.long_term_steps = 3
    p.pred_len = 3
    p.seq_len = 96
    p.forecast_every = 1
    p.predictor = _StubPredictor(per_t_pred)
    p._last_features = {dc: np.zeros(4, dtype=np.float32) for dc in p.dc_ids}
    p._last_forecast_step = {dc: -10**9 for dc in p.dc_ids}
    p._last_per_t_pred = None
    p._dirty_steps = 0
    p._shift_cfg = TimeCAPGodEyeProvider._resolve_shift_cfg(shift_cfg)
    # _aggregate_dc needs per-turbine max power; stub a flat 10 kW.
    p.predictor.max_power_kw = {1: 10.0}
    return p


def test_get_features_stashes_shifted_pred_for_crd():
    """After get_features, the CRD accessor must return the SHIFTED wind —
    proving agent obs and R_forecast derive from the same wrong forecast."""
    raw = {1: _pred([2.0, 4.0, 6.0])}  # kW
    p = _make_provider_with_stub(raw, {"enabled": True, "factor": 2.0})
    try:
        p.get_features(0)
    except Exception:
        # _aggregate_dc may need more predictor internals than stubbed; the
        # stash happens before aggregation only if features computed — so a
        # failure here invalidates the test rather than passing vacuously.
        pytest.fail("get_features raised with stubbed predictor")
    wind = p.get_predicted_wind_w_per_dc(horizon=0)
    assert wind is not None
    # 2.0 kW × shift 2.0 = 4.0 kW = 4000 W
    assert wind[0] == pytest.approx(4000.0)


def test_get_features_unshifted_when_disabled():
    raw = {1: _pred([2.0, 4.0, 6.0])}
    p = _make_provider_with_stub(raw, {"enabled": False})
    try:
        p.get_features(0)
    except Exception:
        pytest.fail("get_features raised with stubbed predictor")
    wind = p.get_predicted_wind_w_per_dc(horizon=0)
    assert wind[0] == pytest.approx(2000.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
