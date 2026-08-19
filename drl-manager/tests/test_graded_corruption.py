"""Graded corruption modes for the severity sweep (reviewer #4, 2026-08-19).

The paper reports two discrete corruption points (Blend, Shuffle). A reviewer
asked for a degradation CURVE. These modes interpolate to the existing
endpoints, so the sweep's extreme cell reproduces the published one exactly.
"""
import numpy as np
import pytest

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv


class _Perturber:
    """Bind only the pure perturbation method (no env/gateway needed)."""

    def __init__(self, monkeypatch, mode, eps):
        monkeypatch.setenv("FORECAST_PERTURB_MODE", mode)
        monkeypatch.setenv("FORECAST_PERTURB_EPS", str(eps))
        self._perturb_this_episode = True

    def run(self, sm, st, lm, lp, step=7):
        return HierarchicalMultiDCEnv._perturb_forecast(
            self, np.array(sm, dtype=np.float64), np.array(st, dtype=np.float64),
            np.array(lm, dtype=np.float64), np.array(lp, dtype=np.float64), step)


SM = [0.9, 0.7, 0.4, 0.2, 0.1]
ST = [0.5, -0.2, 0.3, -0.4, 0.1]
LM = [0.8, 0.6, 0.5, 0.3, 0.2]
LP = [0.1, 0.3, 0.5, 0.7, 0.9]


class TestEndpointEquivalence:
    def test_panti_at_one_equals_anti(self, monkeypatch):
        a = _Perturber(monkeypatch, "panti", 1.0).run(SM, ST, LM, LP)
        b = _Perturber(monkeypatch, "anti", 0.0).run(SM, ST, LM, LP)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-6)

    def test_pshuffle_at_one_equals_shuffle(self, monkeypatch):
        a = _Perturber(monkeypatch, "pshuffle", 1.0).run(SM, ST, LM, LP)
        b = _Perturber(monkeypatch, "shuffle", 0.0).run(SM, ST, LM, LP)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-6)

    def test_zero_severity_is_identity(self, monkeypatch):
        for mode in ("panti", "pshuffle", "bias", "blend", "noise"):
            out = _Perturber(monkeypatch, mode, 0.0).run(SM, ST, LM, LP)
            np.testing.assert_allclose(out[0], SM, atol=1e-9)
            np.testing.assert_allclose(out[2], LM, atol=1e-9)


class TestMonotoneSeverity:
    def test_panti_moves_monotonically_towards_the_mirror(self, monkeypatch):
        target = 1.0 - np.array(SM)
        prev = np.abs(np.array(SM) - target).sum()
        for a in (0.25, 0.5, 0.75, 1.0):
            out = _Perturber(monkeypatch, "panti", a).run(SM, ST, LM, LP)
            d = np.abs(out[0] - target).sum()
            assert d < prev + 1e-12
            prev = d
        assert prev == pytest.approx(0.0, abs=1e-6)

    def test_pshuffle_touches_k_sites(self, monkeypatch):
        out = _Perturber(monkeypatch, "pshuffle", 0.4).run(SM, ST, LM, LP)
        rev = np.array(SM)[::-1]
        assert np.allclose(out[0][:2], rev[:2])      # k = round(0.4*5) = 2
        assert np.allclose(out[0][2:], np.array(SM)[2:])

    def test_bias_shifts_levels_and_clips(self, monkeypatch):
        out = _Perturber(monkeypatch, "bias", 0.3).run(SM, ST, LM, LP)
        assert out[0][0] == pytest.approx(1.0)        # 0.9 + 0.3 clipped
        assert out[0][4] == pytest.approx(0.4)
        np.testing.assert_allclose(out[1], ST, atol=1e-9)   # trend untouched

    def test_bias_negative_direction(self, monkeypatch):
        out = _Perturber(monkeypatch, "bias", -0.3).run(SM, ST, LM, LP)
        assert out[0][0] == pytest.approx(0.6)
        assert out[0][4] == pytest.approx(0.0)        # 0.1 - 0.3 clipped


class TestHygiene:
    def test_unknown_mode_is_identity(self, monkeypatch):
        out = _Perturber(monkeypatch, "not_a_mode", 0.7).run(SM, ST, LM, LP)
        np.testing.assert_allclose(out[0], SM, atol=1e-9)

    def test_output_is_float32(self, monkeypatch):
        out = _Perturber(monkeypatch, "panti", 0.5).run(SM, ST, LM, LP)
        for x in out:
            assert x.dtype == np.float32

    def test_inactive_episode_skips_perturbation(self, monkeypatch):
        p = _Perturber(monkeypatch, "panti", 1.0)
        p._perturb_this_episode = False
        out = p.run(SM, ST, LM, LP)
        np.testing.assert_allclose(out[0], SM, atol=1e-9)
