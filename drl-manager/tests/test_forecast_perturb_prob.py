"""FORECAST_PERTURB_PROB episode-level curriculum gating (kn2 weapon).

Contract:
- PROB unset/<=0 + MODE set  -> every episode perturbed (historical always-on).
- PROB in (0,1] + MODE set   -> Bernoulli(prob) per episode via gym-seeded RNG.
- MODE none                  -> never perturbed, regardless of PROB.
- An unselected episode passes forecasts through IDENTICALLY even with MODE=anti.
No Java gateway needed: the env object is built via __new__ and only the two
pure-ish methods under test are exercised.
"""
import numpy as np
import pytest

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv


def _bare_env(rng_seed=0):
    env = object.__new__(HierarchicalMultiDCEnv)
    env._np_random = np.random.default_rng(rng_seed)  # gym's np_random backing attr
    return env


@pytest.fixture
def forecasts():
    sm = np.array([0.1, 0.9, 0.4], dtype=np.float32)
    st = np.array([0.2, -0.3, 0.0], dtype=np.float32)
    lm = np.array([0.6, 0.2, 0.8], dtype=np.float32)
    lp = np.array([0.3, 0.7, 0.5], dtype=np.float32)
    return sm, st, lm, lp


def test_mode_none_never_selected(monkeypatch):
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "none")
    monkeypatch.setenv("FORECAST_PERTURB_PROB", "1.0")
    assert _bare_env()._draw_perturb_episode() is False


def test_prob_unset_keeps_always_on(monkeypatch):
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "anti")
    monkeypatch.delenv("FORECAST_PERTURB_PROB", raising=False)
    assert _bare_env()._draw_perturb_episode() is True


def test_prob_one_always_selects(monkeypatch):
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "anti")
    monkeypatch.setenv("FORECAST_PERTURB_PROB", "1.0")
    env = _bare_env()
    assert all(env._draw_perturb_episode() for _ in range(20))


def test_prob_fraction_mixes_episodes(monkeypatch):
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "anti")
    monkeypatch.setenv("FORECAST_PERTURB_PROB", "0.25")
    env = _bare_env(rng_seed=7)
    draws = [env._draw_perturb_episode() for _ in range(400)]
    frac = sum(draws) / len(draws)
    assert 0.15 < frac < 0.35, f"Bernoulli(0.25) draw fraction off: {frac}"


def test_unselected_episode_is_identity(monkeypatch, forecasts):
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "anti")
    env = _bare_env()
    env._perturb_this_episode = False
    out = env._perturb_forecast(*forecasts, sim_step=123)
    for got, exp in zip(out, forecasts):
        np.testing.assert_array_equal(got, exp)


def test_selected_episode_is_perturbed(monkeypatch, forecasts):
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "anti")
    env = _bare_env()
    env._perturb_this_episode = True
    sm, st, lm, lp = forecasts
    out_sm, out_st, out_lm, out_lp = env._perturb_forecast(sm, st, lm, lp, sim_step=123)
    np.testing.assert_allclose(out_sm, 1.0 - sm, rtol=1e-6)
    np.testing.assert_allclose(out_st, -st, rtol=1e-6)
    np.testing.assert_allclose(out_lm, 1.0 - lm, rtol=1e-6)
    np.testing.assert_allclose(out_lp, 1.0 - lp, rtol=1e-6)


def test_missing_flag_defaults_to_perturbed(monkeypatch, forecasts):
    # Old checkpoint-eval paths never call reset-side lottery init; the getattr
    # default must preserve the historical always-on behaviour.
    monkeypatch.setenv("FORECAST_PERTURB_MODE", "anti")
    env = _bare_env()
    assert not hasattr(env, "_perturb_this_episode")
    out_sm, *_ = env._perturb_forecast(*forecasts, sim_step=1)
    np.testing.assert_allclose(out_sm, 1.0 - forecasts[0], rtol=1e-6)
