"""Estimator tests for the real-error audit.

The audit's numbers become the sole parameter source for the Scheme 2-E primary error, so
the estimators have to be shown to recover known truth before anyone reads what they say
about the real checkpoint. Every test here builds data with the answer already known.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timecap_error_audit as A  # noqa: E402

RNG = np.random.default_rng(20260902)
N_ANCHORS, N_LEADS = 4000, 6


def _synth(lam, b_ols, mu=500.0, sigma=1.0, n=N_ANCHORS, leads=N_LEADS):
    """pred - mu = lam*(truth - mu) + b_ols + eps, with the parameters known by construction."""
    truth = mu + RNG.normal(0.0, 200.0, size=(n, leads))
    eps = RNG.normal(0.0, sigma, size=(n, leads))
    pred = mu + lam * (truth - mu) + b_ols + eps
    return pred, truth, mu


class TestLambdaAndInterceptRecovery:
    @pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_lambda_is_recovered(self, lam):
        pred, truth, mu = _synth(lam, b_ols=0.0)
        fit = A.fit_lambda_b(pred, truth, mu)
        assert np.allclose(fit["lambda_lead"], lam, atol=0.02), fit["lambda_lead"]

    def test_ols_intercept_is_recovered(self):
        pred, truth, mu = _synth(0.4, b_ols=-37.0)
        fit = A.fit_lambda_b(pred, truth, mu)
        assert np.allclose(fit["b_ols_lead"], -37.0, atol=1.0)

    def test_mean_bias_is_not_the_ols_intercept_when_lambda_differs_from_one(self):
        """The work order's b is mean(pred - truth); it equals the fitted intercept only
        at lambda = 1. Emitting both is deliberate, and this pins the distinction."""
        mu, lam, b = 500.0, 0.4, 0.0
        pred, truth, _ = _synth(lam, b_ols=b, mu=mu)
        fit = A.fit_lambda_b(pred, truth, mu)
        # mean(truth - mu) is ~0 here, so make the level deliberately off-centre instead.
        truth2 = truth + 300.0
        pred2 = mu + lam * (truth2 - mu) + b
        fit2 = A.fit_lambda_b(pred2, truth2, mu)
        assert np.allclose(fit2["b_ols_lead"], b, atol=1.0)
        # (lam - 1) * mean(truth - mu), using the realized mean so the check is exact.
        expected_mean_bias = (lam - 1.0) * (truth2 - mu).mean(axis=0)
        assert np.allclose(fit2["b_mean_bias_lead"], expected_mean_bias, atol=1e-6)
        assert not np.allclose(fit2["b_mean_bias_lead"], fit2["b_ols_lead"], atol=1.0)

    def test_ols_residual_is_zero_mean_and_the_mean_bias_one_is_not(self):
        mu, lam = 500.0, 0.4
        truth = mu + 300.0 + RNG.normal(0.0, 200.0, size=(N_ANCHORS, N_LEADS))
        pred = mu + lam * (truth - mu)
        fit = A.fit_lambda_b(pred, truth, mu)
        expected = (1.0 - lam) * (truth - mu).mean(axis=0)
        assert np.allclose(fit["resid_mean_lead_with_mean_bias"], expected, atol=1e-6)
        assert np.all(fit["resid_var_lead_with_ols_intercept"]
                      <= fit["resid_var_lead_with_mean_bias"] + 1e-6)

    def test_residual_variance_matches_the_injected_noise(self):
        pred, truth, mu = _synth(0.6, b_ols=10.0, sigma=7.0)
        fit = A.fit_lambda_b(pred, truth, mu)
        assert np.allclose(np.sqrt(fit["resid_var_lead_with_ols_intercept"]), 7.0, atol=0.4)

    def test_constant_truth_does_not_blow_up(self):
        truth = np.full((50, N_LEADS), 500.0)
        pred = np.full((50, N_LEADS), 480.0)
        fit = A.fit_lambda_b(pred, truth, 500.0)
        assert np.all(np.isfinite(fit["lambda_lead"]))
        assert np.allclose(fit["lambda_lead"], 0.0)


class TestAr1:
    def test_recovers_a_known_autocorrelation(self):
        rho, n, leads = 0.8, 6000, 200
        e = np.empty((n, leads))
        e[:, 0] = RNG.normal(size=n)
        s = np.sqrt(1 - rho ** 2)
        for l in range(1, leads):
            e[:, l] = rho * e[:, l - 1] + s * RNG.normal(size=n)
        assert A.ar1_along_lead(e) == pytest.approx(rho, abs=0.03)

    def test_white_noise_gives_about_zero(self):
        assert A.ar1_along_lead(RNG.normal(size=(4000, 200))) == pytest.approx(0.0, abs=0.03)


class TestLeanTimeOptimism:
    def test_a_flat_mean_forecast_is_always_optimistic_below_the_mean(self):
        """The pilot's mechanism in its purest form: predict mu, always."""
        mu = 500.0
        truth = mu + RNG.normal(0.0, 200.0, size=(500, 20))
        pred = np.full_like(truth, mu)
        out = A.lean_time_optimism(pred, truth, mu)
        assert out["p_over_given_lean"] == 1.0
        assert out["mean_signed_error_given_lean"] > 0

    def test_a_perfect_forecast_is_never_optimistic(self):
        mu = 500.0
        truth = mu + RNG.normal(0.0, 200.0, size=(500, 20))
        out = A.lean_time_optimism(truth.copy(), truth, mu)
        assert out["p_over_given_lean"] == 0.0
        assert out["mean_signed_error_given_lean"] == pytest.approx(0.0)

    def test_no_lean_cells_is_reported_not_divided_by_zero(self):
        mu = 0.0
        truth = np.full((10, 5), 100.0)
        out = A.lean_time_optimism(truth, truth, mu)
        assert out["n_lean_cells"] == 0 and out["p_over_given_lean"] is None


class TestPeakRates:
    def test_a_perfect_forecast_misses_nothing(self):
        truth = RNG.normal(500, 100, size=(200, 144))
        r = A.peak_rates(truth.copy(), truth)
        assert r["miss_rate"] == pytest.approx(0.0)
        assert r["false_peak_rate"] == pytest.approx(0.0)

    def test_an_independent_forecast_sits_at_the_base_rate(self):
        truth = RNG.normal(500, 100, size=(4000, 144))
        pred = RNG.normal(500, 100, size=(4000, 144))
        r = A.peak_rates(pred, truth)
        assert r["false_peak_rate"] == pytest.approx(r["base_rate_not_true_peak"], abs=0.02)

    def test_base_rate_follows_the_percentile(self):
        truth = RNG.normal(500, 100, size=(200, 144))
        r = A.peak_rates(truth.copy(), truth)
        assert r["base_rate_not_true_peak"] == pytest.approx(A.PEAK_Q / 100.0, abs=0.02)


class TestPhaseLag:
    def test_a_known_shift_is_found(self):
        base = np.sin(np.arange(400) / 9.0) * 100 + 500
        shift = 7
        truth = np.stack([base[i:i + 144] for i in range(50)])
        pred = np.stack([base[i + shift:i + shift + 144] for i in range(50)])
        out = A.phase_lag(pred, truth)
        # pred[k] = truth[k + shift]; the estimator compares p[k] against t[k + lag],
        # so the match is at lag = +shift.
        assert out["median_lag"] == pytest.approx(shift, abs=1)

    def test_zero_shift_is_found(self):
        truth = RNG.normal(500, 100, size=(50, 144))
        out = A.phase_lag(truth.copy(), truth)
        assert out["median_lag"] == 0.0
        assert out["undefined_fraction"] == 0.0

    def test_a_flat_forecast_is_undefined_not_noise_argmax(self):
        truth = RNG.normal(500, 100, size=(30, 144))
        pred = np.full_like(truth, 500.0)
        out = A.phase_lag(pred, truth)
        assert out["undefined_fraction"] == 1.0 and out["median_lag"] is None


class TestRanking:
    MU = {0: 500.0, 1: 500.0, 2: 500.0}

    def test_perfect_ranking(self):
        t = {d: RNG.normal(500, 100, size=(40, 20)) for d in (0, 1, 2)}
        r = A.ranking_stats({d: v.copy() for d, v in t.items()}, t, [0, 1, 2], self.MU)
        assert r["argmax_hit_rate"] == 1.0
        assert r["mean_kendall_tau"] == pytest.approx(1.0)

    def test_independent_ranking_lands_near_chance(self):
        t = {d: RNG.normal(500, 100, size=(200, 60)) for d in (0, 1, 2)}
        p = {d: RNG.normal(500, 100, size=(200, 60)) for d in (0, 1, 2)}
        r = A.ranking_stats(p, t, [0, 1, 2], self.MU)
        assert r["argmax_hit_rate"] == pytest.approx(A.RANDOM_ARGMAX, abs=0.05)
        assert r["mean_kendall_tau"] == pytest.approx(0.0, abs=0.06)

    def test_constant_mu_baseline_beats_uniform_random_when_levels_differ(self):
        """Separated levels are exactly why 1/3 is the wrong reference: a ranker that
        knows only the long-run order already wins most comparisons."""
        mu = {0: 900.0, 1: 600.0, 2: 300.0}
        t = {d: RNG.normal(mu[d], 100, size=(300, 40)) for d in (0, 1, 2)}
        p = {d: RNG.normal(500, 100, size=(300, 40)) for d in (0, 1, 2)}   # no information
        r = A.ranking_stats(p, t, [0, 1, 2], mu)
        assert r["constant_mu_argmax_hit_rate"] > 0.8
        assert r["argmax_hit_rate"] == pytest.approx(A.RANDOM_ARGMAX, abs=0.06)
        assert r["argmax_lift_over_constant_mu"] < 0

    def test_lift_is_zero_when_the_model_is_the_constant_ranker(self):
        mu = {0: 900.0, 1: 600.0, 2: 300.0}
        t = {d: RNG.normal(mu[d], 100, size=(200, 30)) for d in (0, 1, 2)}
        p = {d: np.full((200, 30), mu[d]) for d in (0, 1, 2)}
        r = A.ranking_stats(p, t, [0, 1, 2], mu)
        assert r["argmax_lift_over_constant_mu"] == pytest.approx(0.0)
        assert r["kendall_lift_over_constant_mu"] == pytest.approx(0.0)

    def test_kendall_tau_three_items(self):
        assert A.kendall_tau_3([3, 2, 1], [3, 2, 1]) == pytest.approx(1.0)
        assert A.kendall_tau_3([3, 2, 1], [1, 2, 3]) == pytest.approx(-1.0)


class TestFrozenSetup:
    def test_setup_matches_the_work_order(self):
        assert A.DC_TURBINES == {0: (12, 36), 1: (95, 91), 2: (96,)}
        assert A.YEAR == 2020 and A.STRIDE == 240 and A.LABEL_OFFSET == 0
        assert A.SEQ == 96 and A.PRED == 144 and A.PEAK_Q == 75

    def test_probe_reads_no_carbon_artifact(self):
        src = open(A.__file__).read()
        body = src.split('"""', 2)[-1]
        # Directory and file names that exist only under the scheduling results, so none
        # of them can be a field name the probe legitimately emits. ("verdicts" is one of
        # the probe's OWN output fields, which is why bare English words are useless here.)
        for forbidden in ("stage_a_out", "_out/", "ladder_v2", "PILOT_",
                          "conf_tier", "pilot_tier", "s2_manifest"):
            assert forbidden not in body, f"the probe reaches into {forbidden}"
        # The only inputs it may name are the split CSVs and the checkpoint.
        assert "windProduction/split" in body and "ckpt_best.pth" in body
