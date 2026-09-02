"""Acceptance tests for the perturbed God's Eye observation provider.

The three the work order names, plus the equivalence that makes the first one meaningful:
the aggregation has to be the SAME function TimeCAPGodEyeProvider uses, or "identical to
godeye" would only mean "identical to my own copy of godeye".
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.baselines import forecast_perturb as fp  # noqa: E402
from src.prediction.perturbed_godeye_provider import (  # noqa: E402
    DEFAULT_TIER, PerturbedGodEyeProvider, SUPPORTED_TIERS, from_config)

FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Prtv", "T2m",
            "Sp", "RelH", "Wspd_w", "Wdir_w", "Patv"]
DC_ASSIGN = {0: [12, 36], 1: [95, 91], 2: [96]}
PRED = 144


def _write(tmp, tid, n=1200, seed=0):
    rng = np.random.default_rng(seed + tid)
    patv = np.abs(np.sin(np.arange(n) / 37.0) * 400 + rng.normal(0, 60, n)) + 5.0
    data = {c: rng.normal(10, 2, n) for c in FEATURES if c != "Patv"}
    data["Patv"] = patv
    data["TurbID"] = tid
    data["Tmstamp"] = pd.date_range("2020-01-01", periods=n, freq="10min")
    p = os.path.join(tmp, f"Turbine_{tid}_2020.csv")
    pd.DataFrame(data).to_csv(p, index=False)
    return p


@pytest.fixture()
def paths(tmp_path):
    return {t: _write(str(tmp_path), t) for ts in DC_ASSIGN.values() for t in ts}


def _provider(paths, tier=DEFAULT_TIER, **kw):
    return PerturbedGodEyeProvider(DC_ASSIGN, paths, perturb_tier=tier,
                                   pred_len=PRED, feature_columns=FEATURES, **kw)


class TestGodeyeIsTheIdentity:
    def test_godeye_series_is_the_truth_bit_for_bit(self, paths):
        p = _provider(paths, "godeye")
        for d, tids in DC_ASSIGN.items():
            for t in tids:
                got = p._perturbed_series(t, d, 200)
                assert np.array_equal(got, p.true_series(t, 200))

    def test_godeye_features_equal_the_features_of_the_true_future(self, paths):
        p = _provider(paths, "godeye")
        feats = p.get_features(200)
        for d, tids in DC_ASSIGN.items():
            truth_feats = p._aggregate_dc({t: p.true_series(t, 200) for t in tids}, tids)
            assert np.array_equal(feats[d], truth_feats)

    def test_aggregation_is_the_timecap_provider_s_aggregation(self, paths):
        """Otherwise "identical to godeye" would only mean "identical to my own copy"."""
        from src.prediction.timecap_godeye_provider import TimeCAPGodEyeProvider
        p = _provider(paths, "godeye")
        # The aggregation is a pure function of (per-turbine arrays, max powers, windows);
        # build the reference without loading a checkpoint.
        ref = TimeCAPGodEyeProvider.__new__(TimeCAPGodEyeProvider)
        ref.short_term_steps, ref.long_term_steps, ref.pred_len = 3, PRED, PRED
        ref.predictor = type("P", (), {"max_power_kw": p.max_power_kw})()
        for d, tids in DC_ASSIGN.items():
            per_t = {t: p.true_series(t, 300) for t in tids}
            assert np.array_equal(p._aggregate_dc(per_t, tids),
                                  ref._aggregate_dc(per_t, tids))


class TestLeadZeroAndScope:
    @pytest.mark.parametrize("tier", ["s30", "shrink50", "anti", "shuffle"])
    def test_lead_zero_is_never_corrupted(self, paths, tier):
        p = _provider(paths, tier)
        for d, tids in DC_ASSIGN.items():
            for t in tids:
                assert p._perturbed_series(t, d, 250)[0] == p.true_series(t, 250)[0]

    @pytest.mark.parametrize("tier", ["s30", "shrink50", "anti"])
    def test_only_leads_one_and_beyond_move(self, paths, tier):
        p = _provider(paths, tier)
        got = p._perturbed_series(12, 0, 250)
        truth = p.true_series(12, 250)
        assert got[0] == truth[0]
        assert not np.array_equal(got[1:], truth[1:])

    def test_the_world_is_never_touched(self, paths):
        """The provider must not mutate the truth it serves; the settlement reads it."""
        p = _provider(paths, "anti")
        before = {t: p.truth[t].copy() for t in p.truth}
        p.get_features(100)
        p.get_features(400)
        for t, arr in before.items():
            assert np.array_equal(arr, p.truth[t])


class TestDeterminism:
    @pytest.mark.parametrize("tier", ["s30", "shrink50", "anti", "shuffle"])
    def test_two_providers_agree_bit_for_bit(self, paths, tier):
        a, b = _provider(paths, tier), _provider(paths, tier)
        for step in (0, 137, 500):
            fa, fb = a.get_features(step), b.get_features(step)
            for d in DC_ASSIGN:
                assert np.array_equal(fa[d], fb[d]), (tier, step, d)

    def test_repeat_calls_are_stable(self, paths):
        p = _provider(paths, "s30")
        assert np.array_equal(p.get_features(210)[0], p.get_features(210)[0])

    def test_reset_does_not_change_the_view(self, paths):
        p = _provider(paths, "shrink50")
        first = p.get_features(180)[1].copy()
        p.reset()
        assert np.array_equal(p.get_features(180)[1], first)

    def test_episode_key_is_shared_across_sites_and_data_derived(self, paths):
        p = _provider(paths, "s30")
        assert len(p._episode_key) == 64
        q = _provider({k: v for k, v in paths.items()}, "s30")
        assert p._episode_key == q._episode_key


class TestTierChangesTheFeatures:
    @pytest.mark.parametrize("tier", ["s30", "shrink50", "anti", "shuffle"])
    def test_features_differ_from_godeye(self, paths, tier):
        clean = _provider(paths, "godeye").get_features(260)
        dirty = _provider(paths, tier).get_features(260)
        assert any(not np.array_equal(clean[d], dirty[d]) for d in DC_ASSIGN), tier

    def test_shrink50_pulls_the_level_toward_the_flat_mean(self, paths):
        """The pilot's mechanism: the amplitude of the future profile is compressed."""
        g = _provider(paths, "godeye")
        s = _provider(paths, "shrink50")
        truth = g._perturbed_series(12, 0, 300)
        shrunk = s._perturbed_series(12, 0, 300)
        assert shrunk[1:].std() < truth[1:].std()

    def test_anti_is_not_the_env_side_value_mirror(self, paths):
        """The env's FORECAST_PERTURB_MODE=anti computes 1 - feature. This anti reverses
        time. The two share a name and nothing else; this pins which one is wired."""
        p = _provider(paths, "anti")
        got = p._perturbed_series(12, 0, 300)
        series = p.truth[12]
        expected = np.maximum(0.0, series[::-1][300:300 + PRED])
        assert np.array_equal(got[1:], expected[1:])


class TestConfigKeys:
    def test_mode_key_is_validated(self, paths):
        cfg = {"green_oracle_mode": "godeye", "perturb_tier": "s30"}
        with pytest.raises(ValueError, match="perturbed_godeye"):
            from_config(cfg, DC_ASSIGN, paths)

    def test_tier_key_is_read_and_defaults_to_the_identity(self, paths):
        p = from_config({"green_oracle_mode": "perturbed_godeye"}, DC_ASSIGN, paths)
        assert p.perturb_tier == DEFAULT_TIER == "godeye"
        q = from_config({"green_oracle_mode": "perturbed_godeye",
                         "perturb_tier": "shrink50"}, DC_ASSIGN, paths)
        assert q.perturb_tier == "shrink50"

    def test_unknown_tier_fails_loudly_rather_than_falling_back_to_clean(self, paths):
        with pytest.raises(ValueError, match="unknown perturb_tier"):
            from_config({"green_oracle_mode": "perturbed_godeye",
                         "perturb_tier": "shrink42"}, DC_ASSIGN, paths)

    def test_calibrated_tier_requires_its_params(self, paths):
        with pytest.raises(ValueError, match="needs error_params"):
            _provider(paths, "calibrated_shrink_v1")

    def test_requested_tiers_are_all_supported(self):
        for tier in ("godeye", "s30", "shrink50", "anti"):
            assert tier in SUPPORTED_TIERS

    def test_describe_records_what_ran(self, paths):
        d = _provider(paths, "shrink50").describe()
        assert d["provider"] == "perturbed_godeye" and d["perturb_tier"] == "shrink50"
        assert d["uses_error_params"] is False and len(d["episode_key"]) == 64


class TestInterfaceParity:
    def test_step_and_get_matches_get_features(self, paths):
        p = _provider(paths, "s30")
        assert np.array_equal(p.step_and_get(120)[0], p.get_features(120)[0])

    def test_warmup_and_update_are_noops(self, paths):
        p = _provider(paths, "s30")
        before = p.get_features(150)[0].copy()
        p.warmup(0)
        p.update(150)
        assert np.array_equal(p.get_features(150)[0], before)
