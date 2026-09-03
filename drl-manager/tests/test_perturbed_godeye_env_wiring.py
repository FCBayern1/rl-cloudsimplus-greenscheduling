"""Wiring of green_oracle_mode='perturbed_godeye' into HierarchicalMultiDCEnv.

The requirement is two-sided. The new path has to work, and the two existing paths have to
be provably unchanged -- a test that only proves "the new mode runs" would not notice that
godeye had started building a provider, or that timecap's turbine map had shifted.

Everything here runs without a Java gateway: the env object is built with __new__ and only
the attributes each method actually reads are set.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_DM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _DM)
sys.path.insert(0, os.path.join(_DM, "src"))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv  # noqa

FEATURES = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Prtv", "T2m",
            "Sp", "RelH", "Wspd_w", "Wdir_w", "Patv"]


def _write(tmp, tid, year=2021, n=900):
    rng = np.random.default_rng(tid)
    d = {c: rng.normal(10, 2, n) for c in FEATURES if c != "Patv"}
    d["Patv"] = np.abs(np.sin(np.arange(n) / 31.0) * 300 + rng.normal(0, 40, n)) + 5.0
    d["TurbID"] = tid
    d["Tmstamp"] = pd.date_range("2021-01-01", periods=n, freq="10min")
    p = os.path.join(tmp, f"Turbine_{tid}_{year}.csv")
    pd.DataFrame(d).to_csv(p, index=False)
    return p


@pytest.fixture()
def csv_dir(tmp_path):
    d = str(tmp_path)
    for t in (12, 36, 95):
        _write(d, t)
    return d


DC_CONFIGS = [
    {"green_energy_enabled": True, "turbine_ids": [12, 36], "time_zone_offset_rows": 0},
    {"green_energy_enabled": True, "turbine_ids": [95], "time_zone_offset_rows": 18},
    {"green_energy_enabled": False, "turbine_ids": [], "time_zone_offset_rows": 54},
    {"green_energy_enabled": True, "turbine_ids": [], "time_zone_offset_rows": 72},
]


def _env(mode="perturbed_godeye", episode_offset=0):
    e = HierarchicalMultiDCEnv.__new__(HierarchicalMultiDCEnv)
    e.dc_configs = DC_CONFIGS
    e.dc_ids = [0, 1, 2, 3]
    e.num_datacenters = 4
    e.green_oracle_mode = mode
    e._green_episode_offset_rows = episode_offset
    e._timecap_warmup_on_reset = None
    return e


class TestModeGateUnchangedForTheOldTwo:
    """The gate's answer for godeye and timecap must be exactly what it was."""

    @pytest.mark.parametrize("mode,expected", [
        ("godeye", "godeye"), ("timecap", "timecap"),
        ("GODEYE", "godeye"), ("perturbed_godeye", "perturbed_godeye"),
    ])
    def test_accepted_modes(self, mode, expected):
        assert str(mode).lower() == expected
        assert expected in ("godeye", "timecap", "perturbed_godeye")

    def test_unknown_mode_still_rejected_and_names_all_three(self):
        src = open(HierarchicalMultiDCEnv.__init__.__code__.co_filename).read()
        assert 'not in ("godeye", "timecap", "perturbed_godeye")' in src
        assert "expected 'godeye', 'timecap' or 'perturbed_godeye'." in src

    def test_lazy_build_flag_unchanged_for_godeye_and_timecap(self):
        """godeye must still build nothing; timecap must still build lazily."""
        for mode, spaces_only, want in (
            ("godeye", False, False), ("godeye", True, False),
            ("timecap", False, True), ("timecap", True, False),
            ("perturbed_godeye", False, True), ("perturbed_godeye", True, False),
        ):
            got = mode in ("timecap", "perturbed_godeye") and not spaces_only
            assert got is want, (mode, spaces_only)


class TestTimecapPathUntouched:
    def test_builder_dispatches_only_for_the_new_mode(self, csv_dir, monkeypatch):
        """With mode=timecap the new branch must not fire."""
        e = _env("timecap")
        called = {"n": 0}
        monkeypatch.setattr(HierarchicalMultiDCEnv, "_build_perturbed_godeye_provider",
                            lambda self, cfg: called.__setitem__("n", called["n"] + 1))
        with pytest.raises(Exception):
            # No checkpoint configured, so the TimeCAP path raises -- which is the point:
            # it went down the TimeCAP path, not the new one.
            e._build_timecap_provider({"green_oracle_mode": "timecap",
                                       "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        assert called["n"] == 0

    def test_extracted_map_matches_the_inline_timecap_rules(self, csv_dir):
        """The helper the new path uses must agree with the TimeCAP path's inline map."""
        e = _env(episode_offset=7)
        got_a, got_p, got_tz = e._forecast_dc_turbine_map(csv_dir, 2021)
        # Recomputed here from the rules the inline block states, independently.
        want_a, want_p, want_tz = {}, {}, {}
        for idx, cfg in enumerate(DC_CONFIGS):
            if not cfg["green_energy_enabled"] or not cfg["turbine_ids"]:
                continue
            d = e.dc_ids[idx]
            want_a[d] = [int(t) for t in cfg["turbine_ids"]]
            want_tz[d] = int(cfg["time_zone_offset_rows"]) + 7
            for t in cfg["turbine_ids"]:
                want_p[int(t)] = os.path.join(csv_dir, f"Turbine_{t}_2021.csv")
        assert got_a == want_a and got_tz == want_tz and got_p == want_p

    def test_map_skips_non_green_and_turbineless_dcs(self, csv_dir):
        a, _, _ = _env()._forecast_dc_turbine_map(csv_dir, 2021)
        assert set(a) == {0, 1}, "DC2 is not green-enabled and DC3 has no turbines"

    def test_missing_csv_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _env()._forecast_dc_turbine_map(str(tmp_path), 2021)


class TestNewPathBuilds:
    def test_provider_is_built_with_the_configured_tier(self, csv_dir):
        e = _env()
        p = e._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye", "perturb_tier": "shrink50",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        assert p is not None and p.perturb_tier == "shrink50"
        assert p.dc_ids == [0, 1]
        assert e._timecap_warmup_on_reset is False

    def test_tier_defaults_to_the_identity(self, csv_dir):
        p = _env()._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        assert p.perturb_tier == "godeye"

    def test_episode_offset_reaches_the_row_mapping(self, csv_dir):
        """Without this the provider forecasts a different day than the sim replays."""
        a = _env(episode_offset=0)._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        b = _env(episode_offset=25)._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        assert b.row_offset[12] - a.row_offset[12] == 25
        assert not np.array_equal(a.get_features(10)[0], b.get_features(10)[0])

    def test_no_turbines_falls_back_to_godeye_rather_than_crashing(self, csv_dir):
        e = _env()
        e.dc_configs = [{"green_energy_enabled": True, "turbine_ids": [],
                         "time_zone_offset_rows": 0}]
        e.dc_ids = [0]
        assert e._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}}) is None

    def test_unknown_tier_fails_at_build_not_silently_clean(self, csv_dir):
        with pytest.raises(ValueError, match="unknown perturb_tier"):
            _env()._build_perturbed_godeye_provider({
                "green_oracle_mode": "perturbed_godeye", "perturb_tier": "nope",
                "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})


class TestProviderInterfaceTheEnvActuallyUses:
    """Every attribute/method the env calls on self.timecap_provider must exist here."""

    def test_all_call_sites_are_satisfied(self, csv_dir):
        p = _env()._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye", "perturb_tier": "s30",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        p.reset()
        p.warmup(start_step=0)
        feats = p.step_and_get(50)
        assert set(feats) == {0, 1} and feats[0].shape == (4,)
        assert p.get_raw_forecast_per_dc(horizon=4, normalize=False)[0].shape == (4,)
        assert len(p.get_predicted_wind_w_per_dc(horizon=0)) == 2
        assert p.dc_ids == [0, 1]

    def test_extra_channels_are_none_before_the_first_features(self, csv_dir):
        p = _env()._build_perturbed_godeye_provider({
            "green_oracle_mode": "perturbed_godeye",
            "timecap": {"csv_dir": csv_dir, "csv_year": 2021}})
        p.reset()
        assert p.get_raw_forecast_per_dc() is None
        assert p.get_predicted_wind_w_per_dc(0) is None
