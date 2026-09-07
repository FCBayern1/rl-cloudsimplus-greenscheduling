"""The Python forecast providers must read the wind rows the simulator burns (2026-09-06):
in COMPRESSED SPLINE mode the Java provider skips 12 rows, so the per-DC offset handed to
TimeCAP / perturbed-godeye must carry the same shift."""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "g1" / "compressed_timecap_s2"))
from gym_cloudsimplus.envs.hierarchical_multidc_env import simulator_row_shift, SIM_SPLINE_SKIP_ROWS  # noqa: E402

WIND = REPO_ROOT.parent / "cloudsimplus-gateway" / "src" / "main" / "resources" / "windProduction"


def test_shift_is_the_java_spline_skip_only_in_compressed_spline_mode():
    assert SIM_SPLINE_SKIP_ROWS == 12
    assert simulator_row_shift({"time_scaling_mode": "COMPRESSED"}) == 12                                # default SPLINE
    assert simulator_row_shift({"time_scaling_mode": "COMPRESSED", "green_interpolation_mode": "STEP"}) == 0
    assert simulator_row_shift({"time_scaling_mode": "REAL_TIME"}) == 0


def test_provider_rows_match_the_simulator_rows_after_the_shift():
    import ladder_run as lr
    from src.prediction.perturbed_godeye_provider import from_config
    blk = {"compressed_power_divisor": 3000.0, "wind_csv_year": 2021, "min_time_between_events": 1.0,
           "green_oracle_mode": "perturbed_godeye", "perturb_tier": "godeye",
           "datacenters": [{"datacenter_id": 0, "time_zone_offset_rows": 18, "turbine_ids": [22, 81], "green_energy_enabled": True,
                            "time_scaling_mode": "COMPRESSED"}]}
    offset = 4240
    dc = blk["datacenters"][0]
    tz = {0: dc["time_zone_offset_rows"] + offset + simulator_row_shift(dc)}          # the env's composition, fixed
    paths = {t: str(WIND / "split" / f"Turbine_{t}_2021.csv") for t in (22, 81)}
    prov = from_config({**blk, "dc_tz_offsets": tz}, {0: [22, 81]}, paths)
    G, _ = lr.truth_curve(blk, offset, 400)                                             # simulator green at obs row t
    t = 40; clock = t + 1
    prov.step_and_get(clock)
    s = (np.asarray(prov.true_series(22, clock)) + np.asarray(prov.true_series(81, clock))) * 1000.0 / 3000.0
    # provider index i at this clock is the simulator's green at obs row t + i (index 0 = now)
    assert np.abs(s[:60] - G[0, t: t + 60]).max() < 1e-6
    # and without the shift it would be 12 rows stale
    assert np.abs(s[:60] - G[0, t - 12: t + 48]).max() > 1.0


def test_future_series_uses_provider_index_h_for_step_h():
    from gym_cloudsimplus.envs.hierarchical_multidc_env import future_series_from_raw
    raw = np.array([100.0, 200.0, 300.0, 400.0]) * 3000.0        # provider: now, +1, +2, +3 (W before the divisor)
    out = future_series_from_raw(99.0, raw, 6, 3000.0)
    assert out.tolist() == [99.0, 200.0, 300.0, 400.0, 99.0, 99.0]   # index 0 measured present; h -> raw[h]; tail repeats


def test_observation_row_is_the_clock_minus_the_event_spacing():
    from gym_cloudsimplus.envs.hierarchical_multidc_env import obs_row_from_clock
    assert obs_row_from_clock(4.0, 1.0, 99) == 3                # certification twin: step 3 at clock 4.0
    assert obs_row_from_clock(33.01, 0.01, 99) == 33            # the discarded 0.01 s trial
    assert obs_row_from_clock(0.0, 1.0, 0) == 0                 # reset: never negative
    assert obs_row_from_clock(None, 1.0, 7) == 7                # no clock published: the counter


def test_provider_shrink_tiers_equal_the_ladders_rungs_on_one_window():
    import ladder_run as lr
    from src.prediction.perturbed_godeye_provider import from_config
    from gym_cloudsimplus.envs.hierarchical_multidc_env import simulator_row_shift
    blk = {"compressed_power_divisor": 3000.0, "wind_csv_year": 2021, "min_time_between_events": 1.0,
           "green_oracle_mode": "perturbed_godeye",
           "datacenters": [{"datacenter_id": 0, "time_zone_offset_rows": 18, "turbine_ids": [22, 81], "green_energy_enabled": True,
                            "time_scaling_mode": "COMPRESSED"}]}
    offset = 4240; dc = blk["datacenters"][0]
    tz = {0: dc["time_zone_offset_rows"] + offset + simulator_row_shift(dc)}
    paths = {t: str(WIND / "split" / f"Turbine_{t}_2021.csv") for t in (22, 81)}
    G, _ = lr.truth_curve(blk, offset, 400)
    mu = lr._mu_w(blk)
    t = 40; clock = t + 1
    for tier, rung in (("shrink75", "shrink_0.75"), ("shrink50", "shrink_0.5"), ("shrink0", "shrink_0")):
        prov = from_config({**blk, "dc_tz_offsets": tz, "perturb_tier": tier}, {0: [22, 81]}, paths)
        prov.step_and_get(clock)
        raw = prov.get_raw_forecast_per_dc(horizon=60, normalize=False)[0] / 3000.0
        want = lr.rung_curve(G, rung, mu, seed_key="x")[0, t:t + 60]
        # lead 0 is the measured present (the provider never perturbs it); leads 1.. follow the rung
        assert abs(raw[0] - G[0, t]) < 1e-6
        assert np.abs(raw[1:60] - want[1:60]).max() < 1e-3, tier
