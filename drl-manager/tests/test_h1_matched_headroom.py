"""H1 matched-headroom evaluator units: blindify byte-parity with the env's
none-branch, and shared-route action assembly per temporal mode."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h1_matched_headroom import assemble_actions, blindify


class TestBlindify:
    def test_matches_env_persistence_fills(self):
        obs = {"dc_current_green_power_w": np.array([1500.0, 0.0, 6000.0]),
               "dc_future_short_mean": np.array([0.9, 0.9, 0.9]),
               "dc_future_short_trend": np.array([0.5, -0.5, 0.1]),
               "dc_future_long_mean": np.array([0.8, 0.8, 0.8]),
               "dc_future_long_peak_timing": np.array([0.1, 0.9, 0.3]),
               "batch_cloudlet_forecast_gain": np.array([0.4, 0.0]),
               "batch_cloudlet_time_to_best_green": np.array([0.2, 1.0]),
               "batch_cloudlet_best_now_carbon": np.array([0.3, 0.6]),
               "batch_cloudlet_best_future_carbon": np.array([0.1, 0.6])}
        b = blindify(obs, green_high=3000.0)
        assert np.allclose(b["dc_future_short_mean"], [0.5, 0.0, 1.0])
        assert np.allclose(b["dc_future_short_trend"], 0.0)
        assert np.allclose(b["dc_future_long_mean"], [0.5, 0.0, 1.0])
        assert np.allclose(b["dc_future_long_peak_timing"], 0.5)
        assert np.allclose(b["batch_cloudlet_forecast_gain"], 0.0)
        assert np.allclose(b["batch_cloudlet_time_to_best_green"], 1.0)
        assert np.allclose(b["batch_cloudlet_best_future_carbon"], [0.3, 0.6])

    def test_original_untouched(self):
        obs = {"dc_current_green_power_w": np.array([100.0]),
               "dc_future_short_mean": np.array([0.9])}
        blindify(obs, 3000.0)
        assert abs(obs["dc_future_short_mean"][0] - 0.9) < 1e-9


class TestAssembleActions:
    ROUTE = np.array([3, 1, 5])
    MI = np.array([100.0, 0.0, 50.0])          # slot 1 is padding

    def test_immediate_never_defers(self):
        a = assemble_actions("immediate", self.ROUTE, self.MI,
                             np.array([0.9, 0.9, 0.9]), np.zeros(3),
                             np.zeros(3, bool), defer_idx=8)
        assert a == [3, 1, 5]

    def test_blindgate_uses_blind_p_only(self):
        a = assemble_actions("blindgate", self.ROUTE, self.MI,
                             np.array([0.6, 0.6, 0.4]), np.array([0.0, 0.0, 0.9]),
                             np.zeros(3, bool), defer_idx=8)
        assert a == [8, 1, 5]                  # padding slot keeps route value

    def test_ftgate_uses_ft_p_only(self):
        a = assemble_actions("ftgate", self.ROUTE, self.MI,
                             np.array([0.9, 0.9, 0.9]), np.array([0.4, 0.9, 0.6]),
                             np.zeros(3, bool), defer_idx=8)
        assert a == [3, 1, 8]

    def test_oraclegate_uses_teacher_flags(self):
        a = assemble_actions("oraclegate", self.ROUTE, self.MI,
                             np.zeros(3), np.zeros(3),
                             np.array([True, True, False]), defer_idx=8)
        assert a == [8, 1, 5]

    def test_shared_route_choice_across_modes(self):
        # the route target for a routed slot must be identical in every mode
        for m in ("immediate", "blindgate", "ftgate", "oraclegate"):
            a = assemble_actions(m, self.ROUTE, self.MI, np.zeros(3),
                                 np.zeros(3), np.zeros(3, bool), 8)
            assert a[0] == 3 and a[2] == 5
