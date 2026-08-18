"""Prescreen gate units: hazard closed form and gate semantics."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqt2_prescreen import hazard_p_end_within, TroughIndex, gate_flags


class TestHazardPosterior:
    def test_fresh_trough_mostly_short(self):
        # age 0, budget 1500: all short troughs end in time (w=0.8), no long do
        assert abs(hazard_p_end_within(0, 1500) - 0.8) < 1e-9

    def test_age_reveals_long_trough(self):
        # age 1600 > short_max: posterior = long; budget 1000 -> min end 2700
        assert hazard_p_end_within(1600, 1000) == 0.0
        # same age, budget 2900: end by 4500 from U[2700,4500]|D>1600 -> full
        assert abs(hazard_p_end_within(1600, 2900) - 1.0) < 1e-9

    def test_deep_age_hazard_rises(self):
        # age 3000 in a long trough: at most 1500 s remain -> certain end
        assert abs(hazard_p_end_within(3000, 1500) - 1.0) < 1e-9

    def test_zero_budget_never(self):
        assert hazard_p_end_within(100, 0) == 0.0


class TestTroughIndexAndGates:
    IDX = TroughIndex([{"start": 100, "dur": 400, "kind": "short"}])

    def test_query(self):
        assert self.IDX.query(99) == (False, 0.0, 0.0)
        in_t, age, res = self.IDX.query(300)
        assert in_t and age == 200.0 and res == 200.0

    def _g(self):
        return {"batch_cloudlet_mi": np.array([40e6, 0.0]),
                "batch_cloudlet_pes": np.array([1.0, 0.0]),
                "batch_cloudlet_time_to_deadline": np.array([0.5, 0.0]),  # x3600=1800
                "batch_cloudlet_deadline_present": np.array([1.0, 0.0]),
                "global_deferred_count": np.array([0.0])}

    def test_clairvoyant_uses_residual_vs_budget(self):
        # budget = 1800-1000-120 = 680
        g = self._g()
        assert gate_flags("clairvoyant", g, 2, 3600.0, True, 50, 600)[0]
        assert not gate_flags("clairvoyant", g, 2, 3600.0, True, 50, 700)[0]

    def test_naive_defers_whenever_dark_and_budget(self):
        g = self._g()
        assert gate_flags("naive", g, 2, 3600.0, True, 50, 99999)[0]
        assert not gate_flags("naive", g, 2, 3600.0, False, 0, 0)[0]

    def test_nowait_never(self):
        assert not gate_flags("nowait", self._g(), 2, 3600.0, True, 0, 1)[0]
