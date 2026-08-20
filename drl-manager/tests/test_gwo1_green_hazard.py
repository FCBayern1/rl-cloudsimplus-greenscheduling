"""gwo1 green-window hazard posterior + freeze (5080, 2026-08-20).

The trough q*=0.5 was frozen on a decision set that skips green arrivals, so
the green-side rule needs its own posterior and its own freeze. These lock
the closed form against the ON ~ U[1500,2700] law and the freeze against
result-driven threshold picking.
"""
import pytest

from gen_sqt2 import ON_LO, ON_HI
from gwo1_green_hazard_calibrate import (CANDIDATES, accuracies, freeze,
                                         green_intervals, green_p_ends_within)


class TestPosterior:
    def test_no_need_is_zero(self):
        assert green_p_ends_within(500.0, 0.0) == 0.0

    def test_fresh_window_fits_any_job_shorter_than_on_min(self):
        # age 0: remaining is at least ON_LO, so a job of 1000 s never overflows
        assert green_p_ends_within(0.0, 1000.0) == 0.0
        assert green_p_ends_within(0.0, float(ON_LO)) == 0.0

    def test_past_on_max_is_certain(self):
        assert green_p_ends_within(float(ON_HI), 100.0) == 1.0

    def test_monotone_in_age(self):
        ps = [green_p_ends_within(a, 600.0) for a in (0, 500, 1000, 1600, 2000, 2600)]
        assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:]))

    def test_monotone_in_need(self):
        ps = [green_p_ends_within(2000.0, n) for n in (50, 200, 600, 900)]
        assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:]))

    def test_closed_form_below_on_lo(self):
        # age 1000 < ON_LO: remaining ~ U[500, 1700]; P(rem < 800) = 300/1200
        assert green_p_ends_within(1000.0, 800.0) == pytest.approx(0.25)

    def test_closed_form_above_on_lo(self):
        # age 2100 >= ON_LO: remaining ~ U[0, 600]; P(rem < 300) = 0.5
        assert green_p_ends_within(2100.0, 300.0) == pytest.approx(0.5)

    def test_differs_from_the_trough_posterior(self):
        """The whole reason for a separate freeze: different question."""
        from sqt2_prescreen import hazard_p_end_within
        a, b = 2000.0, 600.0
        assert green_p_ends_within(a, b) != pytest.approx(hazard_p_end_within(a, b))


class TestGreenIntervals:
    def test_complement_of_troughs(self):
        troughs = [{"start": 100, "dur": 50}, {"start": 400, "dur": 30},
                   {"start": 900, "dur": 20}]
        # includes the LEADING span 0..100 (it has a following trough, so the
        # "next onset reachable?" question is well defined there)
        assert green_intervals(troughs) == [(0.0, 100.0, 50), (150.0, 400.0, 30),
                                            (430.0, 900.0, 20)]

    def test_carries_the_following_trough_duration(self):
        spans = green_intervals([{"start": 0, "dur": 10}, {"start": 500, "dur": 77}])
        assert spans == [(10.0, 500.0, 77)]   # needed for "next onset reachable?"

    def test_leading_span_included_trailing_excluded(self):
        spans = green_intervals([{"start": 200, "dur": 40}, {"start": 900, "dur": 10}])
        assert spans[0] == (0.0, 200.0, 40)          # leading: kept
        assert all(s[0] < 910 for s in spans)        # trailing (>=910): dropped,
        # a job there has no following trough and can never reach a next onset

    def test_single_trough_has_no_bounded_green_span(self):
        assert green_intervals([{"start": 0, "dur": 10}]) == []


class TestFreeze:
    def test_picks_argmax_accuracy(self):
        res = {f"green_hazard@{q:.2f}": {"acc": 0.60} for q in CANDIDATES}
        res["green_hazard@0.40"] = {"acc": 0.81}
        q, acc = freeze(res)
        assert q == 0.40 and acc == 0.81

    def test_trivial_baselines_are_reported(self):
        dset = [(True, 0.0, 100.0, 500.0, 2.0), (False, 0.0, 100.0, 500.0, 1.0)]
        res = accuracies(dset)
        assert res["always_wait"]["acc"] == 0.5
        assert res["never_wait"]["acc"] == 0.5
        assert res["always_wait"]["acc_mi"] == pytest.approx(2 / 3)

    def test_candidate_grid_matches_the_trough_side(self):
        assert CANDIDATES == (0.25, 0.40, 0.50, 0.60)
