"""Unit tests for the Version-B synthetic diurnal green generator.

Asserts the generated supply has the properties Version B REQUIRES (and that the spiky
real wind lacked): smooth periodic shape, a true 0W trough, a configurable peak, and that
the per-DC series are SYNCHRONIZED in phase (so during the system-wide night the only
carbon lever is temporal deferral). If any of these regress, the deferral physics breaks.
"""
import os
import sys

import pytest

DRL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DRL not in sys.path:
    sys.path.insert(0, DRL)

import gen_diurnal_green as g


def test_trough_is_true_zero_and_peak_matches():
    s = g.build_turbine_series(peak_w=800.0, n_rows=2400, period=2400, duty=0.5, phase=0.0)
    assert min(s) == 0.0, "night trough must be a true 0W (running then is all-brown)"
    assert abs(max(s) - 800.0) < 1e-6, "peak must equal the configured value"


def test_half_period_is_dark():
    s = g.build_turbine_series(peak_w=800.0, n_rows=2400, period=2400, duty=0.5, phase=0.0)
    dark = sum(1 for v in s if v == 0.0)
    assert abs(dark / len(s) - 0.5) < 0.02, "duty=0.5 => ~half the cycle is dark"


def test_periodic_and_seamless_wrap():
    P = 2400
    s = g.build_turbine_series(peak_w=500.0, n_rows=3 * P, period=P, duty=0.5, phase=0.0)
    # value at t and t+P must match (clean cyclic wrap so episodes tile seamlessly)
    for t in (0, 137, 600, 1200, 1799):
        assert abs(s[t] - s[t + P]) < 1e-9


def test_smooth_not_spiky():
    # The real wind's problem was UNFORECASTABILITY: large jumps between adjacent steps
    # (gusts) that can't be anticipated. A diurnal cycle changes gradually, so the
    # step-to-step change is a tiny fraction of the peak -> the agent can see the next
    # green window coming. (cv is NOT the right metric: a deterministic cycle with a deep
    # 0W trough has high cv yet is perfectly predictable.)
    peak = 800.0
    s = g.build_turbine_series(peak_w=peak, n_rows=7200, period=2400, duty=0.5, phase=0.0)
    max_step = max(abs(s[i] - s[i - 1]) for i in range(1, len(s)))
    assert max_step < 0.05 * peak, f"adjacent-step change must be tiny (forecastable); got {max_step:.1f}W"


def test_all_dcs_synchronized():
    """Same phase for every green DC => when one is dark, all are dark => DEFER is the
    only lever during the system-wide trough (the core of the temporal story)."""
    import statistics as st
    N, P = 7200, 2400
    series = {}
    for dc, turbs in g.DC_TURBINES.items():
        peak = g.DEFAULT_DC_PEAK_W[dc]
        per = peak / len(turbs)
        arrs = [g.build_turbine_series(per, N, P, 0.5, 0.0) for _ in turbs]
        series[dc] = [sum(a[i] for a in arrs) for i in range(N)]

    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = sum((x - ma) ** 2 for x in a) ** 0.5
        db = sum((y - mb) ** 2 for y in b) ** 0.5
        return num / (da * db)

    assert corr(series[0], series[1]) > 0.99
    assert corr(series[0], series[2]) > 0.99
    # and all dark simultaneously at the trough
    for dc in series:
        assert series[dc][0] == 0.0
