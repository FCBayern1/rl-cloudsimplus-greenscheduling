"""P0-C: the three evaluation windows are stratified by GREEN SCARCITY and are
disjoint from the training window (Codex re-ruling 2026-08-23).

The first version stratified by row position. That is not the same thing: it
left one window (offset 4036) overlapping the training window by 3164 of 7200
rows, and it did not control the variable that decides whether routing has any
room to matter. Green scarcity is that variable, so the strata are terciles of
the mean turbine power over the window the episode actually reads.
"""
import bisect
import csv
import functools
import json
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "calib" / "p0c_green_windows.json"
WIND = ROOT.parent / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
TURBINES = [12, 36, 95, 91, 96]
MULT = 1009


@pytest.fixture(scope="module")
def art():
    assert ART.is_file(), f"frozen artifact missing: {ART}"
    return json.loads(ART.read_text())


@functools.lru_cache(maxsize=1)
def _prefix():
    series = {}
    for t in TURBINES:
        with open(WIND / f"Turbine_{t}_2021.csv") as fh:
            series[t] = [float(r["power_kw"] or 0) for r in csv.DictReader(fh)]
    n = min(len(v) for v in series.values())
    total = [sum(series[t][i] for t in TURBINES) for i in range(n)]
    pre = [0.0]
    for v in total:
        pre.append(pre[-1] + v)
    return tuple(pre), n


def window_mean(offset, warm, ep, tz):
    pre, _ = _prefix()
    a, b = offset + warm, offset + warm + ep + tz
    return (pre[b] - pre[a]) / (b - a)


def span(offset, warm, ep):
    """Rows the episode reads, ignoring per-DC tz shift."""
    return offset + warm, offset + warm + ep


def test_schedule_is_full_cycle(art):
    assert math.gcd(MULT, art["green_episode_offset_range"]) == 1


def test_offsets_match_the_formula(art):
    r = art["green_episode_offset_range"]
    for w in art["windows"]:
        assert (MULT * w["episode_index_k"]) % r == w["offset_rows"], w


def test_recorded_scarcity_matches_the_wind_data(art):
    d = art["safe_domain"]
    for w in art["windows"]:
        got = window_mean(w["offset_rows"], d["warmup_rows"], d["episode_rows"],
                          d["max_tz_offset_rows"])
        assert abs(got - w["mean_total_kw"]) < 0.5, (w, got)


def test_windows_span_low_mid_high_scarcity(art):
    d = art["safe_domain"]
    grid = [window_mean(o, d["warmup_rows"], d["episode_rows"], d["max_tz_offset_rows"])
            for o in range(0, d["safe_offset_max"] + 1, 50)]
    grid.sort()
    pct = {w["stratum"]: 100 * bisect.bisect_left(grid, w["mean_total_kw"]) / len(grid)
           for w in art["windows"]}
    assert pct["low"] < 33.4, pct
    assert 33.3 < pct["mid"] < 66.7, pct
    assert pct["high"] > 66.6, pct


def test_no_window_overlaps_the_training_window(art):
    """The training run is open-book on k=0; an evaluation window that shares rows
    with it is partly in-sample."""
    d = art["safe_domain"]
    t0, t1 = span(0, d["warmup_rows"], d["episode_rows"])
    for w in art["windows"]:
        a, b = span(w["offset_rows"], d["warmup_rows"], d["episode_rows"])
        assert min(b, t1) - max(a, t0) <= 0, f"{w['stratum']} overlaps training window"
        assert w["overlap_rows_with_training"] == 0


def test_windows_are_mutually_disjoint(art):
    d = art["safe_domain"]
    spans = sorted(span(w["offset_rows"], d["warmup_rows"], d["episode_rows"])
                   for w in art["windows"])
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert b1 <= a2, f"windows overlap: {(a1, b1)} vs {(a2, b2)}"


def test_largest_window_fits_the_trace(art):
    d = art["safe_domain"]
    _, n = _prefix()
    worst = max(w["offset_rows"] for w in art["windows"])
    assert worst + d["warmup_rows"] + d["episode_rows"] + d["max_tz_offset_rows"] <= n


def test_reset_cost_is_bounded(art):
    assert max(w["episode_index_k"] for w in art["windows"]) <= 100


def test_training_window_is_recorded_as_low_green(art):
    """The paper must not claim k=0 represents the year."""
    assert art["training_window"]["percentile"] < 25


def test_evaluator_exposes_reset_skip():
    src = (ROOT / "src" / "baselines" / "evaluate.py").read_text()
    assert '"--reset-skip"' in src
