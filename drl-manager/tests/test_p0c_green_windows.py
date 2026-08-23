"""P0-C: the three evaluation windows are stratified by GREEN SCARCITY and are
disjoint from the training window (Codex re-ruling 2026-08-23).

Two earlier versions were wrong and both are guarded against here.

The first stratified by row position, which is not the same thing: it left one
window (offset 4036) overlapping the training window by 3164 of 7200 rows, and
it did not control the variable that decides whether routing has any room to
matter.

The second stratified by scarcity but computed it wrongly. It summed all five
turbines on the same row and padded the span by max_tz=108. Neither matches what
the episode reads: tz=108 belongs to DC_APAC, which carries no turbines at all,
and the five turbines sit at three different offsets (0, 18, 54). The tests
below recompute scarcity from the CSVs under the real per-turbine read
semantics, so the artifact cannot drift back.
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
CONFIG = ROOT.parent / "config_C.yml"

# turbine -> timezone offset in rows, from the datacentre that owns it
TURBINE_TZ = {12: 0, 36: 0, 95: 18, 91: 18, 96: 54}
MULT = 1009


@pytest.fixture(scope="module")
def art():
    assert ART.is_file(), f"frozen artifact missing: {ART}"
    return json.loads(ART.read_text())


@functools.lru_cache(maxsize=1)
def _prefix():
    """Per-turbine prefix sums. Kept per turbine, never pooled across turbines,
    because each one is read at its own timezone offset."""
    series = {}
    for t in TURBINE_TZ:
        with open(WIND / f"Turbine_{t}_2021.csv") as fh:
            series[t] = [float(r["power_kw"] or 0) for r in csv.DictReader(fh)]
    n = min(len(v) for v in series.values())
    pre = {}
    for t, v in series.items():
        acc = [0.0]
        for x in v[:n]:
            acc.append(acc[-1] + x)
        pre[t] = tuple(acc)
    return pre, n


def window_mean(offset, warm, ep):
    """Mean total green power over the window the episode actually reads: each
    turbine contributes its own tz-shifted span of the same length."""
    pre, _ = _prefix()
    total = 0.0
    for t, tz in TURBINE_TZ.items():
        a = offset + warm + tz
        total += (pre[t][a + ep] - pre[t][a]) / ep
    return total


def span(offset, warm, ep, tz=0):
    return offset + warm + tz, offset + warm + tz + ep


def test_warmup_rows_comes_from_the_config_not_the_artifact():
    """`warmup_rows: 13` was an invented constant that both the artifact and an
    external reviewer reproduced. The simulator reads simulation_warmup_rows,
    which is absent everywhere and therefore 0."""
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    art_val = json.loads(ART.read_text())["safe_domain"]["warmup_rows"]
    for key in ("common", "experiment_p0cprobe_van",
                "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b",
                "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan"):
        assert cfg.get(key, {}).get("simulation_warmup_rows", 0) == art_val, key


def test_turbine_timezones_match_the_config():
    """The stratifier is only meaningful if these offsets are the ones the
    simulator uses. This is the test that would have caught max_tz=108."""
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    dcs = cfg["experiment_p0cprobe_van"]["datacenters"]
    got = {}
    for dc in dcs:
        for t in dc.get("turbine_ids") or []:
            got[t] = dc["time_zone_offset_rows"]
    assert got == TURBINE_TZ, got
    green_tz = {dc["time_zone_offset_rows"] for dc in dcs if dc.get("turbine_ids")}
    all_tz = {dc["time_zone_offset_rows"] for dc in dcs}
    assert 108 in all_tz and 108 not in green_tz, "tz=108 must be a turbine-free DC"


def test_artifact_records_per_turbine_semantics(art):
    s = art["stratifier"]
    assert {int(k): v for k, v in s["turbine_timezones_rows"].items()} == TURBINE_TZ
    assert "max_tz_offset_rows" not in art["safe_domain"], "blanket max_tz is the old bug"
    assert art["safe_domain"]["max_turbine_tz_offset_rows"] == max(TURBINE_TZ.values())


def test_schedule_is_full_cycle(art):
    assert math.gcd(MULT, art["green_episode_offset_range"]) == 1


def test_offsets_match_the_formula(art):
    r = art["green_episode_offset_range"]
    for w in art["windows"]:
        assert (MULT * w["episode_index_k"]) % r == w["offset_rows"], w


def test_recorded_scarcity_matches_the_wind_data(art):
    d = art["safe_domain"]
    for w in art["windows"]:
        got = window_mean(w["offset_rows"], d["warmup_rows"], d["episode_rows"])
        assert abs(got - w["mean_total_kw"]) < 0.05, (w["stratum"], got, w["mean_total_kw"])
    got0 = window_mean(0, d["warmup_rows"], d["episode_rows"])
    assert abs(got0 - art["training_window"]["mean_total_kw"]) < 0.05, got0


def test_recorded_percentiles_match_the_reference_domain(art):
    """Percentiles are quoted against the offsets the schedule can produce."""
    d = art["safe_domain"]
    grid = sorted(window_mean(o, d["warmup_rows"], d["episode_rows"])
                  for o in range(0, art["green_episode_offset_range"], 50))
    for w in art["windows"] + [art["training_window"]]:
        pct = 100 * bisect.bisect_left(grid, w["mean_total_kw"]) / len(grid)
        assert abs(pct - w["percentile"]) < 1.5, (w.get("stratum", "training"), pct, w["percentile"])


def test_windows_span_low_mid_high_scarcity(art):
    d = art["safe_domain"]
    grid = sorted(window_mean(o, d["warmup_rows"], d["episode_rows"])
                  for o in range(0, art["green_episode_offset_range"], 50))
    pct = {w["stratum"]: 100 * bisect.bisect_left(grid, w["mean_total_kw"]) / len(grid)
           for w in art["windows"]}
    assert pct["low"] < 33.4, pct
    assert 33.3 < pct["mid"] < 66.7, pct
    assert pct["high"] > 66.6, pct


def test_no_window_overlaps_the_training_window(art):
    """Training is open-book on k=0. Checked per turbine: a shared tz cancels in
    the comparison, but the assertion should not depend on that."""
    d = art["safe_domain"]
    for tz in set(TURBINE_TZ.values()):
        t0, t1 = span(0, d["warmup_rows"], d["episode_rows"], tz)
        for w in art["windows"]:
            a, b = span(w["offset_rows"], d["warmup_rows"], d["episode_rows"], tz)
            assert min(b, t1) - max(a, t0) <= 0, (w["stratum"], tz)
    for w in art["windows"]:
        assert w["overlap_rows_with_training"] == 0


def test_windows_are_mutually_disjoint(art):
    d = art["safe_domain"]
    for tz in set(TURBINE_TZ.values()):
        spans = sorted(span(w["offset_rows"], d["warmup_rows"], d["episode_rows"], tz)
                       for w in art["windows"])
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            assert b1 <= a2, f"windows overlap at tz={tz}: {(a1, b1)} vs {(a2, b2)}"


def test_largest_window_fits_the_trace(art):
    d = art["safe_domain"]
    _, n = _prefix()
    worst = max(w["offset_rows"] for w in art["windows"])
    assert worst + d["warmup_rows"] + d["episode_rows"] + max(TURBINE_TZ.values()) <= n


def test_offset_range_is_within_the_exact_safe_bound(art):
    """44950 is kept deliberately below the corrected bound so the frozen
    offsets do not move. It must never exceed it."""
    d = art["safe_domain"]
    _, n = _prefix()
    exact = n - d["warmup_rows"] - d["episode_rows"] - max(TURBINE_TZ.values()) \
        - d["timecap_horizon_reserve_rows"]
    assert exact == d["safe_offset_max_exact"], exact
    assert art["green_episode_offset_range"] <= exact


def test_reset_cost_is_bounded(art):
    assert max(w["episode_index_k"] for w in art["windows"]) <= 100


def test_training_window_is_recorded_as_low_green(art):
    """The paper must not claim k=0 represents the year."""
    assert art["training_window"]["percentile"] < 25


def test_evaluator_exposes_reset_skip():
    src = (ROOT / "src" / "baselines" / "evaluate.py").read_text()
    assert '"--reset-skip"' in src
