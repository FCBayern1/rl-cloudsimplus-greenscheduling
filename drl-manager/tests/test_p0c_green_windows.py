"""P0-C: the three stratified green windows are a frozen, reproducible artifact.

Codex ruling 2026-08-23 requires the evaluation windows to be computed from the
safe offset domain, fixed before any main-arm run, identical across every arm /
seed / condition, and reached EXPLICITLY rather than produced implicitly by
consecutive resets.
"""
import json
import math
import pathlib

import pytest

ART = pathlib.Path(__file__).resolve().parent.parent / "calib" / "p0c_green_windows.json"
MULT = 1009


@pytest.fixture(scope="module")
def art():
    assert ART.is_file(), f"frozen artifact missing: {ART}"
    return json.loads(ART.read_text())


def test_schedule_is_full_cycle(art):
    """gcd(1009, range)==1, otherwise the formula cannot reach every offset."""
    assert math.gcd(MULT, art["green_episode_offset_range"]) == 1


def test_offsets_match_the_formula(art):
    r = art["green_episode_offset_range"]
    for w in art["windows"]:
        assert (MULT * w["episode_index_k"]) % r == w["offset_rows"], w


def test_one_window_per_stratum_and_well_separated(art):
    ws = sorted(art["windows"], key=lambda w: w["offset_rows"])
    assert len(ws) == 3
    r = art["green_episode_offset_range"]
    for i, w in enumerate(ws):
        lo, hi = i * r // 3, (i + 1) * r // 3
        assert lo <= w["offset_rows"] < hi, f"window {w} outside stratum [{lo},{hi})"
        # inner 50% of the stratum: never adjacent to a boundary
        assert lo + (hi - lo) * 0.25 <= w["offset_rows"] <= lo + (hi - lo) * 0.75


def test_safe_domain_leaves_room_for_episode_horizon_and_tz(art):
    d = art["safe_domain"]
    consumed = (d["warmup_rows"] + d["episode_rows"]
                + d["max_tz_offset_rows"] + d["timecap_horizon_reserve_rows"])
    assert d["safe_offset_max"] + consumed <= d["rows_available"], (
        "the largest offset plus one episode must stay inside the wind trace")


def test_largest_window_still_fits_the_trace(art):
    d = art["safe_domain"]
    worst = max(w["offset_rows"] for w in art["windows"])
    assert worst + d["warmup_rows"] + d["episode_rows"] + d["max_tz_offset_rows"] \
        <= d["rows_available"]


def test_reset_cost_is_bounded(art):
    """Every reset restarts the Java simulation, so the schedule must be cheap."""
    assert max(w["episode_index_k"] for w in art["windows"]) <= 100


def test_evaluator_exposes_reset_skip():
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "baselines" / "evaluate.py").read_text()
    assert '"--reset-skip"' in src, "explicit window selection must be a CLI flag"
