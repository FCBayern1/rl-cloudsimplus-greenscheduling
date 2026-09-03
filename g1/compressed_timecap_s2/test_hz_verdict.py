import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("yaml")
from hz_verdict import judge, LABELS  # noqa: E402

CELLS = [f"c{i}" for i in range(6)]
KS = [2, 10, 18]


def table(intensity, mi=1000.0, drop=(), bad_env=()):
    """intensity: label -> per-grid intensity (constant across the grid unless callable)."""
    rows = {}
    for lab in LABELS:
        for c in CELLS:
            for k in KS:
                if (lab, c, k) in drop:
                    rows[(lab, c, k)] = None
                    continue
                v = intensity[lab](c, k) if callable(intensity[lab]) else intensity[lab]
                rows[(lab, c, k)] = {"carbon": v * mi, "mi": mi, "contract_ok": True,
                                     "static_ok": (lab, c, k) not in bad_env, "cap_ok": True}
    return rows


def test_textbook_pass():
    # blind 30, clean 10, shrink gives back 60% (22), shuffle/anti worse than blind
    out = judge(table({"blind": 30.0, "clean": 10.0, "primary": 22.0,
                       "shuffle": 32.0, "anti": 40.0}), CELLS, KS)
    assert out["verdict"] == "PASS_HZ_DISCOVERY"
    assert out["gates"] == {"g0_contract": True, "g1_clean_load_bearing": True,
                            "g2_primary_hurts": True, "g3_negative_controls": True}
    assert abs(out["retention_pooled"]["primary"] - 0.4) < 1e-9
    assert out["retention_pooled"]["shuffle"] < 0 and out["controls_worse_than_blind"]["anti"]


def test_missing_run_is_invalid_not_stop():
    out = judge(table({"blind": 30.0, "clean": 10.0, "primary": 22.0, "shuffle": 32.0, "anti": 40.0},
                      drop={("primary", "c3", 10)}), CELLS, KS)
    assert out["verdict"] == "INVALID_INCOMPLETE_DATA"
    assert ("c3", "primary", 10, "missing") in out["problems"]


def test_planner_env_mismatch_is_invalid_data():
    out = judge(table({"blind": 30.0, "clean": 10.0, "primary": 22.0, "shuffle": 32.0, "anti": 40.0},
                      bad_env={("clean", "c0", 2)}), CELLS, KS)
    assert out["gates"]["g0_contract"] is False
    assert out["grids_valid"] == 17 and out["verdict"] == "INVALID_INCOMPLETE_DATA"


def test_contract_failure_voids_the_grid_and_the_verdict_proceeds():
    rows = table({"blind": 30.0, "clean": 10.0, "primary": 22.0, "shuffle": 32.0, "anti": 40.0})
    rows[("primary", "c5", 18)]["contract_ok"] = False
    out = judge(rows, CELLS, KS)
    assert out["grids_voided"] == [("c5", 18)] and out["grids_valid"] == 17
    assert out["gates"]["g0_contract"] is True
    assert out["strict_reading_all_runs_contract_green"] is False
    assert out["verdict"] == "PASS_HZ_DISCOVERY"


def test_voided_grids_cannot_satisfy_direction_counts():
    # primary hurts in only four cells; void one of them -> 3 adverse cells -> STOP
    prim = lambda c, k: 30.0 if c in ("c0", "c1", "c2", "c3") else 9.9  # noqa: E731
    rows = table({"blind": 30.0, "clean": 10.0, "primary": prim, "shuffle": 32.0, "anti": 40.0})
    for k in KS:
        rows[("anti", "c3", k)]["contract_ok"] = False
    out = judge(rows, CELLS, KS)
    assert out["g2_cells_adverse"] == 3 and out["verdict"] == "STOP_HZ"


def test_clean_not_load_bearing_stops():
    out = judge(table({"blind": 30.0, "clean": 29.0, "primary": 31.0, "shuffle": 32.0, "anti": 40.0}), CELLS, KS)
    assert out["gates"]["g1_clean_load_bearing"] is False and out["verdict"] == "STOP_HZ"


def test_primary_harmless_stops_even_when_controls_fail():
    # primary 3% worse than clean (below the 5% raise) and retention 0.985 (above 0.5)
    out = judge(table({"blind": 30.0, "clean": 10.0, "primary": 10.3, "shuffle": 32.0, "anti": 40.0}), CELLS, KS)
    assert out["gates"]["g2_primary_hurts"] is False and out["verdict"] == "STOP_HZ"


def test_raise_threshold_is_inclusive_at_five_percent():
    out = judge(table({"blind": 30.0, "clean": 10.0, "primary": 10.5, "shuffle": 32.0, "anti": 40.0}), CELLS, KS)
    assert out["gates"]["g2_primary_hurts"] is True


def test_direction_gate_needs_four_cells_and_two_windows():
    # primary hurts strongly in three cells only -> pooled raise passes, direction fails
    prim = lambda c, k: 30.0 if c in ("c0", "c1", "c2") else 9.9  # noqa: E731
    out = judge(table({"blind": 30.0, "clean": 10.0, "primary": prim, "shuffle": 32.0, "anti": 40.0}), CELLS, KS)
    assert out["g2_cells_adverse"] == 3 and out["gates"]["g2_primary_hurts"] is False


def test_undefined_per_grid_retention_is_excluded_not_zeroed():
    # in cell c5 clean equals blind: per-grid retention undefined there, pooled still fine
    clean = lambda c, k: 30.0 if c == "c5" else 10.0  # noqa: E731
    out = judge(table({"blind": 30.0, "clean": clean, "primary": 22.0, "shuffle": 32.0, "anti": 40.0}), CELLS, KS)
    assert out["retention_per_grid_median"]["primary"] is not None
    assert out["g1_cells_favourable"] == 5
