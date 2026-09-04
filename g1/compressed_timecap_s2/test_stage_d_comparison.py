import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_d_comparison import CLEAN_TIER, CORRUPT_TIER, compare  # noqa: E402

SEEDS = [20260904, 20260905, 20260906, 20260907, 20260908]
GPU = "NVIDIA GH200 120GB, 580.00"


def table_for(intensities, seeds=SEEDS):
    """intensities: {line: (clean, corrupt)} -> a row table with one row per cell/window."""
    t = {}
    for L, (c0, c1) in intensities.items():
        for s in seeds:
            t[(L, s, CLEAN_TIER[L])] = [{"carbon": c0 * 1000, "mi": 1000.0}]
            if c1 is not None:
                t[(L, s, CORRUPT_TIER)] = [{"carbon": c1 * 1000, "mi": 1000.0}]
    return t


BASE = {"NV": (1.00, None), "V": (0.80, 1.10), "NE": (1.00, None), "E": (0.82, 0.90),
        "NC": (1.00, None), "C": (0.85, 1.05), "RCV": (0.90, 1.20)}


def platforms(tag=GPU, seeds=SEEDS):
    return {f"root:{s}": tag for s in seeds}


def test_refuses_a_mixed_platform_table():
    p = platforms()
    p["root:20260908"] = "NVIDIA GeForce RTX 5080, 580.17"
    out = compare(table_for(BASE), p)
    assert out["status"] == "REFUSED_MIXED_PLATFORMS" and len(out["platforms"]) == 2


def test_single_platform_table_is_produced():
    out = compare(table_for(BASE), platforms())
    assert out["status"] == "OK" and out["platform"] == GPU
    assert out["seeds"] == SEEDS


def test_forecast_value_uses_each_line_matched_no_forecast():
    out = compare(table_for(BASE), platforms())["median"]
    assert abs(out["V"]["forecast_value"] - 0.20) < 1e-9        # against N_V
    assert abs(out["E"]["forecast_value"] - 0.18) < 1e-9        # against N_E
    assert abs(out["C"]["forecast_value"] - 0.15) < 1e-9        # against N_C
    assert abs(out["RCV"]["forecast_value"] - 0.10) < 1e-9      # risk lines share N_V


def test_containment_is_relative_to_vanilla():
    out = compare(table_for(BASE), platforms())["median"]
    vinc = (1.10 - 0.80) / 0.80
    einc = (0.90 - 0.82) / 0.82
    assert abs(out["V"]["corruption_increment"] - vinc) < 1e-9
    assert abs(out["E"]["containment_vs_vanilla"] - (1 - einc / vinc)) < 1e-9
    assert abs(out["V"]["containment_vs_vanilla"] - 0.0) < 1e-9


def test_missing_corrupt_row_leaves_the_line_without_an_increment():
    t = table_for(BASE)
    del t[("C", 20260904, CORRUPT_TIER)]
    out = compare(t, platforms())
    assert "corruption_increment" not in out["per_seed"][20260904]["C"]
    assert "corruption_increment" in out["median"]["C"]     # other seeds still contribute
