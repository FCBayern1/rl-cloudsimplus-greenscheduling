import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hz_decomp import decompose  # noqa: E402

CELLS = ["c1", "c2"]
KS = [26, 34]


def rows_for(inten, contract=None):
    """inten: {arm: intensity}; every grid has mi 100 so pooled intensity == inten."""
    contract = contract or {}
    return {(a, c, k): {"carbon": inten[a] * 100.0, "mi": 100.0,
                        "contract_ok": contract.get((a, c, k), True)}
            for a in ("B", "S", "ST") for c in CELLS for k in KS}


def test_shares_when_site_choice_captures_most():
    out = decompose(rows_for({"B": 1.0, "S": 0.7, "ST": 0.6}), CELLS, KS)
    r = out["all_grids"]
    assert out["status"] == "OK"
    assert abs(r["spatial_capturable"] - 0.3) < 1e-12 and abs(r["temporal_increment"] - 0.1) < 1e-12
    assert abs(r["total_lever"] - 0.4) < 1e-12 and abs(r["spatial_share"] - 0.75) < 1e-12


def test_shares_when_timing_captures_most():
    r = decompose(rows_for({"B": 1.0, "S": 0.95, "ST": 0.6}), CELLS, KS)["all_grids"]
    assert abs(r["spatial_share"] - 0.125) < 1e-12


def test_missing_grid_is_incomplete_not_guessed():
    rows = rows_for({"B": 1.0, "S": 0.7, "ST": 0.6})
    rows[("S", "c2", 34)] = None
    out = decompose(rows, CELLS, KS)
    assert out["status"] == "INCOMPLETE" and out["missing"] == [("S", "c2", 34)]


def test_contract_failure_is_reported_and_clean_subset_recomputed():
    rows = rows_for({"B": 1.0, "S": 0.7, "ST": 0.6}, contract={("S", "c1", 26): False})
    out = decompose(rows, CELLS, KS)
    assert out["contract_failures"]["S"] == [("c1", 26)] and out["contract_failures"]["B"] == []
    assert out["all_grids"]["grids"] == 4 and out["clean_grids"]["grids"] == 3
    assert abs(out["clean_grids"]["spatial_share"] - 0.75) < 1e-12


def test_s_worse_than_blind_gives_negative_spatial_share_not_an_error():
    r = decompose(rows_for({"B": 1.0, "S": 1.2, "ST": 0.6}), CELLS, KS)["all_grids"]
    assert r["spatial_capturable"] < 0 and r["spatial_share"] < 0
