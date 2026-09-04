import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("yaml")
from stage_d_longrun_verdict import CELLS, CLEAN, LINES, TIERS, judge, judge_seed  # noqa: E402

OFFS = [13016, 21088, 29160, 37232, 45304, 48230]
KS = list(range(6))


def rows_for(C, init_scale=1.3, reward_scale=-100.0, bad=()):
    """C: intensities per (line, tier) as kg per MI-unit; init rows scaled up (worse)."""
    rows = {}
    for L in LINES:
        for t in TIERS[L]:
            for c in CELLS:
                for k in KS:
                    inten = C[(L, t)]
                    row = {"carbon": inten * 1000.0, "mi": 1000.0, "comp": 1.0, "ontime": 1.0, "forced": 0,
                           "cap": 0, "reward": reward_scale * inten, "offset": OFFS[k], "tier_effective": None}
                    if (L, t, c, k) in bad:
                        row = dict(row, ontime=0.9)
                    rows[(L, "final", t, c, k)] = row
        for c in CELLS:
            for k in KS:
                inten = C[(L, CLEAN[L])] * init_scale
                rows[(L, "init", CLEAN[L], c, k)] = {"carbon": inten * 1000.0, "mi": 1000.0, "comp": 1.0,
                                                     "ontime": 1.0, "forced": 0, "cap": 0,
                                                     "reward": reward_scale * inten, "offset": OFFS[k],
                                                     "tier_effective": None}
    return rows


CRD = {"NE": {"dr_std": 0.05, "rho_routing_mean": 0.6, "rho_forecast_mean": 0.2},
       "E": {"dr_std": 0.05, "rho_routing_mean": 0.6, "rho_forecast_mean": 0.2}}


def good_C():
    # NV 1.00; V clean 0.80 (uses forecast), shrink 1.10 (hurt, gives back all);
    # NE 1.00; E clean 0.82; E shrink 0.90 (increment 9.8% vs vanilla 37.5% -> < half)
    return {("NV", "hollow"): 1.00, ("V", "godeye"): 0.80, ("V", "calibrated_shrink_v1"): 1.10,
            ("V", "shuffle"): 1.15, ("V", "anti"): 1.20, ("NE", "hollow"): 1.00, ("E", "godeye"): 0.82,
            ("E", "calibrated_shrink_v1"): 0.90, ("E", "shuffle"): 0.95, ("E", "anti"): 1.00}


def test_pass_on_five_good_seeds():
    recs = {s: judge_seed(rows_for(good_C()), OFFS, CRD) for s in range(5)}
    out = judge(recs)
    assert out["verdict"] == "PASS_STAGE_D", out
    assert out["counts"] == {"g1": 5, "g2": 5, "g3": 5, "g4": 5, "g5": 5}


def test_step2_stop_when_vanilla_ignores_forecast():
    C = dict(good_C()); C[("V", "godeye")] = 0.99; C[("V", "calibrated_shrink_v1")] = 1.00
    recs = {s: judge_seed(rows_for(C), OFFS, CRD) for s in range(5)}
    assert judge(recs)["verdict"] == "STOP_STAGE_D_STEP2"


def test_step3_stop_when_eucrd_does_not_resist():
    C = dict(good_C()); C[("E", "calibrated_shrink_v1")] = 1.12   # increment 36.6% ~ vanilla's
    recs = {s: judge_seed(rows_for(C), OFFS, CRD) for s in range(5)}
    out = judge(recs)
    assert out["verdict"] == "STOP_STAGE_D_STEP3" and out["counts"]["g4"] == 0


def test_direction_needs_four_of_five():
    recs = {s: judge_seed(rows_for(good_C()), OFFS, CRD) for s in range(5)}
    C = dict(good_C()); C[("V", "godeye")] = 0.99; C[("V", "calibrated_shrink_v1")] = 1.00
    recs[0] = judge_seed(rows_for(C), OFFS, CRD)
    assert judge(recs)["verdict"] == "PASS_STAGE_D"          # 4/5 still pass
    recs[1] = judge_seed(rows_for(C), OFFS, CRD)
    assert judge(recs)["verdict"] == "STOP_STAGE_D_STEP2"    # 3/5 fail


def test_clean_contract_failure_is_stop_contract_never_voided():
    bad = {("V", "godeye", CELLS[0], 0)}
    recs = {s: judge_seed(rows_for(good_C(), bad=bad), OFFS, CRD) for s in range(5)}
    out = judge(recs)
    assert out["verdict"] == "STOP_STAGE_D_CONTRACT" and out["contract_failed_seeds"] == list(range(5))


def test_corrupted_e_contract_failure_fails_gate5_and_v_is_reported_only():
    bad_e = {("E", "calibrated_shrink_v1", CELLS[0], 0)}
    rec = judge_seed(rows_for(good_C(), bad=bad_e), OFFS, CRD)
    assert rec["gates"]["g5"] is False and rec["e_corrupted_contract_bad"] == [(CELLS[0], 0)]
    bad_v = {("V", "calibrated_shrink_v1", CELLS[1], 2)}
    rec = judge_seed(rows_for(good_C(), bad=bad_v), OFFS, CRD)
    assert rec["gates"]["g2"] is True and rec["v_corrupted_contract_bad"] == [(CELLS[1], 2)]


def test_missing_row_or_wrong_offset_is_invalid():
    rows = rows_for(good_C())
    rows.pop(("E", "final", "anti", CELLS[2], 3))
    assert judge_seed(rows, OFFS, CRD)["verdict"] == "INVALID_DATA"
    rows = rows_for(good_C())
    rows[("V", "final", "godeye", CELLS[0], 0)]["offset"] = 2018
    assert judge_seed(rows, OFFS, CRD)["verdict"] == "INVALID_DATA"


def test_reward_carbon_opposition_stops():
    recs = {s: judge_seed(rows_for(good_C(), reward_scale=+100.0), OFFS, CRD) for s in range(5)}
    out = judge(recs)
    assert out["verdict"] == "STOP_STAGE_D_STEP2" and len(out["codirection_failed_seeds"]) == 5


def test_row_count_is_360_main_plus_144_init():
    rows = rows_for(good_C())
    assert sum(1 for k in rows if k[1] == "final") == 360
    assert sum(1 for k in rows if k[1] == "init") == 144
