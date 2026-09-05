import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offset_gates import ALL_ARMS, BLIND_ARMS, freeze_blind_star, judge  # noqa: E402
from test_option_gates import _ledger, _row  # noqa: E402


def _rows(carbon):
    return {(a, k): _row(carbon=carbon.get(a, 10.0), hold_refused=0) for a in ALL_ARMS for k in range(6)}


def test_freeze_picks_the_lowest_pooled_blind_and_needs_every_window():
    rows = _rows({"fixed_off_8": 9.0})
    fz = freeze_blind_star(rows, list(range(6)))
    assert fz["status"] == "FROZEN" and fz["blind_star"] == "fixed_off_8"
    assert set(fz["pooled_carbon"]) == set(BLIND_ARMS)
    rows[("reactive_off", 3)] = None
    assert freeze_blind_star(rows, list(range(6)))["status"] == "INCOMPLETE"


def test_judge_uses_the_frozen_blind_star_and_stops_in_order():
    rows = _rows({"fixed_off_8": 9.0, "oracle_off": 6.5, "shuffle_off": 9.7, "anti_off": 9.6})
    led = {k: _ledger() for k in rows}
    refs = {"B": [10.0] * 6, "ST": [6.0] * 6}
    fz = freeze_blind_star(rows, list(range(6)))
    out = judge(rows, led, refs, fz)
    assert out["verdict"] == "PASS_GATES_1_2_3_PROCEED_TO_GATE4" and out["gate2"]["blind_star"] == "fixed_off_8"
    assert judge(rows, led, refs, {"status": "MISSING"})["verdict"] == "INVALID_BLIND_STAR_NOT_FROZEN"
    weak = {k: dict(v, carbon=(8.0 if k[0] == "oracle_off" else v["carbon"])) for k, v in rows.items()}
    assert judge(weak, led, refs, fz)["verdict"].startswith("STOP_GATE1_FAIL_ACTION_SPACE_LINE_ENDS")
    # a blind that beats the oracle by the margin ends the line at gate 2, and blind* is the
    # frozen one even if another blind would now look better
    strong = {k: dict(v, carbon=(6.6 if k[0] == "fixed_off_8" else v["carbon"])) for k, v in rows.items()}
    out = judge(strong, led, refs, fz)
    assert out["verdict"].startswith("STOP_GATE2") and out["gate2"]["blind_star"] == "fixed_off_8"
    bad = dict(rows); bad[("fixed_off_72", 2)] = _row(forced=1)
    assert judge(bad, led, refs, fz)["verdict"] == "STOP_GATE3"
