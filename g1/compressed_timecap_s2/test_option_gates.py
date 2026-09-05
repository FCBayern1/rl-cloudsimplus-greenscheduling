import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from option_gates import BLIND_ARMS, capture, gate1, gate2, gate3, gate3_row, judge  # noqa: E402


def _row(**kw):
    base = {"carbon": 1.0, "completion": 1.0, "ontime": 1.0, "forced": 0, "created": 2, "released": 2,
            "release_unknown": 0, "release_failed": 0, "held_open": 0, "hold_refused": 0, "hold_masked": 0,
            "term_green": 1, "term_margin": 1, "route_to_start_max_steps": 1.0, "start_unknown": 0, "stale": 0}
    base.update(kw)
    return base


def _ledger(n=2):
    return [{"id": str(i), "t_s": "10.0", "stale": "False"} for i in range(n)]


def test_gate3_row_clean_and_each_violation_named():
    assert gate3_row(_row(), _ledger()) == []
    assert any("ontime" in v for v in gate3_row(_row(ontime=0.99), _ledger()))
    assert any("forced" in v for v in gate3_row(_row(forced=1), _ledger()))
    assert any("hold_refused" in v for v in gate3_row(_row(hold_refused=1), _ledger()))
    assert gate3_row(_row(hold_refused=1), _ledger(), analytic=False) == []
    assert any("route_to_start" in v for v in gate3_row(_row(route_to_start_max_steps=2.0), _ledger()))
    assert any("held_open" in v for v in gate3_row(_row(held_open=1, released=1), _ledger()))
    assert any("ledger rows" in v for v in gate3_row(_row(), _ledger(3)))
    assert any("duplicate" in v for v in gate3_row(_row(), [{"id": "1", "t_s": "1", "stale": "False"}] * 2))
    assert any("start event" in v for v in gate3_row(_row(), [{"id": "0", "t_s": "", "stale": "False"},
                                                              {"id": "1", "t_s": "3", "stale": "False"}]))


def test_gate3_requires_every_arm_and_window_clean():
    rows = {("oracle_opt", 0): _row(), ("always_hold", 0): _row(ontime=0.9)}
    led = {k: _ledger() for k in rows}
    g = gate3(rows, led)
    assert g["pass"] is False and "always_hold:k0" in g["violations"]
    rows[("always_hold", 0)] = _row()
    assert gate3(rows, led)["pass"] is True
    rows[("nowait_opt", 0)] = None
    assert gate3(rows, led)["violations"]["nowait_opt:k0"] == ["missing row"]


def test_capture_and_gate1_with_denominator_validity():
    assert capture(10.0, 6.0, 7.0) == 0.75
    assert capture(10.0, 9.5, 7.0) is None                       # gap 5% < 10% of C_B
    b, st = [10.0] * 6, [6.0] * 6
    assert gate1(b, st, [6.5] * 6)["pass"] is True                # capture 0.875
    g = gate1(b, st, [7.5] * 6)                                   # 0.625 pooled
    assert g["pass"] is False and g["verdict"] == "FAIL"
    # pooled passes but two windows below 0.70 -> fail (all but one must pass)
    orc = [6.0, 6.0, 6.0, 6.0, 8.0, 8.0]
    assert gate1(b, st, orc)["pass"] is False
    assert gate1(b, st, [6.0, 6.0, 6.0, 6.0, 6.0, 8.0])["pass"] is True
    # invalid denominators: pooled invalid, or too few valid windows
    assert gate1(b, [9.5] * 6, [7.0] * 6)["verdict"] == "INVALID_DENOMINATOR"
    st2 = [6.0, 6.0, 6.0, 9.5, 9.5, 9.5]
    assert gate1(b, st2, [6.5] * 6)["verdict"] == "INVALID_DENOMINATOR"


def test_gate2_conditions_and_the_executor_red_flag():
    blinds = {n: [10.0] * 6 for n in BLIND_ARMS}
    blinds["reactive_opt"] = [9.0] * 6                             # blind* = reactive_opt
    g = gate2([8.0] * 6, blinds, [9.5] * 6, [9.6] * 6)
    assert g["blind_star"] == "reactive_opt" and g["pass"] is True
    # oracle only 3% below blind*: fails condition 1
    assert gate2([8.8] * 6, blinds, [9.5] * 6, [9.6] * 6)["pass"] is False
    # a wrong forecast also beats the blind by 5%: the executor carries the gain
    g = gate2([8.0] * 6, blinds, [8.4] * 6, [9.6] * 6)
    assert g["pass"] is False and g["verdict"] == "FAIL_EXECUTOR_CARRIES_THE_GAIN"
    # oracle not below a control on the pooled sum
    assert gate2([8.0] * 6, blinds, [7.9] * 6, [9.6] * 6)["pass"] is False


def test_judge_order_stops_at_the_first_failing_gate():
    arms = ("oracle_opt", "shuffle_opt", "anti_opt", "shrink_opt", "persistence_opt",
            "climatology_opt", "reactive_opt", "nowait_opt", "always_hold")
    rows = {(a, k): _row(carbon={"oracle_opt": 6.5, "shuffle_opt": 9.7, "anti_opt": 9.6}.get(a, 10.0))
            for a in arms for k in range(6)}
    led = {k: _ledger() for k in rows}
    refs = {"B": [10.0] * 6, "ST": [6.0] * 6}
    assert judge(rows, led, refs)["verdict"] == "PASS_GATES_1_2_3_PROCEED_TO_GATE4"
    bad = dict(rows); bad[("always_hold", 3)] = _row(forced=1)
    assert judge(bad, led, refs)["verdict"] == "STOP_GATE3"
    weak = {k: dict(v, carbon=(8.0 if k[0] == "oracle_opt" else v["carbon"])) for k, v in rows.items()}
    assert judge(weak, led, refs)["verdict"].startswith("STOP_GATE1_FAIL")
    strong_blind = {k: dict(v, carbon=(6.6 if k[0] == "reactive_opt" else v["carbon"])) for k, v in rows.items()}
    assert judge(strong_blind, led, refs)["verdict"].startswith("STOP_GATE2_FAIL")
