import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_cert_verdict import development_windows, error_gate, mechanism_control  # noqa: E402


def _row(c, comp=1.0, ontime=1.0, forced=0):
    return {"carbon": c, "completion": comp, "ontime": ontime, "forced": forced}


def _rows(b=1.0, st=0.7, sh=1.1, an=1.2, n=12, **over):
    rows = {}
    for k in range(n):
        rows[("reactive_wait_planner", k)] = _row(b)
        rows[("godeye", k)] = _row(st)
        rows[("shuffle", k)] = _row(sh)
        rows[("anti", k)] = _row(an)
    for key, r in over.items():
        arm, k = key.rsplit("_k", 1)
        rows[(arm, int(k))] = r
    return rows


def test_mechanism_control_needs_st_below_blind_controls_not_below_and_contract():
    ks = list(range(12))
    assert mechanism_control(_rows(), ks)["pass"] is True
    assert mechanism_control(_rows(st=1.0), ks)["pass"] is False                   # ST not below
    assert mechanism_control(_rows(sh=0.9), ks)["pass"] is False                   # a control beats the blind
    m = mechanism_control(_rows(godeye_k3=_row(0.7, ontime=0.9)), ks)
    assert m["pass"] is False and m["contract_violations"] == ["godeye:k3"]
    rows = _rows(); rows[("anti", 5)] = None
    assert mechanism_control(rows, ks)["verdict"] == "INVALID_INCOMPLETE"


def test_development_windows_take_the_hash_earliest_passing_six():
    pool = list(range(100, 1300, 100))                     # 12 offsets in hash order
    rows = _rows()
    rows[("godeye", 1)] = _row(0.95)                       # k1: 5 % relative gap -> fails
    rows[("godeye", 4)] = _row(0.999)                      # k4 fails
    d = development_windows(rows, pool, c_brown_ref=1.0)
    assert d["status"] == "OK" and d["dev_k"] == [0, 2, 3, 5, 6, 7]
    assert d["dev_offsets"] == [100, 300, 400, 600, 700, 800]
    assert d["table"][1]["pass"] is False and d["table"][0]["abs_gate"] == 0.05
    # absolute gate: a 30 % relative gap that is below 5 % of the brown reference fails
    tiny = {k: v for k, v in _rows(b=0.01, st=0.007).items()}
    assert development_windows(tiny, pool, c_brown_ref=1.0)["status"] == "STOP_WINDOW_SPLIT"


def test_error_gate_needs_pooled_ratio_and_four_windows():
    rows = _rows()
    for k in range(6):
        rows[("calibrated_shrink_hz_v2", k)] = _row(0.8)   # 0.8/0.7 = 1.14 on every window
    g = error_gate(rows, [0, 1, 2, 3, 4, 5])
    assert g["pass"] is True and g["windows_above"] == 6
    rows[("calibrated_shrink_hz_v2", 0)] = _row(0.6); rows[("calibrated_shrink_hz_v2", 1)] = _row(0.6)
    rows[("calibrated_shrink_hz_v2", 2)] = _row(0.6)
    assert error_gate(rows, [0, 1, 2, 3, 4, 5])["pass"] is False           # only 3 windows above
    for k in range(6):
        rows[("calibrated_shrink_hz_v2", k)] = _row(0.72)  # ratio 1.03 < 1.05
    assert error_gate(rows, [0, 1, 2, 3, 4, 5])["verdict"] == "STOP_ERROR_NOT_LOAD_BEARING"
    assert error_gate({}, [0])["verdict"] == "INVALID_INCOMPLETE"
