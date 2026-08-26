"""gate-only BC 探针纯函数测试(Codex 四判据口径)。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_gate_bc import (FC_MARGIN_MIN, bc_probe_verdict, clair_hold_label,
                          degeneracy_check)
from tb12_direction_gate import movement_gate_verdict


def mv(pooled, valid, positive):
    return {"pooled_movement": pooled, "offsets_valid": valid,
            "offsets_positive": positive}


def fit(acc, loss):
    return {"acc": acc, "loss": loss}


def test_clair_hold_label_matches_runner_quantization():
    # 窗 [t·600,(t+1)·600):释放落在本窗内 ⇒ route;落在之后 ⇒ hold
    assert clair_hold_label(1200.0, t=0) is True      # 1200 >= 600 -> hold
    assert clair_hold_label(600.0, t=0) is True       # 边界:下一窗起点 -> hold
    assert clair_hold_label(599.0, t=0) is False      # 本窗内 -> route
    assert clair_hold_label(1200.0, t=1) is True      # 1200 >= 1200 -> hold
    assert clair_hold_label(1199.0, t=1) is False


def test_degeneracy_rejects_all_hold_and_all_route():
    assert not degeneracy_check(np.full(100, 0.9))[0]      # 全 hold
    assert not degeneracy_check(np.full(100, 0.1))[0]      # 全 route
    ok, d = degeneracy_check(np.concatenate([np.full(50, 0.9), np.full(50, 0.1)]))
    assert ok and abs(d["frac_hold"] - 0.5) < 1e-12
    assert not degeneracy_check([])[0]


def test_degeneracy_boundary_at_5_percent():
    v = np.concatenate([np.full(5, 0.9), np.full(95, 0.1)])
    assert degeneracy_check(v)[0]                          # 恰 5% -> 通过
    v2 = np.concatenate([np.full(4, 0.9), np.full(96, 0.1)])
    assert not degeneracy_check(v2)[0]                     # 4% -> 退化


def test_verdict_all_pass():
    ok, v = bc_probe_verdict(True, mv(0.30, 5, 5), mv(0.02, 5, 2),
                             True, {"frac_hold": 0.4},
                             fit(0.93, 0.20), fit(0.71, 0.55))
    assert ok and v["ALL_PASS"]
    assert v["C2_fc_beats_nofc"]["margin"] > FC_MARGIN_MIN


def test_verdict_fails_when_nofc_matches_fc():
    # 标签可被无预报臂同样拟合 -> 不能证明预测载重
    ok, v = bc_probe_verdict(True, mv(0.30, 5, 5), mv(0.29, 5, 5),
                             True, {"frac_hold": 0.4},
                             fit(0.93, 0.20), fit(0.93, 0.20))
    assert not ok and not v["C2_fc_beats_nofc"]["ok"]


def test_verdict_fails_on_degenerate_gate():
    ok, v = bc_probe_verdict(True, mv(0.30, 5, 5), mv(0.01, 5, 1),
                             False, {"frac_hold": 0.99},
                             fit(0.93, 0.20), fit(0.60, 0.66))
    assert not ok and not v["C3_not_degenerate"]["ok"]


def test_verdict_fails_when_fc_direction_fails():
    # v3 事故指纹:移动量在噪声级 -> C1 不过
    fc_ok, fc_det = movement_gate_verdict(
        [{"offset": o, "job_rank": 0, "target_hold": True,
          "p_ck0": 0.45, "p_ck50": 0.4505} for o in range(5)])
    ok, v = bc_probe_verdict(fc_ok, fc_det, mv(0.0, 5, 0),
                             True, {"frac_hold": 0.4},
                             fit(0.80, 0.40), fit(0.70, 0.55))
    assert not ok and not v["C1_fc_direction"]["ok"]


def test_verdict_requires_fc_fit_better_not_just_movement():
    # 移动差够大但拟合不优于 nofc -> C2 仍不过(防止只靠 gate 偏置漂移)
    ok, v = bc_probe_verdict(True, mv(0.30, 5, 5), mv(0.05, 5, 3),
                             True, {"frac_hold": 0.4},
                             fit(0.70, 0.55), fit(0.70, 0.55))
    assert not ok and not v["C2_fc_beats_nofc"]["ok"]
