"""方向门纯函数测试(Codex ② 重做版:分歧作业 signed 移动)。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_direction_gate import (movement_gate_verdict, pooled_auc,
                                 worthy_labels, GAP_MIN)


def mk(off, target_hold, p0, p5):
    return {"offset": off, "job_rank": 0, "target_hold": target_hold,
            "p_ck0": p0, "p_ck50": p5}


def test_correct_movement_both_directions_passes():
    # clair 说等的 p_hold 上移,clair 说走的下移 -> PASS
    samples = []
    for off in range(5):
        samples += [mk(off, True, 0.5, 0.7), mk(off, False, 0.5, 0.3)]
    ok, d = movement_gate_verdict(samples)
    assert ok and abs(d["pooled_movement"] - 0.2) < 1e-12
    assert d["offsets_valid"] == 5 and d["offsets_positive"] == 5


def test_uniform_drift_up_fails():
    # 全体 p_hold 同涨(defer 漂移):target=route 的样本贡献负移动 -> 池化 0
    samples = []
    for off in range(5):
        samples += [mk(off, True, 0.5, 0.7), mk(off, False, 0.5, 0.7)]
    ok, d = movement_gate_verdict(samples)
    assert not ok and abs(d["pooled_movement"]) < 1e-12


def test_frozen_policy_zero_movement_fails():
    samples = [mk(off, bool(off % 2), 0.6, 0.6) for off in range(5)]
    ok, d = movement_gate_verdict(samples)
    assert not ok and d["pooled_movement"] == 0.0


def test_no_disagreement_undefined_fail():
    ok, d = movement_gate_verdict([])
    assert not ok and d["undefined"] == "no_disagreement_samples"


def test_fewer_than_4_valid_offsets_fails_even_if_positive():
    samples = [mk(off, True, 0.4, 0.9) for off in range(3)]   # 仅 3 个有效 offset
    ok, d = movement_gate_verdict(samples)
    assert not ok and d["offsets_valid"] == 3


def test_4_of_5_positive_passes_3_of_5_fails():
    good = [mk(off, True, 0.4, 0.8) for off in range(4)]       # 4 正
    bad1 = [mk(4, True, 0.8, 0.4)]                             # 1 负
    ok, d = movement_gate_verdict(good + bad1)
    assert ok and d["offsets_positive"] == 4
    bad2 = [mk(3, True, 0.8, 0.2), mk(4, True, 0.8, 0.2)]
    ok2, d2 = movement_gate_verdict(good[:3] + bad2)
    assert not ok2 and d2["offsets_positive"] == 3


def test_pooled_threshold_gap_min():
    hi = [mk(off, True, 0.5, 0.5 + 0.051) for off in range(5)]
    ok, _ = movement_gate_verdict(hi)
    assert ok
    lo = [mk(off, True, 0.5, 0.5 + 0.049) for off in range(5)]
    ok2, d2 = movement_gate_verdict(lo)
    assert not ok2 and d2["pooled_movement"] < GAP_MIN


def test_auc_helper_still_sane():
    assert pooled_auc([0.9], [0.1]) == 1.0
    assert np.isnan(pooled_auc([], [0.1]))


def test_worthy_labels_from_teacher_releases():
    arrivals = np.array([0.0, 1200.0, 2400.0])
    releases = [0.0, 4800.0, 2400.0]
    lab = worthy_labels(releases, arrivals)
    assert lab == {0: False, 1: True, 2: False}
