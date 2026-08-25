"""方向门纯函数测试(Codex Step 3 附加门口径)。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_direction_gate import (direction_gate_verdict, pooled_auc,
                                 worthy_labels, GAP_MIN)


def mk(off, worthy, p):
    return {"offset": off, "job_rank": 0, "worthy": worthy,
            "p_hold": p, "logit_margin": 0.0}


def test_clean_separation_passes():
    samples = []
    for off in range(6):
        samples += [mk(off, True, 0.8), mk(off, False, 0.2)]
    ok, d = direction_gate_verdict(samples)
    assert ok and d["mean_gap"] > GAP_MIN and d["offsets_positive"] == 6
    assert abs(d["pooled_auc_diagnostic"] - 1.0) < 1e-12


def test_uniform_p_hold_fails_gap():
    # v2 事故指纹:p_hold 全 1.0(全 defer)→ 无分离 → FAIL
    samples = []
    for off in range(6):
        samples += [mk(off, True, 1.0), mk(off, False, 1.0)]
    ok, d = direction_gate_verdict(samples)
    assert not ok and abs(d["mean_gap"]) < 1e-12


def test_empty_class_is_undefined_fail():
    ok, d = direction_gate_verdict([mk(0, True, 0.9), mk(1, True, 0.8)])
    assert not ok and not d["both_classes_nonempty"]


def test_gap_threshold_around_0p05():
    def run(hi, lo):
        samples = []
        for off in range(6):
            samples += [mk(off, True, hi), mk(off, False, lo)]
        return direction_gate_verdict(samples)
    ok_hi, d_hi = run(0.551, 0.50)     # gap 0.051 > 0.05 -> PASS
    assert ok_hi and d_hi["mean_gap"] > GAP_MIN
    ok_lo, d_lo = run(0.549, 0.50)     # gap 0.049 < 0.05 -> FAIL
    assert not ok_lo and d_lo["mean_gap"] < GAP_MIN


def test_offsets_positive_minimum_4_of_6():
    samples = []
    for off in range(4):                       # 4 个正向
        samples += [mk(off, True, 0.9), mk(off, False, 0.1)]
    for off in (4, 5):                         # 2 个反向
        samples += [mk(off, True, 0.1), mk(off, False, 0.9)]
    ok, d = direction_gate_verdict(samples)
    assert d["offsets_positive"] == 4 and ok
    # 3/6 则 FAIL
    samples2 = samples[:6] + [mk(3, True, 0.1), mk(3, False, 0.9)] \
        + samples[8:]
    ok2, d2 = direction_gate_verdict(samples2)
    assert d2["offsets_positive"] < 4 and not ok2


def test_auc_handles_ties():
    assert pooled_auc([0.5], [0.5]) == 0.5
    assert pooled_auc([0.9, 0.8], [0.1]) == 1.0
    assert np.isnan(pooled_auc([], [0.1]))


def test_worthy_labels_from_teacher_releases():
    arrivals = np.array([0.0, 1200.0, 2400.0])
    releases = [0.0, 4800.0, 2400.0]          # 作业1 等待,0/2 立即
    lab = worthy_labels(releases, arrivals)
    assert lab == {0: False, 1: True, 2: False}
