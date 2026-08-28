"""窗长特征纯函数测试(预注册 PREREG_WINDOW_FEATURES 判据口径)。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_window_features import (W1_MEDIAN_MIN, W2_ACC_GAIN_MIN,
                                  green_threshold, probe_verdict,
                                  window_features_at, window_stats)


def test_threshold_is_the_frozen_60th_percentile():
    s = np.arange(100, dtype=float)
    assert abs(green_threshold(s) - np.percentile(s, 60)) < 1e-12


def test_window_remaining_counts_contiguous_above_threshold():
    # 行 0..4 在窗内(值 10),之后掉出
    s = np.array([10, 10, 10, 10, 10, 0, 0, 0, 10, 10], dtype=float)
    f = window_features_at(s, t=0, theta=5.0, runtime_rows=3)
    assert f[0] == 5, f"当前窗剩余应为 5,实得 {f[0]}"


def test_onset_when_outside_window():
    s = np.array([0, 0, 0, 10, 10, 10, 0, 0], dtype=float)
    f = window_features_at(s, t=0, theta=5.0, runtime_rows=2)
    assert f[0] == 0 and f[1] == 3, f"窗外剩余应 0、起点应 3,实得 {f[:2]}"


def test_fits_current_is_ratio_to_runtime():
    s = np.array([10] * 8 + [0] * 4, dtype=float)
    f = window_features_at(s, t=0, theta=5.0, runtime_rows=4)
    assert abs(f[3] - 8 / 4) < 1e-12, "fits_current 应为 窗剩余/作业时长"


def test_all_zero_series_gives_no_window():
    s = np.zeros(20)
    f = window_features_at(s, t=0, theta=0.5, runtime_rows=3)
    assert f[0] == 0 and f[2] == 0 and f[4] == 0


def test_window_stats_median_and_cv():
    s = np.array([10, 10, 0, 10, 10, 10, 10, 0, 10], dtype=float)
    st = window_stats(s, theta=5.0)
    assert st["n_windows"] == 3 and st["median_rows"] == 2.0


def test_verdict_requires_both_w1_and_w2():
    ok, v = probe_verdict(0.09, 0.80, 0.75, True, True)
    assert ok and v["W1_median_gap"]["ok"] and v["W2_acc_gain"]["ok"]
    # W1 过而 W2 不过
    ok2, v2 = probe_verdict(0.09, 0.76, 0.75, True, True)
    assert not ok2 and v2["W1_median_gap"]["ok"] and not v2["W2_acc_gain"]["ok"]
    # W2 过而 W1 不过(现役水平 0.0072)
    ok3, v3 = probe_verdict(0.0072, 0.80, 0.75, True, True)
    assert not ok3 and not v3["W1_median_gap"]["ok"]


def test_verdict_fails_on_degeneracy():
    assert not probe_verdict(0.09, 0.80, 0.75, False, True)[0]
    assert not probe_verdict(0.09, 0.80, 0.75, True, False)[0]


def test_frozen_thresholds_match_prereg():
    assert W1_MEDIAN_MIN == 0.05 and W2_ACC_GAIN_MIN == 0.03
