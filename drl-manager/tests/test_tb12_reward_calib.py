"""tb12_reward_calib 纯函数测试:fixed_max 提案与真值表逻辑。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_reward_calib import propose_fixed_max, truth_table, CAP


def test_propose_fixed_max_leaves_2x_headroom():
    kgs = np.array([1e-4, 5e-4, 5.4e-4])
    fm = propose_fixed_max(kgs)
    assert abs(kgs.max() / fm - CAP / 2.0) < 1e-12  # max 比值 = 1.5


def test_truth_table_order_match_when_uncapped():
    # 无封顶时 ΣĈ 与物理 kg 严格等比 → 排序必然一致
    per_kg = {"a": [1e-4, 2e-4], "b": [5e-5, 5e-5], "c": [4e-4, 4e-4]}
    steps = {k: 2 for k in per_kg}
    fm = propose_fixed_max(np.concatenate([np.asarray(v) for v in per_kg.values()]))
    tt = truth_table(per_kg, steps, fm)
    assert tt["_order_match"]
    assert tt["_order_phys"] == ["b", "a", "c"]
    assert all(tt[a]["cap_hits"] == 0 for a in per_kg)


def test_truth_table_detects_cap_arbitrage():
    # 复现事故:分母过小 → 集中爆发被封顶抹掉,排序反转被检出
    per_kg = {"burst": [0.0, 0.0, 1.0e-3],       # 物理更差(集中)
              "spread": [3.0e-4, 3.0e-4, 3.0e-4]}  # 物理更好(分散)
    steps = {k: 3 for k in per_kg}
    fm = 2e-05  # 事故分母:两臂大量步撞 3.0 封顶
    tt = truth_table(per_kg, steps, fm)
    assert tt["burst"]["phys_kg"] > tt["spread"]["phys_kg"]
    assert tt["burst"]["sum_chat"] < tt["spread"]["sum_chat"]  # 奖励空间反转
    assert not tt["_order_match"]
    assert tt["burst"]["cap_hits"] >= 1


def test_assert_year_consistency_locks_training_distribution():
    import pytest
    from tb12_reward_calib import assert_year_consistency
    assert_year_consistency({"csv_year": 2021}, 2021)  # 通过
    with pytest.raises(SystemExit):
        assert_year_consistency({"csv_year": 2021}, 2020)  # 2020 事故复现
    with pytest.raises(SystemExit):
        assert_year_consistency({}, 2021)  # 缺键也拒绝
