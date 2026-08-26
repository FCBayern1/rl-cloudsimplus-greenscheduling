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


# ---- v3 defer 计费哨兵(Codex Step 1)----

def _ev(t, job, act, r):
    return {"t": t, "job": job, "act": act, "r": r}


def test_audit_first_defer_exactly_minus_base():
    from tb12_reward_calib import audit_defer_charges
    ok, d = audit_defer_charges([
        _ev(0, 0, "defer", -0.5), _ev(1, 0, "defer", 0.0),
        _ev(2, 0, "route", 0.37)], ledger_sum=-0.5, n_jobs=5)
    assert ok and d["per_job"][0]["first_defer_r"] == -0.5


def test_audit_catches_v2_free_first_defer():
    # v2 事故:首次 defer 免费(r=0)-> 哨兵必须抓出
    from tb12_reward_calib import audit_defer_charges
    ok, d = audit_defer_charges([_ev(0, 0, "defer", 0.0)])
    assert not ok and d["fails"][0][1] == "first_defer_not_minus_base"


def test_audit_catches_repeat_base_charge():
    from tb12_reward_calib import audit_defer_charges
    ok, d = audit_defer_charges([
        _ev(0, 0, "defer", -0.5), _ev(1, 0, "defer", -0.5)])
    assert not ok and any(f[1] == "repeat_base_charge" for f in d["fails"])


def test_audit_trust_window_excludes_post_force_events():
    # 强制路由后槽归因失效:-2.5 的 per-action 路由奖励被错标 defer,
    # 必须因 forced_mark 截断而不产生误报
    from tb12_reward_calib import audit_defer_charges
    ok, d = audit_defer_charges([
        _ev(28, 0, "defer", -0.5),
        {"t": 122, "forced_mark": True},
        _ev(122, 0, "defer", -2.5),        # 实为 job0 强制路由的 per-action
        _ev(175, 0, "defer", 0.37)],       # 实为后续作业路由奖励
        ledger_sum=-2.5, n_jobs=5)         # 账本:5 作业各收一次 base
    assert ok and d["trust_cutoff_t"] == 122


def test_audit_ledger_integer_multiple_and_range():
    from tb12_reward_calib import audit_defer_charges
    ev = [_ev(0, 0, "defer", -0.5)]
    ok, _ = audit_defer_charges(ev, ledger_sum=-1.0, n_jobs=5)   # n=2 ∈ [1,5]
    assert ok
    ok2, d2 = audit_defer_charges(ev, ledger_sum=-0.75, n_jobs=5)  # 非整数倍
    assert not ok2 and d2["fails"][0][1] == "not_integer_multiple_of_base"
    ok3, d3 = audit_defer_charges(ev, ledger_sum=-3.0, n_jobs=5)   # n=6 > 5
    assert not ok3 and d3["fails"][0][1] == "multiple_out_of_range"
    ok4, d4 = audit_defer_charges(ev, ledger_sum=0.0, n_jobs=5)    # n=0 < 1(已见 defer)
    assert not ok4


def test_audit_encounter_invariance_via_ledger():
    # 同一作业 defer 2 次 vs 100 次:账本都恰为 -0.5(增量为 0)
    from tb12_reward_calib import audit_defer_charges
    short = [_ev(0, 0, "defer", -0.5), _ev(1, 0, "route", 0.4)]
    long = [_ev(0, 0, "defer", -0.5)] + \
        [_ev(t, 0, "defer", 0.0) for t in range(1, 100)]
    for ev in (short, long):
        ok, d = audit_defer_charges(ev, ledger_sum=-0.5, n_jobs=5)
        assert ok and round(d["per_job"]["_ledger" if False else 0]["n_defer_trusted"]) >= 1


def test_audit_urgency_increment_allowed_not_double_base():
    from tb12_reward_calib import audit_defer_charges
    ok, _ = audit_defer_charges([
        _ev(0, 0, "defer", -0.5), _ev(1, 0, "defer", -0.1),
        _ev(2, 0, "defer", -0.333)])
    assert ok
