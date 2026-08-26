"""表示能力审计纯函数测试(Codex 冻结门 G1–G4)。"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_repr_audit import (BLOCKS_POSITIVE_MIN, POOLED_MIN, audit_verdict,
                             block_bootstrap_ci, signed_paired_score)


def mk(block, s):
    return {"block": block, "s": s, "offset": 0, "job_rank": 0}


def test_signed_score_sign_convention():
    # clair 说等(y=1):fc 的 p_hold 更高 ⇒ 正分
    assert signed_paired_score(True, 0.8, 0.5) > 0
    assert signed_paired_score(True, 0.3, 0.5) < 0
    # clair 说走(y=0):fc 的 p_hold 更低 ⇒ 正分
    assert signed_paired_score(False, 0.2, 0.6) > 0
    assert signed_paired_score(False, 0.9, 0.6) < 0
    assert signed_paired_score(True, 0.5, 0.5) == 0.0


def test_bootstrap_ci_positive_when_signal_strong():
    per_block = {str(b): [0.3, 0.35, 0.28] for b in range(5)}
    r = block_bootstrap_ci(per_block, n_resample=2000)
    assert r["ci_low"] > 0 and r["n_blocks"] == 5


def test_bootstrap_ci_straddles_zero_when_one_block_dominates():
    # 单块强正、其余为零 ⇒ 按块重采样必然出现全零组合 ⇒ 下界 ≤ 0
    per_block = {"0": [1.0] * 5, "1": [0.0] * 5, "2": [0.0] * 5,
                 "3": [0.0] * 5, "4": [0.0] * 5}
    r = block_bootstrap_ci(per_block, n_resample=4000)
    assert r["ci_low"] <= 0


def test_bootstrap_undefined_on_empty():
    r = block_bootstrap_ci({"0": [], "1": []})
    assert "undefined" in r and np.isnan(r["ci_low"])


def test_verdict_all_pass():
    samples = [mk(b, 0.2) for b in range(5) for _ in range(4)]
    per_block = {str(b): [0.2] * 4 for b in range(5)}
    boot = block_bootstrap_ci(per_block, n_resample=2000)
    ok, v = audit_verdict(samples, per_block, boot, True, True)
    assert ok and v["G1_pooled"]["pooled_E_s"] >= POOLED_MIN
    assert v["G2_blocks_positive"]["n_positive"] == 5


def test_verdict_fails_when_pooled_below_threshold():
    per_block = {str(b): [0.04] * 4 for b in range(5)}
    samples = [mk(b, 0.04) for b in range(5) for _ in range(4)]
    ok, v = audit_verdict(samples, per_block,
                          block_bootstrap_ci(per_block, n_resample=1000),
                          True, True)
    assert not ok and not v["G1_pooled"]["ok"]


def test_verdict_fails_on_3_of_5_blocks_even_if_pooled_high():
    # 两块强正拉高池化,但只有 3 块为正 ⇒ G2 拦下
    per_block = {"0": [1.0], "1": [1.0], "2": [0.1],
                 "3": [-0.2], "4": [-0.2]}
    samples = [mk(int(b), s) for b, v in per_block.items() for s in v]
    ok, v = audit_verdict(samples, per_block,
                          block_bootstrap_ci(per_block, n_resample=1000),
                          True, True)
    assert v["G2_blocks_positive"]["n_positive"] == 3 and not ok


def test_verdict_fails_on_degeneracy_either_arm():
    per_block = {str(b): [0.2] * 4 for b in range(5)}
    samples = [mk(b, 0.2) for b in range(5) for _ in range(4)]
    boot = block_bootstrap_ci(per_block, n_resample=1000)
    assert not audit_verdict(samples, per_block, boot, False, True)[0]
    assert not audit_verdict(samples, per_block, boot, True, False)[0]


def test_verdict_undefined_without_disagreement_samples():
    ok, v = audit_verdict([], {}, {"ci_low": 1.0}, True, True)
    assert not ok and "undefined" in v


def test_frozen_offset_spec_is_nonoverlapping_and_uniform():
    p = pathlib.Path(__file__).resolve().parents[1] / "calib/tb12_repaudit_offsets.json"
    spec = json.loads(p.read_text())
    offs = spec["offsets"]
    assert len(offs) == 60 == len(set(offs))
    gaps = np.diff(offs)
    assert gaps.min() >= 300                      # 非重叠(每集 300 行)
    assert gaps.max() - gaps.min() <= 1           # 全年均匀
    blocks = spec["time_blocks_5fold"]
    assert len(blocks) == 5
    assert sorted(o for v in blocks.values() for o in v) == sorted(offs)
    for v in blocks.values():                      # 每块时间连续
        assert v == sorted(v) and len(v) == 12


# ---- Run 2 仪器修复:每作业等权/作业内类别平衡(Codex 冻结的四项测试)----

def _idx(spec):
    """spec: {(offset, rank): [y, y, ...]} -> (index, y)"""
    index, y = [], []
    for (off, rank), ys in spec.items():
        for t, yy in enumerate(ys):
            index.append((off, t, 0, rank))
            y.append(float(yy))
    return index, np.asarray(y)


def test_weight_job_waiting_1_step_equals_job_waiting_20_steps():
    """测试①:等 1 步与等 20 步的作业**总权重相同**。"""
    from tb12_repr_audit import per_job_balanced_weights
    index, y = _idx({(0, 0): [1, 0],                    # 等 1 步后释放
                     (0, 1): [1] * 20 + [0]})           # 等 20 步后释放
    w = per_job_balanced_weights(index, y)
    tot = {}
    for i, (off, _t, _s, r) in enumerate(index):
        tot[(off, r)] = tot.get((off, r), 0.0) + w[i]
    assert abs(tot[(0, 0)] - tot[(0, 1)]) < 1e-9


def test_weight_within_delayed_job_hold_and_route_each_half():
    """测试②:延迟作业内 hold/route 各占该作业总权重的一半。"""
    from tb12_repr_audit import per_job_balanced_weights
    index, y = _idx({(0, 0): [1] * 9 + [0]})            # H=9, R=1
    w = per_job_balanced_weights(index, y)
    wh = sum(w[i] for i in range(len(y)) if y[i] > 0.5)
    wr = sum(w[i] for i in range(len(y)) if y[i] <= 0.5)
    assert abs(wh - wr) < 1e-9
    assert abs(w[0] * 9 - w[-1] * 1) < 1e-9             # 1/(2H)*H == 1/(2R)*R


def test_majority_class_hold_is_not_upweighted():
    """测试③:hold=1 是多数类,不得被错误增权(单样本权重必须更小)。"""
    from tb12_repr_audit import per_job_balanced_weights
    index, y = _idx({(0, 0): [1] * 19 + [0]})           # hold 占 95%
    w = per_job_balanced_weights(index, y)
    w_hold = w[0]
    w_route = w[-1]
    assert w_hold < w_route                              # 多数类单样本权重更小
    assert abs(w_hold * 19 - w_route) < 1e-9             # 两类总权重相等
    # 与朴素 pos_weight>1 的对比:那会让多数类总权重更大(本方案不会)
    assert sum(w[i] for i in range(19)) <= sum(w) / 2 + 1e-9


def test_synthetic_90_10_separable_is_learnable_and_passes_G4():
    """测试④:90/10 可分合成数据能学出两类并通过 G4。"""
    import torch
    from tb12_repr_audit import per_job_balanced_weights
    from tb12_gate_bc import degeneracy_check, train_gate
    rng = np.random.default_rng(0)
    # 每个"作业"9 个 hold(特征均值 −1)+1 个 route(特征均值 +3),线性可分
    spec, feats = {}, []
    for j in range(40):
        ys = [1] * 9 + [0]
        spec[(0, j)] = ys
        for yy in ys:
            feats.append(rng.normal(-1.0 if yy else 3.0, 0.3, size=4))
    index, y = _idx(spec)
    X = torch.tensor(np.asarray(feats), dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    w = per_job_balanced_weights(index, y)
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(),
                              torch.nn.Linear(16, 1))
    gate, fit = train_gate(mlp, X, Y, steps=2000, lr=1e-3, batch_size=256,
                           seed=0, weights=w)
    assert fit["acc"] > 0.95                             # 两类都学出来了
    with torch.no_grad():
        p = torch.sigmoid(gate(X).reshape(-1)).numpy()
    ok, dg = degeneracy_check(p)
    assert ok, f"G4 未通过: {dg}"
    assert 0.05 <= dg["frac_hold"] <= 0.95
