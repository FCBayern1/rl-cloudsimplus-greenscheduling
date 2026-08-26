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
