#!/usr/bin/env python3
"""TB12 表示能力审计(Codex 2026-08-26 批准的新预注册,非旧 BC 门重判)。

解决的唯一问题:小样本 BC 探针只有 11 个分歧作业,无法区分"预报表示无效"
与"探针样本不足"。本审计用 60 个互不重叠、全年均匀的 offset,做**时间分块
5-fold out-of-fold** 预测,主判断**只在 clair≠greenfollow 的分歧作业上**。

主统计量(仅分歧作业,首次 eligible 决策点):
    s_j = (2·y_j − 1) · (p_fc,j − p_nofc,j),   y_j=1 ⟺ clair 认为应当等待
p 为 **OOF** 预测(该作业所属时间块从未参与训练)。

冻结门:G1 池化 E[s]≥+0.05;G2 ≥4/5 块为正;G3 block bootstrap 95% CI 下界>0;
G4 两臂均不退化。总体准确率与 gate-on-q 拟合仅作描述,**不得**替代分歧集
判据、**不得**据以宣称 trunk 无罪(Codex 更正)。

held-out T116+117 全程不读。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_gate_bc import (assert_corpora_aligned, build_bc_corpus,  # noqa: E402
                          cache_q_and_labels, degeneracy_check, train_gate)
from tb12_direction_gate import corpus_sha  # noqa: E402

POOLED_MIN = 0.05
BLOCKS_POSITIVE_MIN = 4


# ---------------------------------------------------------------- 纯函数
def signed_paired_score(y_hold, p_fc, p_nofc):
    """s = (2y−1)(p_fc − p_nofc);y=1 ⟺ clair 说等。"""
    return (2.0 * float(bool(y_hold)) - 1.0) * (float(p_fc) - float(p_nofc))


def block_bootstrap_ci(per_block_scores, n_resample=10000, seed=20260826,
                       alpha=0.05):
    """按**块**重采样(有放回),对池化均值取 (alpha/2) 分位作为 CI 下界。
    per_block_scores: {block_id: [s_j, ...]}。空块跳过。"""
    blocks = [np.asarray(v, dtype=float) for v in per_block_scores.values()
              if len(v)]
    if not blocks:
        return {"undefined": "no_nonempty_blocks", "ci_low": float("nan")}
    rng = np.random.default_rng(seed)
    k = len(blocks)
    means = np.empty(n_resample)
    for b in range(n_resample):
        idx = rng.integers(0, k, size=k)
        means[b] = np.concatenate([blocks[i] for i in idx]).mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"ci_low": lo, "ci_high": hi, "n_blocks": k,
            "n_resample": n_resample, "seed": seed}


def audit_verdict(samples, per_block, boot, degen_fc, degen_nofc,
                  pooled_min=POOLED_MIN, blocks_min=BLOCKS_POSITIVE_MIN):
    """G1–G4 机械合成。samples: [{block, s, ...}]。"""
    if not samples:
        return False, {"undefined": "no_disagreement_samples"}
    pooled = float(np.mean([s["s"] for s in samples]))
    block_means = {str(b): float(np.mean(v)) for b, v in per_block.items() if len(v)}
    n_pos = sum(1 for v in block_means.values() if v > 0)
    g1 = pooled >= pooled_min
    g2 = n_pos >= blocks_min
    g3 = bool(boot.get("ci_low", float("nan")) > 0)
    g4 = bool(degen_fc and degen_nofc)
    ok = bool(g1 and g2 and g3 and g4)
    return ok, {
        "G1_pooled": {"ok": bool(g1), "pooled_E_s": pooled, "min": pooled_min,
                      "n_samples": len(samples)},
        "G2_blocks_positive": {"ok": bool(g2), "n_positive": n_pos,
                               "n_blocks": len(block_means), "min": blocks_min,
                               "per_block_E_s": block_means},
        "G3_block_bootstrap": {"ok": g3, **boot},
        "G4_no_degeneracy": {"ok": g4, "fc_ok": bool(degen_fc),
                             "nofc_ok": bool(degen_nofc)},
        "ALL_PASS": ok,
    }


# ---------------------------------------------------------------- 执行
def oof_predictions(ck0, data, blocks, steps, lr, batch_size, seed):
    """时间分块 5-fold:第 k 折在其余块上训练,只对第 k 块产出 OOF p_hold。"""
    import torch
    q, y, index, gate0 = cache_q_and_labels(ck0, data)
    off_of = np.array([off for off, _, _, _ in index])
    oof_p = np.full(len(index), np.nan)
    fold_fit = {}
    for k, offs_k in blocks.items():
        held = np.isin(off_of, np.asarray(offs_k))
        if held.sum() == 0 or (~held).sum() == 0:
            continue
        tr = torch.tensor(~held)
        gate, fit = train_gate(gate0, q[tr], y[tr], steps, lr, batch_size, seed)
        with torch.no_grad():
            p = torch.sigmoid(gate(q).reshape(-1)).numpy()
        oof_p[held] = p[held]
        fold_fit[str(k)] = fit
        print(f"    fold {k}: train n={int((~held).sum())} "
              f"held n={int(held.sum())} acc(train)={fit['acc']:.4f}", flush=True)
    return oof_p, index, y.numpy(), fold_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None, help="experiment 名;构建审计 corpus")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fc-corpus", default=None)
    ap.add_argument("--nofc-corpus", default=None)
    ap.add_argument("--ck0", default=None)
    ap.add_argument("--spec", default="calib/tb12_repaudit_offsets.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--json-out", default="repr_audit.json")
    a = ap.parse_args()

    spec = json.loads((pathlib.Path(__file__).resolve().parent
                       / a.spec).read_text())
    if a.build:
        import tb12_gate_bc
        tb12_gate_bc.CALIB_OFFSETS = list(spec["offsets"])   # 冻结清单驱动
        build_bc_corpus(a.build, a.out)
        return

    fc = np.load(a.fc_corpus, allow_pickle=True)["corpus"][0]
    nofc = np.load(a.nofc_corpus, allow_pickle=True)["corpus"][0]
    assert_corpora_aligned(fc, nofc)
    blocks = {k: v for k, v in spec["time_blocks_5fold"].items()}
    block_of = {int(o): int(k) for k, v in blocks.items() for o in v}

    arms = {}
    for tag, data in (("fc", fc), ("nofc", nofc)):
        print(f"[AUDIT {tag}] blocked 5-fold OOF", flush=True)
        p, index, yv, fold_fit = oof_predictions(
            a.ck0, data, blocks, a.steps, a.lr, a.batch_size, a.seed)
        acc = float(((p > 0.5).astype(float) == yv)[~np.isnan(p)].mean())
        dg_ok, dg = degeneracy_check(p[~np.isnan(p)])
        arms[tag] = {"p": p, "index": index, "y": yv, "fold_fit": fold_fit,
                     "oof_acc_descriptive": acc, "degen_ok": dg_ok, "degen": dg}
        print(f"[AUDIT {tag}] OOF 总体准确率(描述性)={acc:.4f} "
              f"非退化={dg_ok}(hold率 {dg.get('frac_hold', float('nan')):.3f})",
              flush=True)

    # 主统计量:仅分歧作业的首次 eligible 决策点
    pos_fc = {(off, r): i for i, (off, t, s, r) in enumerate(arms["fc"]["index"])}
    pos_no = {(off, r): i for i, (off, t, s, r) in enumerate(arms["nofc"]["index"])}
    samples, per_block = [], {k: [] for k in blocks}
    for off, ep in fc.items():
        first = ep["first_seen"]
        for r, target_hold in ep["disagree_targets"].items():
            t_s = tuple(first[r])
            key = (int(off), int(r))
            i_fc = next((i for i, (o, t, s, rr) in enumerate(arms["fc"]["index"])
                         if (o, rr) == key and (t, s) == t_s), None)
            i_no = next((i for i, (o, t, s, rr) in enumerate(arms["nofc"]["index"])
                         if (o, rr) == key and (t, s) == t_s), None)
            if i_fc is None or i_no is None:
                continue
            pf, pn = arms["fc"]["p"][i_fc], arms["nofc"]["p"][i_no]
            if np.isnan(pf) or np.isnan(pn):
                continue
            s = signed_paired_score(target_hold, pf, pn)
            b = block_of[int(off)]
            samples.append({"offset": int(off), "job_rank": int(r),
                            "target_hold": bool(target_hold),
                            "p_fc": float(pf), "p_nofc": float(pn),
                            "s": float(s), "block": b})
            per_block[str(b)].append(s)

    boot = block_bootstrap_ci(per_block, spec["bootstrap_resamples"], a.seed)
    ok, verdict = audit_verdict(samples, per_block, boot,
                                arms["fc"]["degen_ok"], arms["nofc"]["degen_ok"])
    for k in ("G1_pooled", "G2_blocks_positive", "G3_block_bootstrap",
              "G4_no_degeneracy"):
        print(f"[AUDIT] {k}: {'PASS' if verdict[k]['ok'] else '**FAIL**'}", flush=True)
    print(f"[AUDIT] ALL: {'PASS' if ok else 'FAIL'}", flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"spec_sha256": corpus_sha(pathlib.Path(__file__).resolve().parent / a.spec),
         "fc_corpus_sha256": corpus_sha(a.fc_corpus),
         "nofc_corpus_sha256": corpus_sha(a.nofc_corpus),
         "ck0": a.ck0, "steps": a.steps, "lr": a.lr,
         "batch_size": a.batch_size, "seed": a.seed,
         "verdict": verdict, "samples": samples,
         "descriptive": {t: {"oof_acc": arms[t]["oof_acc_descriptive"],
                             "degen": arms[t]["degen"],
                             "fold_fit": arms[t]["fold_fit"]}
                         for t in ("fc", "nofc")}}, indent=1))
    print("REPR AUDIT DONE", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
