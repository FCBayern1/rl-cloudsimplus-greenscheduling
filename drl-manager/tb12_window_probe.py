#!/usr/bin/env python3
"""TB12 窗长特征探针 —— 主判定(预注册 PREREG_WINDOW_FEATURES.md)。

复用表示审计 Run 2 的 corpus 与仪器:同一批 60 offset / 5 时间连续块 /
blocked OOF / 42 个分歧作业 / 同 ck0 / 同 steps-lr-batch-seed /
每作业等权 + 作业内类别平衡 BCE。

唯一改动:**fc 侧的 gate 输入从 q 变成 q ⊕ 窗长特征**;nofc 侧保持 q 不变。
窗长特征按**真实未来风**离线算(无需仿真):TB12 中 episode 行号 = offset + t
(run_episode 把 time_zone_offset_rows 设为 offset,接线哨兵已逐位核验过)。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_run import load_scaled  # noqa: E402
from tb12_gate_bc import cache_q_and_labels, degeneracy_check, train_gate  # noqa: E402
from tb12_repr_audit import per_job_balanced_weights  # noqa: E402
from tb12_window_features import (green_threshold, probe_verdict,  # noqa: E402
                                  window_features_at)
from tb12_direction_gate import corpus_sha  # noqa: E402

N_WIN_FEAT = 5


def build_window_matrix(index, series, theta, runtime_rows):
    """对每个决策点 (offset, t, slot, rank) 算五个窗长特征。行号 = offset + t。"""
    out = np.zeros((len(index), N_WIN_FEAT), dtype=np.float64)
    cache = {}
    for i, (off, t, _slot, _rank) in enumerate(index):
        key = (int(off) + int(t))
        v = cache.get(key)
        if v is None:
            v = window_features_at(series, key, theta, runtime_rows)
            cache[key] = v
        out[i] = v
    # 标准化(与 gate 的 q 同量级,避免大数吃掉梯度)
    sd = out.std(axis=0)
    keep = sd > 1e-12
    out[:, keep] = (out[:, keep] - out[:, keep].mean(axis=0)) / sd[keep]
    out[:, ~keep] = 0.0
    return out, [bool(x) for x in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-corpus", default="calib/tb12_repaudit_corpus_fc.npz")
    ap.add_argument("--nofc-corpus", default="calib/tb12_repaudit_corpus_nofc.npz")
    ap.add_argument("--ck0", required=True)
    ap.add_argument("--spec", default="calib/tb12_repaudit_offsets.json")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--json-out", default="../local_eval_rt/audit/tb12_window_probe.json")
    a = ap.parse_args()
    import torch

    spec = json.loads((pathlib.Path(__file__).resolve().parent / a.spec).read_text())
    blocks = spec["time_blocks_5fold"]
    fc = np.load(a.fc_corpus, allow_pickle=True)["corpus"][0]
    nofc = np.load(a.nofc_corpus, allow_pickle=True)["corpus"][0]

    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    rt_rows = int(round(art["rt_h"] * 3600.0 / 600.0))
    series = load_scaled((100, 101), 2021)
    theta = green_threshold(series)
    print(f"[WP] θ={theta:.4f} W  作业时长={rt_rows} 行", flush=True)

    arms = {}
    for tag, data, add_win in (("fc", fc, True), ("nofc", nofc, False)):
        q, y, index, gate0 = cache_q_and_labels(a.ck0, data)
        X = q.numpy().astype(np.float64)
        if add_win:
            Wm, keep = build_window_matrix(index, series, theta, rt_rows)
            X = np.concatenate([X, Wm], axis=1)
            print(f"[WP] {tag}: q={q.shape[1]} 维 + 窗长 {Wm.shape[1]} 维 "
                  f"(非常量 {sum(keep)}/{N_WIN_FEAT})", flush=True)
        Xt = torch.tensor(X, dtype=torch.float32)
        w = per_job_balanced_weights(index, y.numpy())
        off_of = np.array([o for o, _, _, _ in index])
        oof = np.full(len(index), np.nan)
        in_dim = Xt.shape[1]
        for k, offs_k in sorted(blocks.items()):
            held = np.isin(off_of, np.asarray(offs_k))
            if held.sum() == 0 or (~held).sum() == 0:
                continue
            torch.manual_seed(a.seed)
            mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 64), torch.nn.Tanh(),
                                      torch.nn.Linear(64, 1))
            g, _ = train_gate(mlp, Xt[torch.tensor(~held)], y[torch.tensor(~held)],
                              a.steps, a.lr, a.batch_size, a.seed,
                              weights=w[~held])
            with torch.no_grad():
                p = torch.sigmoid(g(Xt).reshape(-1)).numpy()
            oof[held] = p[held]
        ok = ~np.isnan(oof)
        acc = float(((oof[ok] > 0.5).astype(float) == y.numpy()[ok]).mean())
        dg_ok, dg = degeneracy_check(oof[ok])
        arms[tag] = {"p": oof, "index": index, "acc": acc,
                     "degen_ok": dg_ok, "degen": dg}
        print(f"[WP] {tag}: OOF acc={acc:.4f}  非退化={dg_ok} "
              f"(hold率 {dg.get('frac_hold', float('nan')):.3f})", flush=True)

    # 42 个分歧作业首次 eligible 决策点
    gaps, samples = [], []
    for off, ep in fc.items():
        for r, target in ep["disagree_targets"].items():
            ts = tuple(ep["first_seen"][r])
            i_fc = next((i for i, (o, t, s, rr) in enumerate(arms["fc"]["index"])
                         if (o, rr) == (int(off), int(r)) and (t, s) == ts), None)
            i_no = next((i for i, (o, t, s, rr) in enumerate(arms["nofc"]["index"])
                         if (o, rr) == (int(off), int(r)) and (t, s) == ts), None)
            if i_fc is None or i_no is None:
                continue
            pf, pn = arms["fc"]["p"][i_fc], arms["nofc"]["p"][i_no]
            if np.isnan(pf) or np.isnan(pn):
                continue
            gaps.append(abs(float(pf) - float(pn)))
            samples.append({"offset": int(off), "job_rank": int(r),
                            "target_hold": bool(target),
                            "p_fc": float(pf), "p_nofc": float(pn)})
    med = float(np.median(gaps)) if gaps else float("nan")
    ok, verdict = probe_verdict(med, arms["fc"]["acc"], arms["nofc"]["acc"],
                                arms["fc"]["degen_ok"], arms["nofc"]["degen_ok"])
    print(f"\n[WP] 分歧作业 n={len(gaps)}  |p_fc-p_nofc| 中位 = {med:.4f} "
          f"(现役 0.0072,判据 ≥0.05)", flush=True)
    for k in ("W1_median_gap", "W2_acc_gain", "W3_no_degeneracy"):
        print(f"[WP] {k}: {'PASS' if verdict[k]['ok'] else '**FAIL**'}", flush=True)
    print(f"[WP] ALL: {'PASS -> 方向有救' if ok else 'FAIL -> TB12 这条线到头'}",
          flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"theta_w": theta, "runtime_rows": rt_rows, "n_disagreement": len(gaps),
         "median_gap": med, "verdict": verdict,
         "fc_corpus_sha256": corpus_sha(a.fc_corpus),
         "nofc_corpus_sha256": corpus_sha(a.nofc_corpus),
         "samples": samples}, indent=1, ensure_ascii=False))
    print("WINDOW PROBE DONE", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
