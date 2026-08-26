#!/usr/bin/env python3
"""TB12 gate-only BC 探针(Codex 裁定 2026-08-26)。

回答的问题:**架构在 TB12 上到底能不能表达并学会逐作业的等/不等选择。**
不是"PPO 能不能自己发现",那是下一杆。

为什么"gate-only"在 TB12 是结构性保证而非约定:
tb12 块已开 `factorized_temporal_gate: true`,前向为

    gate_logit = temporal_gate(q)                    # 只依赖 cloudlet query q
    P(defer)   = sigmoid(gate_logit)                 # 精确二元
    P(dc_j)    = (1-sigmoid(gate_logit))·softmax(scores)_j

route 的**条件分布** softmax(scores) 完全不含 temporal_gate 的参数,因此
只训练 temporal_gate 时空间 route 被结构性冻结(不依赖任何"我们不去动它"
的约定)。同理 q 不依赖 temporal_gate ⇒ **q 在整个 BC 训练期是常量**,
可以一次性缓存,BC 退化为在冻结表示上训练一个 6.3k 参数的 MLP。

对称性(Codex:唯一差异保持 forecast_mode):
- 两臂共用**同一个 ck0**(v3 两臂 ck0 是不同随机初始化,不能各用各的);
- 两臂共用**同一套 clair 标签**、同样的 step 数、lr、batch、loss 权重;
- 两臂唯一差异 = 各自 forecast_mode 下的 obs 内容(⇒ q 不同);
- 驱动轨迹同为冻结 greenfollow(离线释放计划,不依赖 obs)⇒ 两臂动力学
  逐位相同,由 slotmap/first_seen 一致性哨兵核验。

held-out(T116+117)全程不读;仅用训练/校准分布 T100+101/2021。
"""
import argparse
import copy
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_run import FrozenPolicies, load_scaled, ROW_S  # noqa: E402
from tb12_direction_gate import (CALIB_OFFSETS, GAP_MIN,  # noqa: E402
                                 OFFSETS_POSITIVE_MIN, corpus_sha,
                                 movement_gate_verdict, worthy_labels)

FC_MARGIN_MIN = 0.02      # "fc 明显优于 nofc":池化移动差的下限
DEGEN_MIN_FRAC = 0.05     # 非全等/非全不等:两类预测各占比下限


# ---------------------------------------------------------------- 纯函数
def clair_hold_label(release_sorted_rank, t, row_s=ROW_S):
    """与 runner 同一量化语义:计划释放落在当前窗之后 ⇒ 该决策点应 hold。"""
    return bool(release_sorted_rank >= t * row_s + row_s - 1e-9)


def degeneracy_check(p_hold_values, thr=0.5, min_frac=DEGEN_MIN_FRAC):
    """非全等、非全不等:argmax 两类各占比 ≥ min_frac。"""
    if not len(p_hold_values):
        return False, {"undefined": "no_samples"}
    arr = np.asarray(p_hold_values, dtype=float)
    frac_hold = float((arr > thr).mean())
    ok = min_frac <= frac_hold <= 1.0 - min_frac
    return ok, {"frac_hold": frac_hold, "min_frac": min_frac,
                "n": int(arr.size)}


def bc_probe_verdict(fc_move_ok, fc_move_detail, nofc_move_detail,
                     fc_degen_ok, fc_degen_detail, fc_fit, nofc_fit,
                     margin_min=FC_MARGIN_MIN):
    """Codex 四判据合成:
      1) fc 方向 gap ≥ +0.05 且 ≥4/5 有效 offset 正向(= movement_gate_verdict);
      2) fc 明显优于 nofc(池化移动差 ≥ margin_min,且 fc 拟合优于 nofc);
      3) 非全等、非全不等(fc 的 gate 不退化);
    全过 ⇒ 架构可表达可学 ⇒ 允许进入 PPO+常驻锚。"""
    fc_pool = fc_move_detail.get("pooled_movement", float("nan"))
    nofc_pool = nofc_move_detail.get("pooled_movement", float("nan"))
    margin = fc_pool - nofc_pool
    fc_better = bool(margin >= margin_min and fc_fit["acc"] > nofc_fit["acc"])
    ok = bool(fc_move_ok and fc_better and fc_degen_ok)
    return ok, {
        "C1_fc_direction": {"ok": bool(fc_move_ok), **fc_move_detail},
        "C2_fc_beats_nofc": {"ok": fc_better, "fc_pooled": fc_pool,
                             "nofc_pooled": nofc_pool, "margin": margin,
                             "margin_min": margin_min,
                             "fc_acc": fc_fit["acc"], "nofc_acc": nofc_fit["acc"],
                             "fc_loss": fc_fit["loss"], "nofc_loss": nofc_fit["loss"]},
        "C3_not_degenerate": {"ok": bool(fc_degen_ok), **fc_degen_detail},
        "ALL_PASS": ok,
    }


# ---------------------------------------------------------------- corpus
def build_bc_corpus(experiment, out_path, turbines=(100, 101), year=2021):
    """冻结 greenfollow 驱动;记录 obs 序列、逐步 slotmap、clair/gf 标签。
    v3 方向门 corpus 保持不动(已随 v3 判决封存),此处另建 v4 双臂 corpus。"""
    from src.baselines.evaluate import load_config
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from oracle_slack_planner import drain_action
    cfg0 = load_config(experiment)
    cfg0.pop("py4j_port", None)
    cfg0.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg0.setdefault("output_dir", "/tmp/tb12_gw")
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    arrivals = np.array([j[1] for j in art["jobs"]], dtype=float)
    pol = FrozenPolicies(art["rt_h"] * 3600.0, art["slack_h"] * 3600.0)
    ref = load_scaled(turbines, year)
    order = np.argsort(arrivals)
    data = {}
    for off in CALIB_OFFSETS:
        w_ep = ref[off:off + 300]
        rel_gf = pol.releases("greenfollow", w_ep, arrivals)   # 驱动 = 冻结 teacher
        rel_cl = pol.releases("clair", w_ep, arrivals)         # 标签 = clair
        gf_rank = {r: worthy_labels(rel_gf, arrivals)[int(j)]
                   for r, j in enumerate(order)}
        cl_rank = {r: worthy_labels(rel_cl, arrivals)[int(j)]
                   for r, j in enumerate(order)}
        disagree = {r: cl_rank[r] for r in gf_rank if gf_rank[r] != cl_rank[r]}
        rel_gf_sorted = [rel_gf[i] for i in order]
        rel_cl_sorted = [rel_cl[i] for i in order]

        cfg = copy.deepcopy(cfg0)
        cfg["green_episode_offset_range"] = 0
        for dc in cfg["datacenters"]:
            dc["time_zone_offset_rows"] = int(off)
        env = HierarchicalMultiDCEnv(cfg)
        try:
            obs, _ = env.reset(seed=1)
            g0 = float(np.asarray(obs["global"]["dc_current_green_power_w"]).reshape(-1)[0])
            if abs(g0 - float(ref[int(off)])) > 1e-3:
                sys.exit(f"接线哨兵失败 off={off}: {g0} != {ref[int(off)]}")
            batch = env.global_routing_batch_size
            released, done, t = 0, False, 0
            obs_seq, slotmaps, first_seen = [], {}, {}
            while not done and t < 300:
                g = obs["global"]
                obs_seq.append({k: np.asarray(v).copy() for k, v in g.items()})
                mi = np.asarray(g["batch_cloudlet_mi"]).reshape(-1)[:batch]
                acts, k, smap = [], 0, {}
                for i in range(batch):
                    if mi[i] <= 0:
                        acts.append(0)
                        continue
                    j = released + k
                    smap[i] = j
                    if j not in first_seen:
                        first_seen[j] = (t, i)
                    hold = (j < len(rel_gf_sorted)
                            and rel_gf_sorted[j] >= t * ROW_S + ROW_S - 1e-9)
                    acts.append(1 if hold else 0)
                    k += 1
                slotmaps[t] = smap
                released += sum(1 for i in range(batch)
                                if mi[i] > 0 and acts[i] == 0)
                obs, _, term, trunc, _ = env.step(
                    {"global": acts,
                     "local": {0: drain_action(env.get_local_action_masks(0))}})
                done = term or trunc
                t += 1
        finally:
            env.close()
        data[str(off)] = {
            "obs_seq": obs_seq, "slotmaps": slotmaps, "first_seen": first_seen,
            "gf_by_rank": gf_rank, "clair_by_rank": cl_rank,
            "disagree_targets": disagree,
            "gf_releases_sorted": [float(x) for x in rel_gf_sorted],
            "clair_releases_sorted": [float(x) for x in rel_cl_sorted],
        }
        n_dec = sum(len(v) for v in slotmaps.values())
        print(f"[BC-CORPUS {experiment.split('rl_')[-1]:>12} off={off:>6}] "
              f"steps={len(obs_seq)} 决策点={n_dec} 分歧={len(disagree)}", flush=True)
    np.savez_compressed(out_path, corpus=np.array([data], dtype=object),
                        experiment=experiment)
    print(f"BC CORPUS SAVED {out_path} sha256={corpus_sha(out_path)[:16]}", flush=True)


def assert_corpora_aligned(fc_data, nofc_data):
    """两臂动力学必须逐位相同(驱动是离线 greenfollow 计划,与 obs 无关)。"""
    for off in fc_data:
        a, b = fc_data[off], nofc_data[off]
        if len(a["obs_seq"]) != len(b["obs_seq"]):
            sys.exit(f"corpus 步数不一致 off={off}")
        if a["first_seen"] != b["first_seen"]:
            sys.exit(f"corpus first_seen 不一致 off={off}")
        if a["slotmaps"] != b["slotmaps"]:
            sys.exit(f"corpus slotmap 不一致 off={off}")
        if a["disagree_targets"] != b["disagree_targets"]:
            sys.exit(f"corpus 分歧集不一致 off={off}")
    print("[BC] 两臂 corpus 动力学一致性哨兵通过", flush=True)


# ---------------------------------------------------------------- q 缓存
def cache_q_and_labels(ck0, data):
    """在冻结 corpus 上滚 trunk,用 forward hook 抓 temporal_gate 的输入 q。
    q 不依赖 temporal_gate ⇒ 整个 BC 期恒定,只需缓存一次。"""
    import torch
    from tb12_rl_eval import FullActionHead
    from ray.rllib.core.columns import Columns

    head = FullActionHead(pathlib.Path(ck0).resolve())
    module = head.module
    if not getattr(module, "factorized_temporal_gate", False):
        sys.exit("模块未开 factorized_temporal_gate —— gate-only 前提不成立")
    grab = {}

    def _hook(_m, inp):
        grab["q"] = inp[0].detach()
    h = module.temporal_gate.register_forward_pre_hook(_hook)

    qs, labels, index = [], [], []
    try:
        for off, ep in data.items():
            head.reset()
            rel_cl = ep["clair_releases_sorted"]
            for t, obs in enumerate(ep["obs_seq"]):
                batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)[None, ...])
                                       for k, v in obs.items()
                                       if np.asarray(v).dtype.kind in "ifub"}}
                state = head.state
                if state is None:
                    init = module.get_initial_state()
                    if init:
                        state = {k: torch.as_tensor(np.asarray(v))[None, ...]
                                 for k, v in init.items()}
                if state:
                    batch[Columns.STATE_IN] = state
                with torch.no_grad():
                    out = module.forward_inference(batch)
                head.state = out.get(Columns.STATE_OUT) or None
                q_step = grab["q"].reshape(-1, grab["q"].shape[-1])   # (N_b, D)
                for slot, rank in ep["slotmaps"].get(t, {}).items():
                    if rank >= len(rel_cl):
                        continue
                    qs.append(q_step[int(slot)].clone())
                    labels.append(1.0 if clair_hold_label(rel_cl[rank], t) else 0.0)
                    index.append((int(off), int(t), int(slot), int(rank)))
    finally:
        h.remove()
    gate0 = copy.deepcopy(module.temporal_gate)
    return (torch.stack(qs), torch.tensor(labels), index, gate0)


# ---------------------------------------------------------------- BC 训练
def train_gate(gate, q, y, steps, lr, batch_size, seed):
    """在冻结 q 上以 BCE 训练 temporal_gate。两臂预算逐位相同。"""
    import torch
    g = copy.deepcopy(gate)
    for p in g.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    n = q.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=gen)
        opt.zero_grad()
        logit = g(q[idx]).reshape(-1)
        loss = lossf(logit, y[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        logit = g(q).reshape(-1)
        loss = float(lossf(logit, y))
        acc = float(((torch.sigmoid(logit) > 0.5).float() == y).float().mean())
    return g, {"loss": loss, "acc": acc, "n": int(n),
               "label_hold_frac": float(y.mean())}


def p_hold_at(gate, q, index, want):
    """want: {(offset, rank)} -> 取该作业首次 eligible 决策点的 p_hold。"""
    import torch
    with torch.no_grad():
        p = torch.sigmoid(gate(q).reshape(-1)).numpy()
    out = {}
    for i, (off, t, slot, rank) in enumerate(index):
        key = (off, rank)
        if key in want and key not in out and (t, slot) == want[key]:
            out[key] = float(p[i])
    return out


def loo_direction(ck0, data, index_qy, steps, lr, batch_size, seed):
    """留一偏移交叉验证(报告项,非判据):每折在其余 5 个 offset 上训练,
    在留出 offset 的分歧作业上量方向移动 —— 排除"6.3k 参数记忆 ~600 个
    决策点"的可能。判据仍按 Codex 冻结的三条(in-sample)。"""
    import torch
    q, y, index, gate0 = index_qy
    folds, samples = {}, []
    offs = sorted({off for off, _, _, _ in index})
    for held in offs:
        if not data[str(held)]["disagree_targets"]:
            continue
        mask = torch.tensor([off != held for off, _, _, _ in index])
        if int(mask.sum()) == 0:
            continue
        gate, _ = train_gate(gate0, q[mask], y[mask], steps, lr, batch_size, seed)
        want = {(held, int(r)): tuple(data[str(held)]["first_seen"][r])
                for r in data[str(held)]["disagree_targets"]}
        p0 = p_hold_at(gate0, q, index, want)
        p1 = p_hold_at(gate, q, index, want)
        fold = [{"offset": held, "job_rank": r,
                 "target_hold": bool(data[str(held)]["disagree_targets"][r]),
                 "p_ck0": p0[(held, r)], "p_ck50": p1[(held, r)]}
                for (o_, r) in sorted(want) if (held, r) in p0 and (held, r) in p1]
        samples += fold
        folds[str(held)] = float(np.mean(
            [(f["p_ck50"] - f["p_ck0"]) * (1.0 if f["target_hold"] else -1.0)
             for f in fold])) if fold else float("nan")
    ok, det = movement_gate_verdict(samples)
    return {"held_out_ok_if_gating": ok, "per_fold_movement": folds, **det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=None,
                    help="experiment 名;构建该臂 BC corpus")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fc-corpus", default=None)
    ap.add_argument("--nofc-corpus", default=None)
    ap.add_argument("--ck0", default=None, help="两臂共用的起始 checkpoint")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--json-out", default="bc_probe.json")
    a = ap.parse_args()

    if a.build:
        build_bc_corpus(a.build, a.out)
        return

    fc = np.load(a.fc_corpus, allow_pickle=True)["corpus"][0]
    nofc = np.load(a.nofc_corpus, allow_pickle=True)["corpus"][0]
    assert_corpora_aligned(fc, nofc)

    res, gates = {}, {}
    for tag, data in (("fc", fc), ("nofc", nofc)):
        q, y, index, gate0 = cache_q_and_labels(a.ck0, data)
        gate, fit = train_gate(gate0, q, y, a.steps, a.lr, a.batch_size, a.seed)
        want = {(int(off), int(r)): tuple(ep["first_seen"][r])
                for off, ep in data.items() for r in ep["disagree_targets"]}
        p0 = p_hold_at(gate0, q, index, want)
        p1 = p_hold_at(gate, q, index, want)
        samples = [{"offset": off, "job_rank": r,
                    "target_hold": bool(data[str(off)]["disagree_targets"][r]),
                    "p_ck0": p0[(off, r)], "p_ck50": p1[(off, r)]}
                   for (off, r) in sorted(want) if (off, r) in p0 and (off, r) in p1]
        ok, det = movement_gate_verdict(samples)
        import torch
        with torch.no_grad():
            p_all = torch.sigmoid(gate(q).reshape(-1)).numpy()
        dg_ok, dg = degeneracy_check(p_all)
        loo = loo_direction(a.ck0, data, (q, y, index, gate0),
                            a.steps, a.lr, a.batch_size, a.seed)
        res[tag] = {"fit": fit, "move_ok": ok, "move": det,
                    "degen_ok": dg_ok, "degen": dg, "samples": samples,
                    "loo_diagnostic": loo}
        gates[tag] = gate
        print(f"[BC {tag:>5}] loss={fit['loss']:.4f} acc={fit['acc']:.4f} "
              f"n={fit['n']} 标签hold率={fit['label_hold_frac']:.3f} | "
              f"移动 {det.get('pooled_movement', float('nan')):+.4f} "
              f"pos={det.get('offsets_positive', 0)}/{det.get('offsets_valid', 0)} "
              f"| 非退化={dg_ok}(hold率 {dg.get('frac_hold', float('nan')):.3f})",
              flush=True)
        print(f"[BC {tag:>5}] 留一诊断(非判据): 移动 "
              f"{loo.get('pooled_movement', float('nan')):+.4f} "
              f"pos={loo.get('offsets_positive', 0)}/{loo.get('offsets_valid', 0)}",
              flush=True)

    ok, verdict = bc_probe_verdict(
        res["fc"]["move_ok"], res["fc"]["move"], res["nofc"]["move"],
        res["fc"]["degen_ok"], res["fc"]["degen"],
        res["fc"]["fit"], res["nofc"]["fit"])
    for k in ("C1_fc_direction", "C2_fc_beats_nofc", "C3_not_degenerate"):
        print(f"[BC-GATE] {k}: {'PASS' if verdict[k]['ok'] else '**FAIL**'}", flush=True)
    print(f"[BC-GATE] ALL: {'PASS -> 可进 PPO+常驻锚' if ok else 'FAIL -> 断点在表示层/gate 架构'}",
          flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(
        {"ck0": a.ck0, "steps": a.steps, "lr": a.lr,
         "batch_size": a.batch_size, "seed": a.seed,
         "fc_corpus_sha256": corpus_sha(a.fc_corpus),
         "nofc_corpus_sha256": corpus_sha(a.nofc_corpus),
         "verdict": verdict, "arms": res}, indent=1))
    print("BC PROBE DONE", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
