#!/usr/bin/env python3
"""TB12 逐作业方向门(Codex 新执行顺序 Step 3 附加门,2026-08-26)。

不用随机采样动作代替概率分析:在**同一份冻结 observation corpus**上读取
raw p_hold(defer 选项的 softmax 概率),按冻结 teacher 的 worthy/not-worth
标签检验方向分离。

corpus 构建(--build):每偏移用 teacher(greenfollow 冻结释放计划)驱动
episode,记录整段 obs 序列 + 每个作业的首次 eligible decision(步,槽)+
worthy 标签(teacher 释放时刻 > 到达 + 半步 ⇒ 值得等)。npz 落盘并记 sha256。

判定(Codex ② 重做,2026-08-26):greenfollow 标签只有 3 个 offset 同时含
两类(≥4/6 数学不可达)且不能证明预测载重 —— 改为:
  - corpus 仍 = greenfollow 固定驱动;
  - 只取 **clair 与 greenfollow 分歧** 的作业,以 clair 为目标方向;
  - 判定量 = fc 从 ck0→ck50 的 signed p_hold 移动
    (target=hold ⇒ +Δp;target=route ⇒ −Δp);
  - 池化 mean movement ≥ +0.05;
  - 有效 offset(含 ≥1 分歧作业,预期 5 个)中 ≥4 个方向为正;
    有效数 <4 ⇒ undefined FAIL;
  - 每作业只取首次 eligible decision;附 p_hold 分位数与 logit margin。
训练侧诊断(采样 defer 率、defer/route advantage、TD residual)由
report_training_diagnostics 从 progress.csv 读取。
"""
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tb12_run import FrozenPolicies, load_scaled, ROW_S  # noqa: E402

CALIB_OFFSETS = [4000, 12000, 20000, 28000, 36000, 44000]
GAP_MIN = 0.05
OFFSETS_POSITIVE_MIN = 4


# ---------------------------------------------------------------- 纯函数
def worthy_labels(releases, arrivals, half_step=ROW_S / 2):
    """teacher 释放晚于到达半步以上 ⇒ 该作业值得等待。按到达序返回。"""
    order = np.argsort(arrivals)
    return {int(j): bool(releases[j] > arrivals[j] + half_step) for j in order}


def first_eligible_map(arrivals, max_steps=300, row_s=ROW_S):
    """作业(按到达序编号 j)的首次 eligible decision:(step, slot)。
    槽位语义与 run 循环一致:pending 队列按到达序占据有效槽 0..k。"""
    order = np.argsort(arrivals)
    out = {}
    for rank, j in enumerate(order):
        step = int(np.ceil(arrivals[j] / row_s))
        # 该步时,先到且未释放的作业占前排槽;保守取"仍 pending 的更早作业数"
        # 上界 = rank(teacher corpus 中,首现步槽位由 corpus 构建时实测覆盖)。
        out[int(j)] = (min(step, max_steps - 1), rank)
    return out


def pooled_auc(p_worthy, p_not):
    """Mann-Whitney AUC:P(p_hold(worthy) > p_hold(not_worth)),平局计 0.5。"""
    if not p_worthy or not p_not:
        return float("nan")
    wins = ties = 0
    for a in p_worthy:
        for b in p_not:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(p_worthy) * len(p_not))


def movement_gate_verdict(samples, gap_min=GAP_MIN,
                          offsets_positive_min=OFFSETS_POSITIVE_MIN):
    """samples: [{offset, job_rank, target_hold, p_ck0, p_ck50}](仅分歧作业)。
    movement = (p_ck50 − p_ck0) × (+1 if target_hold else −1)。机械判定。"""
    if not samples:
        return False, {"undefined": "no_disagreement_samples"}
    movs = [(s_["p_ck50"] - s_["p_ck0"]) * (1.0 if s_["target_hold"] else -1.0)
            for s_ in samples]
    pooled = float(np.mean(movs))
    per_off = {}
    for s_, m in zip(samples, movs):
        per_off.setdefault(s_["offset"], []).append(m)
    signs = {off: float(np.mean(ms)) > 0 for off, ms in per_off.items()}
    n_valid = len(signs)
    n_pos = sum(signs.values())
    ok = pooled >= gap_min and n_valid >= offsets_positive_min \
        and n_pos >= offsets_positive_min
    return ok, {"pooled_movement": pooled, "gap_min": gap_min,
                "offsets_valid": n_valid, "offsets_positive": n_pos,
                "offsets_positive_min": offsets_positive_min,
                "per_offset_mean_movement": {str(k): float(np.mean(v))
                                             for k, v in per_off.items()},
                "n_samples": len(samples)}


def direction_gate_verdict(samples):
    """samples: [{offset, job, worthy, p_hold, logit_margin}]。机械判定。"""
    pw = [s["p_hold"] for s in samples if s["worthy"]]
    pn = [s["p_hold"] for s in samples if not s["worthy"]]
    both = bool(pw) and bool(pn)
    gap = (float(np.mean(pw)) - float(np.mean(pn))) if both else float("nan")
    per_off = {}
    for s in samples:
        per_off.setdefault(s["offset"], {"w": [], "n": []})[
            "w" if s["worthy"] else "n"].append(s["p_hold"])
    signs = {}
    for off, d in per_off.items():
        signs[off] = (float(np.mean(d["w"])) - float(np.mean(d["n"]))) > 0 \
            if d["w"] and d["n"] else None
    n_pos = sum(1 for v in signs.values() if v is True)
    n_def = sum(1 for v in signs.values() if v is not None)
    ok = both and gap >= GAP_MIN and n_pos >= OFFSETS_POSITIVE_MIN
    qs = {f"p{q}": float(np.percentile([s["p_hold"] for s in samples], q))
          for q in (10, 50, 90)} if samples else {}
    return ok, {"both_classes_nonempty": both, "mean_gap": gap,
                "gap_min": GAP_MIN, "offsets_positive": n_pos,
                "offsets_defined": n_def,
                "offsets_positive_min": OFFSETS_POSITIVE_MIN,
                "per_offset_sign": {str(k): v for k, v in signs.items()},
                "pooled_auc_diagnostic": pooled_auc(pw, pn),
                "n_worthy": len(pw), "n_not_worth": len(pn),
                "p_hold_quantiles": qs}


def corpus_sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- corpus 构建
def build_corpus_episode(cfg0, off, releases, arrivals, ref_series):
    """teacher 驱动一集,记录 obs 序列 + 实测首现 (step, slot)。"""
    import copy
    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from oracle_slack_planner import drain_action
    cfg = copy.deepcopy(cfg0)
    cfg["green_episode_offset_range"] = 0
    for dc in cfg["datacenters"]:
        dc["time_zone_offset_rows"] = int(off)
    env = HierarchicalMultiDCEnv(cfg)
    try:
        obs, _ = env.reset(seed=1)
        g0 = float(np.asarray(obs["global"]["dc_current_green_power_w"]).reshape(-1)[0])
        if abs(g0 - float(ref_series[int(off)])) > 1e-3:
            sys.exit(f"接线哨兵失败 off={off}: {g0} != {ref_series[int(off)]}")
        batch = env.global_routing_batch_size
        order = np.argsort(arrivals)
        rel_sorted = [releases[i] for i in order]
        released, done, t = 0, False, 0
        obs_seq, first_seen = [], {}
        while not done and t < 300:
            g = obs["global"]
            obs_seq.append({k: np.asarray(v).copy() for k, v in g.items()})
            mi = np.asarray(g["batch_cloudlet_mi"]).reshape(-1)[:batch]
            acts, k = [], 0
            for i in range(batch):
                if mi[i] <= 0:
                    acts.append(0)
                    continue
                j = released + k                    # 到达序第 j 个作业占此槽
                if j not in first_seen:
                    first_seen[j] = (t, i)          # 实测首次 eligible (步,槽)
                # 与校准 runner 同一量化语义(Codex ④):窗内释放即路由
                hold = (j < len(rel_sorted)
                        and rel_sorted[j] >= t * ROW_S + ROW_S - 1e-9)
                acts.append(1 if hold else 0)
                k += 1
            released += sum(1 for i in range(batch)
                            if mi[i] > 0 and acts[i] == 0)
            obs, _, term, trunc, info = env.step(
                {"global": acts, "local": {0: drain_action(env.get_local_action_masks(0))}})
            done = term or trunc
            t += 1
        return obs_seq, first_seen
    finally:
        env.close()


def build_corpus(experiment, out_path):
    from src.baselines.evaluate import load_config
    cfg0 = load_config(experiment)
    cfg0.pop("py4j_port", None)
    cfg0.setdefault("gateway_log_dir", "/tmp/tb12_gw")
    cfg0.setdefault("output_dir", "/tmp/tb12_gw")
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())
    arrivals = np.array([j[1] for j in art["jobs"]], dtype=float)
    pol = FrozenPolicies(art["rt_h"] * 3600.0, art["slack_h"] * 3600.0)
    ref = load_scaled((100, 101), 2021)
    data = {}
    for off in CALIB_OFFSETS:
        w_ep = ref[off:off + 300]
        rel = pol.releases("greenfollow", w_ep, arrivals)      # 冻结 teacher
        rel_clair = pol.releases("clair", w_ep, arrivals)      # 目标策略
        order = np.argsort(arrivals)
        gf = worthy_labels(rel, arrivals)
        cl = worthy_labels(rel_clair, arrivals)
        # rank(到达序) -> 原作业号,与 run 循环的 j 语义一致
        rank_worthy = {rank: gf[int(j)] for rank, j in enumerate(order)}
        rank_clair = {rank: cl[int(j)] for rank, j in enumerate(order)}
        disagree = {rank: rank_clair[rank] for rank in rank_worthy
                    if rank_worthy[rank] != rank_clair[rank]}   # rank -> target_hold
        obs_seq, first_seen = build_corpus_episode(cfg0, off, rel, arrivals, ref)
        data[str(off)] = {"obs_seq": obs_seq, "first_seen": first_seen,
                          "worthy_by_rank": rank_worthy,
                          "clair_by_rank": rank_clair,
                          "disagree_targets": disagree,
                          "teacher_releases": [float(r) for r in rel],
                          "clair_releases": [float(r) for r in rel_clair]}
        print(f"[CORPUS off={off:>6}] steps={len(obs_seq)} jobs={len(first_seen)} "
              f"gf_worthy={sum(rank_worthy.values())}/5 分歧={len(disagree)} "
              f"targets={disagree}", flush=True)
    np.savez_compressed(out_path, corpus=np.array([data], dtype=object),
                        experiment=experiment)
    print(f"CORPUS SAVED {out_path} sha256={corpus_sha(out_path)[:16]}", flush=True)


# ---------------------------------------------------------------- p_hold 判定
def read_p_hold(ckpt, data):
    """在冻结 corpus 上逐步推进 GTrXL 状态,返回 {(offset,rank): (p_hold, margin)}。"""
    import torch
    from tb12_rl_eval import FullActionHead
    from ray.rllib.core.columns import Columns

    class ProbHead(FullActionHead):
        def step_probs(self, obs):
            batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)[None, ...])
                                   for k, v in obs.items()
                                   if np.asarray(v).dtype.kind in "ifub"}}
            state = self.state
            if state is None:
                init = self.module.get_initial_state()
                if init:
                    state = {k: torch.as_tensor(np.asarray(v))[None, ...]
                             for k, v in init.items()}
            if state:
                batch[Columns.STATE_IN] = state
            with torch.no_grad():
                out = self.module.forward_inference(batch)
            self.state = out.get(Columns.STATE_OUT) or None
            logits = out[Columns.ACTION_DIST_INPUTS].detach().reshape(-1)
            n_opt = logits.numel() // self.n_slots
            lg = logits.reshape(self.n_slots, n_opt)
            return torch.softmax(lg, -1).numpy(), lg.numpy()

    head = ProbHead(pathlib.Path(ckpt).resolve())
    out = {}
    for off, ep in data.items():
        head.reset()
        want = {}
        for rank, (t, slot) in ep["first_seen"].items():
            want.setdefault(int(t), []).append((int(rank), int(slot)))
        for t, obs in enumerate(ep["obs_seq"]):
            probs, logits = head.step_probs(obs)     # 状态必须逐步推进
            for rank, slot in want.get(t, []):
                n_opt = probs.shape[1]
                defer_idx = n_opt - 1                # 选项 = [dc0..dcN-1, defer]
                out[(int(off), rank)] = (
                    float(probs[slot, defer_idx]),
                    float(logits[slot, defer_idx] - logits[slot, :defer_idx].max()))
    return out


def judge_movement(ck0, ck50, corpus_path, json_out):
    """Codex ② 门:分歧作业上 fc 的 ck0→ck50 signed p_hold 移动。"""
    raw = np.load(corpus_path, allow_pickle=True)
    data = raw["corpus"][0]
    p0 = read_p_hold(ck0, data)
    p5 = read_p_hold(ck50, data)
    samples = []
    for off, ep in data.items():
        for rank, target_hold in ep["disagree_targets"].items():
            key = (int(off), int(rank))
            if key in p0 and key in p5:
                samples.append({"offset": int(off), "job_rank": int(rank),
                                "target_hold": bool(target_hold),
                                "p_ck0": p0[key][0], "p_ck50": p5[key][0],
                                "margin_ck0": p0[key][1], "margin_ck50": p5[key][1]})
    ok, det = movement_gate_verdict(samples)
    qs = {tag: {f"p{q}": float(np.percentile([s[f"p_{tag}"] for s in samples], q))
                for q in (10, 50, 90)} if samples else {}
          for tag in ("ck0", "ck50")}
    print(f"[DIRECTION-MOVE] {'PASS' if ok else '**FAIL**'} "
          f"pooled={det.get('pooled_movement', float('nan')):.4f} "
          f"pos={det.get('offsets_positive', 0)}/{det.get('offsets_valid', 0)} "
          f"n={det.get('n_samples', 0)}", flush=True)
    pathlib.Path(json_out).write_text(json.dumps(
        {"ck0": str(ck0), "ck50": str(ck50),
         "corpus_sha256": corpus_sha(corpus_path), "ok": ok, "detail": det,
         "p_hold_quantiles": qs, "samples": samples}, indent=1))
    return ok


def report_training_diagnostics(progress_csv):
    """训练侧诊断(报告项,非判定):采样 defer 率、defer advantage、TD residual。"""
    import pandas as pd
    d = pd.read_csv(progress_csv)
    out = {}
    for label, col in (("adv_defer", "learners/global_policy/v32_adv_defer"),
                       ("td_abs_defer", "learners/global_policy/v32_td_abs_defer"),
                       ("defer_count", "learners/global_policy/v32_adv_defer_count")):
        if col in d.columns:
            x = d[col].dropna()
            if len(x):
                out[label] = {"first": float(x.iloc[0]), "last": float(x.iloc[-1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--experiment", default="experiment_tb12_rl_fc_v3s50k")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ck0", default=None)
    ap.add_argument("--ck50", default=None)
    ap.add_argument("--json-out", default="direction_movement.json")
    a = ap.parse_args()
    if a.build:
        build_corpus(a.experiment, a.corpus)
    if a.ck0 and a.ck50:
        ok = judge_movement(a.ck0, a.ck50, a.corpus, a.json_out)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
