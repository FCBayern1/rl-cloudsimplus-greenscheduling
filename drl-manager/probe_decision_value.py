#!/usr/bin/env python3
"""决策价值探针：p 的分布、H(Y|盲状态)、动作分歧 —— 第 11 考场的前三道离线门。

诊断(5080 2026-08-21)说前十个考场断在"信息不改变动作"。这个探针把它
从推论变成实测:对每一个绿窗决策,同时算

  Y     真实结果  —— 等待到底值不值(用实现的绿电序列,clairvoyant 视角)
  p_hat 盲估计    —— 只用 tau<=t 的可见量 + 注册的 ON/OFF 律

然后回答三个问题:
  1. p_hat 的分布压在边界上,还是跨过决策阈 p*?
  2. 给定 p_hat,Y 还剩多少不确定性?(H(Y|p_hat) —— 条件信息门)
  3. 在哪些 p* 上,最强盲与 clairvoyant 的动作会分歧?(动作分歧门)

盲状态【不含】rem_green —— 那是未来信息。盲只知道 green_age(当前绿窗
已经持续多久),这正是本问题的信息不对称所在。
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle_slack_planner import WARMUP_ROWS                      # noqa: E402
from sqt2_prescreen import (HORIZON_S, MARGIN_S, MIPS, ON_HI, ON_LO,  # noqa: E402
                            TroughIndex, green_p_ends_within)
from teacher_reward_audit import effective_budget, episode_offset  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
ANCHORS = (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)


def p_star(c_brown, c_green, l_expire):
    """盲策略的等待阈:p > p* 才等。

    p*(L=C_brown) = 0  -> 永远等占优(前十考场:backstop 兜底,等错只多烧棕电)
    p*(L->inf)    -> 1  -> 永不等占优(硬完成合同,丢作业不可接受)
    只有 C_brown < L < inf 时 p* 才在 (0,1) 内部。
    """
    return (l_expire - c_brown) / (l_expire - c_green)


def collect(schedule_art, trace_csv, offset_range=180000):
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / schedule_art).read_text())
    ti = TroughIndex(art["troughs"], horizon=art["rows"])
    rows = list(csv.DictReader(open(trace_csv)))
    out = []
    for k in ANCHORS:
        off = episode_offset(k, offset_range)
        for r in rows:
            arr = float(r["arrival_time"])
            if arr >= HORIZON_S:
                continue
            mi = float(r["length"])
            pes = max(1, int(r["pes_required"]))
            rt = max(1.0, mi / (pes * MIPS))
            base = int(WARMUP_ROWS + off + arr)
            in_tr, _, _, rem_green, green_age = ti.query(base)
            if in_tr:
                continue                       # 只看绿窗域
            budget = effective_budget(float(r["deadline"]) - arr, rt,
                                      MARGIN_S, HORIZON_S - arr)
            nxt = ti.next_trough_dur(base)
            # --- 真实结果 Y:等到下一个绿窗起点,是否既等得起、又跑得完 ---
            spills_now = rem_green < rt
            reachable = (rem_green + nxt) <= budget
            if np.isfinite(nxt):
                nxt_start = base + rem_green + nxt
                _, _, _, nxt_rem, _ = ti.query(int(nxt_start))
                fits_next = nxt_rem >= rt
            else:
                fits_next = False
            Y = bool(spills_now and reachable and fits_next)
            # --- 盲估计:只用 green_age / budget / runtime + 注册律 ---
            #  P(现在跑会溢出) x P(等得起下一个绿窗)
            p_spill = green_p_ends_within(green_age, rt)
            #  等得起的概率:剩余绿电 + 槽长 <= budget。盲只知道律。
            #  E[剩余绿电|age] 与槽长分布 -> 用注册律的保守闭式近似
            lo = max(green_age, ON_LO)
            exp_rem = 0.0 if green_age >= ON_HI else (ON_HI - lo) / 2.0
            p_reach = 1.0 if budget >= exp_rem + 4500 else (
                0.0 if budget <= exp_rem + 300 else
                (budget - exp_rem - 300) / (4500 - 300))
            out.append((p_spill * p_reach, Y, mi, budget, rt, green_age))
    return out


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def report(data, label):
    ph = np.array([d[0] for d in data])
    Y = np.array([d[1] for d in data], dtype=bool)
    mi = np.array([d[2] for d in data], dtype=float)
    print(f"\n{'='*66}\n{label}   n={len(data)}  MI={mi.sum():.3e}")
    print(f"  真实 P(等待值得) = {Y.mean():.4f}   (MI 加权 {mi[Y].sum()/mi.sum():.4f})")

    print("\n  p_hat 分布:")
    edges = [0, .05, .2, .4, .6, .8, .95, 1.001]
    for a, b in zip(edges, edges[1:]):
        m = (ph >= a) & (ph < b)
        if m.sum():
            print(f"    [{a:.2f},{b:.2f})  n={m.sum():>5} ({100*m.mean():>5.1f}%)"
                  f"  实测 P(Y)={Y[m].mean():.3f}")
    interior = ((ph > 0.05) & (ph < 0.95)).mean()
    print(f"  -> p_hat 落在 (0.05,0.95) 内部的比例: {100*interior:.1f}%")

    H_Y = entropy(Y.mean())
    bins = np.digitize(ph, [.05, .2, .4, .6, .8, .95])
    H_cond = sum((bins == b).mean() * entropy(Y[bins == b].mean())
                 for b in np.unique(bins) if (bins == b).sum())
    print(f"\n  H(Y)          = {H_Y:.4f} bit")
    print(f"  H(Y | p_hat)  = {H_cond:.4f} bit   (条件信息门)")
    print(f"  I(Y ; p_hat)  = {H_Y - H_cond:.4f} bit"
          f"   —— 盲状态解释掉了 {100*(H_Y-H_cond)/max(H_Y,1e-9):.1f}%")

    print(f"\n  动作分歧（最强盲用阈 p*，clairvoyant 用真值 Y）:")
    print(f"    {'p*':>6}{'盲的动作':>10}{'分歧作业':>10}{'分歧MI':>10}")
    for ps in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        blind = ph > ps
        dis = blind != Y
        act = "全等" if blind.all() else ("全不等" if not blind.any() else "混合")
        print(f"    {ps:>6.1f}{act:>10}{100*dis.mean():>9.1f}%"
              f"{100*mi[dis].sum()/mi.sum():>9.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="calib/gwo1_schedule.json")
    ap.add_argument("--trace", default=str(
        _REPO / "cloudsimplus-gateway/src/main/resources/traces/gwo1_n1200_x130.csv"))
    a = ap.parse_args()
    print("p* = (L_expire - C_brown) / (L_expire - C_green)")
    for L, tag in ((0.55, "L=C_brown（前十考场：backstop 兜底）"),
                   (1.1, "L=2xC_brown"), (5.5, "L=10xC_brown"),
                   (1e9, "L->inf（硬完成合同）")):
        print(f"  {tag:<34} p* = {p_star(0.55, 0.01, L):.4f}")
    report(collect(a.schedule, a.trace), f"gwo1 绿窗域  {pathlib.Path(a.trace).name}")
