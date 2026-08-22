#!/usr/bin/env python3
"""逐状态 DP 盲(Codex P0-3 的"最强因果盲"):(相位, 窗龄, 剩余预算) 上的最优停时。

全部表 2020 拟合冻结:
  cost20(phase, age)  = 该状态下立即释放的经验期望棕电占比
                        (对 2020 每一行直接算 [r, r+rt) 的实现棕占比再按状态分桶)
  h_on(a), h_off(o)   = 窗/隙在龄 a 结束的经验危险率
  V[b][s]             = min( cost20(s), E_{s'|s} V[b-1][s'] )   b = 剩余预算行数

评测(2021)时逐行因果走:状态由已实现的 2021 序列给出(τ<=t),
释放判据 cost20(s) <= E[V[b-1][s']],b=0 强制释放(latest-start 兜底)。
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage15_continuous import ROW_S, binarize

MAX_AGE = 72          # 龄上限(行);更老的并入末桶


def row_states(on):
    """每行 (phase, age_rows),age 封顶 MAX_AGE。"""
    phase = on.astype(np.int8)
    age = np.zeros(len(on), dtype=np.int32)
    for i in range(1, len(on)):
        age[i] = age[i - 1] + 1 if on[i] == on[i - 1] else 0
    return phase, np.minimum(age, MAX_AGE)


def fit_tables(watts, thr, rt):
    """2020 冻结表:cost20[phase][age] 与危险率 h[phase][age]。"""
    on = watts > thr
    n = len(on)
    rt_rows = max(1, int(round(rt / ROW_S)))
    cum = np.concatenate(([0], np.cumsum(on.view(np.int8))))
    cost = np.empty(n)
    for r in range(n):
        b = min(n, r + rt_rows)
        green = cum[b] - cum[r]
        cost[r] = 1.0 - green / rt_rows
    phase, age = row_states(on)
    c_tab = np.zeros((2, MAX_AGE + 1))
    h_tab = np.zeros((2, MAX_AGE + 1))
    for p in (0, 1):
        for a in range(MAX_AGE + 1):
            m = (phase == p) & (age == a)
            m = m[:-1]                       # 需要看下一行
            if m.sum() == 0:
                c_tab[p, a] = c_tab[p, a - 1] if a else (0.0 if p else 1.0)
                h_tab[p, a] = h_tab[p, a - 1] if a else 0.5
                continue
            c_tab[p, a] = cost[:-1][m].mean()
            nxt_flip = phase[1:][m] != p
            h_tab[p, a] = nxt_flip.mean()
    return c_tab, h_tab


def solve_dp(c_tab, h_tab, b_rows):
    """V[b, p, a];返回 V 与释放策略 rel[b, p, a](True=现在放)。"""
    V = np.empty((b_rows + 1, 2, MAX_AGE + 1))
    rel = np.zeros((b_rows + 1, 2, MAX_AGE + 1), dtype=bool)
    V[0] = c_tab
    rel[0] = True
    for b in range(1, b_rows + 1):
        for p in (0, 1):
            a = np.arange(MAX_AGE + 1)
            a_next = np.minimum(a + 1, MAX_AGE)
            flip = h_tab[p, a]
            cont = flip * V[b - 1, 1 - p, 0] + (1 - flip) * V[b - 1, p, a_next]
            stay = c_tab[p, a]
            rel[b, p] = stay <= cont
            V[b, p] = np.minimum(stay, cont)
    return V, rel


def dp_blind_releases(jobs, watts_eval, thr, rel_policy):
    """2021 因果执行:状态取自已实现序列,查冻结策略表。"""
    on = watts_eval > thr
    phase, age = row_states(on)
    out = []
    for mi, pes, rt, arr, slack, _ in jobs:
        r = int(arr // ROW_S)
        b_rows = int(slack // ROW_S)
        released = None
        for k in range(b_rows + 1):
            rr = min(r + k, len(on) - 1)
            b_left = b_rows - k
            if rel_policy[b_left, phase[rr], age[rr]]:
                released = max(arr, rr * ROW_S)
                break
        out.append(released if released is not None
                   else arr + slack)
    return out


# ---------------- P0-2: 连续功率 DP (Codex 2026-08-22) ----------------
POWER_BINS = 6      # G/p_job 的桶: 0, (0,.5), [.5,1), [1,2), [2,4), >=4


def power_bin(g_over_p):
    if g_over_p <= 0: return 0
    if g_over_p < 0.5: return 1
    if g_over_p < 1.0: return 2
    if g_over_p < 2.0: return 3
    if g_over_p < 4.0: return 4
    return 5


def fit_tables_cont(watts, scale, p_job, thr, rt):
    """连续功率 DP 的 2020 冻结表。

    状态 s = (power_bin, phase, age)。立即成本 = 该状态下起跑 [r, r+rt) 的
    【连续瓦特】期望棕电占比 sum(max(0, p - G))/(p*rt)(单作业口径 —— 块状
    区域作业几乎不重叠,自拥挤可忽略,与 headline 计分一致)。
    转移 = 2020 相邻行的经验频率。
    """
    G = watts * scale
    on = watts > thr
    n = len(on)
    rt_rows = max(1, int(round(rt / ROW_S)))
    covered = np.minimum(G, p_job)
    cum = np.concatenate(([0.0], np.cumsum(covered)))
    cost = np.empty(n)
    for r in range(n):
        b = min(n, r + rt_rows)
        cost[r] = 1.0 - (cum[b] - cum[r]) / (p_job * rt_rows)
    phase, age = row_states(on)
    pb = np.array([power_bin(g / p_job) for g in G], dtype=np.int32)
    S = POWER_BINS * 2 * (MAX_AGE + 1)
    def sid(i):
        return (pb[i] * 2 + phase[i]) * (MAX_AGE + 1) + age[i]
    c_tab = np.full(S, np.nan)
    cnt = np.zeros(S)
    csum = np.zeros(S)
    T = {}
    for i in range(n - 1):
        s = sid(i)
        csum[s] += cost[i]; cnt[s] += 1
        T.setdefault(s, {}).setdefault(sid(i + 1), 0)
        T[s][sid(i + 1)] += 1
    seen = cnt > 0
    c_tab[seen] = csum[seen] / cnt[seen]
    c_tab[~seen] = np.nanmean(c_tab)          # 未见状态回退全局均值
    P = {s: {t: c / sum(d.values()) for t, c in d.items()}
         for s, d in T.items()}
    return c_tab, P, (pb, phase, age)


def solve_dp_cont(c_tab, P, b_rows):
    S = len(c_tab)
    V = np.empty((b_rows + 1, S))
    rel = np.zeros((b_rows + 1, S), dtype=bool)
    V[0] = c_tab; rel[0] = True
    for b in range(1, b_rows + 1):
        cont = c_tab.copy()                   # 无转移数据的状态视为吸收
        for s, dist in P.items():
            cont[s] = sum(p * V[b - 1][t] for t, p in dist.items())
        rel[b] = c_tab <= cont
        V[b] = np.minimum(c_tab, cont)
    return V, rel


def dp_cont_releases(jobs, watts_eval, scale, p_job, thr, rel_policy):
    G = watts_eval * scale
    on = watts_eval > thr
    phase, age = row_states(on)
    pb = np.array([power_bin(g / p_job) for g in G], dtype=np.int32)
    def sid(i):
        return (pb[i] * 2 + phase[i]) * (MAX_AGE + 1) + age[i]
    out = []
    for mi, pes, rt, arr, slack, _ in jobs:
        r = int(arr // ROW_S)
        b_rows = int(slack // ROW_S)
        released = None
        for k in range(b_rows + 1):
            rr = min(r + k, len(on) - 1)
            if rel_policy[b_rows - k, sid(rr)]:
                released = max(arr, rr * ROW_S)
                break
        out.append(released if released is not None else arr + slack)
    return out
