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
