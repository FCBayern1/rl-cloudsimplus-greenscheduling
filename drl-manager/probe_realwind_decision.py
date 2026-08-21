#!/usr/bin/env python3
"""同一个决策价值探针,跑在真实 SDWPF 风电上 —— 第 11 考场的可行性前测。

合成绿电的 ON~U[1500,2700] 变异系数只有 0.16(近乎确定),真实风电是 2.4
(重尾)。假设:重尾会把盲估计 p_hat 从边界推向内部,从而让"信息改变动作"。

盲臂用【经验危险率】而非注册闭式律 —— 真实风电没有注册律,盲只能从历史
估计。估计器本身必须是因果的(只用 tau<=t 的历史)。
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from probe_decision_value import entropy                     # noqa: E402
from sqt2_prescreen import HORIZON_S, MARGIN_S, MIPS         # noqa: E402
from teacher_reward_audit import effective_budget            # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
_W = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
ROW_S = 600                      # SDWPF 是 10 分钟一行


def on_flags(turbine, year=2021, q=30):
    rows = list(csv.DictReader(open(_W / f"Turbine_{turbine}_{year}.csv")))
    p = np.array([float(r["power_kw"]) for r in rows])
    thr = np.percentile(p[p > 0], q)
    return (p > thr)


def segments(on):
    d = np.diff(np.concatenate(([0], on.view(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e


def build_index(on):
    """秒级 (rem_on, on_age, next_off_dur) 查询,不展开成 31M 数组。"""
    s, e = segments(on)
    return s * ROW_S, e * ROW_S            # ON 区间边界（秒）


def query(starts, ends, t):
    i = np.searchsorted(ends, t, "right")
    if i < len(starts) and starts[i] <= t < ends[i]:
        rem, age = ends[i] - t, t - starts[i]
        nxt_off = (starts[i + 1] - ends[i]) if i + 1 < len(starts) else np.inf
        nxt_len = (ends[i + 1] - starts[i + 1]) if i + 1 < len(starts) else 0
        return True, float(rem), float(age), float(nxt_off), float(nxt_len)
    return False, 0.0, 0.0, 0.0, 0.0


def empirical_p(durs, age, need):
    """P(当前 ON 段在 need 秒内结束 | 已持续 age) —— 经验生存函数,无参数假设。"""
    alive = durs[durs > age]
    if alive.size == 0:
        return 1.0
    return float((alive <= age + need).mean())


def run(turbine, trace_csv, n_offsets=10, seed=20260821):
    on = on_flags(turbine)
    starts, ends = build_index(on)
    durs = (ends - starts).astype(float)
    rows = list(csv.DictReader(open(trace_csv)))
    rng = np.random.default_rng(seed)
    span = ends[-1] - HORIZON_S
    offs = [int(x) for x in rng.integers(0, span, n_offsets)]
    out = []
    for off in offs:
        for r in rows:
            arr = float(r["arrival_time"])
            if arr >= HORIZON_S:
                continue
            mi = float(r["length"]); pes = max(1, int(r["pes_required"]))
            rt = max(1.0, mi / (pes * MIPS))
            in_on, rem, age, nxt_off, nxt_len = query(starts, ends, off + arr)
            if not in_on:
                continue
            budget = effective_budget(float(r["deadline"]) - arr, rt,
                                      MARGIN_S, HORIZON_S - arr)
            Y = bool(rem < rt and (rem + nxt_off) <= budget and nxt_len >= rt)
            p_spill = empirical_p(durs, age, rt)
            offd = np.diff(np.concatenate(
                ([0.0], (starts[1:] - ends[:-1]).astype(float))))
            gaps = (starts[1:] - ends[:-1]).astype(float)
            p_reach = float((gaps <= max(0.0, budget)).mean())
            out.append((p_spill * p_reach, Y, mi))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbines", default="100,101,124")
    ap.add_argument("--trace", default=str(
        _REPO / "cloudsimplus-gateway/src/main/resources/traces/gwo1_n1200_x130.csv"))
    a = ap.parse_args()
    print(f"{'涡轮':>6}{'n':>7}{'P(Y)':>8}{'p̂内部':>8}{'H(Y)':>8}"
          f"{'H(Y|p̂)':>9}{'解释掉':>8}{'最小分歧MI':>11}")
    for t in [int(x) for x in a.turbines.split(",")]:
        d = run(t, a.trace)
        if not d:
            print(f"{t:>6}  (无绿窗决策)"); continue
        ph = np.array([x[0] for x in d]); Y = np.array([x[1] for x in d], bool)
        mi = np.array([x[2] for x in d])
        interior = ((ph > 0.05) & (ph < 0.95)).mean()
        H = entropy(Y.mean())
        bins = np.digitize(ph, [.05, .2, .4, .6, .8, .95])
        Hc = sum((bins == b).mean() * entropy(Y[bins == b].mean())
                 for b in np.unique(bins) if (bins == b).sum())
        best = min(100 * mi[(ph > ps) != Y].sum() / mi.sum()
                   for ps in np.linspace(0, 1, 21))
        print(f"{t:>6}{len(d):>7}{Y.mean():>8.4f}{100*interior:>7.1f}%"
              f"{H:>8.4f}{Hc:>9.4f}{100*(H-Hc)/max(H,1e-9):>7.1f}%{best:>10.1f}%")
