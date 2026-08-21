#!/usr/bin/env python3
"""门 4：EVPI —— 完美未来信息在真实损失函数下值多少碳。

EVPI = J(最强因果盲) − J(clairvoyant)，分母用 nowait 的【整个 episode 总碳】，
这样它和判决线 −5% 同底可比（不是绿窗决策集的份额 —— 那个口径会把上界
放大约 1.5 倍，2026-08-20 已踩过一次）。

最强盲的构造：把盲可见状态离散化成桶，每个桶里选【实测平均损失更小】的
动作。这是在同一批数据上拟合的经验 Bayes 最优 —— 它比任何可实现的盲策略
都强（in-sample 优势），所以算出的 EVPI 是【下界】。方向是安全的：下界都
过线，真实 EVPI 只会更大。

损失表（M1 的全部意义在 penalty 那一项）：
    run now   : 溢出的那段烧棕电
    wait ok   : 全绿
    wait fail : regime A -> backstop 强制执行，全棕（等待免费，p*=0）
                regime B -> 作业过期，L_expire = k x C_brown（p* 进入内部）
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from oracle_slack_planner import WARMUP_ROWS                       # noqa: E402
from probe_realwind_decision import build_index, on_flags, query   # noqa: E402
from sqt2_prescreen import HORIZON_S, MARGIN_S, MIPS, TroughIndex  # noqa: E402
from teacher_reward_audit import effective_budget, episode_offset  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
C_BROWN, C_GREEN = 0.55, 0.01
ANCHORS = (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)


def costs(mi, runtime, rem_green, nxt_off, nxt_len, budget, k_expire):
    """(run_now, wait) 两个动作的碳。"""
    brown_frac = max(0.0, runtime - rem_green) / runtime
    run = mi * (brown_frac * C_BROWN + (1 - brown_frac) * C_GREEN)
    if rem_green + nxt_off <= budget and nxt_len >= runtime:
        wait = mi * C_GREEN
    else:
        wait = mi * C_BROWN * k_expire     # k=1 -> backstop; k>1 -> M1 过期
    return run, wait


def blind_bins(green_age, budget, runtime):
    """盲可见状态的离散化。只用 tau<=t 的量 —— 没有 rem_green。"""
    return (int(np.digitize(green_age, [600, 1800, 3600, 7200])),
            int(np.digitize(budget, [600, 1800, 3600, 7200])),
            int(np.digitize(runtime, [300, 900, 1800, 3600])))


def evaluate(decisions, k_expire):
    """decisions: (mi, runtime, rem_green, nxt_off, nxt_len, budget, blind_key)"""
    run = np.array([costs(*d[:6], k_expire)[0] for d in decisions])
    wait = np.array([costs(*d[:6], k_expire)[1] for d in decisions])
    keys = [d[6] for d in decisions]

    j_clair = np.minimum(run, wait).sum()
    j_nowait = run.sum()
    j_always = wait.sum()
    # 最强盲：逐桶取实测均值更小的动作（in-sample 最优 -> EVPI 下界）
    idx = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, []).append(i)
    j_blind = 0.0
    waits = 0
    for k, ii in idx.items():
        ii = np.array(ii)
        if wait[ii].mean() < run[ii].mean():
            j_blind += wait[ii].sum(); waits += len(ii)
        else:
            j_blind += run[ii].sum()
    return dict(j_clair=j_clair, j_blind=j_blind, j_nowait=j_nowait,
                j_always=j_always, n=len(decisions), n_bins=len(idx),
                blind_wait_frac=waits / len(decisions))


def collect_synth(trace_csv, schedule_art="calib/gwo1_schedule.json"):
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / schedule_art).read_text())
    ti = TroughIndex(art["troughs"], horizon=art["rows"])
    rows = list(csv.DictReader(open(trace_csv)))
    dec, total_mi, nowait_brown = [], 0.0, 0.0
    for k in ANCHORS:
        off = episode_offset(k, 180000)
        for r in rows:
            arr = float(r["arrival_time"])
            if arr >= HORIZON_S:
                continue
            mi = float(r["length"]); pes = max(1, int(r["pes_required"]))
            rt = max(1.0, mi / (pes * MIPS))
            base = int(WARMUP_ROWS + off + arr)
            total_mi += mi
            nowait_brown += mi * sum(1 for d in range(int(rt))
                                     if ti.query(base + d)[0]) / max(1.0, int(rt))
            in_tr, _, _, rem, age = ti.query(base)
            if in_tr:
                continue
            bud = effective_budget(float(r["deadline"]) - arr, rt, MARGIN_S,
                                   HORIZON_S - arr)
            if bud <= 0:
                continue
            nxt = ti.next_trough_dur(base)
            if np.isfinite(nxt):
                _, _, _, nlen, _ = ti.query(int(base + rem + nxt))
            else:
                nxt, nlen = 1e18, 0.0
            dec.append((mi, rt, rem, nxt, nlen, bud, blind_bins(age, bud, rt)))
    ep_carbon = C_BROWN * nowait_brown + C_GREEN * (total_mi - nowait_brown)
    return dec, ep_carbon


def collect_real(turbine, trace_csv, rt_x, bud_x, horizon, n_off=8,
                 seed=20260821):
    s, e = build_index(on_flags(turbine))
    rows = list(csv.DictReader(open(trace_csv)))
    rng = np.random.default_rng(seed)
    offs = rng.integers(0, int(e[-1] - horizon), n_off)
    dec, total_mi, nowait_brown = [], 0.0, 0.0
    for off in offs:
        for r in rows:
            arr = float(r["arrival_time"]) * (horizon / 7200.0)
            if arr >= horizon:
                continue
            mi = float(r["length"]) * rt_x
            pes = max(1, int(r["pes_required"]))
            rt = max(1.0, mi / (pes * MIPS))
            slack = (float(r["deadline"]) - float(r["arrival_time"])
                     - float(r["length"]) / (pes * MIPS))
            bud = min(slack * bud_x, horizon - arr - rt - 120)
            total_mi += mi
            ok, rem, age, noff, nlen = query(s, e, off + arr)
            brown = 0.0 if (ok and rem >= rt) else (
                rt if not ok else min(rt, rt - rem))
            nowait_brown += mi * brown / rt
            if not ok or bud <= 0:
                continue
            dec.append((mi, rt, rem, noff, nlen, bud,
                        blind_bins(age, bud, rt)))
    ep_carbon = C_BROWN * nowait_brown + C_GREEN * (total_mi - nowait_brown)
    return dec, ep_carbon


def report(label, dec, ep_carbon):
    print(f"\n{'='*72}\n{label}   n={len(dec)}  episode总碳(nowait)={ep_carbon:.4e}")
    print(f"{'L_expire':>12}{'p*':>8}{'盲等待率':>10}{'EVPI(碳)':>11}"
          f"{'/总碳':>9}{'过-5%?':>8}")
    for k in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        r = evaluate(dec, k)
        ps = (k * C_BROWN - C_BROWN) / (k * C_BROWN - C_GREEN)
        evpi = r["j_blind"] - r["j_clair"]
        rel = -evpi / ep_carbon
        print(f"{k:>10.1f}x{ps:>8.3f}{100*r['blind_wait_frac']:>9.1f}%"
              f"{evpi:>11.3e}{100*rel:>8.2f}%{'✅' if rel <= -0.05 else '❌':>7}")
    r = evaluate(dec, 1.0)
    print(f"  桶数={r['n_bins']}  (最强盲在同一批数据上拟合 -> EVPI 是下界)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-only", action="store_true")
    a = ap.parse_args()
    T = _REPO / "cloudsimplus-gateway/src/main/resources/traces"
    if not a.real_only:
        d, c = collect_synth(str(T / "gwo1_n1200_x130.csv"))
        report("合成 gwo1 (x130, horizon 2h)", d, c)
    d, c = collect_real(100, str(T / "gwo1_n1200_x130.csv"), 8, 8, 57600)
    report("真实 SDWPF T100 (rt x8, bud x8, horizon 16h)", d, c)
