#!/usr/bin/env python3
"""相变图第一阶段：真实风电 x 等待风险 x 空间自由度 的离线扫描（Codex 2026-08-22）。

机制与 gwo1 不同 —— 这是"低绿电到达"的正控制场:
    作业随时到达;释放后不可抢占;deadline 前必须开始(latest start),
    否则过期(M1:失效 MI 计入未完成,不折碳)。
    决策 = 释放时刻。价值藏在两个盲测不准的量里:
      (a) 棕电期:绿窗多久才来(OFF 重尾, CV~2.4)
      (b) 绿窗口:这个窗装不装得下 runtime(ON 重尾,中位 30min)

三个策略,全部逐作业、无容量耦合(第一阶段口径;拥堵是仿真阶段的事):
    nowait      到达即跑
    blind       因果动态策略:每 600s 一行重估
                  ON  内: P(窗还能装下 runtime | 窗龄) >= theta -> 释放
                  OFF 内: 等;到 latest-start 兜底释放(永不过期)
                theta 在同一数据上网格搜索取最优 -> in-sample 最强
                (动态盲,严格强于此前 probe_evpi 的静态二元盲)
    clairvoyant 知道整条未来:在 {到达, 各绿窗起点, latest-start} 里挑
                棕电重叠最小的释放时刻

空间轴:k 个独立涡轮 = k 个无容量 DC。
    blind-spatial: 释放时选"当下是绿"的 DC(否则第 1 个)
    clair-spatial: 选 (DC, 时刻) 联合最优
这给出无容量情形下空间对时间价值的替代率。

输出每格:EVPI(vs 动态最强盲, 分母 = nowait 全 episode 碳)、动作分歧 MI%、
headroom(clair vs nowait)。Codex 预注册门:分歧>=10% MI, EVPI>=10%, headroom>=8%。
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_REPO = pathlib.Path(__file__).resolve().parent.parent
_W = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
ROW_S = 600
C_BROWN, C_GREEN = 0.55, 0.01
THETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.1)     # 1.1 = 从不进窗(纯 latest-start)


def load_segments(turbine, year=2021, q=30):
    rows = list(csv.DictReader(open(_W / f"Turbine_{turbine}_{year}.csv")))
    p = np.array([float(r["power_kw"]) for r in rows])
    on = p > np.percentile(p[p > 0], q)
    d = np.diff(np.concatenate(([0], on.view(np.int8), [0])))
    s = np.where(d == 1)[0] * ROW_S
    e = np.where(d == -1)[0] * ROW_S
    return s.astype(float), e.astype(float)


def green_overlap(starts, ends, a, b):
    """[a,b) 与 ON 段的重叠秒数。"""
    if b <= a:
        return 0.0
    i = max(0, np.searchsorted(ends, a, "right"))
    tot = 0.0
    while i < len(starts) and starts[i] < b:
        tot += max(0.0, min(ends[i], b) - max(starts[i], a))
        i += 1
    return tot


def state(starts, ends, t):
    """(in_on, age)。"""
    i = np.searchsorted(ends, t, "right")
    if i < len(starts) and starts[i] <= t:
        return True, t - starts[i]
    return False, 0.0


def fit_prob_table(durs, rt, max_age_rows=600):
    """F[a_row] = P(窗总长 >= age + rt | 总长 > age),经验生存函数。"""
    out = np.empty(max_age_rows)
    for j in range(max_age_rows):
        a = j * ROW_S
        alive = durs[durs > a]
        out[j] = 0.0 if alive.size == 0 else (alive >= a + rt).mean()
    return out


def clair_release(starts, ends, t0, rt, latest):
    """棕电重叠最小的释放时刻(平局取最早)。返回 (release, brown_frac)。"""
    cands = [t0, latest]
    i = np.searchsorted(starts, t0, "left")
    while i < len(starts) and starts[i] <= latest:
        cands.append(starts[i])
        i += 1
    best = (1.1, t0)
    for s in sorted(set(cands)):
        bf = 1.0 - green_overlap(starts, ends, s, s + rt) / rt
        if bf < best[0] - 1e-12:
            best = (bf, s)
    return best[1], best[0]


def blind_release(starts, ends, F, t0, rt, latest, theta):
    """因果动态盲:逐行走,ON 内 F(age)>=theta 就释放,否则 latest 兜底。"""
    t = t0
    while t < latest:
        in_on, age = state(starts, ends, t)
        if in_on and F[min(int(age // ROW_S), len(F) - 1)] >= theta:
            return t
        nxt = t - (t % ROW_S) + ROW_S           # 下一行边界
        t = min(nxt, latest)
    return latest


def scan_cell(starts, ends, durs, jobs, offs, thetas=THETAS):
    """jobs: (mi, rt, slack)。返回该格的指标字典。"""
    per_theta_blind = {th: 0.0 for th in thetas}
    j_now = j_clair = tot_mi = 0.0
    div_mi = 0.0
    F_cache = {}
    releases = []
    # 每个作业独立到达时刻:off 只定块,块内均匀铺 24h —— 否则全部作业挤在
    # 同一瞬间,slack=5h 与 10h 会给出逐位相同的退化结果(首扫实测)。
    rng = np.random.default_rng(int(offs[0]) + len(jobs))
    for off in offs:
        arr_off = rng.uniform(0, 24 * 3600.0, size=len(jobs))
        for (mi, rt, slack), da in zip(jobs, arr_off):
            t0 = float(off) + float(da)
            latest = t0 + max(0.0, slack)
            tot_mi += mi
            bf_now = 1.0 - green_overlap(starts, ends, t0, t0 + rt) / rt
            j_now += mi * (bf_now * C_BROWN + (1 - bf_now) * C_GREEN)
            rel_c, bf_c = clair_release(starts, ends, t0, rt, latest)
            j_clair += mi * (bf_c * C_BROWN + (1 - bf_c) * C_GREEN)
            key = int(rt)
            if key not in F_cache:
                F_cache[key] = fit_prob_table(durs, rt)
            F = F_cache[key]
            rels = {}
            for th in thetas:
                r = blind_release(starts, ends, F, t0, rt, latest, th)
                bf = 1.0 - green_overlap(starts, ends, r, r + rt) / rt
                per_theta_blind[th] += mi * (bf * C_BROWN + (1 - bf) * C_GREEN)
                rels[th] = r
            releases.append((mi, rel_c, rels))
    th_star = min(per_theta_blind, key=per_theta_blind.get)
    j_blind = per_theta_blind[th_star]
    for mi, rel_c, rels in releases:
        if abs(rels[th_star] - rel_c) > ROW_S:
            div_mi += mi
    return {"j_nowait": j_now, "j_blind": j_blind, "j_clair": j_clair,
            "theta_star": th_star, "n": len(releases),
            "evpi_rel": (j_clair - j_blind) / j_now,
            "headroom_rel": (j_clair - j_now) / j_now,
            "blind_vs_now": (j_blind - j_now) / j_now,
            "divergence_mi": div_mi / tot_mi}


def scan_cell_kdc(seg_list, durs_list, jobs, offs):
    """k 个独立涡轮 DC。blind-spatial 选当下是绿的 DC;clair 选 (DC,t) 联合最优。
    时间盲的 theta 沿用单 DC 情形的搜索(对每 DC 用各自 F)。"""
    j_now = j_blind = j_clair = 0.0
    F = [ {} for _ in seg_list ]
    rng = np.random.default_rng(int(offs[0]) + 7 * len(jobs))
    for off in offs:
        arr_off = rng.uniform(0, 24 * 3600.0, size=len(jobs))
        for (mi, rt, slack), da in zip(jobs, arr_off):
            t0 = float(off) + float(da); latest = t0 + max(0.0, slack)
            # nowait-spatial: 挑当下绿的 DC(等价 blind 空间启发)
            bfs = [1.0 - green_overlap(s, e, t0, t0 + rt) / rt
                   for s, e in seg_list]
            j_now += mi * (min(bfs) * C_BROWN + (1 - min(bfs)) * C_GREEN)
            # blind: 在"当下最绿"的 DC 上跑单 DC 动态盲(theta=0.5 固定,
            # 单 DC 扫描显示 0.25-0.75 差异小;固定避免 k 倍搜索成本)
            d = int(np.argmax([state(s, e, t0)[0] for s, e in seg_list])) \
                if any(state(s, e, t0)[0] for s, e in seg_list) else \
                int(np.argmin(bfs))
            s_, e_ = seg_list[d]
            key = int(rt)
            if key not in F[d]:
                F[d][key] = fit_prob_table(durs_list[d], rt)
            r = blind_release(s_, e_, F[d][key], t0, rt, latest, 0.5)
            bf = 1.0 - green_overlap(s_, e_, r, r + rt) / rt
            j_blind += mi * (bf * C_BROWN + (1 - bf) * C_GREEN)
            # clair: (DC, t) 联合
            best = 1.1
            for s2, e2 in seg_list:
                _, bfc = clair_release(s2, e2, t0, rt, latest)
                best = min(best, bfc)
            j_clair += mi * (best * C_BROWN + (1 - best) * C_GREEN)
    return {"j_nowait": j_now, "j_blind": j_blind, "j_clair": j_clair,
            "evpi_rel": (j_clair - j_blind) / j_now,
            "headroom_rel": (j_clair - j_now) / j_now}


def job_mix(trace_csv, rt_scale, n_sample, seed):
    """从 gwo1 trace 取 MI/PES 分布,slack 由扫描轴单独给。"""
    rows = list(csv.DictReader(open(trace_csv)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n_sample, len(rows)), replace=False)
    out = []
    for i in idx:
        mi = float(rows[i]["length"]) * rt_scale
        pes = max(1, int(rows[i]["pes_required"]))
        out.append((mi, max(1.0, mi / (pes * 40000.0))))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbine", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=300)
    ap.add_argument("--n-offs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()
    starts, ends = load_segments(a.turbine)
    durs = ends - starts
    trace = str(_REPO / "cloudsimplus-gateway/src/main/resources/traces"
                / "gwo1_n1200_x130.csv")
    rng = np.random.default_rng(a.seed)
    span = ends[-1] - 24 * 3600
    offs = rng.integers(0, int(span), a.n_offs)

    print("=" * 78)
    print(f"轴 1x2: 等待风险(slack) x 作业时长   涡轮 T{a.turbine} 单 DC")
    print(f"{'rt中位':>8}{'slack':>7}{'θ*':>6}{'blind vs now':>14}"
          f"{'clair vs now':>14}{'EVPI':>9}{'分歧MI':>9}  门(≥10/≥10/≥8)")
    for rt_scale, rt_tag in ((8.0, "64min"), (16.0, "2.1h"), (32.0, "4.3h")):
        base = job_mix(trace, rt_scale, a.n_jobs, a.seed)
        for slack_h in (2, 5, 10, 15):
            jobs = [(mi, rt, slack_h * 3600.0) for mi, rt in base]
            r = scan_cell(starts, ends, durs, jobs, offs)
            g = ("✅" if (r["divergence_mi"] >= 0.10
                          and -r["evpi_rel"] >= 0.10
                          and -r["headroom_rel"] >= 0.08) else "❌")
            print(f"{rt_tag:>8}{slack_h:>6}h{r['theta_star']:>6.2f}"
                  f"{100*r['blind_vs_now']:>13.2f}%{100*r['headroom_rel']:>13.2f}%"
                  f"{100*r['evpi_rel']:>8.2f}%{100*r['divergence_mi']:>8.1f}%  {g}")


def run_kdc_axis(n_jobs, n_offs, seed, turbines=(100, 101, 124, 10, 102, 103, 104, 105)):
    trace = str(_REPO / "cloudsimplus-gateway/src/main/resources/traces"
                / "gwo1_n1200_x130.csv")
    segs = {t: load_segments(t) for t in turbines}
    rng = np.random.default_rng(seed)
    span = min(e[-1] for _, e in segs.values()) - 24 * 3600
    offs = rng.integers(0, int(span), n_offs)
    print("=" * 78)
    print(f"轴 3: 空间自由度(k 个独立涡轮 DC)   rt中位 64min, slack 10h")
    print(f"{'k':>4}{'blind vs now':>14}{'clair vs now':>14}{'EVPI':>9}"
          f"{'EVPI 相对 k=1':>14}")
    base = job_mix(trace, 8.0, n_jobs, seed)
    jobs = [(mi, rt, 10 * 3600.0) for mi, rt in base]
    e1 = None
    for k in (1, 2, 4, 8):
        sl = [segs[t] for t in turbines[:k]]
        dl = [e - s for s, e in sl]
        r = scan_cell_kdc(sl, dl, jobs, offs)
        e1 = e1 or r["evpi_rel"]
        print(f"{k:>4}{100*(r['j_blind']-r['j_nowait'])/r['j_nowait']:>13.2f}%"
              f"{100*r['headroom_rel']:>13.2f}%{100*r['evpi_rel']:>8.2f}%"
              f"{100*r['evpi_rel']/e1:>13.1f}%")


def run_heldout(n_jobs, n_offs, seed, turbine=124):
    trace = str(_REPO / "cloudsimplus-gateway/src/main/resources/traces"
                / "gwo1_n1200_x130.csv")
    starts, ends = load_segments(turbine)
    durs = ends - starts
    rng = np.random.default_rng(seed + 1)
    offs = rng.integers(0, int(ends[-1] - 24 * 3600), n_offs)
    print("=" * 78)
    print(f"held-out 涡轮 T{turbine}(未参与任何标定)  rt中位 64min")
    print(f"{'slack':>7}{'θ*':>6}{'blind vs now':>14}{'clair vs now':>14}"
          f"{'EVPI':>9}{'分歧MI':>9}")
    base = job_mix(trace, 8.0, n_jobs, seed)
    for slack_h in (5, 10, 15):
        jobs = [(mi, rt, slack_h * 3600.0) for mi, rt in base]
        r = scan_cell(starts, ends, durs, jobs, offs)
        print(f"{slack_h:>6}h{r['theta_star']:>6.2f}"
              f"{100*r['blind_vs_now']:>13.2f}%{100*r['headroom_rel']:>13.2f}%"
              f"{100*r['evpi_rel']:>8.2f}%{100*r['divergence_mi']:>8.1f}%")
