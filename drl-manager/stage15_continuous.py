#!/usr/bin/env python3
"""Stage 1.5:连续瓦特碳积分 + 年份冻结盲 + artifact 固化(Codex P0-1/2/3)。

与 phase_scan_offline 的三个区别:
  P0-1  碳不再按二值窗口计,而是流体积分:
            brown(t) = max(0, D(t) - G(t)),  green(t) = min(D(t), G(t))
        D(t) 由该策略全部作业的实际释放时刻叠加(pes x 2.541 W),
        G(t) 是原始瓦特序列 —— 绿电共享/自拥挤自动包含在内。
  P0-2  到达过程固化成 artifact(块 offset、逐作业到达、MI/PES 抽样索引、
        slack、horizon 40h、绿电缩放),Stage 2 用同一份。
  P0-3  盲的窗长生存表 F 与阈值 theta 在 2020 年拟合并冻结,2021 年评测,
        2022 年完全不碰(留给正式 held-out)。二值化阈值同样取自 2020。

clairvoyant 仍按逐作业二值重叠选释放时刻(它看不到别的作业),但按流体口径
计分 —— 这低估 clair,方向保守。

判决口径(Codex 更正):verdict = (C_clair - C_blind) / C_blind。
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

_REPO = pathlib.Path(__file__).resolve().parent.parent
_W = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
ROW_S = 600
C_BROWN, C_GREEN = 0.55, 0.01
W_PER_PE = 2.541
THETAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.1)


def load_watts(turbines, year):
    gs = None
    for t in turbines:
        rows = list(csv.DictReader(open(_W / f"Turbine_{t}_{year}.csv")))
        p = np.array([float(r["power_kw"]) for r in rows])
        gs = p if gs is None else gs[:len(p)] + p[:len(gs)]
    return gs * 1000.0            # -> W(缩放在 ρ 处统一处理,artifact 记录)


def binarize(watts, thr):
    on = watts > thr
    d = np.diff(np.concatenate(([0], on.view(np.int8), [0])))
    s = np.where(d == 1)[0] * ROW_S
    e = np.where(d == -1)[0] * ROW_S
    return s.astype(float), e.astype(float)


def fit_prob_table(durs, rt, max_age_rows=600):
    out = np.empty(max_age_rows)
    for j in range(max_age_rows):
        a = j * ROW_S
        alive = durs[durs > a]
        out[j] = 0.0 if alive.size == 0 else (alive >= a + rt).mean()
    return out


def green_overlap(starts, ends, a, b):
    if b <= a:
        return 0.0
    i = max(0, np.searchsorted(ends, a, "right"))
    tot = 0.0
    while i < len(starts) and starts[i] < b:
        tot += max(0.0, min(ends[i], b) - max(starts[i], a))
        i += 1
    return tot


def state(starts, ends, t):
    i = np.searchsorted(ends, t, "right")
    if i < len(starts) and starts[i] <= t:
        return True, t - starts[i]
    return False, 0.0


def clair_release(starts, ends, t0, rt, latest):
    cands = [t0, latest]
    i = np.searchsorted(starts, t0, "left")
    while i < len(starts) and starts[i] <= latest:
        cands.append(starts[i]); i += 1
    best = (2.0, t0)
    for s in sorted(set(cands)):
        bf = 1.0 - green_overlap(starts, ends, s, s + rt) / rt
        if bf < best[0] - 1e-12:
            best = (bf, s)
    return best[1]


def blind_release(starts, ends, F_of_rt, t0, rt, latest, theta):
    t = t0
    while t < latest:
        in_on, age = state(starts, ends, t)
        if in_on:
            F = F_of_rt
            if F[min(int(age // ROW_S), len(F) - 1)] >= theta:
                return t
        t = min(t - (t % ROW_S) + ROW_S, latest)
    return latest


def fluid_carbon(jobs, releases, watts, green_scale):
    """流体积分:D(t) 叠加全部作业,G(t) = watts x green_scale。返回碳与能量。"""
    n = len(watts)
    D = np.zeros(n)
    for (mi, pes, rt, *_), rel in zip(jobs, releases):
        a = int(rel // ROW_S)
        b = min(n, int((rel + rt) // ROW_S) + 1)
        if a >= n:
            continue
        # 端行按覆盖比例计
        for r in range(a, b):
            lo, hi = r * ROW_S, (r + 1) * ROW_S
            frac = max(0.0, min(rel + rt, hi) - max(rel, lo)) / ROW_S
            D[r] += pes * W_PER_PE * frac
    G = watts * green_scale
    brownE = np.maximum(0.0, D - G).sum() * ROW_S
    greenE = np.minimum(D, G).sum() * ROW_S
    return C_BROWN * brownE + C_GREEN * greenE, brownE, greenE


def make_jobs(trace_csv, rt_scale, n_blocks, jobs_per_block, slack_s, year_s,
              seed):
    rows = list(csv.DictReader(open(trace_csv)))
    rng = np.random.default_rng(seed)
    offs = np.sort(rng.integers(0, int(year_s - 40 * 3600), n_blocks))
    jobs = []
    for off in offs:
        idx = rng.choice(len(rows), size=jobs_per_block, replace=False)
        arr = rng.uniform(0, 24 * 3600.0, size=jobs_per_block)
        for i, da in zip(idx, arr):
            mi = float(rows[i]["length"]) * rt_scale
            pes = max(1, int(rows[i]["pes_required"]))
            rt = max(1.0, mi / (pes * 40000.0))
            jobs.append((mi, pes, rt, float(off) + float(da), slack_s,
                         int(i)))
    return jobs, offs


def run(cal_year=2020, eval_year=2021, turbines=(100, 101), rho=1.0,
        rt_scale=8.0, slack_h=15.0, n_blocks=20, jobs_per_block=150,
        seed=20260822, q=30):
    trace = str(_REPO / "cloudsimplus-gateway/src/main/resources/traces"
                / "gwo1_n1200_x130.csv")
    art = {"turbines": list(turbines), "cal_year": cal_year,
           "eval_year": eval_year, "rho": rho, "rt_scale": rt_scale,
           "slack_h": slack_h, "n_blocks": n_blocks,
           "jobs_per_block": jobs_per_block, "seed": seed,
           "binarize_percentile": q, "horizon_h": 40,
           "w_per_pe": W_PER_PE, "c_brown": C_BROWN, "c_green": C_GREEN}

    # ---- 2020:拟合并冻结(阈值、F 表、theta) ----
    w_cal = load_watts(turbines, cal_year)
    thr = float(np.percentile(w_cal[w_cal > 0], q))
    s_c, e_c = binarize(w_cal, thr)
    durs_cal = e_c - s_c
    art["binarize_thr_w"] = thr

    jobs_cal, offs_cal = make_jobs(trace, rt_scale, n_blocks, jobs_per_block,
                                   slack_h * 3600.0, len(w_cal) * ROW_S, seed)
    # ρ 定标:绿电缩放使 mean(G) = mean(D_nowait)/ρ,在 2020 上冻结
    rel_now = [j[3] for j in jobs_cal]
    D_mean = sum(j[1] * W_PER_PE * j[2] for j in jobs_cal) / (len(w_cal) * ROW_S)
    green_scale = D_mean / rho / w_cal.mean()
    art["green_scale"] = green_scale
    art["demand_mean_w"] = D_mean

    F_cache = {}
    def F_for(rt):
        key = int(rt // 600)
        if key not in F_cache:
            F_cache[key] = fit_prob_table(durs_cal, rt)
        return F_cache[key]

    def eval_policies(jobs, watts, segs):
        s_, e_ = segs
        rel_n = [j[3] for j in jobs]
        rel_c = [clair_release(s_, e_, j[3], j[2], j[3] + j[4]) for j in jobs]
        out = {}
        for th in THETAS:
            rel_b = [blind_release(s_, e_, F_for(j[2]), j[3], j[2],
                                   j[3] + j[4], th) for j in jobs]
            out[th] = (fluid_carbon(jobs, rel_b, watts, green_scale)[0], rel_b)
        c_now = fluid_carbon(jobs, rel_n, watts, green_scale)[0]
        c_cla = fluid_carbon(jobs, rel_c, watts, green_scale)[0]
        return c_now, c_cla, out, rel_c

    print(f"[cal {cal_year}] thr={thr:.0f}W scale={green_scale:.3g} "
          f"rho={rho} jobs={len(jobs_cal)}")
    c_now, c_cla, by_th, _ = eval_policies(jobs_cal, w_cal, (s_c, e_c))
    th_star = min(by_th, key=lambda k: by_th[k][0])
    art["theta_star"] = th_star
    print(f"  theta*={th_star}  (blind 碳按 theta: "
          + " ".join(f"{k}:{by_th[k][0]:.3g}" for k in THETAS) + ")")

    # ---- 2021:全部冻结量下评测 ----
    w_ev = load_watts(turbines, eval_year)
    s_v, e_v = binarize(w_ev, thr)             # 阈值冻结自 2020
    jobs_ev, offs_ev = make_jobs(trace, rt_scale, n_blocks, jobs_per_block,
                                 slack_h * 3600.0, len(w_ev) * ROW_S,
                                 seed + 1)
    c_now, c_cla, by_th, rel_c = eval_policies(jobs_ev, w_ev, (s_v, e_v))
    c_blind, rel_b = by_th[th_star]
    c_await, _ = by_th[1.1]                    # 永不进窗 = 等到 latest-start
    tot_mi = sum(j[0] for j in jobs_ev)
    div = sum(j[0] for j, rb, rc in zip(jobs_ev, rel_b, rel_c)
              if abs(rb - rc) > ROW_S) / tot_mi
    res = {"c_nowait": c_now, "c_blind_frozen": c_blind, "c_clair": c_cla,
           "c_alwayswait": c_await,
           "verdict_clair_vs_blind": (c_cla - c_blind) / c_blind,
           "evpi_vs_nowait_denom": (c_cla - c_blind) / c_now,
           "blind_vs_nowait": (c_blind - c_now) / c_now,
           "headroom_clair_vs_nowait": (c_cla - c_now) / c_now,
           "divergence_mi": div}
    art["offsets_cal"] = [int(x) for x in offs_cal]
    art["offsets_eval"] = [int(x) for x in offs_ev]
    art["results_eval_year"] = res
    print(f"[eval {eval_year}] (冻结 thr/F/theta,绿电缩放同一份)")
    print(f"  nowait      {c_now:.4g}")
    print(f"  always-wait {c_await:.4g}   ({100*(c_await-c_now)/c_now:+.1f}% vs now)")
    print(f"  blind(冻结) {c_blind:.4g}   ({100*res['blind_vs_nowait']:+.1f}% vs now)")
    print(f"  clairvoyant {c_cla:.4g}   ({100*res['headroom_clair_vs_nowait']:+.1f}% vs now)")
    print(f"  ★ verdict (clair-blind)/blind = {100*res['verdict_clair_vs_blind']:+.2f}%")
    print(f"    分歧 MI = {100*div:.1f}%")
    ok = (res["verdict_clair_vs_blind"] <= -0.08 and div >= 0.10
          and c_blind < c_now and c_blind < c_await)
    print(f"  门(clair vs blind<=-8%, 分歧>=10%, 盲严格优于两个极限): "
          f"{'✅ 过' if ok else '❌ 不过'}")
    return art, res, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--slack-h", type=float, default=15.0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    art, res, ok = run(rho=a.rho, slack_h=a.slack_h)
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(art, indent=1))
        print(f"artifact -> {a.json_out}")


def coordinated_bound(jobs, watts, green_scale):
    """协调 clairvoyant 的碳下界(放松:作业可分割可抢占,速率 <= pes x W)。

    在共享绿电下,逐作业贪心 clair 会自拥挤;真正的信息价值上界是联合调度:
    按窗口截止的 EDF 水填充,逐行把绿电能量分给窗口内的作业(速率封顶)。
    这是运输问题的多拟阵贪心,对"最大绿电吸收"是精确的;对不可分割的真问题
    是上界 -> 碳是下界 -> EVPI 的上界。方向:有利于"过线",过不了就是真死。
    """
    n = len(watts)
    G = watts * green_scale
    evs = []
    for mi, pes, rt, arr, slack, _ in jobs:
        evs.append((arr, arr + slack + rt, pes * W_PER_PE,
                    pes * W_PER_PE * rt))
    evs.sort()
    totalE = sum(e[3] for e in evs)
    import heapq
    absorbed = 0.0
    heap = []          # (window_end, rate, remaining)
    i = 0
    for r in range(n):
        t0, t1 = r * ROW_S, (r + 1) * ROW_S
        while i < len(evs) and evs[i][0] < t1:
            heapq.heappush(heap, [evs[i][1], evs[i][2], evs[i][3]])
            i += 1
        g_left = G[r] * ROW_S
        if g_left <= 0 or not heap:
            continue
        keep = []
        while heap and g_left > 1e-9:
            it = heapq.heappop(heap)
            if it[0] < t0:                     # 窗口已过
                continue
            take = min(it[1] * ROW_S, it[2], g_left)
            absorbed += take
            g_left -= take
            it[2] -= take
            if it[2] > 1e-9:
                keep.append(it)
        for it in keep:
            heapq.heappush(heap, it)
    brown = totalE - absorbed
    return C_BROWN * brown + C_GREEN * absorbed, absorbed / totalE


def coordinated_blind_contig(jobs, watts, green_scale):
    """连续版协调盲(严格因果):绿电跟随准入。

    逐行向前;只用当前实测 G(r) 与自己已释放作业的需求 D_run(r):
    空余绿电够一个 pending 作业的功率就放行(EDF 按 latest-start),
    到 latest-start 的作业无条件放行(永不过期)。不看任何未来 G。
    """
    n = len(watts)
    G = watts * green_scale
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][3])
    D = np.zeros(n)
    releases = [0.0] * len(jobs)
    import heapq
    pend = []                     # (latest_start, idx)
    i = 0
    for r in range(n):
        t = r * ROW_S
        while i < len(order) and jobs[order[i]][3] < t + ROW_S:
            j = order[i]
            heapq.heappush(pend, (jobs[j][3] + jobs[j][4], j))
            i += 1
        while pend:
            latest, j = pend[0]
            p = jobs[j][1] * W_PER_PE
            if latest <= t:                       # 兜底:到点必须放
                heapq.heappop(pend)
                releases[j] = max(jobs[j][3], float(latest))
                a = int(releases[j] // ROW_S)
                b = min(n, a + int(jobs[j][2] // ROW_S) + 1)
                D[a:b] += p
            elif G[r] - D[r] >= p:                # 空余绿电够 -> 放行
                heapq.heappop(pend)
                releases[j] = max(jobs[j][3], float(t))
                a = r
                b = min(n, a + int(jobs[j][2] // ROW_S) + 1)
                D[a:b] += p
            else:
                break
    return releases


def coordinated_clair_contig(jobs, watts, green_scale):
    """连续版协调 clairvoyant:按 latest-start 序贪心插入,每个作业在自己
    窗口内选【给定已放置作业后增量棕电最小】的整块位置。看得到全部未来 G。"""
    n = len(watts)
    G = watts * green_scale
    D = np.zeros(n)
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][3] + jobs[i][4])
    releases = [0.0] * len(jobs)
    for j in order:
        mi, pes, rt, arr, slack, _ = jobs[j]
        p = pes * W_PER_PE
        rt_rows = int(rt // ROW_S) + 1
        best = (float("inf"), arr)
        r_lo = int(arr // ROW_S)
        r_hi = int((arr + slack) // ROW_S)
        for r in range(r_lo, r_hi + 1):
            seg = slice(r, min(n, r + rt_rows))
            inc = (np.maximum(0.0, D[seg] + p - G[seg])
                   - np.maximum(0.0, D[seg] - G[seg])).sum()
            if inc < best[0] - 1e-9:
                best = (inc, max(arr, r * ROW_S))
        releases[j] = best[1]
        a = int(best[1] // ROW_S)
        D[a:min(n, a + rt_rows)] += p
    return releases
