#!/usr/bin/env python3
"""第十考场离线证明 v3 —— 容量耦合(条件④硬关) + 分位数拆解。

相对 v2 新增:
  (1) 容量模型:并发上限 K + FIFO 排队。关键耦合是【排队后的实际开始时刻回过头
      决定还装不装得下绿窗】——碳排按实际执行窗 [s, s+L) 积分,而策略选的是 r。
      拥堵把作业推出绿窗 = 真实的条件④ 检验。
  (2) 分位数拆解:clairvoyant vs 最强盲臂的碳差按作业 runtime 十分位拆开,
      检查优势是否全靠尾部鲸鱼作业。
合同:作业 ontime 当且仅当 实际开始 s_i <= a_i + B_i(即排队没吃掉预算)。
K* = 控制臂 nowait 仍守住其自身天花板的最紧容量(天花板实测,不预设 99.5%)。
"""
import argparse, heapq
import numpy as np

def build_green(T, rng, on_lo=1500, on_hi=2700, short_p=0.8, sh=(300,1500), lo=(2700,4500)):
    g = np.zeros(T, dtype=np.int8); t = 0
    while t < T:
        on = int(rng.integers(on_lo, on_hi)); g[t:t+on] = 1; t += on
        if t >= T: break
        t += int(rng.integers(*(sh if rng.random() < short_p else lo)))
    return g

def build_ci(T, period, amp, g, rho, base=1.0):
    t = np.arange(T)
    indep = np.sin(2*np.pi*t/period)
    w = max(1,int(period)); s = np.convolve(g.astype(float), np.ones(w)/w, mode="same")
    s = (s - s.mean())/(s.std()+1e-12)
    return base*(1.0 + amp*((1-rho)*indep + rho*(-np.clip(s,-2,2)/2.0)))

def fifo_start(r, L, K):
    """并发上限 K 的 FIFO 准入:按释放时刻排队,返回实际开始时刻 s。"""
    order = np.argsort(r, kind="stable")
    s = np.empty_like(r); busy = []          # 小顶堆:运行中作业的完成时刻
    for i in order:
        t = r[i]
        while busy and busy[0] <= t: heapq.heappop(busy)
        if len(busy) >= K:
            t = heapq.heappop(busy)          # 等到最早一个空位
        s[i] = t; heapq.heappush(busy, t + L[i])
    return s

def carbon_of(cost_pt, s, L, mi, csum=None):
    if csum is None: csum = np.concatenate([[0.0], np.cumsum(cost_pt)])
    # 排队可能把开始时刻推到接近时间轴末端;超出视界的部分不计碳(并在 ontime 上已被判罚)
    hi = np.clip(s + L, 0, len(csum) - 1); lo = np.clip(s, 0, len(csum) - 1)
    return (csum[hi] - csum[lo]) * (mi / L)

def peak_conc(s, L, T):
    occ = np.zeros(T+2)
    np.add.at(occ, np.clip(s,0,T+1), 1); np.add.at(occ, np.clip(s+L,0,T+1), -1)
    return int(np.cumsum(occ)[:T].max())

def anchor(seed, T, n, slack, run, ci_period, amp, rho, theta_q, Ks):
    rng = np.random.default_rng(seed)
    g = build_green(T, rng); c = build_ci(T, ci_period, amp, g, rho)
    cost = np.where(g==1, 0.0, c)
    csum = np.concatenate([[0.0], np.cumsum(cost)])   # 窗口和 O(1)
    ccsum = np.concatenate([[0.0], np.cumsum(c)])     # 纯碳强度的前缀和(确定性,盲臂可解析获知)
    a = rng.integers(0, T-slack[1]-run[1]-2, size=n)
    B = rng.integers(slack[0], slack[1], size=n)
    L = rng.integers(run[0], run[1], size=n); mi = L.astype(float)
    # 注册分布(独立种子)用于冻结盲阈值
    reg = np.random.default_rng(9999); gr = build_green(T, reg)
    cr = build_ci(T, ci_period, amp, gr, rho)
    theta = float(np.quantile(cr, theta_q))
    d = np.diff(np.concatenate([[0], gr, [0]]))
    ages, rems = [], []
    for st, en in zip(np.flatnonzero(d==1), np.flatnonzero(d==-1)):
        dur = en-st; ages += list(range(dur)); rems += list(range(dur,0,-1))
    ages = np.array(ages); rems = np.array(rems); AM = int(ages.max())+1
    exp_rem = np.array([rems[ages==k].mean() if (ages==k).any() else 0.0 for k in range(AM)])
    gage = np.where(g==1, 0, -1).astype(np.int64); run_ = 0
    for t in range(T):
        run_ = run_+1 if g[t]==1 else 0
        gage[t] = run_-1 if g[t]==1 else -1

    onset = np.zeros(T, dtype=bool)
    onset[0] = g[0] == 1
    onset[1:] = (g[1:] == 1) & (g[:-1] == 0)          # 绿窗起点,当下可观测
    R = {k: np.empty(n, dtype=np.int64) for k in
         ("nowait","naive_green","naive_carbon","green_age","onset_wait","clock_carbon","clock_onset","combo","clair")}
    spill = np.zeros(n)   # (A) 诊断:naive_green 释放后有多少 runtime 落在绿窗之外
    for i in range(n):
        lo_, hi_ = int(a[i]), int(min(a[i]+B[i], T-L[i]-1))
        win = np.arange(lo_, hi_+1)
        R["nowait"][i] = lo_
        gi = np.flatnonzero(g[win]==1); R["naive_green"][i] = win[gi[0]] if gi.size else hi_
        ci_ = np.flatnonzero(c[win]<=theta); R["naive_carbon"][i] = win[ci_[0]] if ci_.size else hi_
        ok = np.flatnonzero((g[win]==1) & (exp_rem[np.clip(gage[win],0,AM-1)] >= L[i]))
        R["green_age"][i] = win[ok[0]] if ok.size else (win[gi[0]] if gi.size else hi_)
        ow = np.flatnonzero(onset[win])
        R["onset_wait"][i] = win[ow[0]] if ow.size else (win[gi[0]] if gi.size else hi_)
        rg = R["naive_green"][i]
        spill[i] = float((g[rg:rg+L[i]] == 0).sum())   # 溢出到棕电的步数
        ccum = ccsum[win+L[i]] - ccsum[win]          # 纯 c 的窗口积分(不含绿电信息)
        r_clock = win[int(np.argmin(ccum))]
        R["clock_carbon"][i] = min(win[gi[0]], r_clock) if gi.size else r_clock
        R["clock_onset"][i] = win[ow[0]] if ow.size else win[int(np.argmin(ccum))]
        if ow.size:                 R["combo"][i] = win[ow[0]]
        elif ok.size:               R["combo"][i] = win[ok[0]]
        elif gi.size:               R["combo"][i] = win[gi[0]]
        else:                       R["combo"][i] = win[int(np.argmin(ccum))]
        cs = csum[win+L[i]] - csum[win]
        R["clair"][i] = win[int(np.argmin(cs))]

    latest = a + B
    out = {"K": {}, "L": L, "mi": mi}
    out["spill_frac"] = float(np.median(spill / L))
    out["spill_any"] = 100*float((spill > 0).mean())
    for K in Ks:
        rec = {}
        for k, r in R.items():
            s = fifo_start(r, L, K)
            cb = carbon_of(cost, s, L, mi, csum)
            rec[k] = {"carbon": float(cb.sum()),
                      "ontime": 100*float(mi[s<=latest].sum()/mi.sum()),
                      "qdelay": float(np.median(s-r)),
                      "peak": peak_conc(s, L, T),
                      "per_job": cb}
        out["K"][K] = rec
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=72000); p.add_argument("--anchors", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=600)
    p.add_argument("--ci-amp", type=float, default=0.5); p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--Ks", default="1000,60,45,35,28,22,16")
    p.add_argument("--ci-period", type=int, default=8640)
    a = p.parse_args()
    Ks = [int(x) for x in a.Ks.split(",")]
    rows = [anchor(1009*k, a.T, a.n_jobs, (200,3000), (100,1000), a.ci_period,
                   a.ci_amp, a.rho, 0.30, Ks) for k in range(a.anchors)]
    ARMS = ("nowait","naive_green","naive_carbon","green_age","onset_wait","clock_carbon","clock_onset","combo","clair")
    med = lambda K,k,f: float(np.median([r["K"][K][k][f] for r in rows]))
    print(f"anchors={len(rows)} n={a.n_jobs} amp={a.ci_amp} rho={a.rho}  (K=1000 视作无限容量)")
    # 提供的负载:mean = n*E[L]/T (与臂无关的标准 offered load);peak 取无容量约束时的实测峰值
    meanL = float(np.mean(np.concatenate([r["L"] for r in rows])))
    offered_mean = a.n_jobs * meanL / a.T
    offered_peak = float(np.median([r["K"][Ks[0]]["clair"]["peak"] for r in rows]))
    print(f"(A)诊断 naive_green: 溢出到棕电的 runtime 占比 中位={np.median([r['spill_frac'] for r in rows]):.3f}"
          f"  有溢出的作业占比={np.median([r['spill_any'] for r in rows]):.1f}%")
    print(f"提供负载: 均值并发={offered_mean:.1f}  峰值并发={offered_peak:.0f}  (E[L]={meanL:.0f})")
    print(f"{'K':>6} {'均值利用率':>11} {'峰值利用率':>11} {'nowait_ontime':>14} {'clair vs best':>14} {'同号':>7} {'peak_c/peak_b':>14} {'qdelay_c':>9}")
    K0 = Ks[0]
    t0 = {k: med(K0,k,"carbon") for k in ARMS}
    print("  各盲臂总碳(K=%d,越低越强): %s" % (K0, {k: round(t0[k]/1e6,2) for k in ARMS}))
    for K in Ks:
        tots = {k: med(K,k,"carbon") for k in ARMS}
        bb = min(("nowait","naive_green","naive_carbon","green_age","onset_wait","clock_carbon","clock_onset","combo"), key=lambda k: tots[k])
        per = [100*(r["K"][K]["clair"]["carbon"]-r["K"][K][bb]["carbon"])/max(r["K"][K][bb]["carbon"],1e-12) for r in rows]
        same = sum(1 for x in per if x < 0)
        um = 100*offered_mean/K; up = 100*offered_peak/K
        print(f"{K:>6} {um:>10.1f}% {up:>10.1f}% {med(K,'nowait','ontime'):>13.2f}% "
              f"{np.median(per):>13.2f}% {same:>4}/10 {med(K,'clair','peak'):>6.0f}/{med(K,bb,'peak'):<7.0f} {med(K,'clair','qdelay'):>8.0f}")
    # 分位数拆解(无限容量档)
    K0 = Ks[0]
    tots = {k: med(K0,k,"carbon") for k in ARMS}
    bb = min(("nowait","naive_green","naive_carbon","green_age","onset_wait","clock_carbon","clock_onset","combo"), key=lambda k: tots[k])
    print(f"\n分位数拆解 (K={K0}, 最强盲臂={bb}): clairvoyant 的碳节省按 runtime 十分位")
    L_all = np.concatenate([r["L"] for r in rows])
    d_all = np.concatenate([r["K"][K0][bb]["per_job"]-r["K"][K0]["clair"]["per_job"] for r in rows])
    q = np.quantile(L_all, np.linspace(0,1,11))
    tot_sav = d_all.sum()
    print(f"{'十分位':>8} {'runtime范围':>16} {'节省占比':>10} {'累计':>8}")
    cum = 0.0
    for i in range(10):
        m = (L_all>=q[i]) & (L_all<=q[i+1] if i==9 else L_all<q[i+1])
        sh = 100*d_all[m].sum()/tot_sav; cum += sh
        print(f"{i+1:>8} {f'{q[i]:.0f}-{q[i+1]:.0f}':>16} {sh:>9.1f}% {cum:>7.1f}%")

if __name__ == "__main__":
    main()
