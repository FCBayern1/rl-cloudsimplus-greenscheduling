#!/usr/bin/env python3
"""第十考场离线证明 v2 —— 补齐 5080 指出的三处硬伤。

v1 的三个缺陷,逐条修:
  (1) 盲策略集不完整。时变碳强度下 c(t) 与 g(t) 一样是【当下可观测】的,
      所以"等到便宜"是一个零信息就能做的策略。SQT2 的教训是最强盲基线正是
      那个利用了免费信息的最笨策略。本版把盲候选扩成四个,判据改为
      clairvoyant 必须打赢【最强的那个】>= 5%。
  (2) 碳排按 runtime 积分,不取释放时刻点值。作业跑 L 秒,碳排是
      sum_{t=r}^{r+L-1} (MI/L) * [g(t)=0] * c(t) —— 窗口均值比点值平滑,
      会削薄 clairvoyant 的优势,长作业(MI 权重最大)被削得最狠。
  (3) 输出各臂释放时刻的并发直方图与峰值并发。SQT2 的条件 ④ 就死在
      "两个会等的臂在同一时刻释放同一批作业"。

决策信号 vs 计分口径(关键的不对称,刻意如此):
  盲策略只能看【当下】 cost_point(t) = 0 if g(t) else c(t);
  clairvoyant 看得到未来,按【积分】碳排挑释放点;
  但所有臂都按【积分】碳排计分。
"""
import argparse, json
import numpy as np

def build_green(T, rng, on_lo=1500, on_hi=2700, short_p=0.8,
                sh=(300, 1500), lo=(2700, 4500)):
    g = np.zeros(T, dtype=np.int8); t = 0
    while t < T:
        on = int(rng.integers(on_lo, on_hi)); g[t:t+on] = 1; t += on
        if t >= T: break
        t += int(rng.integers(*(sh if rng.random() < short_p else lo)))
    return g

def build_ci(T, period, amp, phase, g, rho, base=1.0):
    """c(t) = (1-rho)*独立正弦 + rho*(-绿电滑动均值)。rho=0 独立, rho=1 现实反相关。"""
    t = np.arange(T)
    indep = np.sin(2*np.pi*(t/period + phase))
    w = max(1, int(period)); k = np.ones(w)/w
    s = np.convolve(g.astype(float), k, mode="same")
    s = (s - s.mean())/(s.std()+1e-12)
    return base * (1.0 + amp * ((1-rho)*indep + rho*(-np.clip(s,-2,2)/2.0)))

def integ_cost(cost_pt, r, L, mi):
    """作业在 [r, r+L) 的积分碳排,功率 = mi/L。"""
    return float(cost_pt[r:r+L].sum()) * (mi / L)

def reservation_values(cost_samples, kmax):
    """最优停时的保留值:v_0 = E[cost](到期强派), v_k = E[min(cost, v_{k-1})]。"""
    v = np.empty(kmax+1); v[0] = cost_samples.mean()
    for k in range(1, kmax+1):
        v[k] = np.minimum(cost_samples, v[k-1]).mean()
    return v

def run_anchor(seed, T, n_jobs, slack_lo, slack_hi, run_lo, run_hi,
               ci_period, ci_amp, rho, theta_q, stop_grid):
    rng = np.random.default_rng(seed)
    g = build_green(T, rng)
    c = build_ci(T, ci_period, ci_amp, 0.0, g, rho)
    cost_pt = np.where(g == 1, 0.0, c)                 # 当下可观测的单位功率碳价

    a = rng.integers(0, T - slack_hi - run_hi - 2, size=n_jobs)
    B = rng.integers(slack_lo, slack_hi, size=n_jobs)
    L = rng.integers(run_lo, run_hi, size=n_jobs)
    mi = L.astype(float)                                # 单位功率 -> MI 正比 runtime

    # 冻结的盲阈值:theta = cost 分布的 theta_q 分位(在"注册分布"上算,不看本锚未来)
    reg = np.random.default_rng(9999)
    gr = build_green(T, reg); cr = build_ci(T, ci_period, ci_amp, 0.0, gr, rho)
    reg_cost = np.where(gr == 1, 0.0, cr)
    theta = float(np.quantile(cr, theta_q))   # 阈值定义在 c 分布上(与比较对象一致)
    vres = reservation_values(reg_cost[np.random.default_rng(7).integers(0, T, 20000)],
                              int(slack_hi))

    # 绿窗年龄的注册后验:E[剩余ON时长 | 已持续 age],纯零信息(只用注册分布)
    ages, rems = [], []
    d = np.diff(np.concatenate([[0], gr, [0]]))
    for st, en in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        dur = en - st
        for age in range(dur):
            ages.append(age); rems.append(dur - age)
    ages = np.array(ages); rems = np.array(rems)
    AMAX = int(ages.max()) + 1
    exp_rem = np.zeros(AMAX)
    for age in range(AMAX):
        m = ages == age
        exp_rem[age] = rems[m].mean() if m.any() else 0.0
    # 本锚绿窗年龄
    gage = np.zeros(T, dtype=np.int64); run = 0
    for t in range(T):
        run = run + 1 if g[t] == 1 else 0
        gage[t] = run - 1 if g[t] == 1 else -1

    arms = {k: np.empty(n_jobs, dtype=np.int64) for k in
            ("nowait", "naive_green", "naive_carbon", "stopping", "green_age", "clair")}
    for i in range(n_jobs):
        lo_, hi_ = int(a[i]), int(min(a[i]+B[i], T-L[i]-1))
        win = np.arange(lo_, hi_+1)
        arms["nowait"][i] = lo_
        gi = np.flatnonzero(g[win] == 1)
        arms["naive_green"][i] = win[gi[0]] if gi.size else hi_
        ci_ = np.flatnonzero(c[win] <= theta)
        arms["naive_carbon"][i] = win[ci_[0]] if ci_.size else hi_
        # 最优停时:剩余预算 k 时,若 cost_pt <= v[k] 则停
        rem = hi_ - win
        take = np.flatnonzero(cost_pt[win] <= vres[np.minimum(rem, len(vres)-1)])
        arms["stopping"][i] = win[take[0]] if take.size else hi_
        # green_age:零信息的 hazard 类比 —— 绿电中,且注册后验说"预计剩余 >= 本作业 runtime"才放行
        ok = np.flatnonzero((g[win] == 1) &
                            (exp_rem[np.clip(gage[win], 0, AMAX-1)] >= L[i]))
        arms["green_age"][i] = win[ok[0]] if ok.size else (
            win[gi[0]] if gi.size else hi_)     # 预算内等不到足够长的绿窗 -> 退回第一个绿
        # clairvoyant:按积分碳排最小
        costs = np.array([integ_cost(cost_pt, int(r), int(L[i]), mi[i]) for r in win])
        arms["clair"][i] = win[int(np.argmin(costs))]

    out = {"green_ratio": float(g.mean()), "theta": theta}
    tot = {}
    for k, r in arms.items():
        tot[k] = float(sum(integ_cost(cost_pt, int(r[i]), int(L[i]), mi[i]) for i in range(n_jobs)))
        # 峰值并发(释放时刻直方图的最大重叠)
        occ = np.zeros(T+1)
        np.add.at(occ, r, 1); np.add.at(occ, r+L, -1)
        out[f"peak_conc_{k}"] = int(np.cumsum(occ)[:T].max())
    out["carbon"] = tot
    best_blind = min(("nowait","naive_green","naive_carbon","stopping","green_age"), key=lambda k: tot[k])
    out["best_blind"] = best_blind
    out["impr_vs_best_blind"] = 100*(tot["clair"]-tot[best_blind])/max(tot[best_blind],1e-12)
    for k in ("nowait","naive_green","naive_carbon","stopping","green_age"):
        out[f"impr_vs_{k}"] = 100*(tot["clair"]-tot[k])/max(tot[k],1e-12)
    # clair_forgone: 最强盲臂严格劣于 clairvoyant 的 MI 占比
    bb = arms[best_blind]
    worse = np.array([integ_cost(cost_pt,int(bb[i]),int(L[i]),mi[i]) >
                      integ_cost(cost_pt,int(arms["clair"][i]),int(L[i]),mi[i])+1e-12
                      for i in range(n_jobs)])
    out["clair_forgone_mi"] = 100*float(mi[worse].sum()/mi.sum())
    return out

def main():
    p = argparse.ArgumentParser()
    for k, v in [("--T",72000),("--anchors",10),("--n-jobs",1200),
                 ("--slack-lo",200),("--slack-hi",3000),
                 ("--run-lo",100),("--run-hi",1000),("--ci-period",8640)]:
        p.add_argument(k, type=int, default=v)
    p.add_argument("--ci-amp", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--theta-q", type=float, default=0.30)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    rows = [run_anchor(1009*k, a.T, a.n_jobs, a.slack_lo, a.slack_hi, a.run_lo, a.run_hi,
                       a.ci_period, a.ci_amp, a.rho, a.theta_q, None) for k in range(a.anchors)]
    med = lambda key: float(np.median([r[key] for r in rows]))
    print(f"锚点={len(rows)}  绿电占比={med('green_ratio'):.3f}  runtime~U[{a.run_lo},{a.run_hi}]  rho={a.rho}  amp={a.ci_amp}")
    print("  clairvoyant 相对各盲臂的碳中位改善（负=更好）:")
    for k in ("nowait","naive_green","naive_carbon","stopping","green_age"):
        s = sum(1 for r in rows if r[f"impr_vs_{k}"] < 0)
        print(f"    vs {k:14} {med(f'impr_vs_{k}'):+7.2f}%   同号 {s}/{len(rows)}")
    from collections import Counter
    bb = Counter(r["best_blind"] for r in rows)
    sbest = sum(1 for r in rows if r["impr_vs_best_blind"] < 0)
    print(f"  最强盲臂分布: {dict(bb)}")
    print(f"  ★ vs 最强盲臂  {med('impr_vs_best_blind'):+7.2f}%   同号 {sbest}/{len(rows)}   门槛 <=-5%")
    print(f"  clair_forgone MI 占比 = {med('clair_forgone_mi'):.2f}%   (SQT2 恒为 0)")
    print("  峰值并发（条件④ 早期预警,clair≈naive 则标红）:")
    for k in ("nowait","naive_green","naive_carbon","stopping","green_age","clair"):
        print(f"    {k:14} {med(f'peak_conc_{k}'):8.0f}")
    gate = med('impr_vs_best_blind') <= -5.0 and sbest >= 8
    print(f"  判决: {'✅ 过' if gate else '❌ 不过'}")
    if a.json: json.dump({"params":vars(a),"rows":rows}, open(a.json,"w"), indent=1)

if __name__ == "__main__":
    main()
