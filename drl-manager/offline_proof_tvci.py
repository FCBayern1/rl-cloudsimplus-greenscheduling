#!/usr/bin/env python3
"""第十考场离线可行性证明:时变电网碳强度 (time-varying carbon intensity, TVCI)。

HANDOVER_NEW_SCENARIO.md §3 末尾的硬规矩:建场景之前先用纯离线计算回答
clairvoyant 相对 naive 的逐作业碳差,过 >=10% 中位改善 + >=8/10 锚点同号才动代码。

模型(与 SQT2 的 clair_forgone 分解同形):
  给定绿电可用曲线 g(t) in {0,1} 与棕电碳强度曲线 c(t) > 0,作业 i 有到达 a_i、
  计算量 MI_i、预算 B_i(latest start = a_i + B_i)。释放时刻 r 的碳排:
        carbon(i, r) = 0                    若 g(r) = 1   (绿电,碳 ~ 0)
                     = K * MI_i * c(r)      否则          (棕电,按当刻强度计价)
  三个策略:
    nowait      r = a_i
    naive       r = 第一个 t >= a_i 且 g(t)=1;若预算内无绿,r = a_i + B_i (backstop)
    clairvoyant r = argmin_{r in [a_i, a_i+B_i]} carbon(i, r)

⚠️ 设计陷阱(本脚本要验的核心):现实电网里 c(t) 与 g(t) 是【反相关】的
   —— 风大 → 电网更干净。若照搬这个相关性,"等到绿"同时也是"等到最干净",
   naive 依然无悔,直接重蹈 SQT2 覆辙。所以本脚本把 c(t) 与 g(t) 的相位关系
   作为可扫参数,用来定位"预报才有价值"的区域(若存在)。
"""
import argparse, json
import numpy as np

def build_green(T, rng, on_lo=1500, on_hi=2700, short_p=0.8,
                sh=(300, 1500), lo=(2700, 4500)):
    """SQT2 式同步方波:ON ~ U[on_lo,on_hi],OFF 双峰。返回 g(t) in {0,1}。"""
    g = np.zeros(T, dtype=np.int8); t = 0
    while t < T:
        on = rng.integers(on_lo, on_hi); g[t:t+on] = 1; t += on
        if t >= T: break
        if rng.random() < short_p: off = rng.integers(*sh)
        else:                      off = rng.integers(*lo)
        t += off
    return g

phase_rho=[0.0]
def build_ci(T, period, amp, phase, base=1.0, mode="sine", g=None):
    """棕电碳强度 c(t)。amp = (max-min)/(2*base),phase 以周期为单位 [0,1)。"""
    t = np.arange(T)
    if mode == "sine":
        return base * (1.0 + amp * np.sin(2*np.pi*(t/period + phase)))
    if mode == "square":
        return base * (1.0 + amp * np.sign(np.sin(2*np.pi*(t/period + phase))))
    if mode == "mix":
        # c(t) = (1-rho) * 独立成分 + rho * 反相关成分。rho=0 完全独立,rho=1 完全跟随绿电。
        w = max(1, int(period)); k = np.ones(w) / w
        s = np.convolve(g.astype(float), k, mode="same")
        s = (s - s.mean()) / (s.std() + 1e-12)
        indep = np.sin(2*np.pi*(t/period + phase))
        rho = float(phase_rho[0])
        mixed = (1.0 - rho) * indep + rho * (-np.clip(s, -2, 2) / 2.0)
        return base * (1.0 + amp * mixed)
    if mode in ("anticorr", "corr"):
        # 现实电网:可再生出力高 -> 电网更干净。用 g 的滑动均值驱动 c。
        w = max(1, int(period)); k = np.ones(w) / w
        s = np.convolve(g.astype(float), k, mode="same")
        s = (s - s.mean()) / (s.std() + 1e-12)
        sign = -1.0 if mode == "anticorr" else +1.0
        return base * (1.0 + amp * sign * np.clip(s, -2, 2) / 2.0)
    raise ValueError(mode)

def policies(g, c, arrivals, budgets):
    """返回 (nowait, naive, clair) 三个数组:各作业的 释放时刻 与 单位MI碳。"""
    T = len(g)
    r_now = arrivals.copy()
    r_nai = np.empty_like(arrivals); r_cla = np.empty_like(arrivals)
    u_now = np.empty(len(arrivals)); u_nai = np.empty(len(arrivals)); u_cla = np.empty(len(arrivals))
    for i, (a, B) in enumerate(zip(arrivals, budgets)):
        hi = min(a + B, T - 1)
        win = np.arange(a, hi + 1)
        cost = np.where(g[win] == 1, 0.0, c[win])       # 单位 MI 碳
        # naive: 预算内第一个绿;没有则 backstop 到最晚
        gi = np.flatnonzero(g[win] == 1)
        r_nai[i] = win[gi[0]] if gi.size else hi
        # clairvoyant: 预算内碳最小(并列取最早,避免无谓延迟)
        r_cla[i] = win[int(np.argmin(cost))]
        u_now[i] = 0.0 if g[a] == 1 else c[a]
        u_nai[i] = 0.0 if g[r_nai[i]] == 1 else c[r_nai[i]]
        u_cla[i] = cost.min()
    return (r_now, u_now), (r_nai, u_nai), (r_cla, u_cla)

def one_anchor(T, seed, ci_period, ci_amp, ci_phase, ci_mode, n_jobs,
               slack_lo, slack_hi, mi_lo, mi_hi):
    rng = np.random.default_rng(seed)
    g = build_green(T, rng)
    c = build_ci(T, ci_period, ci_amp, ci_phase, mode=ci_mode, g=g)
    a = rng.integers(0, T - slack_hi - 1, size=n_jobs)
    B = rng.integers(slack_lo, slack_hi, size=n_jobs)
    mi = rng.uniform(mi_lo, mi_hi, size=n_jobs)
    (rn, un), (ra, ua), (rc, uc) = policies(g, c, a, B)
    Cn, Ca, Cc = un*mi, ua*mi, uc*mi                     # 逐作业碳
    tot = lambda x: float(x.sum())
    diff_mask = (ra != rc)
    return {
        "carbon_nowait": tot(Cn), "carbon_naive": tot(Ca), "carbon_clair": tot(Cc),
        "impr_nowait_to_naive": 100*(tot(Ca)-tot(Cn))/max(tot(Cn),1e-12),
        "impr_naive_to_clair":  100*(tot(Cc)-tot(Ca))/max(tot(Ca),1e-12),
        "mi_share_release_differs": 100*float(mi[diff_mask].sum()/mi.sum()),
        "jobs_release_differs": 100*float(diff_mask.mean()),
        "green_ratio": float(g.mean()),
        # SQT2 的判死指标:naive 已经拿到 clairvoyant 的多少
        "clair_forgone_mi_share": 100*float(mi[(Ca > Cc + 1e-12)].sum()/mi.sum()),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=72000)
    p.add_argument("--anchors", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=8000)
    p.add_argument("--slack-lo", type=int, default=200)
    p.add_argument("--slack-hi", type=int, default=3000)
    p.add_argument("--mi-lo", type=float, default=1.0)
    p.add_argument("--mi-hi", type=float, default=3.0)
    p.add_argument("--ci-period", type=int, default=8640)
    p.add_argument("--ci-amp", type=float, default=0.5)
    p.add_argument("--ci-phase", type=float, default=0.0)
    p.add_argument("--ci-mode", default="sine", choices=["sine", "square", "anticorr", "corr", "mix"])
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    phase_rho[0] = a.rho
    rows = [one_anchor(a.T, 1009*k, a.ci_period, a.ci_amp, a.ci_phase, a.ci_mode,
                       a.n_jobs, a.slack_lo, a.slack_hi, a.mi_lo, a.mi_hi)
            for k in range(a.anchors)]
    med = lambda k: float(np.median([r[k] for r in rows]))
    same = sum(1 for r in rows if r["impr_naive_to_clair"] < 0)
    print(f"锚点数={len(rows)}  绿电占比中位={med('green_ratio'):.3f}")
    print(f"  nowait→naive   碳中位改善 = {med('impr_nowait_to_naive'):+7.2f}%   (杠杆价值)")
    print(f"  naive→clair    碳中位改善 = {med('impr_naive_to_clair'):+7.2f}%   (预报内容价值)  同号 {same}/{len(rows)}")
    print(f"  释放时刻不同的 MI 占比中位 = {med('mi_share_release_differs'):.2f}%")
    print(f"  clair_forgone MI 占比中位  = {med('clair_forgone_mi_share'):.2f}%   (SQT2 恒为 0)")
    gate = (med('impr_naive_to_clair') <= -10.0) and same >= 8
    print(f"  预注册门槛(中位≤-10% 且 ≥8/10 同号): {'✅ 过' if gate else '❌ 不过'}")
    if a.json:
        json.dump({"params": vars(a), "rows": rows,
                   "median_impr_naive_to_clair": med('impr_naive_to_clair'),
                   "same_sign": same, "gate_pass": gate}, open(a.json, "w"), indent=1)

if __name__ == "__main__":
    main()
