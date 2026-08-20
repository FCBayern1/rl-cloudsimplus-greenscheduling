#!/usr/bin/env python3
"""最优盲策略的动态规划 —— 用逆向归纳算出真实信息价值 VoI。

停止手搓盲臂:四轮下来每加一个盲臂优势就削一层(−16.43 → −11.78),没有终点。
绿电是已注册的马尔可夫更新过程(ON/OFF 时长分布已知),c(t) 是确定性的,
所以"只用免费信息的最优策略"可以精确算出来:

  状态  (t, s∈{绿,棕}, 当前状态已持续 age, 剩余预算 b)   作业 runtime L 为参数
  动作  现在放行 / 再等一步
  V(t,s,age,b) = min( 放行的期望碳, E[V(t+1, s', age', b−1)] )
  V(t,s,age,0) = 放行的期望碳                      (预算耗尽,强制放行)

放行的期望碳用【已注册的时长分布】对未来绿电求期望,用【确定性 c(t)】的真值:
  E[carbon | t,s,age] = (MI/L) · Σ_{k<L} (1 − f(s,age,k)) · c(t+k)
其中 f(s,age,k) = P(k 步后为绿 | 当前 s 已持续 age),由更新过程正向递推得到。

得到的 dp_blind 是信息论上界:任何手搓盲臂都不可能更好。
    VoI = clairvoyant − dp_blind
不再是"我们还没想到更好的盲臂"的残差。
"""
import argparse
import numpy as np

# 与 offline_proof_tvci3.build_green 一致的已注册分布(单位:仿真步)
ON_LO, ON_HI = 1500, 2700
SHORT_P, SH_LO, SH_HI = 0.8, 300, 1500
LONG_LO, LONG_HI = 2700, 4500


def survival(pmf):
    """S[a] = P(时长 > a)。"""
    return np.concatenate([[1.0], 1.0 - np.cumsum(pmf)])


def duration_pmfs(dt):
    """离散到 DP 时间单位 dt,返回 (ON pmf, OFF pmf)。"""
    amax = int(np.ceil(LONG_HI / dt)) + 2
    on = np.zeros(amax); off = np.zeros(amax)
    for v in range(ON_LO, ON_HI):
        on[min(v // dt, amax - 1)] += 1.0 / (ON_HI - ON_LO)
    for v in range(SH_LO, SH_HI):
        off[min(v // dt, amax - 1)] += SHORT_P / (SH_HI - SH_LO)
    for v in range(LONG_LO, LONG_HI):
        off[min(v // dt, amax - 1)] += (1 - SHORT_P) / (LONG_HI - LONG_LO)
    return on, off


def green_prob_table(dt, kmax):
    """f[s, age, k] = P(k 个 DP 步后为绿 | 当前状态 s 已持续 age)。"""
    on_pmf, off_pmf = duration_pmfs(dt)
    S_on, S_off = survival(on_pmf), survival(off_pmf)
    A = len(on_pmf)
    # 危险率:h[a] = P(再持续一步 | 已持续 a)
    h_on = np.where(S_on[:-1] > 0, S_on[1:] / np.maximum(S_on[:-1], 1e-300), 0.0)
    h_off = np.where(S_off[:-1] > 0, S_off[1:] / np.maximum(S_off[:-1], 1e-300), 0.0)
    n = 2 * A                                  # 状态编码 idx = s*A + age
    P = np.zeros((n, n))
    for a in range(A - 1):
        P[0 * A + a, 0 * A + a + 1] = h_off[a]      # 棕 -> 棕(变老)
        P[0 * A + a, 1 * A + 0] = 1 - h_off[a]      # 棕 -> 绿(重置)
        P[1 * A + a, 1 * A + a + 1] = h_on[a]       # 绿 -> 绿
        P[1 * A + a, 0 * A + 0] = 1 - h_on[a]       # 绿 -> 棕
    P[0 * A + A - 1, 1 * A + 0] = 1.0
    P[1 * A + A - 1, 0 * A + 0] = 1.0
    is_green = np.zeros(n); is_green[A:] = 1.0
    f = np.zeros((n, kmax + 1)); dist = np.eye(n)
    for k in range(kmax + 1):
        f[:, k] = dist @ is_green
        dist = dist @ P
    return f.reshape(2, A, kmax + 1), A


def solve_dp(c, dt, L_units, budget_units, f, A):
    """逆向归纳。c 已按 dt 聚合(每个 DP 步的平均碳强度)。返回 policy[t, s, age, b]。"""
    T = len(c)
    kmax = f.shape[2] - 1
    Lk = min(L_units, kmax)
    # 放行的期望碳(不含 MI/L 常数,单调性不受影响)
    rel = np.zeros((T, 2, A))
    brown = 1.0 - f[:, :, :Lk]                       # (2, A, Lk)
    cpad = np.concatenate([c, np.repeat(c[-1], Lk + 1)])
    for t in range(T):
        rel[t] = brown @ cpad[t:t + Lk]
    B = budget_units
    V = np.empty((2, A, B + 1)); pol = np.zeros((T, 2, A, B + 1), dtype=bool)
    V[:] = rel[T - 1][:, :, None]
    for t in range(T - 2, -1, -1):
        Vn = V.copy()
        # 期望的"再等一步"价值:按转移聚合下一状态
        cont = np.empty_like(Vn)
        on_pmf, off_pmf = duration_pmfs(dt)
        S_on, S_off = survival(on_pmf), survival(off_pmf)
        h_on = np.where(S_on[:-1] > 0, S_on[1:] / np.maximum(S_on[:-1], 1e-300), 0.0)
        h_off = np.where(S_off[:-1] > 0, S_off[1:] / np.maximum(S_off[:-1], 1e-300), 0.0)
        nx = np.arange(1, A + 1).clip(max=A - 1)
        cont[0] = h_off[:, None] * Vn[0][nx] + (1 - h_off)[:, None] * Vn[1][0][None, :]
        cont[1] = h_on[:, None] * Vn[1][nx] + (1 - h_on)[:, None] * Vn[0][0][None, :]
        rl = rel[t][:, :, None]
        Vw = np.concatenate([rl, cont[:, :, :B]], axis=2)   # b=0 强制放行
        V = np.minimum(rl, Vw)
        pol[t] = rl <= Vw
        V[:, :, 0] = rel[t]                                # 预算耗尽
        pol[t][:, :, 0] = True
    return pol


def rollout(pol, g, c, dt, a, B, L, mi, A):
    """在【真实实现的绿电轨迹】上跟随 DP 策略,返回释放时刻。"""
    T = len(g)
    gage = np.zeros(T, dtype=np.int64); run = 0
    for t in range(T):
        run = run + 1 if g[t] == 1 else 0
        gage[t] = run - 1
    bage = np.zeros(T, dtype=np.int64); run = 0
    for t in range(T):
        run = run + 1 if g[t] == 0 else 0
        bage[t] = run - 1
    Tn = pol.shape[0]; Bn = pol.shape[3] - 1
    out = np.empty(len(a), dtype=np.int64)
    # 价值函数是粗粒度的(dt 步一格),但【决策时机是逐步的】——
    # 否则会错过绿窗起点最多 dt-1 步,那是离散化 handicap 而非策略本身的限制。
    for i in range(len(a)):
        lo, hi = int(a[i]), int(min(a[i] + B[i], T - L[i] - 1))
        win = np.arange(lo, hi + 1)
        sv = g[win].astype(np.int64)
        age = np.where(sv == 1, gage[win], bage[win]) // dt
        # 预算格向【上】取整:不足一格的余量仍是可用预算。向下取整会在 hi-dt+1 处
        # 误判为"预算耗尽"而强制放行(实测让 DP 提前 59 步释放,是它输给 onset_wait 的主因)。
        bb = np.clip(-(-(hi - win) // dt), 0, Bn)
        dec = pol[np.clip(win // dt, 0, Tn - 1), sv, np.clip(age, 0, A - 1), bb]
        out[i] = win[int(np.argmax(dec))] if dec.any() else hi
    return out


def main():
    import offline_proof_tvci3 as tv
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=72000); p.add_argument("--anchors", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=2000); p.add_argument("--dt", type=int, default=60)
    p.add_argument("--ci-period", type=int, default=7080); p.add_argument("--ci-amp", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--buckets", default="150,300,450,600,750,900")
    a_ = p.parse_args()
    dt = a_.dt; Bu = 3000 // dt
    L_BUCKETS = [int(x) for x in a_.buckets.split(',')]
    print(f"DP: dt={dt}s  时间格={a_.T//dt}  预算格={Bu}  runtime 档={L_BUCKETS}")
    tot = {k: 0.0 for k in ("dp_blind", "clair", "combo", "onset_wait", "green_age",
                            "naive_green", "onset_causal")}
    for k_ in range(a_.anchors):
        rng = np.random.default_rng(1009 * k_)
        g = tv.build_green(a_.T, rng); c = tv.build_ci(a_.T, a_.ci_period, a_.ci_amp, g, a_.rho)
        cost = np.where(g == 1, 0.0, c)
        csum = np.concatenate([[0.0], np.cumsum(cost)])
        cd = c[: (a_.T // dt) * dt].reshape(-1, dt).mean(1)          # 聚合到 DP 步
        f, A = green_prob_table(dt, max(L_BUCKETS) // dt + 2)
        arr = rng.integers(0, a_.T - 3000 - 1000 - 2, size=a_.n_jobs)
        Bg = rng.integers(200, 3000, size=a_.n_jobs)
        L = rng.integers(100, 1000, size=a_.n_jobs); mi = L.astype(float)
        r_dp = np.full(a_.n_jobs, -1, dtype=np.int64)
        # 每个作业分配到【最近】的桶,保证全覆盖。此前用 |L-Lb|<=tol 会漏掉区间外的
        # 作业,它们的 r_dp 保持 np.empty 的未初始化值 —— 静默污染结果。
        bidx = np.argmin(np.abs(L[:, None] - np.array(L_BUCKETS)[None, :]), axis=1)
        for j, Lb in enumerate(L_BUCKETS):
            m = bidx == j
            if not m.any(): continue
            pol = solve_dp(cd, dt, Lb // dt, Bu, f, A)
            r_dp[m] = rollout(pol, g, c, dt, arr[m], Bg[m], L[m], mi[m], A)
        assert (r_dp >= 0).all(), "有作业未被任何 runtime 桶覆盖"
        cb = lambda r: float(((csum[np.clip(r+L,0,len(csum)-1)] - csum[np.clip(r,0,len(csum)-1)]) * (mi/L)).sum())
        tot["dp_blind"] += cb(r_dp)
        # 因果版 onset:逐步走,只用当下可观测的绿窗起点;到 hi 强制放行。
        onset = np.zeros(a_.T, bool); onset[0] = g[0] == 1
        onset[1:] = (g[1:] == 1) & (g[:-1] == 0)
        r_oc = np.empty(a_.n_jobs, dtype=np.int64)
        for i in range(a_.n_jobs):
            lo_, hi_ = int(arr[i]), int(min(arr[i] + Bg[i], a_.T - L[i] - 1))
            w = np.arange(lo_, hi_ + 1); ow = np.flatnonzero(onset[w])
            r_oc[i] = w[ow[0]] if ow.size else hi_
        tot["onset_causal"] += cb(r_oc)
        res = tv.anchor(1009*k_, a_.T, a_.n_jobs, (200,3000), (100,1000), a_.ci_period,
                        a_.ci_amp, a_.rho, 0.30, [10**9])
        for k in ("clair","combo","onset_wait","green_age","naive_green"):
            tot[k] += res["K"][10**9][k]["carbon"]
    print("\n各臂总碳(越低越强):")
    for k in sorted(tot, key=lambda k: tot[k]):
        print(f"  {k:12} {tot[k]/1e6:8.3f}")
    voi = 100*(tot["clair"]-tot["dp_blind"])/tot["dp_blind"]
    print(f"\n★ VoI = clairvoyant vs dp_blind = {voi:+.2f}%   门槛 <=-5%  {'✅' if voi<=-5 else '❌'}")
    # 正确性检查只对【因果】臂成立。offline_proof_tvci3 里的 onset_wait/green_age/combo
    # 用整窗扫描挑 fallback 分支 = 偷看未来,DP 输给它们不构成 DP 有误(5080 已确认并
    # 用 tests/test_blind_arms_are_causal.py 固化了这条判据)。
    ok = tot["dp_blind"] <= tot["onset_causal"] * 1.001
    print(f"  正确性自检 dp_blind <= onset_causal(因果版): "
          f"{'✅ 通过' if ok else '❌ DP 实现有误'}"
          f"   ({tot['dp_blind']/1e6:.3f} vs {tot['onset_causal']/1e6:.3f})")
    print(f"    参考(偷看版,不作判据): onset_wait={tot['onset_wait']/1e6:.3f} "
          f"combo={tot['combo']/1e6:.3f}")


if __name__ == "__main__":
    main()
