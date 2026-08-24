#!/usr/bin/env python3
"""EU-CRD 训练早期止损监视(5080 c61b615 的三判据,对应他们 v4 的三个静默灾难)。

用法: tb12_crd_stoploss.py <progress.csv> [--min-iters 8]
配置正确 != 通道有信号:v4 时配置全对、代码全走到、零报错,但
Δr 恒零 / quarantine 惰性 / gate 退化 —— 跑完整批才发现。这三条在
前几十次迭代就能止损。
"""
import argparse
import csv
import sys


def pick(cols, *cands):
    for c in cands:
        for k in cols:
            if c in k:
                return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("progress")
    ap.add_argument("--min-iters", type=int, default=8)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.progress)))
    if len(rows) < a.min_iters:
        print(f"仅 {len(rows)} 次迭代(<{a.min_iters}),继续观察")
        return
    cols = rows[0].keys()
    dr = pick(cols, "crd_dr")
    rho = pick(cols, "crd_rho_routing")
    ct = pick(cols, "crd_c_t")
    fails = []

    def vals(k):
        return [float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")]

    if dr:
        v = vals(dr)
        mean = sum(v)/len(v)
        std = (sum((x-mean)**2 for x in v)/len(v))**0.5
        dead = abs(mean) < 1e-12 and std < 1e-12
        print(f"[{'STOP' if dead else 'ok  '}] crd_dr 均值={mean:.3e} 标准差={std:.3e}"
              + ("   <- Δr 通道死(v4 BUG2):通道名字在,里面不是那个量" if dead else ""))
        if dead: fails.append("crd_dr恒零")
    else:
        print("[warn] progress.csv 无 crd_dr 列 —— EU-CRD 通道未导出,本身就是问题")
        fails.append("crd_dr列缺失")
    if rho:
        v = vals(rho)
        pinned = all(x > 0.985 for x in v)
        print(f"[{'STOP' if pinned else 'ok  '}] crd_rho_routing 范围=[{min(v):.3f},{max(v):.3f}]"
              + ("   <- 份额量纲病,quarantine 失效(v4 BUG1)" if pinned else ""))
        if pinned: fails.append("rho贴0.99")
    if ct:
        v = vals(ct)
        lo = all(x < 0.02 for x in v); hi = all(x > 0.98 for x in v)
        print(f"[{'STOP' if (lo or hi) else 'ok  '}] crd_c_t 范围=[{min(v):.3f},{max(v):.3f}]"
              + ("   <- gate 退化成单通道" if (lo or hi) else ""))
        if lo or hi: fails.append("c_t退化")
    if fails:
        print(f"\nSTOP-LOSS 触发: {fails} —— 按 5080 判据立即停训检查,不要跑完整批")
        sys.exit(1)
    print("\n三判据通过,通道有信号")


if __name__ == "__main__":
    main()
