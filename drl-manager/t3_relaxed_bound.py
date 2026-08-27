#!/usr/bin/env python3
"""T3 B 部分:松弛下界(工单 1f2f20a / 预注册 T3_HEADROOM_PREREG)。

    carbon_relaxed = Σ_j  min_{dc, s 可行}  carbon_j(dc, s)

每个作业**独立**取其可行窗口内碳最低的 (DC, 起始时刻),**忽略容量与争用**
⇒ 乐观 ⇒ **真下界**。命名纪律:这是"松弛下界",**不是**"可省量"。

标定(实测,非猜测):
- 行对齐 = offset + **13**(与工单 warmup 13 行独立吻合)
- 绿电缩放 **k = 2/3 W per CSV-kW**
  两者由观测序列(400 步 × 3 DC)与 CSV 互相关求得,残差 RMSE 1.18e-5 W、
  三 DC 相关系数均 1.000000。
- 1 步 = 1 绿电行(time_scaling_mode COMPRESSED)

功率模型(与仿真一致):RS500A 64 核 / 214 W 峰值 / 24% 静态 ⇒ 动态跨度
162.64 W;作业边际功率 P_j = 162.64 × pes_j / 64。
**注意**:本下界只覆盖**可调度的边际能量**;仿真总能量里 57–65% 是静态/闲置,
调度器动不了。因此 (盲−下界)/盲 是可省量的**上限**,且偏松。
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent
WIND = REPO.parent / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
TRACE = REPO.parent / "cloudsimplus-gateway/src/main/resources/traces/probe_C_2xjob_dl6500.csv"

DC_TURBINES = {0: [12, 36], 1: [95, 91], 2: [96], 3: [], 4: []}
DC_TZ = {0: 0, 1: 18, 2: 54, 3: 72, 4: 108}
BROWN = {0: 0.08, 1: 0.35, 2: 0.55, 3: 0.75, 4: 0.92}     # kg/kWh
GREEN_FACTOR = 0.01                                        # kg/kWh
GREEN_SCALE_W_PER_KW = 2.0 / 3.0                           # 实测标定
WARMUP_ROWS = 13                                           # 实测标定
HOST_PES, HOST_DYNAMIC_W = 64, 214.0 - 214.0 * 0.24        # 162.64 W
VM_PE_MIPS = 40000.0
EPISODE_STEPS = 7200
WINDOWS = {"low": 19171, "mid": 11554, "high": 34306}


def load_green(year=2021):
    ser = {}
    for t in {t for v in DC_TURBINES.values() for t in v}:
        with open(WIND / f"Turbine_{t}_{year}.csv") as f:
            ser[t] = np.array([float(r["power_kw"]) for r in csv.DictReader(f)])
    n = min(len(a) for a in ser.values())
    return ser, n


def dc_green_w(ser, n, dc, base_row, steps=EPISODE_STEPS):
    """DC 在 episode 各步的绿电功率(W)。无涡轮的 DC 恒零。"""
    if not DC_TURBINES[dc]:
        return np.zeros(steps)
    idx = (np.arange(base_row, base_row + steps) + DC_TZ[dc]) % n
    kw = sum(ser[t][idx] for t in DC_TURBINES[dc])
    return kw * GREEN_SCALE_W_PER_KW


def load_jobs():
    with open(TRACE) as f:
        rows = list(csv.DictReader(f))
    arr = np.array([float(r["arrival_time"]) for r in rows])
    ln = np.array([float(r["length"]) for r in rows])
    pes = np.array([float(r["pes_required"]) for r in rows])
    dl = np.array([float(r["deadline"]) for r in rows])
    rt = np.ceil(ln / VM_PE_MIPS).astype(int)              # 步 = 行
    P = HOST_DYNAMIC_W * pes / HOST_PES                     # 边际功率 W
    return arr.astype(int), rt, P, dl.astype(int)


def relaxed_carbon_for_window(ser, n, base_row, arr, rt, P, dl):
    """逐作业取 (dc, s) 上碳最小值之和。前缀和 O(1) 查窗。"""
    steps = EPISODE_STEPS
    green = {d: dc_green_w(ser, n, d, base_row, steps) for d in DC_TURBINES}
    plevels = np.unique(P)
    # cum[d][p][t] = 前 t 步的碳(kg),给定恒定边际功率 p
    cum = {}
    for d in DC_TURBINES:
        g = green[d]
        cum[d] = {}
        for p in plevels:
            brown_w = np.maximum(0.0, p - g)
            green_w = np.minimum(p, g)
            per_step_kg = (brown_w * BROWN[d] + green_w * GREEN_FACTOR) / 3600.0 / 1000.0
            cum[d][float(p)] = np.concatenate([[0.0], np.cumsum(per_step_kg)])
    total, chosen_dc, infeasible = 0.0, np.zeros(len(arr), dtype=int), 0
    for j in range(len(arr)):
        R = int(rt[j])
        lo = int(arr[j])
        hi = min(int(dl[j]) - R, steps - R)                 # 含端点
        if hi < lo:
            infeasible += 1
            hi = lo = min(lo, steps - R)
            if lo < 0:
                continue
        s = np.arange(lo, hi + 1)
        best, best_d = None, -1
        for d in DC_TURBINES:
            c = cum[d][float(P[j])]
            vals = c[s + R] - c[s]
            m = float(vals.min())
            if best is None or m < best:
                best, best_d = m, d
        total += best
        chosen_dc[j] = best_d
    return total, chosen_dc, infeasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="../local_eval_rt/audit/t3_relaxed_bound.json")
    a = ap.parse_args()
    ser, n = load_green()
    arr, rt, P, dl = load_jobs()
    print(f"[T3B] 作业 {len(arr)}  Σ边际能量/窗 = {(P*rt).sum()/3600:.2f} Wh", flush=True)
    out = {"scale_w_per_kw": GREEN_SCALE_W_PER_KW, "warmup_rows": WARMUP_ROWS,
           "windows": {}, "note": "松弛下界,非可省量;仅覆盖可调度边际能量"}
    tot = 0.0
    for name, off in WINDOWS.items():
        c, chosen, infeas = relaxed_carbon_for_window(
            ser, n, off + WARMUP_ROWS, arr, rt, P, dl)
        tot += c
        share = {f"dc{d}": int((chosen == d).sum()) for d in DC_TURBINES}
        out["windows"][name] = {"offset": off, "relaxed_carbon_kg": c,
                                "chosen_dc_counts": share, "infeasible": infeas}
        print(f"[T3B] {name:>5} (offset {off}): 松弛下界 = {c:.6f} kg   "
              f"选中 DC 分布 {share}  不可行 {infeas}", flush=True)
    out["relaxed_total_3windows_kg"] = tot
    print(f"[T3B] 三窗合计松弛下界 = {tot:.6f} kg", flush=True)
    pathlib.Path(a.json_out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("T3B DONE", flush=True)


if __name__ == "__main__":
    main()
