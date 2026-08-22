#!/usr/bin/env python3
"""断口 (A):块状性。作业少而大(尺寸~窗口尺寸)时,流体近似失效,信息价值应回升。

复用 stage15 的全部机器(连续作业、流体计分、因果协调盲、贪心插入 clair)。
预测:信息差随 [每块作业数 N 下降] 与 [runtime 接近窗口尺度] 增大;
N 大或 runtime 远离窗口尺度(过小/过大)时回到 ~2% 的流体死区。
额外红利:N 小时贪心插入 clair 几乎无相互作用,启发式差距小,数字更可信。
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from stage15_continuous import (ROW_S, W_PER_PE, coordinated_blind_contig,
                                coordinated_clair_contig, fluid_carbon,
                                load_watts)

def make_lumpy_jobs(n_blocks, per_block, rt_s, slack_s, year_s, seed, pes=2):
    rng = np.random.default_rng(seed)
    offs = np.sort(rng.integers(0, int(year_s - 60 * 3600), n_blocks))
    jobs = []
    for off in offs:
        for da in rng.uniform(0, 24 * 3600.0, size=per_block):
            mi = rt_s * pes * 40000.0
            jobs.append((mi, pes, float(rt_s), float(off) + float(da),
                         float(slack_s), 0))
    return jobs


def cell(w_eval, jobs, scale):
    rel_n = [j[3] for j in jobs]
    rb = coordinated_blind_contig(jobs, w_eval, scale)
    rc = coordinated_clair_contig(jobs, w_eval, scale)
    c_n = fluid_carbon(jobs, rel_n, w_eval, scale)[0]
    c_b = fluid_carbon(jobs, rb, w_eval, scale)[0]
    c_c = fluid_carbon(jobs, rc, w_eval, scale)[0]
    return c_n, c_b, c_c


if __name__ == "__main__":
    w20 = load_watts((100, 101), 2020)
    w21 = load_watts((100, 101), 2021)
    year_s = len(w21) * ROW_S
    print(f"{'N/块':>6}{'runtime':>9}{'blind vs now':>14}{'clair vs now':>14}"
          f"{'★信息差':>10}")
    for per_block in (2, 5, 15, 50):
        for rt_h in (2, 4, 8):
            jobs = make_lumpy_jobs(40, per_block, rt_h * 3600.0, 15 * 3600.0,
                                   year_s, 20260822)
            # ρ=0.5(绿电富余区,上一轮信息价值的最优区)
            D_mean = sum(j[1] * W_PER_PE * j[2] for j in jobs) / year_s
            scale = D_mean / 0.5 / w20.mean()
            c_n, c_b, c_c = cell(w21, jobs, scale)
            print(f"{per_block:>6}{rt_h:>8}h{100*(c_b-c_n)/c_n:>13.2f}%"
                  f"{100*(c_c-c_n)/c_n:>13.2f}%{100*(c_c-c_b)/c_b:>9.2f}%")
