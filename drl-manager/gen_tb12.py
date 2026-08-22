#!/usr/bin/env python3
"""第 12 考场(块状作业 x 真实风电)场景生成器 —— 烟测版。

三个从 P0 探针学来的硬规则:
  1. MI = rt x 40000(CloudSim length 是每-PE 长度,不除 PES)
  2. p_job 实测 28.75 W/作业(host 静态 23.67 支配);ρ 按它定标
  3. 仿真步长 60s -> 涡轮 CSV 必须重采样成 60s 一行(sim 一行=一步)

产物:重采样涡轮 CSV(tb12 专属 id 9100/9101,不覆盖任何现有文件)、
trace、config block、artifact。
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_W = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
_T = _REPO / "cloudsimplus-gateway/src/main/resources/traces"
TIMESTEP = 60.0
P_JOB_W = 28.75              # P0-3 实测
RHO = 0.5                    # 预注册
MIPS = 40000.0


def resample_turbine(src_id, dst_id, year):
    rows = list(csv.DictReader(open(_W / f"Turbine_{src_id}_{year}.csv")))
    out = [f"timestamp,power_kw"]
    for r in rows:                       # 10min 一行 -> 60s 十行(重复)
        for _ in range(10):
            out.append(f"{r['timestamp']},{float(r['power_kw']):.3f}")
    (_W / f"Turbine_{dst_id}_{year}.csv").write_text("\n".join(out) + "\n")
    return len(out) - 1


def gen_trace(name, per_block, rt_s, slack_s, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i, da in enumerate(np.sort(rng.uniform(0, 24 * 3600.0, per_block))):
        arr = int(da)
        mi = int(rt_s * MIPS)            # 每-PE 长度语义
        rows.append((i, arr, mi, 2, 1000, 500, int(arr + rt_s + slack_s)))
    with open(_T / name, "w") as f:
        f.write("cloudlet_id,arrival_time,length,pes_required,file_size,output_size,deadline\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    return rows


def build_config(base_cfg, trace, wind_rows, divisor, ep_steps, offset_range):
    import copy
    b = copy.deepcopy(base_cfg)
    dc0 = copy.deepcopy(b["datacenters"][0])
    dc0["turbine_ids"] = [9100, 9101]
    b["datacenters"] = [dc0]
    b["cloudlet_trace_file"] = f"traces/{trace}"
    b["simulation_timestep"] = TIMESTEP
    b["max_episode_length"] = ep_steps
    b["compressed_power_divisor"] = float(divisor)
    b["green_episode_offset_range"] = offset_range
    b["experiment_name"] = "tb12_smoke"
    b["simulation_name"] = "TB12_smoke"
    b["preflight_temporal_profile"] = "tb12_lumpy_v0"
    b["obs_cloudlet_mi_high"] = int(2e9)
    b["wandb"] = {"enabled": False}
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--per-block", type=int, default=5)
    ap.add_argument("--rt-h", type=float, default=4.0)
    ap.add_argument("--slack-h", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    n1 = resample_turbine(100, 9100, a.year)
    n2 = resample_turbine(101, 9101, a.year)
    rt_s, slack_s = a.rt_h * 3600.0, a.slack_h * 3600.0
    trace = f"tb12_smoke_n{a.per_block}.csv"
    jobs = gen_trace(trace, a.per_block, rt_s, slack_s, a.seed)

    # ρ 定标:mean demand(到达日口径) / mean green = ρ
    kw = []
    for tid in (9100, 9101):
        rs = list(csv.DictReader(open(_W / f"Turbine_{tid}_{a.year}.csv")))
        kw.append(np.array([float(r["power_kw"]) for r in rs]))
    mean_w_raw = (kw[0][:min(map(len, kw))] + kw[1][:min(map(len, kw))]).mean() * 1000.0
    d_mean = a.per_block * rt_s * P_JOB_W / (24 * 3600.0)
    divisor = mean_w_raw / (d_mean / RHO)

    ep_steps = int((24 * 3600 + slack_s + rt_s + 3600) / TIMESTEP)   # 到达+slack+rt+drain
    offset_range = max(0, min(n1, n2) - ep_steps - 10)

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from src.baselines.evaluate import load_config
    base = load_config("experiment_gwo1_noforecast")
    blk = build_config(base, trace, min(n1, n2), divisor, ep_steps, offset_range)

    cfg_path = _REPO / "config_C.yml"
    txt = cfg_path.read_text()
    if "experiment_tb12_smoke:" in txt:
        raise SystemExit("experiment_tb12_smoke 已存在,先删再生成(不覆盖)")
    with open(cfg_path, "a") as f:
        f.write("\n\n# tb12 烟测(gen_tb12.py 生成,勿手改)\n")
        f.write(yaml.safe_dump({"experiment_tb12_smoke": blk},
                               default_flow_style=False, sort_keys=True,
                               allow_unicode=True, width=4096))
    art = {"scenario": "tb12_smoke", "seed": a.seed, "per_block": a.per_block,
           "rt_h": a.rt_h, "slack_h": a.slack_h, "p_job_w": P_JOB_W,
           "rho": RHO, "divisor": divisor, "timestep": TIMESTEP,
           "ep_steps": ep_steps, "offset_range": offset_range,
           "turbines_src": [100, 101], "turbines_dst": [9100, 9101],
           "year": a.year, "mean_green_w_raw": mean_w_raw,
           "demand_mean_w": d_mean, "mi_rule": "rt*40000 (per-PE length)",
           "jobs": jobs}
    (pathlib.Path(__file__).resolve().parent / "calib/tb12_smoke.json"
     ).write_text(json.dumps(art, indent=1))
    print(f"涡轮重采样 {n1}/{n2} 行;trace {a.per_block} 作业;divisor={divisor:.1f}")
    print(f"ep_steps={ep_steps} (44h @ 60s)  offset_range={offset_range}")


if __name__ == "__main__":
    main()
