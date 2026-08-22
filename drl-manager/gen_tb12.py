#!/usr/bin/env python3
"""第 12 考场场景生成器 v2 —— REAL_TIME 口径(Codex 2026-08-22 审定)。

v1(COMPRESSED + 60s 重采样)已废:它把 1 行压成 1 仿真秒,风窗尺度与离线
模型错位,且用 2021 均值定标违反 2020 冻结纪律。v2 规则:

  time_scaling_mode REAL_TIME     1 行 = 600s,原生 SDWPF 文件直读,零重采样
  simulation_timestep 600         48h episode = 288 步
  green_power_scale 9.05562658195e-5
      仅由 T100+101/2020 标定(mean 529.136954 kW -> ρ=0.5 下 Ḡ=47.9167W),
      永久冻结;任何 held-out 组不得重标,只报告实际 ρ
  MI = rt x 40000                 CloudSim length 为每-PE 长度(P0-3 实测)
  deadline = arrival + slack + runtime + 120   (latest-start 语义)
  horizon 48h                     24h 到达 + 15h slack + 4h rt + 余量,无截断逃单
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
_T = _REPO / "cloudsimplus-gateway/src/main/resources/traces"
TIMESTEP = 600.0
ROW_S = 600.0
GREEN_POWER_SCALE = 9.05562658195e-5     # T100+101/2020 冻结,勿重算
P_JOB_W = 28.75
MIPS = 40000.0
MARGIN_S = 120.0


def gen_trace(name, per_block, rt_s, slack_s, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i, da in enumerate(np.sort(rng.uniform(0, 24 * 3600.0, per_block))):
        arr = int(da)
        mi = int(rt_s * MIPS)                        # 每-PE 长度语义
        dl = int(arr + slack_s + rt_s + MARGIN_S)    # latest-start 语义
        rows.append((i, arr, mi, 2, 1000, 500, dl))
    with open(_T / name, "w") as f:
        f.write("cloudlet_id,arrival_time,length,pes_required,"
                "file_size,output_size,deadline\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    return rows


def build_config(base_cfg, trace, turbines, year, ep_steps, offset_range):
    import copy
    b = copy.deepcopy(base_cfg)
    dc0 = copy.deepcopy(b["datacenters"][0])
    dc0["turbine_ids"] = list(turbines)
    dc0["time_scaling_mode"] = "REAL_TIME"
    b["datacenters"] = [dc0]
    b["cloudlet_trace_file"] = f"traces/{trace}"
    b["simulation_timestep"] = TIMESTEP
    b["max_episode_length"] = ep_steps
    b["green_power_scale"] = GREEN_POWER_SCALE
    b["csv_year"] = year
    b["green_episode_offset_range"] = offset_range
    b["experiment_name"] = "tb12_smoke"
    b["simulation_name"] = "TB12_smoke"
    b["preflight_temporal_profile"] = "tb12_lumpy_v0"
    b["obs_cloudlet_mi_high"] = int(2e9)
    b["wandb"] = {"enabled": False}
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbines", default="100,101")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--per-block", type=int, default=5)
    ap.add_argument("--rt-h", type=float, default=4.0)
    ap.add_argument("--slack-h", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=20260822)
    a = ap.parse_args()

    turbines = tuple(int(x) for x in a.turbines.split(","))
    rt_s, slack_s = a.rt_h * 3600.0, a.slack_h * 3600.0
    trace = f"tb12_n{a.per_block}_rt{int(a.rt_h)}h.csv"
    jobs = gen_trace(trace, a.per_block, rt_s, slack_s, a.seed)

    ep_steps = int(48 * 3600 / TIMESTEP)             # 288
    n_rows = 52560                                   # 原生 SDWPF 年行数
    offset_range = n_rows - ep_steps - 10

    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from src.baselines.evaluate import load_config
    base = load_config("experiment_gwo1_noforecast")
    blk = build_config(base, trace, turbines, a.year, ep_steps, offset_range)

    cfg_path = _REPO / "config_C.yml"
    if "experiment_tb12_smoke:" in cfg_path.read_text():
        raise SystemExit("experiment_tb12_smoke 已存在,先删再生成(不覆盖)")
    with open(cfg_path, "a") as f:
        f.write("\n# tb12 v2 REAL_TIME(gen_tb12.py 生成,勿手改)\n")
        f.write(yaml.safe_dump({"experiment_tb12_smoke": blk},
                               default_flow_style=False, sort_keys=True,
                               allow_unicode=True, width=4096))
    art = {"scenario": "tb12_v2", "seed": a.seed, "per_block": a.per_block,
           "rt_h": a.rt_h, "slack_h": a.slack_h, "p_job_w": P_JOB_W,
           "green_power_scale": GREEN_POWER_SCALE,
           "scale_calibration": "T100+101/2020 mean 529.136954 kW, rho=0.5, FROZEN",
           "timestep": TIMESTEP, "ep_steps": ep_steps,
           "offset_range": offset_range, "turbines": list(turbines),
           "year": a.year, "mi_rule": "rt*40000 per-PE",
           "deadline_rule": "arrival+slack+runtime+120", "jobs": jobs}
    (pathlib.Path(__file__).resolve().parent / "calib/tb12_v2.json"
     ).write_text(json.dumps(art, indent=1))
    print(f"trace={trace} ep_steps={ep_steps} offset_range={offset_range} "
          f"scale={GREEN_POWER_SCALE}")


if __name__ == "__main__":
    main()
