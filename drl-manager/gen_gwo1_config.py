#!/usr/bin/env python3
"""gwo1（第十考场）的四个 experiment block —— 由 SQT2 块克隆 + 定点改写。

克隆而不是手写，理由和 SQT2.2 一样：两个场景必须共享同一套奖励/观测/
Lagrangian 参数，否则 gwo1 的结论无法归因给"决策域 + trace"这两个唯一
的自变量。任何手抄都会引入不受控的第三个变量。

改写的键（且只有这些）：
    block key                 experiment_sqt2*  -> experiment_gwo1*
    cloudlet_trace_file       -> traces/gwo1{ho}_n1200_x130.csv
    datacenters[*].turbine_ids-> +200（cal: 95xx->97xx；ho: 96xx->98xx）
    experiment_name           -> gwo1{ho}_{oracle,noforecast}
    simulation_name           -> GWO1_<experiment_name>
    preflight_temporal_profile-> gwo1_trough_v1

turbine 偏移 +200 是 gen_sqt2.py VARIANTS 注册的值（cal=0/ho=100/
gwo1=200/gwo1ho=300），从 sqt2 块出发一律 +200 即可命中。
"""
import argparse
import copy
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
TURBINE_OFFSET = 200
RUNTIME_SCALE = 1.30        # 预注册 scale（暴露带 [0.20,0.50] 中部）
SCALE_TAG = "x130"
PROFILE = "gwo1_trough_v1"

# (源 block, 目标 block, trace 前缀)
PLAN = [("experiment_sqt2_oracle",     "experiment_gwo1_oracle",     "gwo1"),
        ("experiment_sqt2_noforecast", "experiment_gwo1_noforecast", "gwo1"),
        ("experiment_sqt2ho_oracle",     "experiment_gwo1ho_oracle",     "gwo1ho"),
        ("experiment_sqt2ho_noforecast", "experiment_gwo1ho_noforecast", "gwo1ho")]


def derive(src_block: dict, dst_key: str, trace_prefix: str) -> dict:
    b = copy.deepcopy(src_block)
    name = dst_key[len("experiment_"):]          # gwo1_oracle / gwo1ho_noforecast
    b["cloudlet_trace_file"] = f"traces/{trace_prefix}_n1200_{SCALE_TAG}.csv"
    b["experiment_name"] = name
    b["simulation_name"] = f"GWO1_{name}"
    b["preflight_temporal_profile"] = PROFILE
    # MI 随 runtime 同比放大，观测上界必须跟着放大，否则 max MI 会被截断
    # （preflight 的 "obs bound >= max MI" 门在 x130 上实测抓到 52e6 > 50e6）。
    # 同比放大保持与 SQT2 相同的 1.25 倍余量，不引入新的自由参数。
    b["obs_cloudlet_mi_high"] = int(round(src_block["obs_cloudlet_mi_high"]
                                          * RUNTIME_SCALE))
    for dc in b["datacenters"]:
        dc["turbine_ids"] = [t + TURBINE_OFFSET for t in dc.get("turbine_ids") or []]
    return b


def build(cfg: dict) -> dict:
    out = {}
    for src, dst, prefix in PLAN:
        if src not in cfg:
            raise KeyError(f"源 block 缺失: {src}")
        out[dst] = derive(cfg[src], dst, prefix)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(_REPO / "config_C.yml"))
    ap.add_argument("--append", action="store_true",
                    help="直接追加到 config（默认只打到 stdout 供人工核对）")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    blocks = build(cfg)
    already = [k for k in blocks if k in cfg]
    if already:
        sys.exit(f"拒绝：目标 block 已存在 {already}；先手工删除再重跑（不覆盖已冻结的块）")

    text = ("\n\n# " + "-" * 73 + "\n"
            "# gwo1（第十考场，2026-08-20）：由 SQT2 块克隆，只改 trace / 涡轮块 /\n"
            "# 名称 / preflight profile。奖励、观测、Lagrangian、容量全部与 SQT2 相同，\n"
            "# 所以 gwo1 与 SQT2 的差别被限制在决策域和 trace 这两个自变量上。\n"
            "# 由 gen_gwo1_config.py 生成，不要手工编辑。\n"
            "# " + "-" * 73 + "\n"
            + yaml.safe_dump(blocks, default_flow_style=False, sort_keys=True,
                             allow_unicode=True, width=4096))
    if a.append:
        with open(a.config, "a") as f:
            f.write(text)
        print(f"已追加 {len(blocks)} 个 block 到 {a.config}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
