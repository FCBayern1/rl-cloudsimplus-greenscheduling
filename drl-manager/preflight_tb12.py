#!/usr/bin/env python3
"""tb12 A' 预注册前六项检查(Codex 2026-08-23 裁定,逐项对应)。

用法: preflight_tb12.py <iso_smoke.json>
"""
import json
import pathlib
import sys

import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
P_JOB_W = 28.75
SCALE = 9.05562658195e-5

fails = []


def chk(name, ok, msg):
    print(f"[{'PASS' if ok else '**FAIL**'}] {name:42s} {msg}")
    if not ok:
        fails.append(name)


def main(json_path):
    res = json.loads(pathlib.Path(json_path).read_text())["results"]
    cfg = yaml.safe_load(open(_REPO / "config_C.yml"))["experiment_tb12_iso"]
    dc = cfg["datacenters"][0]
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / "calib/tb12_v2.json").read_text())

    # 1. VM-host 1:1(结构性:等数 RoundRobin;m/l 为零无混布)
    chk("1: VM-host 1:1 by construction",
        dc["initial_s_vm_count"] == dc["host_count_spec_asus_rs500a"]
        and dc["initial_m_vm_count"] == 0 and dc["initial_l_vm_count"] == 0,
        f"S={dc['initial_s_vm_count']} hosts={dc['host_count_spec_asus_rs500a']} "
        f"M={dc['initial_m_vm_count']} L={dc['initial_l_vm_count']} (RoundRobin 等数)")

    # 2. 一 VM 一作业(SpaceShared + vm_pes == job pes)
    chk("2: one job per VM (SpaceShared, pes match)",
        dc["small_vm_pes"] == 2,
        f"small_vm_pes={dc['small_vm_pes']} == job pes=2, 2PE 作业占满 2PE VM")

    # 3. 五臂完成一致,无截断
    by_arm = {}
    for r in res:
        by_arm.setdefault(r["arm"], []).append(r)
    ok3 = all(r["finished"] == 5 and r["ontime"] == 1.0 and r["steps"] < 288
              for r in res)
    chk("3: all arms complete, no truncation",
        ok3, f"{len(res)} cells, finished=5 & ontime=1 & steps<288 everywhere: {ok3}")

    # 4. 无风 off=44000: 五臂总能量极差 <=1%, 碳差 <=1%
    calm = [r for r in res if r["offset"] == 44000]
    ce = [r["energy_wh"] for r in calm]
    cc = [r["carbon_kg"] for r in calm]
    e_rng = (max(ce) - min(ce)) / min(ce)
    c_rng = (max(cc) - min(cc)) / min(cc)
    chk("4: calm-window energy range <=1%",
        e_rng <= 0.01, f"energy range {100*e_rng:.2f}% ({min(ce):.1f}-{max(ce):.1f} Wh)")
    chk("4: calm-window carbon range <=1%",
        c_rng <= 0.01, f"carbon range {100*c_rng:.2f}%")

    # 5. 单作业功率 ~28.75W 且 scale 冻结不变
    nw = [r for r in calm if r["arm"] == "nowait"][0]
    p_meas = nw["energy_wh"] / (5 * art["rt_h"])
    chk("5: single-job power ~= 28.75 W",
        abs(p_meas - P_JOB_W) / P_JOB_W <= 0.05,
        f"measured {p_meas:.2f} W vs calibrated {P_JOB_W} (calm window, no sharing)")
    chk("5: green_power_scale frozen",
        abs(cfg["green_power_scale"] - SCALE) < 1e-15,
        f"{cfg['green_power_scale']} (T100+101/2020 only)")

    # 6. 有风 off=26000: clair 打赢冻结最强盲
    windy = {r["arm"]: r["carbon_kg"] for r in res if r["offset"] == 26000}
    blind_best = min(windy[a] for a in ("greenfollow", "hazard", "dpcont"))
    chk("6: clair beats frozen best blind at off=26000",
        windy["clair"] < blind_best,
        f"clair={windy['clair']:.5f} vs best blind={blind_best:.5f} "
        f"({100*(windy['clair']-blind_best)/blind_best:+.1f}%)")

    print()
    if fails:
        print(f"PREFLIGHT FAILED ({len(fails)}): {fails}")
        sys.exit(1)
    print("PREFLIGHT PASSED - prereg may be frozen")


if __name__ == "__main__":
    main(sys.argv[1])
