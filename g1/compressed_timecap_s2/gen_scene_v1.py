"""Scene v1 configs (reports/SCENE_INTERFACE_DESIGN.md §1–§2, frozen): the D' V block with the
never-used turbines of stage_a_out/scene_v1_isolation.json and the first twelve hash-ordered
2021 windows as the development pool. Two twins: the defer-mode block (step-wise certification
arms B / ST / shuffle / anti / calibrated shrink) and the offset-mode block (the gates of §4).
Nothing else in the blocks changes; the manifest records every source hash.

Usage: python gen_scene_v1.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scene_v1 import ROWS, draw_windows  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out")
BLOCK = "sd_V_s2_r48_w72_c3_n35"
SOURCES = {"defer": "config_stage_d_dprime.yml", "offset": "config_stage_d_dprime_offset.yml",
           "interface": "config_stage_d_dprime_interface.yml"}
TAG_2021 = "scene-interface-v1:2021:"
N_POOL = 12


def scene_block(src_block, dc_turbines, windows, name):
    """Pure: copy of the source block with the turbines replaced per datacentre index and the
    window allowlist set; every other key untouched. dc_turbines: {dc_index: [ids]}."""
    b = copy.deepcopy(src_block)
    dcs = sorted(b["datacenters"], key=lambda d: int(d.get("datacenter_id", 0)))
    for d in dcs:
        i = int(d.get("datacenter_id", 0))
        d["turbine_ids"] = [int(x) for x in dc_turbines.get(str(i), dc_turbines.get(i, []))]
    b["datacenters"] = dcs
    b["green_episode_offset_allowlist"] = ";".join(str(int(w)) for w in windows)
    b["experiment_name"] = name
    b["simulation_name"] = f"SCENE_V1_{name}"
    return b


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    iso = json.load(open(os.path.join(OUT, "scene_v1_isolation.json")))
    dc_turbines = iso["turbines"]["dc_turbines"]
    pool = draw_windows(ROWS[2021], N_POOL, TAG_2021)
    if pool["status"] != "OK":
        raise SystemExit(json.dumps(pool))
    man = {"isolation_sha256": _sha(os.path.join(OUT, "scene_v1_isolation.json")), "dc_turbines": dc_turbines,
           "pool_2021": pool, "configs": {}}
    for mode, src in SOURCES.items():
        cfg = yaml.safe_load(open(os.path.join(HERE, src)))
        blk = scene_block(cfg[BLOCK], dc_turbines, pool["windows"], f"sv1_{mode}_{BLOCK[5:]}")
        out = {k: v for k, v in cfg.items() if not k.startswith("sd_")}
        out[f"sv1_{mode}_{BLOCK[5:]}"] = blk
        name = f"config_scene_v1_{mode}.yml"
        with open(os.path.join(HERE, name), "w") as f:
            yaml.safe_dump(out, f, sort_keys=True)
        man["configs"][mode] = {"file": name, "source": src, "source_sha256": _sha(os.path.join(HERE, src)),
                                "sha256": _sha(os.path.join(HERE, name)), "block": f"sv1_{mode}_{BLOCK[5:]}"}
    with open(os.path.join(OUT, "scene_v1_manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(json.dumps({"dc_turbines": dc_turbines, "pool_2021": pool["windows"], "configs": man["configs"]}, indent=1))


if __name__ == "__main__":
    main()
