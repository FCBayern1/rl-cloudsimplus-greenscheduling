"""Interface memory / throughput smoke (SCENE_INTERFACE_DESIGN §4.4, Addendum A4, ordered):
one development window, the dev twins WITH the candidate key (interface) and WITHOUT it
(offset), same arm (nowait_planner, dispatch now; reads nothing), peak RSS of the evaluator
process and steps per second, plus the per-step size of the key. Rule: dense float32 first;
peak RSS > 1.5x or throughput < 0.5x the reference -> dense float16 (with its own
round-trip test) -> STOP_RESOURCE_INTERFACE. Zero RL. Writes stage_a_out/interface_smoke.json.

Usage: python interface_smoke.py
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "stage_a_out")


def run_once(mode, k, offset):
    import run_stage_a as rs
    cfg, cell = rs.scene_dev_config(mode)
    out_csv = os.path.join(OUT, "interface_smoke", f"{mode}_k{k}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    env = dict(os.environ)
    env.update({"GATEWAY_LIBS": os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"),
                "EVAL_CONFIG_PATH": cfg, "ORACLE_EXPERIMENT": cell, "ORACLE_OFFSET_ROWS": str(offset),
                "ORACLE_WIND_DIR": os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"),
                "PLANNER_EXPECTED_CAP": "640;512;640;512;192", "PLANNER_STATIC_TOTAL_W": "0"})
    cmd = [os.path.join(REPO, "drl-manager/.venv/bin/python"), "-m", "src.baselines.evaluate", "--experiment", cell,
           "--global", "nowait_planner", "--local", "drain", "--episodes", "1", "--seed", "42",
           "--reset-skip", str(k), "--output", out_csv]
    t0 = time.time()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    with open(out_csv.replace(".csv", ".log"), "w") as log:
        rc = subprocess.run(cmd, cwd=os.path.join(REPO, "drl-manager"), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = time.time() - t0
    steps = None
    if rc == 0 and os.path.exists(out_csv):
        import csv
        steps = int(float(list(csv.DictReader(open(out_csv)))[-1].get("episode_length", 0)))
    return {"rc": rc, "wall_s": wall, "steps": steps, "steps_per_s": (steps / wall) if steps else None,
            "peak_rss_children_mb": r1.ru_maxrss / 1024.0, "config": cfg, "block": cell}


def main():
    dev = json.load(open(os.path.join(OUT, "scene_v2_dev.json")))["dev_offsets"]
    k, off = 0, dev[0]
    res = {"window_k": k, "offset": off, "key_bytes_per_step_f32": 128 * 45 * 4}
    res["offset_twin"] = run_once("offset", k, off)          # reference (no key)
    res["interface_twin"] = run_once("interface", k, off)    # dense float32 key
    ref, itf = res["offset_twin"], res["interface_twin"]
    if ref["steps_per_s"] and itf["steps_per_s"]:
        res["throughput_ratio"] = itf["steps_per_s"] / ref["steps_per_s"]
    res["rss_ratio"] = itf["peak_rss_children_mb"] / max(1.0, ref["peak_rss_children_mb"]) if ref["peak_rss_children_mb"] else None
    ok = (res.get("throughput_ratio") or 0) >= 0.5 and (res.get("rss_ratio") or 9) <= 1.5 and itf["rc"] == 0
    res["verdict"] = "PASS_DENSE_F32" if ok else "NEXT_STEP_DENSE_F16"
    with open(os.path.join(OUT, "interface_smoke.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
