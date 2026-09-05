"""Scene + interface design v1 (reports/SCENE_INTERFACE_DESIGN.md, Addenda A/B): the frozen,
scheduling-free arithmetic and the data-isolation rules. Pure functions; no wind value is read.

  dynamic_energy_wh  Σ pes · P_dyn_pe · u · runtime_sec / 3600           (A1 / B1)
  c_brown_ref_kg     E_Wh / 1000 · f_brown_ref [kg per kWh]               (B1)
  headroom_ok        (C_B − C_ST)/C_B ≥ 0.15 and C_B − C_ST ≥ 0.05·C_brown_ref   (§2.2)
  draw_windows       hash-ordered, greedy non-overlapping footprints       (§1, §2.2)
  energy_weighted    coverage weighted by dynamic job energy               (B2)

Usage: python scene_v1.py isolate   -> chooses the turbines (frozen hash rule) and draws the six
                                       2020 confirmation windows; writes stage_a_out/scene_v1_*.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "stage_a_out")
P_DYN_PE_W = (214.0 - 51.4) / 64.0
REL_MIN, ABS_FRAC = 0.15, 0.05
FOOTPRINT_ROWS = 2922
ROWS = {2020: 32224, 2021: 52559}
TAG_WINDOWS = "scene-interface-v1:2020:"


def dynamic_energy_wh(pes, mi, vm_pe_mips, cpu_util, p_dyn_pe_w=P_DYN_PE_W):
    """Σ_jobs pes · P_dyn_pe · u · runtime_sec / 3600, runtime = mi / (mips · u). Pure."""
    pes = np.asarray(pes, dtype=np.float64)
    mi = np.asarray(mi, dtype=np.float64)
    rate = max(1.0, float(vm_pe_mips)) * min(1.0, max(1e-6, float(cpu_util)))
    runtime_sec = mi / rate
    return float(np.sum(pes * p_dyn_pe_w * float(cpu_util) * runtime_sec / 3600.0))


def c_brown_ref_kg(e_dynamic_wh, f_brown_ref_kg_per_kwh):
    return float(e_dynamic_wh) / 1000.0 * float(f_brown_ref_kg_per_kwh)


def mean_brown_factor(datacenters):
    vals = [float(d["brown_carbon_factor"]) for d in datacenters if "brown_carbon_factor" in d]
    if not vals:
        raise ValueError("no brown_carbon_factor in the block")
    return float(np.mean(vals))


def headroom_ok(c_b, c_st, c_brown_ref):
    gap = float(c_b) - float(c_st)
    return bool(c_b > 0 and gap / float(c_b) >= REL_MIN and gap >= ABS_FRAC * float(c_brown_ref))


def draw_windows(n_rows, n_windows, tag, footprint=FOOTPRINT_ROWS, stride=1):
    """Hash-ordered greedy non-overlapping footprints over every legal start (0..n_rows−footprint).
    Reads nothing but the row count. Returns {"status", "windows": [offsets in acceptance order]}."""
    starts = list(range(0, int(n_rows) - int(footprint) + 1, int(stride)))
    order = sorted(starts, key=lambda o: hashlib.sha256(f"{tag}{o}".encode()).hexdigest())
    taken = []
    for o in order:
        if all(o + footprint <= a or a + footprint <= o for a in taken):
            taken.append(o)
            if len(taken) == n_windows:
                break
    status = "OK" if len(taken) == n_windows else "STOP_WINDOW_SPLIT"
    return {"status": status, "tag": tag, "footprint": footprint, "n_rows": int(n_rows), "windows": taken}


def energy_weighted(values, pes, mi, vm_pe_mips, cpu_util):
    """Coverage weighted by each job's dynamic energy (B2). Pure."""
    w = np.array([dynamic_energy_wh([p], [m], vm_pe_mips, cpu_util) for p, m in zip(pes, mi)])
    v = np.asarray(values, dtype=np.float64)
    return float(np.sum(v * w) / np.sum(w)) if np.sum(w) > 0 else 0.0


def ids_mentioned(text):
    """Turbine ids a tracked file refers to: the legacy singular `turbine_id: N`, the list
    form `turbine_ids: [a, b]` (one line) or `- N` items under it, and `Turbine_N_` file
    names. Pure."""
    import re
    ids = set()
    for m in re.finditer(r"\bturbine_id:\s*(\d+)", text):
        ids.add(int(m.group(1)))
    for m in re.finditer(r"\bturbine_ids:\s*\[([^\]]*)\]", text):
        ids.update(int(x) for x in re.findall(r"\d+", m.group(1)))
    for m in re.finditer(r"\bturbine_ids:\s*\n((?:\s*-\s*\d+\s*\n?)+)", text):
        ids.update(int(x) for x in re.findall(r"-\s*(\d+)", m.group(1)))
    for m in re.finditer(r"Turbine_(\d+)_", text):
        ids.add(int(m.group(1)))
    return ids


DATASET_PATHS = ("windProduction/", "scripts/wind/")     # the wind dataset and its preprocessing


def used_in_tracked_files(repo=REPO):
    """Ids referred to by any git-tracked yml/yaml/json/md/py file outside the wind dataset
    itself (the inventory's structured scan missed the legacy singular key; this scan is the
    design's 'never in any tracked experiment config, audit or report' applied literally).
    Files under the dataset and its preprocessing (per-turbine data reports, conversion
    scripts) describe availability of every turbine, not use, and are not counted."""
    import subprocess
    out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True).stdout.split("\n")
    used = {}
    for rel in out:
        if not rel.endswith((".yml", ".yaml", ".json", ".md", ".py")) or "scene_v1_isolation" in rel:
            continue
        if any(p in rel for p in DATASET_PATHS):
            continue
        try:
            text = open(os.path.join(repo, rel), errors="ignore").read()
        except OSError:
            continue
        for i in ids_mentioned(text):
            used.setdefault(i, []).append(rel)
    return used


def isolate():
    from stage_d_prime_turbines import choose, eligible
    used = used_in_tracked_files()
    cands = [i for i in eligible() if i not in used]
    excluded_by_scan = sorted(i for i in eligible() if i in used)
    turbines = choose(cands)
    turbines["excluded_by_tracked_scan"] = {str(i): used[i][:3] for i in excluded_by_scan}
    if turbines["status"] != "OK":
        raise SystemExit(json.dumps(turbines))
    split = os.path.join(REPO, "cloudsimplus-gateway", "src", "main", "resources", "windProduction", "split")
    files = {}
    for i in turbines["turbines"]:
        for y in ROWS:
            p = os.path.join(split, f"Turbine_{i}_{y}.csv")
            files[f"{i}_{y}"] = {"path": os.path.relpath(p, REPO),
                                 "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16],
                                 "rows": sum(1 for _ in open(p)) - 1}
    conf = draw_windows(ROWS[2020], 6, TAG_WINDOWS)
    rec = {"design": "reports/SCENE_INTERFACE_DESIGN.md (v1 + Addenda A, B)", "turbines": turbines,
           "files": files, "design_year": 2021, "confirmation_year": 2020, "confirmation_windows": conf,
           "p_dyn_pe_w": P_DYN_PE_W, "rel_min": REL_MIN, "abs_frac": ABS_FRAC}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scene_v1_isolation.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: rec[k] for k in ("turbines", "confirmation_windows")}, indent=1))
    print("written stage_a_out/scene_v1_isolation.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "isolate":
        isolate()
    else:
        print(__doc__)
