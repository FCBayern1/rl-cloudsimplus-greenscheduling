"""Scheme-2 workload and config generator. Everything derived, nothing hand-written.

The work order (WORKORDER_GPU_COMPRESSED_TIMECAP_SCHEME2.md section 4) freezes the axes;
this module turns them into 108 cells, each a trace CSV plus an experiment block derived
from the frozen C-regime base block by overriding an enumerated list of keys and nothing
else. A test diffs every derived block against the base and fails on any unlisted change,
so a drifting key cannot hide in 300 lines of YAML.

Core closure condition: a job started at its latest legal moment still finishes inside
the 144-row window TimeCAP can see, (s - a) + r <= wait_cap + runtime <= 144, enforced by
the admissible-pair grid and re-checked per trace row. The backstop is the runtime-aware
`latest_start` mode; the legacy 600-second lead would force jobs before the scheduler
ever decided, which is exactly the failure the work order names.

Windows: the simulator's own per-episode offset schedule (1009*k mod range) is kept, and
six k values are chosen by a frozen greedy rule so their windows cannot overlap even if
every episode runs to max_episode_length. k=0 is excluded: it is the historical training
window and stays quarantined. No green value is read anywhere in this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_CONFIG = os.path.join(REPO, "config_C.yml")
BASE_EXPERIMENT = "experiment_g1eval_matchedvan"
TRACE_DIR = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/traces/s2")

RUNTIME_ROWS = (24, 48, 72)
WAIT_CAP_ROWS = (24, 48, 72, 96, 120)
CLOSURE_ROWS = 144
CONCURRENCY = (1, 3, 5)
N_JOBS = (20, 35, 50)
PES_PER_JOB = 2
CPU_UTILISATION = 1.0            # scenario assumption: compute-bound batch jobs
FILE_SIZE, OUTPUT_SIZE = 536, 268
SEED_NAMESPACE = "s2v1"

TRACE_ROWS_MAX = 52559           # every 2021 turbine file, verified by the window gate
EPISODE_ROWS_MAX = 7200          # base block max_episode_length, inherited unchanged
WINDOW_SPACING = 7300            # > EPISODE_ROWS_MAX so windows cannot touch

# The ONLY keys a derived block may change, with their values or value factories.
OVERRIDDEN_KEYS = ("experiment_name", "simulation_name", "cloudlet_trace_file",
                   "defer_deadline_force_mode", "defer_deadline_slack_sec",
                   "cloudlet_cpu_utilization", "green_oracle_mode")
FORCE_MODE = "latest_start"
# The Java latest_start rule is now + runtime + slack >= deadline, and the base block
# carries slack 600 s. Deadline headroom here is at most 120 s, so an inherited slack
# would fire the backstop on every defer attempt before the scheduler ever decided —
# the exact failure the work order names. The smoke run caught it: 8 of 35 jobs
# force-routed, 8 stale reservations, 8 unplanned starts. Slack is therefore pinned to
# zero and the closure semantics is purely runtime-aware.
BACKSTOP_SLACK_SEC = 0.0
# The base block builds a TimeCAP observation provider (23.8M parameters, rebuilt on
# every reset at ~27 s each, plus inference every six steps). No planner arm reads those
# observation features and the simulated physics is identical either way, so Stage A/A'
# run the cheap godeye provider. Stage D (RL) re-decides this knob in its own prereg.
GREEN_ORACLE_MODE = "godeye"


def admissible_pairs():
    return [(r, w) for r in RUNTIME_ROWS for w in WAIT_CAP_ROWS
            if r + w <= CLOSURE_ROWS]


def cells():
    out = []
    for r, w in admissible_pairs():
        for c in CONCURRENCY:
            for n in N_JOBS:
                out.append({"runtime_rows": r, "wait_cap_rows": w,
                            "concurrency": c, "n_jobs": n, "seed": 0})
    return out


def cell_name(cell):
    return (f"s2_r{cell['runtime_rows']}_w{cell['wait_cap_rows']}"
            f"_c{cell['concurrency']}_n{cell['n_jobs']}")


def _payload(cell):
    return json.dumps({**cell, "ns": SEED_NAMESPACE}, sort_keys=True,
                      separators=(",", ":"))


def _seed(cell, domain):
    digest = hashlib.sha256(f"{_payload(cell)}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**31


def arrivals(cell):
    """Partitioned arrivals over the service span, one per interval, never clipped."""
    n, r, c = cell["n_jobs"], cell["runtime_rows"], cell["concurrency"]
    span = int(np.ceil(n * r / c))
    rng = np.random.default_rng(_seed(cell, "arrival"))
    out = np.empty(n, dtype=int)
    for i in range(n):
        lo, hi = (i * span) // n, ((i + 1) * span) // n
        out[i] = lo if hi <= lo else int(rng.integers(lo, hi))
    out.sort()
    return out, span


def trace(cell, vm_pe_mips):
    """Rows of the trace CSV plus the mechanical report the work order requires."""
    a, span = arrivals(cell)
    r, w = cell["runtime_rows"], cell["wait_cap_rows"]
    mi = int(round(r * vm_pe_mips * CPU_UTILISATION))
    rows = [(i, int(a[i]), mi, PES_PER_JOB, FILE_SIZE, OUTPUT_SIZE, int(a[i] + r + w))
            for i in range(cell["n_jobs"])]
    finish_last = max(x[1] for x in rows) + r
    report = {
        "arrival_span": int(a.max() - a.min() + 1),
        "service_span": span,
        "offered_concurrency": round(cell["n_jobs"] * r
                                     / max(finish_last - int(a.min()), 1), 4),
        "runtime_rows": r, "wait_cap_rows": w, "mi_per_job": mi,
        "deadline_max": max(x[6] for x in rows),
        "deadline_reachable": all(x[1] + r <= x[6] <= x[1] + CLOSURE_ROWS for x in rows),
        "closure_ok": r + w <= CLOSURE_ROWS,
        "fits_episode": max(x[6] for x in rows) < EPISODE_ROWS_MAX,
    }
    return rows, report


def trace_text(rows):
    head = "cloudlet_id,arrival_time,length,pes_required,file_size,output_size,deadline"
    return head + "\n" + "\n".join(",".join(str(v) for v in row) for row in rows) + "\n"


def content_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def base_block():
    cfg = yaml.safe_load(open(BASE_CONFIG))
    return copy.deepcopy(cfg[BASE_EXPERIMENT])


def windows(offset_range):
    """Six frozen windows on the simulator's own (1009*k mod range) schedule.

    Greedy over ascending k starting at 1 (k=0 is the historical window, quarantined):
    keep a k when its offset sits at least WINDOW_SPACING from every kept offset and the
    whole worst-case episode stays inside the trace. First three kept are DISCOVERY,
    next three CONFIRMATION.
    """
    kept = []
    k = 1
    while len(kept) < 6:
        off = (1009 * k) % offset_range
        if off + EPISODE_ROWS_MAX <= TRACE_ROWS_MAX and \
                all(abs(off - o) >= WINDOW_SPACING for _kk, o in kept):
            kept.append((k, off))
        k += 1
        if k > 200000:
            raise RuntimeError("no six windows satisfy the spacing rule")
    return {"discovery": kept[:3], "confirmation": kept[3:],
            "spacing": WINDOW_SPACING, "episode_rows_max": EPISODE_ROWS_MAX,
            "offset_range": offset_range, "k0_quarantined": True}


def derived_block(cell, base):
    blk = copy.deepcopy(base)
    name = cell_name(cell)
    blk["experiment_name"] = name
    blk["simulation_name"] = f"S2_{name}"
    blk["cloudlet_trace_file"] = f"traces/s2/{name}.csv"
    blk["defer_deadline_force_mode"] = FORCE_MODE
    blk["defer_deadline_slack_sec"] = BACKSTOP_SLACK_SEC
    blk["cloudlet_cpu_utilization"] = CPU_UTILISATION
    blk["green_oracle_mode"] = GREEN_ORACLE_MODE
    return blk


def generate(out_dir=None, trace_dir=None):
    out_dir = out_dir or HERE
    trace_dir = trace_dir or TRACE_DIR
    os.makedirs(trace_dir, exist_ok=True)
    base = base_block()
    mips = float(base["datacenters"][0]["vm_pe_mips"])
    win = windows(int(base["green_episode_offset_range"]))

    blocks, reports, shas = {}, {}, {}
    for cell in cells():
        rows, rep = trace(cell, mips)
        text = trace_text(rows)
        name = cell_name(cell)
        path = os.path.join(trace_dir, f"{name}.csv")
        tmp = path + ".partial"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
        blocks[name] = derived_block(cell, base)
        rep["cell"] = cell
        reports[name] = rep
        shas[f"traces/s2/{name}.csv"] = content_sha(text)

    # The eval loader merges `common` under the experiment block; without carrying the
    # base file's common section verbatim, the derived experiments would silently lose
    # every key only common provides.
    common = yaml.safe_load(open(BASE_CONFIG)).get("common", {})
    cfg_text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True,
                              default_flow_style=False)
    with open(os.path.join(out_dir, "config_s2.yml"), "w") as f:
        f.write(cfg_text)
    manifest = {
        "cells": len(blocks), "admissible_pairs": len(admissible_pairs()),
        "windows": win, "base_experiment": BASE_EXPERIMENT,
        "base_block_sha": content_sha(json.dumps(base, sort_keys=True, default=str)),
        "config_s2_sha": content_sha(cfg_text),
        "trace_shas": shas, "reports": reports,
        "overridden_keys": list(OVERRIDDEN_KEYS),
        "pes_per_job": PES_PER_JOB, "cpu_utilization": CPU_UTILISATION,
        "vm_pe_mips": mips, "seed_namespace": SEED_NAMESPACE,
    }
    with open(os.path.join(out_dir, "s2_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


if __name__ == "__main__":
    m = generate()
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("trace_shas", "reports")}, indent=2, sort_keys=True))


# ── Scheme 2-E configs: the same 108 blocks on the frozen fresh turbines ────────────

def generate_e(part, out_dir=None):
    """Derive config_s2e_<part>.yml from the S2 blocks by swapping ONLY the green DCs'
    turbine_ids to the frozen E split. Everything else, the traces included, is byte
    identical to the S2 derivation, and a test diffs it to prove that."""
    import json as _json
    out_dir = out_dir or HERE
    split = _json.load(open(os.path.join(HERE, "e_data_split.json")))[part]
    base = base_block()
    common = yaml.safe_load(open(BASE_CONFIG)).get("common", {})
    blocks = {}
    for cell in cells():
        blk = derived_block(cell, base)
        name = cell_name(cell)
        blk["experiment_name"] = name
        blk["simulation_name"] = f"S2E_{part[:4]}_{name}"
        for dc in blk["datacenters"]:
            did = str(dc["datacenter_id"])
            if did in split["dc_map"]:
                dc["turbine_ids"] = [int(t) for t in split["dc_map"][did]]
        blocks[name] = blk
    cfg_text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True,
                              default_flow_style=False)
    path = os.path.join(out_dir, f"config_s2e_{part}.yml")
    with open(path, "w") as f:
        f.write(cfg_text)
    return {"part": part, "blocks": len(blocks), "sha": content_sha(cfg_text),
            "turbines": split["turbines"], "windows_k": split["windows_k"]}


if __name__ == "__main__" and os.environ.get("GEN_S2E"):
    print(json.dumps([generate_e("discovery"), generate_e("confirmation")], indent=2))


# ── Scheme 2-F variant: uniform brown factor + a green-scarcity knob ────────────

F_BROWN_UNIFORM = 0.5      # no clean-DC haven; green already uniform at 0.01

def generate_f(divisor_mult, out_dir=None, part="discovery"):
    """S2-E discovery config with every brown factor set equal and green scaled down.

    The E verdict showed godeye lost to reservation_edf because the blind piled onto DC0
    (brown 0.08, a clean haven) while godeye chased green into dirtier DCs. Removing the
    haven (uniform brown) is necessary; scaling green down (divisor_mult > 1) makes green
    the binding constraint so catching it needs real placement, not just spreading. The
    pilot sweeps divisor_mult to find where godeye beats the blind AND shuffle does not.
    """
    import json as _json
    out_dir = out_dir or HERE
    split = _json.load(open(os.path.join(HERE, "e_data_split.json")))[part]
    base = base_block()
    common = yaml.safe_load(open(BASE_CONFIG)).get("common", {})
    base_div = float(base.get("compressed_power_divisor", 1500.0))
    blocks = {}
    for cell in cells():
        blk = derived_block(cell, base)
        name = cell_name(cell)
        blk["experiment_name"] = name
        blk["simulation_name"] = f"S2F_m{divisor_mult}_{name}"
        blk["compressed_power_divisor"] = base_div * divisor_mult
        for dc in blk["datacenters"]:
            dc["brown_carbon_factor"] = F_BROWN_UNIFORM
            did = str(dc["datacenter_id"])
            if did in split["dc_map"]:
                dc["turbine_ids"] = [int(t) for t in split["dc_map"][did]]
        blocks[name] = blk
    cfg_text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True,
                              default_flow_style=False)
    path = os.path.join(out_dir, f"config_s2f_m{divisor_mult}.yml")
    with open(path, "w") as f:
        f.write(cfg_text)
    return {"divisor_mult": divisor_mult, "brown": F_BROWN_UNIFORM,
            "divisor": base_div * divisor_mult, "blocks": len(blocks),
            "sha": content_sha(cfg_text)}


if __name__ == "__main__" and os.environ.get("GEN_S2F"):
    print(json.dumps([generate_f(m) for m in (1, 2, 4, 8)], indent=2))


# ── Scheme 2-G variant: F (uniform brown + scarcity) with idle hosts powered down ────

def generate_g(divisor_mult, out_dir=None, part="discovery"):
    """The F variant with idle_host_power_down: true on every DC.

    Five lines converged on one wall: waiting burns static idle power and lengthens the
    makespan, so deferring for green is net negative when the fleet never sleeps. With
    power-down an idle host draws zero (DatacenterInstance.hostPowerW), so the temporal
    lever no longer pays a static tax; combined with time-scarce green (divisor_mult) the
    question "does knowing WHEN green comes pay" gets its first fair test.
    """
    import json as _json
    out_dir = out_dir or HERE
    split = _json.load(open(os.path.join(HERE, "e_data_split.json")))[part]
    base = base_block()
    common = yaml.safe_load(open(BASE_CONFIG)).get("common", {})
    base_div = float(base.get("compressed_power_divisor", 1500.0))
    blocks = {}
    for cell in cells():
        blk = derived_block(cell, base)
        name = cell_name(cell)
        blk["experiment_name"] = name
        blk["simulation_name"] = f"S2G_m{divisor_mult}_{name}"
        blk["compressed_power_divisor"] = base_div * divisor_mult
        blk["idle_host_power_down"] = True
        for dc in blk["datacenters"]:
            dc["brown_carbon_factor"] = F_BROWN_UNIFORM
            dc["idle_host_power_down"] = True
            did = str(dc["datacenter_id"])
            if did in split["dc_map"]:
                dc["turbine_ids"] = [int(t) for t in split["dc_map"][did]]
        blocks[name] = blk
    cfg_text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True,
                              default_flow_style=False)
    path = os.path.join(out_dir, f"config_s2g_m{divisor_mult}.yml")
    with open(path, "w") as f:
        f.write(cfg_text)
    return {"divisor_mult": divisor_mult, "idle_power_down": True,
            "blocks": len(blocks), "sha": content_sha(cfg_text)}


if __name__ == "__main__" and os.environ.get("GEN_S2G"):
    for m in (1, 2, 4):
        print(generate_g(m))


# ── Scheme 2-H pilot: F settings with 32-PE jobs (dynamic 81 W vs 51 W idle floor) ───

H_PES = 32

def generate_h(divisor_mult, pilot_cells, out_dir=None, trace_dir=None, part="discovery",
               zero_floor=False):
    """F variant whose pilot cells run 32-PE jobs instead of 2-PE.

    zero_floor=True (Level-1 spiral, config_s2hz_m*.yml): the same fleet on the zero-floor
    host twins (SPEC_ASUS_RS500A_DYN / RS700A_DYN, 0 W idle, 81.3 W per 32-PE job), so the
    simulator's power is exactly the sum of job dynamic power, the physics of toy_lever.py.

    With power-down on all along, energy is dominated by awake host-seconds because a
    2-PE job draws 5 W against a 51 W floor; consolidation beats any forecast-driven
    spreading. At 32 PE the job draws 81 W, above the floor, so fragmentation no longer
    dominates and chasing green can pay. Traces are regenerated with the same arrivals
    and runtimes (only pes and MI-per-job change) under distinct names.
    """
    import json as _json
    out_dir = out_dir or HERE
    trace_dir = trace_dir or TRACE_DIR
    split = _json.load(open(os.path.join(HERE, "e_data_split.json")))[part]
    base = base_block()
    common = yaml.safe_load(open(BASE_CONFIG)).get("common", {})
    mips = float(base["datacenters"][0]["vm_pe_mips"])
    base_div = float(base.get("compressed_power_divisor", 1500.0))
    blocks, shas = {}, {}
    for cell in cells():
        name = cell_name(cell)
        if name not in pilot_cells:
            continue
        rows, _rep = trace(cell, mips)
        rows32 = [(i, a, mi, H_PES, fs, os_, dl) for (i, a, mi, _p, fs, os_, dl) in rows]
        text = trace_text(rows32)
        tpath = os.path.join(trace_dir, f"{name}_pes{H_PES}.csv")
        with open(tpath, "w") as f:
            f.write(text)
        shas[name] = content_sha(text)
        blk = derived_block(cell, base)
        blk["experiment_name"] = name
        tag = "S2HZ" if zero_floor else "S2H"
        blk["simulation_name"] = f"{tag}_m{divisor_mult}_{name}"
        blk["cloudlet_trace_file"] = f"traces/s2/{name}_pes{H_PES}.csv"
        # The C-regime base splits any cloudlet above 8 PEs into 8-PE pieces (MI divided
        # accordingly). A 32-PE job must stay one 32-PE job to draw 81 W on one VM.
        blk["split_large_cloudlets"] = False
        blk["max_cloudlet_pes"] = H_PES
        blk["compressed_power_divisor"] = base_div * divisor_mult
        for dc in blk["datacenters"]:
            dc["brown_carbon_factor"] = F_BROWN_UNIFORM
            did = str(dc["datacenter_id"])
            if did in split["dc_map"]:
                dc["turbine_ids"] = [int(t) for t in split["dc_map"][did]]
            # TB13-v4 map: the host is exposed as 32-PE VMs, as many as the host PEs
            # allow, so a 32-PE job occupies 32 real cores at full dynamic power. The
            # C-regime fleet (2/4/8-PE VMs) time-shared a 32-PE job onto 8 cores and
            # the whole pilot ran at a fifth of the intended draw.
            host_pes = (dc.get("host_count_spec_asus_rs500a", 0) * 64
                        + dc.get("host_count_spec_asus_rs700a", 0) * 128)
            dc["small_vm_pes"] = H_PES
            dc["medium_vm_multiplier"] = 1
            dc["large_vm_multiplier"] = 1
            dc["initial_s_vm_count"] = max(1, host_pes // H_PES)
            dc["initial_m_vm_count"] = 0
            dc["initial_l_vm_count"] = 0
            if zero_floor:
                for legacy in ("host_count_spec_asus_rs500a", "host_count_spec_asus_rs700a"):
                    n_hosts = dc.pop(legacy, 0)
                    if n_hosts:
                        dc[legacy + "_dyn"] = n_hosts
        blocks[name] = blk
    cfg_text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True,
                              default_flow_style=False)
    fname = f"config_s2{'hz' if zero_floor else 'h'}_m{divisor_mult}.yml"
    with open(os.path.join(out_dir, fname), "w") as f:
        f.write(cfg_text)
    return {"divisor_mult": divisor_mult, "pes": H_PES, "blocks": len(blocks),
            "zero_floor": zero_floor, "config": fname,
            "trace_shas": shas, "sha": content_sha(cfg_text)}


PILOT_CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]

if __name__ == "__main__" and os.environ.get("GEN_S2H"):
    for m in (1, 2, 4):
        r = generate_h(m, PILOT_CELLS)
        print({k: r[k] for k in ("divisor_mult", "pes", "blocks", "sha")})

if __name__ == "__main__" and os.environ.get("GEN_S2HZ"):
    for m in (1, 2):
        r = generate_h(m, PILOT_CELLS, zero_floor=True)
        print({k: r[k] for k in ("divisor_mult", "config", "blocks", "sha")})
