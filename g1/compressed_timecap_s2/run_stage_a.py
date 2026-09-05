"""Stage A driver: blinds over the whole grid, freeze one arm, then the oracles.

Order is the registration's and is enforced by the entry point: the oracle phase refuses
to start without a freeze artifact, and the freeze refuses to run unless every blind cell
either passed its contract or is recorded as invalid. Each (arm, cell, window) run is one
`evaluate` subprocess with its own auto-launched JVM on a free port; a finished CSV is
never re-run, so the driver is resumable after an interruption.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_s2 as g  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = g.REPO
OUT = os.path.join(HERE, "stage_a_out")
BLINDS = ("persistence_planner", "climatology_planner", "reactive_wait_planner",
          "nowait_planner", "always_defer")
ORACLES = ("curve_planner", "oracle144_planner")
E_BLINDS = ("nowait_planner", "reactive_wait_planner", "reservation_edf",
            "load_smoothing")
E_ARMS = ("godeye", "calibrated_shrink_v1", "s30", "shuffle", "anti")
TIERS = ("godeye", "s05", "s15", "s30", "s60", "timecap_cal", "shuffle", "anti")
TIERS_V2 = ("godeye", "s05", "s15", "s30", "s60",
            "checkpoint_residual_surrogate_v2", "shuffle", "anti")
WORKERS = int(os.environ.get("S2_WORKERS", "5"))
SEED = 42
CONTRACT = {"completion_rate_mi": 0.995, "ontime_mi_share": 0.995}
ZERO_FIELDS = ("deadline_forced_count", "planner_n_stale_dropped",
               "planner_n_unplanned_start", "planner_n_wrong_dc",
               "planner_n_dispatched_never_started", "planner_running_pes_over_cap")


def windows(which="discovery"):
    return g.windows(44950)[which]


def jobs(arms, cell_names=None, which="discovery"):
    out = []
    names = cell_names or [g.cell_name(c) for c in g.cells()]
    for arm in arms:
        for name in names:
            for k, off in windows(which):
                out.append({"arm": arm, "cell": name, "k": k, "offset": off})
    return out


def e_jobs(part, arms, tier_mode=False):
    """Scheme 2-E jobs: fresh turbines via the part's config, that part's windows."""
    split = json.load(open(os.path.join(HERE, "e_data_split.json")))[part]
    cfg = os.path.join(HERE, f"config_s2e_{part}.yml")
    cal = os.path.join(HERE, "timecap_error_audit.json")
    out = []
    for arm in arms:
        for cell in g.cells():
            name = g.cell_name(cell)
            for k, off in zip(split["windows_k"], split["offsets"]):
                e = {"EVAL_CONFIG_PATH": cfg}
                if tier_mode:
                    e.update({"PLANNER_PERTURB_TIER": arm, "PLANNER_PERTURB_E": "1",
                              "PLANNER_PERTURB_CAL": cal})
                    out.append({"arm": "perturbed_oracle_planner", "cell": name,
                                "k": k, "offset": off, "env": e,
                                "dir": f"e_{part[:4]}_tier_{arm}"})
                else:
                    out.append({"arm": arm, "cell": name, "k": k, "offset": off,
                                "env": e, "dir": f"e_{part[:4]}_{arm}"})
    return out


HZ_BLINDS = E_BLINDS
HZ_ARMS = ("godeye", "calibrated_shrink_v1", "shuffle", "anti")
HZ_MULT = int(os.environ.get("HZ_MULT", "2"))        # x2 is the only verdict scene
HZ_ENV = {"PLANNER_EXPECTED_CAP": "640;512;640;512;192",
          "PLANNER_STATIC_TOTAL_W": "0"}
HZ_PILOT_CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]


def hz_jobs(part, arms, tier_mode=False, mult=None):
    """Scheme 2-HZ jobs (SCHEME2_HZ_PREREG): zero-floor fleet config, that part's windows.

    Discovery windows are the E split's (k=2 read, k=10/18 unread); confirmation
    k=26/34/42 stays sealed until hz_verdict discovery PASSes. The planner's hidden
    quantities (static floor 0, capacity vector) go in every job's env and are read back
    from the result rows by the verdict.
    """
    mult = mult or HZ_MULT
    split = json.load(open(os.path.join(HERE, "e_data_split.json")))[part]
    cfg = os.path.join(HERE, f"config_s2hz_m{mult}.yml")
    cal = os.path.join(HERE, "timecap_error_audit.json")
    out = []
    for arm in arms:
        for name in HZ_PILOT_CELLS:
            for k, off in zip(split["windows_k"], split["offsets"]):
                e = {"EVAL_CONFIG_PATH": cfg, **HZ_ENV}
                if tier_mode:
                    e.update({"PLANNER_PERTURB_TIER": arm, "PLANNER_PERTURB_E": "1",
                              "PLANNER_PERTURB_CAL": cal})
                    out.append({"arm": "perturbed_oracle_planner", "cell": name,
                                "k": k, "offset": off, "env": e,
                                "dir": f"hz_{part[:4]}_m{mult}_tier_{arm}"})
                else:
                    out.append({"arm": arm, "cell": name, "k": k, "offset": off,
                                "env": e, "dir": f"hz_{part[:4]}_m{mult}_{arm}"})
    return out


def hz_freeze(part="discovery", mult=None):
    """One strongest blind by pooled discovery carbon, frozen before any clean number."""
    mult = mult or HZ_MULT
    table = {}
    for arm in HZ_BLINDS:
        vals, bad = [], 0
        for jb in hz_jobs(part, (arm,), mult=mult):
            row = read_cell(jb)
            if row is None or not row["contract_ok"]:
                bad += 1
            else:
                vals.append(row["carbon"])
        table[arm] = {"pooled": (sum(vals) / len(vals)) if vals else None,
                      "valid": len(vals), "invalid": bad}
    everywhere = [a for a in HZ_BLINDS if table[a]["invalid"] == 0]
    art = {"arms": table, "valid_everywhere": everywhere, "mult": mult,
           "expected": len(HZ_PILOT_CELLS) * 3}
    if everywhere:
        art["frozen_blind"] = min(everywhere, key=lambda a: table[a]["pooled"])
        art["status"] = "FROZEN"
    else:
        art["status"] = "STOP_NO_VALID_BLIND"
    with open(os.path.join(OUT, f"hz_blind_freeze_m{mult}.json"), "w") as f:
        f.write(json.dumps(art, sort_keys=True, indent=2))
    return art


def hz_manifest(mult=None):
    """Code, jar, config, audit and per-job environment, hashed, for the HZ prereg."""
    mult = mult or HZ_MULT
    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    jar = os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/"
                             "cloudsimplus-gateway.jar")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=REPO, capture_output=True, text=True).stdout.strip()
    perturb = os.path.join(REPO, "drl-manager/src/baselines/forecast_perturb.py")
    planner = os.path.join(REPO, "drl-manager/src/baselines/global_schedulers.py")
    art = {"commit": commit, "worktree_clean": dirty == "", "mult": mult,
           "jar_sha256": sha(jar),
           "forecast_perturb_sha256": sha(perturb),   # TIERS_E parameters live here
           "global_schedulers_sha256": sha(planner),
           "config": f"config_s2hz_m{mult}.yml",
           "config_sha256": sha(os.path.join(HERE, f"config_s2hz_m{mult}.yml")),
           "audit_sha256": sha(os.path.join(HERE, "timecap_error_audit.json")),
           "planner_env": HZ_ENV, "arms": list(HZ_ARMS), "blinds": list(HZ_BLINDS),
           "cells": HZ_PILOT_CELLS, "seed": SEED,
           "windows": {p: json.load(open(os.path.join(HERE, "e_data_split.json")))[p]
                       for p in ("discovery", "confirmation")},
           "jobs_env": {f"{j['dir']}/{j['cell']}_k{j['k']}": j["env"]
                        for j in hz_jobs("discovery", HZ_BLINDS, mult=mult)
                        + hz_jobs("discovery", HZ_ARMS, tier_mode=True, mult=mult)}}
    with open(os.path.join(OUT, f"hz_manifest_m{mult}.json"), "w") as f:
        f.write(json.dumps(art, sort_keys=True, indent=2))
    return {k: v for k, v in art.items() if k != "jobs_env"}


def stable_region_cells():
    """The cells Stage A froze; A-prime may not enumerate its own."""
    v = json.load(open(os.path.join(OUT, "stage_a_verdict.json")))
    if v.get("verdict") != "PASS_STAGE_A":
        raise RuntimeError("stage A did not pass; there is no region to ladder")
    return [g.cell_name(r["cell"]) for r in v["rows"] if r.get("pass")]


def _paths(j):
    d = os.path.join(OUT, j.get("dir", j["arm"]))
    os.makedirs(d, exist_ok=True)
    stem = f"{j['cell']}_k{j['k']}"
    return os.path.join(d, stem + ".csv"), os.path.join(d, stem + ".log")


def _done(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return bool(rows) and rows[-1].get("completion_rate_mi") not in (None, "")
    except Exception:
        return False


def _reap_orphan_gateways():
    """Kill gateway JVMs whose python parent is gone.

    The env launches every JVM with setsid into its own session, so a group-kill of the
    evaluate process can never reach it, and env-side cleanup only runs on a clean exit.
    An orphaned gateway is precisely a logback-configured java process reparented to
    init (ppid 1); live gateways keep their evaluate as parent. 135 orphans accumulated
    overnight and took 60 GB down, so every run reaps before it starts.
    """
    import re
    reaped = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ")
            if b"java" not in cmd or b"logback" not in cmd:
                continue
            with open(f"/proc/{pid}/stat") as f:
                ppid = int(re.match(r"\d+ \(.*?\) . (\d+)", f.read()).group(1))
            if ppid == 1:
                os.kill(int(pid), 9)
                reaped += 1
        except (OSError, AttributeError, ValueError):
            continue
    if reaped:
        print(f"[janitor] reaped {reaped} orphan gateway JVMs", flush=True)


def run_one(j):
    _reap_orphan_gateways()
    csv_path, log_path = _paths(j)
    if _done(csv_path):
        return "cached"
    env = dict(os.environ)
    env.update({
        "GATEWAY_LIBS": os.path.join(
            REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"),
        "EVAL_CONFIG_PATH": os.path.join(HERE, "config_s2.yml"),
        "ORACLE_WIND_DIR": os.path.join(
            REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"),
        "ORACLE_EXPERIMENT": j["cell"],
        "ORACLE_OFFSET_ROWS": str(j["offset"]),
    })
    env.update(j.get("env", {}))       # a job's own env wins, e.g. the E config path
    cmd = [os.path.join(REPO, "drl-manager/.venv/bin/python"), "-m",
           "src.baselines.evaluate", "--experiment", j["cell"],
           "--global", j["arm"], "--local", "drain", "--episodes", "1",
           "--seed", str(SEED), "--reset-skip", str(j["k"]),
           "--output", csv_path]
    # The evaluate process spawns a JVM child. Killing only the python on timeout or
    # failure orphans the JVM; 135 of those accumulated overnight, exhausted 60 GB and
    # took the whole sweep down. The run therefore gets its own process group and the
    # WHOLE group is killed on the way out, success or not.
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=os.path.join(REPO, "drl-manager"),
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            rc = -1
        finally:
            try:
                os.killpg(proc.pid, 15)
                time.sleep(2)
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
    return "ok" if rc == 0 and _done(csv_path) else "failed"


def sweep(arms, todo=None):
    todo = todo or jobs(arms)
    t0 = time.time()
    counts = collections.Counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for j, res in zip(todo, ex.map(run_one, todo)):
            counts[res] += 1
            n = sum(counts.values())
            if n % 25 == 0 or res == "failed":
                print(f"[{n}/{len(todo)}] {j['arm']} {j['cell']} k{j['k']}: {res} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return dict(counts)


def read_cell(j):
    csv_path, _ = _paths(j)
    if not _done(csv_path):
        return None
    r = list(csv.DictReader(open(csv_path)))[-1]
    # always_defer is not a planner-family arm and emits no planner ledger columns; an
    # absent ledger has nothing to corrupt, so a missing field reads as zero. The env
    # level fields (completion, ontime, deadline_forced_count) exist for every arm and
    # stay mandatory.
    def _z(key):
        return float(r.get(key, 0) or 0)
    ok = all(float(r[k]) >= v for k, v in CONTRACT.items()) and \
        all(_z(z) == 0.0 for z in ZERO_FIELDS)
    return {"carbon": float(r["total_carbon_kg"]), "contract_ok": bool(ok),
            "completion_rate_mi": float(r["completion_rate_mi"]),
            "ontime_mi_share": float(r["ontime_mi_share"]),
            "ledger_columns_absent": [z for z in ZERO_FIELDS if z not in r],
            "violations": {z: _z(z) for z in ZERO_FIELDS if _z(z) != 0.0}}


def freeze_blind():
    """One arm by pooled carbon over every contract-valid cell, before any oracle runs."""
    table, invalid = {}, collections.Counter()
    for arm in BLINDS:
        vals, bad = [], 0
        for j in jobs((arm,)):
            row = read_cell(j)
            if row is None or not row["contract_ok"]:
                bad += 1
            else:
                vals.append(row["carbon"])
        table[arm] = {"pooled": (sum(vals) / len(vals)) if vals else None,
                      "valid_cells": len(vals), "invalid_cells": bad}
        invalid[arm] = bad
    everywhere = [a for a in BLINDS if invalid[a] == 0]
    art = {"arms": table, "valid_everywhere": everywhere,
           "expected_cells": len(g.cells()) * 3}
    if everywhere:
        art["frozen_blind"] = min(everywhere, key=lambda a: table[a]["pooled"])
        art["status"] = "FROZEN"
    else:
        art["status"] = "STOP_NO_VALID_BLIND"
    blob = json.dumps(art, sort_keys=True, indent=2)
    art_path = os.path.join(OUT, "blind_freeze.json")
    with open(art_path, "w") as f:
        f.write(blob)
    art["sha"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return art


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "blinds"
    os.makedirs(OUT, exist_ok=True)
    if phase == "blinds":
        print(json.dumps(sweep(BLINDS), indent=2))
    elif phase == "freeze":
        print(json.dumps(freeze_blind(), indent=2))
    elif phase == "e_blinds":
        todo = e_jobs("discovery", E_BLINDS)
        print(f"e_blinds: {len(todo)} runs on the fresh discovery turbines")
        print(json.dumps(sweep(E_BLINDS, todo=todo), indent=2))
    elif phase == "e_freeze":
        # One strongest blind by pooled total carbon over the three discovery windows,
        # frozen before any clean or corrupted number exists (E prereg section 3).
        table, invalid = {}, {}
        for arm in E_BLINDS:
            vals, bad = [], 0
            for jb in e_jobs("discovery", (arm,)):
                jb2 = dict(jb, dir=jb["dir"])
                row = read_cell(jb2)
                if row is None or not row["contract_ok"]:
                    bad += 1
                else:
                    vals.append(row["carbon"])
            table[arm] = {"pooled": (sum(vals) / len(vals)) if vals else None,
                          "valid": len(vals), "invalid": bad}
        everywhere = [a for a in E_BLINDS if table[a]["invalid"] == 0]
        art = {"arms": table, "valid_everywhere": everywhere}
        if everywhere:
            art["frozen_blind"] = min(everywhere, key=lambda a: table[a]["pooled"])
            art["status"] = "FROZEN"
        else:
            art["status"] = "STOP_NO_VALID_BLIND"
        with open(os.path.join(OUT, "e_blind_freeze.json"), "w") as f:
            f.write(json.dumps(art, sort_keys=True, indent=2))
        print(json.dumps(art, sort_keys=True, indent=2))
    elif phase == "e_main":
        fp = os.path.join(OUT, "e_blind_freeze.json")
        if not os.path.exists(fp) or json.load(open(fp)).get("status") != "FROZEN":
            raise RuntimeError("e_main runs only after the blind freeze is FROZEN")
        todo = e_jobs("discovery", E_ARMS, tier_mode=True)
        print(f"e_main: {len(todo)} runs")
        print(json.dumps(sweep(E_ARMS, todo=todo), indent=2))
    elif phase == "pilot_h":
        # DESIGN_PILOT: F settings with 32-PE jobs. Same cells, arms, window.
        split = json.load(open(os.path.join(HERE, "e_data_split.json")))["discovery"]
        pilot_cells = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
        arms = {"reservation_edf": {"g": "reservation_edf", "tier": False},
                "godeye": {"g": "perturbed_oracle_planner", "tier": "godeye"},
                "shuffle": {"g": "perturbed_oracle_planner", "tier": "shuffle"}}
        todo = []
        for m in (1, 2, 4):
            cfg = os.path.join(HERE, f"config_s2h_m{m}.yml")
            for aname, a in arms.items():
                for cell in pilot_cells:
                    k, off = split["windows_k"][0], split["offsets"][0]
                    # H exposes hosts as 32-PE VMs: the effective capacity the planner
                    # sentinel must see is the VM-PE total per DC, not the C-regime vector.
                    e = {"EVAL_CONFIG_PATH": cfg,
                         "PLANNER_EXPECTED_CAP": "640;512;640;512;192"}
                    if a["tier"]:
                        e.update({"PLANNER_PERTURB_TIER": a["tier"], "PLANNER_PERTURB_E": "1"})
                    todo.append({"arm": a["g"], "cell": cell, "k": k, "offset": off,
                                 "env": e, "dir": f"piloth_m{m}_{aname}"})
        print(f"pilot_h: {len(todo)} runs (32-PE jobs x 3 scarcity x 3 arms x 6 cells)")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "hz_p0":
        # Stage D P0 reward truth table (Codex hard blocker 2): replay four planner-family
        # arms on the V training block (training reward configuration) over the six
        # frozen training windows, before any policy is trained. Windows come from the
        # allowlist: --reset-skip k selects allowlist[k] on both sides.
        # P0_VARIANT selects the reward variant (Addendum C): "" = legacy block,
        # "physical" = config_stage_d_physical.yml; outputs land in p0<variant>_* dirs.
        variant = os.environ.get("P0_VARIANT", "").strip()
        suffix = f"_{variant}" if variant else ""
        man = json.load(open(os.path.join(HERE, f"stage_d_manifest{suffix}.json")))
        allow = [int(w["offset"]) for w in man["train_windows"]]
        cfg = os.path.join(HERE, f"config_stage_d{suffix}.yml")
        cal = os.path.join(HERE, "timecap_error_audit.json")
        cell = "sd_V_s2_r48_w72_c3_n35"
        arms = {"reactive_wait_planner": {"g": "reactive_wait_planner", "tier": False},
                "godeye": {"g": "perturbed_oracle_planner", "tier": "godeye"},
                "calibrated_shrink_v1": {"g": "perturbed_oracle_planner",
                                         "tier": "calibrated_shrink_v1"},
                "always_defer": {"g": "always_defer", "tier": False}}
        if variant == "dprime":
            # P0' (STAGE_D_PRIME_DESIGN §4 step 3): the three behaviours that differ only in
            # timing. ST = godeye (best window, on time), S = godeye with deferral forbidden
            # (start now), always_defer (defer until the mask routes it). Judged on the
            # discounted return by p0_verdict.judge_dprime.
            arms["godeye_nodefer"] = {"g": "perturbed_oracle_planner", "tier": "godeye",
                                      "extra": {"PLANNER_ALLOW_DEFER": "0"}}
        todo = []
        for aname, a in arms.items():
            for k, off in enumerate(allow):
                e = {"EVAL_CONFIG_PATH": cfg, **HZ_ENV, **a.get("extra", {})}
                if a["tier"]:
                    e.update({"PLANNER_PERTURB_TIER": a["tier"], "PLANNER_PERTURB_E": "1",
                              "PLANNER_PERTURB_CAL": cal})
                todo.append({"arm": a["g"], "cell": cell, "k": k, "offset": off,
                             "env": e, "dir": f"p0{suffix}_{aname}"})
        print(f"hz_p0{suffix}: {len(todo)} runs (4 arms x {len(allow)} training windows)")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "hz_opt":
        # Option four-gate rollouts (reports/OPTION_ACTION_DESIGN.md §5–§6, Addendum A):
        # every option arm on the V training block under the option overlay, over the six
        # development windows (the same train windows as P0'). OPT_WINDOWS="0" restricts
        # to k0 for the gate-3 smoke. The step-wise references B and ST of gate 1 are the
        # P0' run-6 rows (same windows, same block, defer mode) and are not rerun here.
        man = json.load(open(os.path.join(HERE, "stage_d_manifest_dprime_option.json")))
        allow = [int(w["offset"]) for w in man["train_windows"]]
        ref = json.load(open(os.path.join(HERE, "stage_d_manifest_dprime.json")))
        if [int(w["offset"]) for w in ref["train_windows"]] != allow:
            raise RuntimeError("option manifest train windows differ from the D' manifest; gate 1 references would not match")
        cfg = os.path.join(HERE, "config_stage_d_dprime_option.yml")
        cal = os.path.join(HERE, "timecap_error_audit.json")
        cell = "sd_V_s2_r48_w72_c3_n35"
        arms = {"oracle_opt": {"g": "perturbed_oracle_planner_opt", "tier": "godeye"},
                "shuffle_opt": {"g": "perturbed_oracle_planner_opt", "tier": "shuffle"},
                "anti_opt": {"g": "perturbed_oracle_planner_opt", "tier": "anti"},
                "shrink_opt": {"g": "perturbed_oracle_planner_opt", "tier": "calibrated_shrink_v1"},
                "persistence_opt": {"g": "persistence_planner_opt", "tier": False},
                "climatology_opt": {"g": "climatology_planner_opt", "tier": False},
                "reactive_opt": {"g": "reactive_wait_planner_opt", "tier": False},
                "nowait_opt": {"g": "nowait_planner", "tier": False},
                "always_hold": {"g": "always_hold", "tier": False}}
        ks = os.environ.get("OPT_WINDOWS", "").strip()
        ks = [int(x) for x in ks.split(",")] if ks else list(range(len(allow)))
        todo = []
        for aname, a in arms.items():
            for k in ks:
                e = {"EVAL_CONFIG_PATH": cfg, **HZ_ENV}
                if a["tier"]:
                    e.update({"PLANNER_PERTURB_TIER": a["tier"], "PLANNER_PERTURB_E": "1",
                              "PLANNER_PERTURB_CAL": cal})
                todo.append({"arm": a["g"], "cell": cell, "k": k, "offset": allow[k],
                             "env": e, "dir": f"opt_{aname}"})
        print(f"hz_opt: {len(todo)} runs ({len(arms)} arms x {len(ks)} windows)")
        print(json.dumps(sweep(tuple(sorted({a['g'] for a in arms.values()})), todo=todo), indent=2))
    elif phase == "hz_manifest":
        print(json.dumps(hz_manifest(), sort_keys=True, indent=2))
    elif phase == "hz_blinds":
        todo = hz_jobs("discovery", HZ_BLINDS)
        print(f"hz_blinds: {len(todo)} runs (x{HZ_MULT})")
        print(json.dumps(sweep(HZ_BLINDS, todo=todo), indent=2))
    elif phase == "hz_freeze":
        print(json.dumps(hz_freeze(), sort_keys=True, indent=2))
    elif phase == "hz_main":
        fp = os.path.join(OUT, f"hz_blind_freeze_m{HZ_MULT}.json")
        if not os.path.exists(fp) or json.load(open(fp)).get("status") != "FROZEN":
            raise RuntimeError("hz_main runs only after the blind freeze is FROZEN")
        todo = hz_jobs("discovery", HZ_ARMS, tier_mode=True)
        print(f"hz_main: {len(todo)} runs (x{HZ_MULT})")
        print(json.dumps(sweep(HZ_ARMS, todo=todo), indent=2))
    elif phase == "hz_corpus":
        # Stage D' timing-selectivity corpus (Q4): the truth-informed planner ST replayed on
        # the development windows under the D' config, dumping every slot decision and the
        # global observation of every step. Frozen once written; timing_selectivity.py scores
        # a V checkpoint on exactly these states.
        man = json.load(open(os.path.join(HERE, "stage_d_manifest_dprime.json")))
        allow = [int(w["offset"]) for w in man["train_windows"]]
        cfg = os.path.join(HERE, "config_stage_d_dprime.yml")
        cal = os.path.join(HERE, "timecap_error_audit.json")
        cell = "sd_V_s2_r48_w72_c3_n35"
        cdir = os.path.join(OUT, "dprime_corpus")
        os.makedirs(cdir, exist_ok=True)
        todo = []
        for k, off in enumerate(allow):
            dump = os.path.join(cdir, f"{cell}_k{k}_decisions.csv")
            if os.path.exists(dump):
                os.remove(dump)                       # the dump appends; one clean file per window
            todo.append({"arm": "perturbed_oracle_planner", "cell": cell, "k": k, "offset": off,
                         "env": {"EVAL_CONFIG_PATH": cfg, **HZ_ENV, "PLANNER_PERTURB_TIER": "godeye",
                                 "PLANNER_PERTURB_E": "1", "PLANNER_PERTURB_CAL": cal,
                                 "EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1"},
                         "dir": "dprime_corpus"})
        print(f"hz_corpus: {len(todo)} ST runs with decision + observation dumps")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "hz_margin_probe":
        # Stage D' mask margin (STAGE_D_PRIME_DESIGN §10): a saturated-dispatch probe on the
        # fixed development load. nowait_planner routes every job the step it appears, so
        # queues form and the route -> exec-start delay is at its worst. The margin is then
        # set mechanically by margin_probe.py: ceil(max_delay / timestep) + 1 steps. Never
        # from carbon or training. Runs under the D' config on the six training windows.
        man = json.load(open(os.path.join(HERE, "stage_d_manifest_dprime.json")))
        allow = [int(w["offset"]) for w in man["train_windows"]]
        if os.environ.get("MARGIN_PROBE_CELLS", "").strip() == "all":
            # design §16 Q3: the six D' cells x six development windows, no forecast effect read
            cfg = os.path.join(HERE, "config_stage_d_eval_dprime_dev.yml")
            todo = [{"arm": "nowait_planner", "cell": f"sde_{c}_godeye", "k": k, "offset": off,
                     "env": {"EVAL_CONFIG_PATH": cfg, **HZ_ENV}, "dir": "dprime_margin_probe_cells"}
                    for c in HZ_PILOT_CELLS for k, off in enumerate(allow)]
        else:
            cfg = os.path.join(HERE, "config_stage_d_dprime.yml")
            cell = "sd_V_s2_r48_w72_c3_n35"
            todo = [{"arm": "nowait_planner", "cell": cell, "k": k, "offset": off,
                     "env": {"EVAL_CONFIG_PATH": cfg, **HZ_ENV}, "dir": "dprime_margin_probe"}
                    for k, off in enumerate(allow)]
        print(f"hz_margin_probe: {len(todo)} saturated-dispatch runs")
        print(json.dumps(sweep(("nowait_planner",), todo=todo), indent=2))
    elif phase == "hz_decomp":
        # Post-verdict mechanism diagnostic (HZ_DECOMPOSITION_DIAGNOSTIC.md): the S arm,
        # the truth-informed planner with deferral forbidden, on the confirmation set.
        # B (frozen blind) and ST (clean godeye) rows already exist and are reused.
        todo = hz_jobs("confirmation", ["godeye"], tier_mode=True)
        for j in todo:
            j["env"] = dict(j["env"], PLANNER_ALLOW_DEFER="0")
            j["dir"] = f"hz_conf_m{HZ_MULT}_tier_godeye_nodefer"
        print(f"hz_decomp: {len(todo)} S runs (x{HZ_MULT}, PLANNER_ALLOW_DEFER=0)")
        print(json.dumps(sweep(["godeye"], todo=todo), indent=2))
    elif phase == "hz_confirm":
        # One-shot. The frozen blind and every arm run on the sealed windows only after
        # the discovery verdict PASSed; the reader, not this phase, decides the outcome.
        vp = os.path.join(OUT, f"hz_verdict_discovery_m{HZ_MULT}.json")
        if not os.path.exists(vp) or json.load(open(vp)).get("verdict") != "PASS_HZ_DISCOVERY":
            raise RuntimeError("hz_confirm runs only after the discovery verdict PASSed")
        fz = json.load(open(os.path.join(OUT, f"hz_blind_freeze_m{HZ_MULT}.json")))
        todo = hz_jobs("confirmation", (fz["frozen_blind"],)) + \
            hz_jobs("confirmation", HZ_ARMS, tier_mode=True)
        print(f"hz_confirm: {len(todo)} runs (x{HZ_MULT})")
        print(json.dumps(sweep(HZ_ARMS, todo=todo), indent=2))
    elif phase == "pilot_hz":
        # DESIGN_PILOT, Level-1 spiral: H fleet on zero-floor hosts (marginal carbon).
        # Blind arms mirror toy_lever.py (nowait = run_now, reactive_wait = myopic) plus
        # the capacity blind; information arms truth / shuffle / anti.
        split = json.load(open(os.path.join(HERE, "e_data_split.json")))["discovery"]
        pilot_cells = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
        arms = {"nowait": {"g": "nowait_planner", "tier": False},
                "reactive_wait": {"g": "reactive_wait_planner", "tier": False},
                "reservation_edf": {"g": "reservation_edf", "tier": False},
                "godeye": {"g": "perturbed_oracle_planner", "tier": "godeye"},
                "shuffle": {"g": "perturbed_oracle_planner", "tier": "shuffle"},
                "anti": {"g": "perturbed_oracle_planner", "tier": "anti"}}
        # Windows: unclaimed k values (S2 burned 1/9/17/25/33/41, E/H hold 2/10/18 and
        # the sealed 26/34/42). Design work lives on k=3,4 so the formal discovery
        # windows k=10/18 stay unread. Offsets follow the simulator's 1009*k mod range.
        ks = [int(x) for x in os.environ.get("PILOT_HZ_K", "3,4").split(",")]
        offset_range = int(g.base_block().get("green_episode_offset_range", 44950))
        todo = []
        for m in (1, 2):
            cfg = os.path.join(HERE, f"config_s2hz_m{m}.yml")
            for aname, a in arms.items():
                for cell in pilot_cells:
                    for k in ks:
                        off = (1009 * k) % offset_range
                        # Zero-floor hosts: the planner must not subtract the C-regime
                        # 332 W fleet floor from green (awake hosts draw 1 W here).
                        e = {"EVAL_CONFIG_PATH": cfg,
                             "PLANNER_EXPECTED_CAP": "640;512;640;512;192",
                             "PLANNER_STATIC_TOTAL_W": "0"}
                        if a["tier"]:
                            e.update({"PLANNER_PERTURB_TIER": a["tier"],
                                      "PLANNER_PERTURB_E": "1"})
                        todo.append({"arm": a["g"], "cell": cell, "k": k, "offset": off,
                                     "env": e, "dir": f"pilothz_m{m}_{aname}"})
        print(f"pilot_hz: {len(todo)} runs (zero-floor hosts x 2 scarcity x 6 arms x 6 cells x k={ks})")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "pilot_g":
        # DESIGN_PILOT: F sweep with idle hosts powered down. Same cells, arms, window.
        split = json.load(open(os.path.join(HERE, "e_data_split.json")))["discovery"]
        pilot_cells = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
        arms = {"reservation_edf": {"g": "reservation_edf", "tier": False},
                "godeye": {"g": "perturbed_oracle_planner", "tier": "godeye"},
                "shuffle": {"g": "perturbed_oracle_planner", "tier": "shuffle"}}
        todo = []
        for m in (1, 2, 4):
            cfg = os.path.join(HERE, f"config_s2g_m{m}.yml")
            for aname, a in arms.items():
                for cell in pilot_cells:
                    k, off = split["windows_k"][0], split["offsets"][0]
                    e = {"EVAL_CONFIG_PATH": cfg}
                    if a["tier"]:
                        e.update({"PLANNER_PERTURB_TIER": a["tier"], "PLANNER_PERTURB_E": "1"})
                    todo.append({"arm": a["g"], "cell": cell, "k": k, "offset": off,
                                 "env": e, "dir": f"pilotg_m{m}_{aname}"})
        print(f"pilot_g: {len(todo)} runs (power-down x 3 scarcity x 3 arms x 6 cells)")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "pilot_f":
        # DESIGN_PILOT (2026-09-03): sweep the green-scarcity knob on the uniform-brown
        # variant to find a setting where godeye beats the strongest blind AND shuffle
        # does not. Discovery windows only; outside every prereg; not a verdict.
        split = json.load(open(os.path.join(HERE, "e_data_split.json")))["discovery"]
        # cells spanning concurrency (where the E effect varied) x two job counts
        pilot_cells = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
        arms = {"reservation_edf": {"g": "reservation_edf", "tier": False},
                "godeye": {"g": "perturbed_oracle_planner", "tier": "godeye"},
                "shuffle": {"g": "perturbed_oracle_planner", "tier": "shuffle"}}
        todo = []
        for m in (1, 2, 4, 8):
            cfg = os.path.join(HERE, f"config_s2f_m{m}.yml")
            for aname, a in arms.items():
                for cell in pilot_cells:
                    for k, off in zip(split["windows_k"][:1], split["offsets"][:1]):
                        e = {"EVAL_CONFIG_PATH": cfg}
                        if a["tier"]:
                            e.update({"PLANNER_PERTURB_TIER": a["tier"],
                                      "PLANNER_PERTURB_E": "1"})
                        todo.append({"arm": a["g"], "cell": cell, "k": k, "offset": off,
                                     "env": e, "dir": f"pilotf_m{m}_{aname}"})
        print(f"pilot_f: {len(todo)} runs (4 scarcity x 3 arms x 6 cells x 1 window)")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "pilot_shrink":
        # DESIGN_PILOT (2026-09-02): amplitude-shrinkage tiers on the DISCOVERY window
        # k=1 only. Exploratory, outside every prereg; results may not enter any verdict.
        # Blind reference = stage A nowait k1; godeye reference = A-prime tier_godeye k1
        # (sigma zero, semantics-invariant between v1 and v2).
        names = stable_region_cells()
        todo = []
        for tier in ("shrink75", "shrink50", "shrink25", "shrink0"):
            e = {"PLANNER_PERTURB_TIER": tier, "PLANNER_PERTURB_V2": "1",
                 "PLANNER_PERTURB_PILOT": "1"}
            for j in jobs(("perturbed_oracle_planner",), cell_names=names,
                          which="discovery"):
                if j["k"] != 1:
                    continue
                todo.append({**j, "dir": f"pilot_tier_{tier}", "env": e})
        print(f"pilot_shrink: {len(todo)} runs (DESIGN_PILOT, discovery k=1 only)")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "ladder_v2":
        # One-shot CONFIRMATION sweep (k=25/33/41): the frozen blind re-run beside the
        # eight v2 tiers on the frozen region. No partial reads; the verdict reader is
        # the only thing that interprets these files.
        names = stable_region_cells()
        cal = os.path.join(HERE, "dc_residual_cal.json")
        todo = [dict(j, dir="conf_nowait_planner")
                for j in jobs(("nowait_planner",), cell_names=names,
                              which="confirmation")]
        for tier in TIERS_V2:
            e = {"PLANNER_PERTURB_TIER": tier, "PLANNER_PERTURB_V2": "1"}
            if tier == "checkpoint_residual_surrogate_v2":
                e["PLANNER_PERTURB_CAL"] = cal
            for j in jobs(("perturbed_oracle_planner",), cell_names=names,
                          which="confirmation"):
                todo.append({**j, "dir": f"conf_tier_{tier}", "env": e})
        print(f"ladder_v2: {len(names)} cells x (1 blind + {len(TIERS_V2)} tiers) "
              f"x 3 confirmation windows = {len(todo)} runs")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "aprime":
        vp = os.path.join(OUT, "stage_a_verdict.json")
        if not os.path.exists(vp):
            raise RuntimeError("no stage A verdict; the ladder runs on its region only")
        names = stable_region_cells()
        cal = os.path.join(HERE, "timecap_cal.json")
        todo = []
        for tier in TIERS:
            e = {"PLANNER_PERTURB_TIER": tier}
            if tier == "timecap_cal":
                e["PLANNER_PERTURB_CAL"] = cal
            for j in jobs(("perturbed_oracle_planner",), cell_names=names):
                todo.append({**j, "dir": f"tier_{tier}", "env": e})
        print(f"ladder: {len(names)} cells x {len(TIERS)} tiers x 3 windows "
              f"= {len(todo)} runs")
        print(json.dumps(sweep(("perturbed_oracle_planner",), todo=todo), indent=2))
    elif phase == "oracles":
        fp = os.path.join(OUT, "blind_freeze.json")
        if not os.path.exists(fp):
            raise RuntimeError("no freeze artifact; the blind phase decides first")
        if json.load(open(fp)).get("status") != "FROZEN":
            raise RuntimeError("blind freeze is not FROZEN; oracles do not run")
        print(json.dumps(sweep(ORACLES), indent=2))
    else:
        raise SystemExit(f"unknown phase {phase!r}")


if __name__ == "__main__":
    main()
