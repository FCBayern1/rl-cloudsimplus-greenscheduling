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
WORKERS = int(os.environ.get("S2_WORKERS", "5"))
SEED = 42
CONTRACT = {"completion_rate_mi": 0.995, "ontime_mi_share": 0.995}
ZERO_FIELDS = ("deadline_forced_count", "planner_n_stale_dropped",
               "planner_n_unplanned_start", "planner_n_wrong_dc",
               "planner_n_dispatched_never_started", "planner_running_pes_over_cap")


def windows():
    return g.windows(44950)["discovery"]


def jobs(arms):
    out = []
    for arm in arms:
        for cell in g.cells():
            name = g.cell_name(cell)
            for k, off in windows():
                out.append({"arm": arm, "cell": name, "k": k, "offset": off})
    return out


def _paths(j):
    d = os.path.join(OUT, j["arm"])
    os.makedirs(d, exist_ok=True)
    stem = f"{j['cell']}_k{j['k']}"
    return os.path.join(d, stem + ".csv"), os.path.join(d, stem + ".log")


def _done(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return bool(rows) and rows[-1].get("completion_rate_mi") not in (None, "")
    except Exception:
        return False


def run_one(j):
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
    cmd = [os.path.join(REPO, "drl-manager/.venv/bin/python"), "-m",
           "src.baselines.evaluate", "--experiment", j["cell"],
           "--global", j["arm"], "--local", "drain", "--episodes", "1",
           "--seed", str(SEED), "--reset-skip", str(j["k"]),
           "--output", csv_path]
    with open(log_path, "w") as log:
        r = subprocess.run(cmd, cwd=os.path.join(REPO, "drl-manager"),
                           env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=3600)
    return "ok" if r.returncode == 0 and _done(csv_path) else "failed"


def sweep(arms):
    todo = jobs(arms)
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
