"""Round 1: freeze the blind, then measure. Two phases, in that order only.

Codex 2026-09-01. Phase A runs the four causal blinds over every required DISCOVERY
instance and freezes ONE arm by pooled carbon, before any oracle is solved and before any
EVPI exists. Phase B then solves the exact model and scores EVPI against that frozen arm.

The order is not a convenience. Choosing the blind after seeing EVPI, or taking the best
blind per instance, would make the denominator a function of the result.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_blinds as cbl  # noqa: E402
import instance_gen as ig  # noqa: E402
import round0 as r0  # noqa: E402
from exact_oracle import solve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TRACKED = ("g1/tb13/round1.py", "g1/tb13/round0.py", "g1/tb13/exact_oracle.py",
           "g1/tb13/causal_blinds.py", "g1/tb13/instance_gen.py",
           "g1/tb13/data_split.txt", "reports/TB13_SCREEN_PREREG.md")
EXPECTED_INSTANCES = 36 * 3 * 3 * 4          # expanded units x n_jobs x wait_cap x budget
EVPI_GATE = 0.15
TIME_LIMIT_S = 30.0
OUTER_WORKERS = 2                            # each CP-SAT already takes four threads


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(round0_dir):
    """Refuse to start unless the tree, the inputs and the provenance all check out."""
    dirty = subprocess.check_output(
        ["git", "-C", REPO, "status", "--porcelain", "--"] + list(TRACKED),
        text=True).strip()
    if dirty:
        raise RuntimeError("refusing to run Round 1 from a dirty tree:\n" + dirty)
    manifest = json.load(open(os.path.join(round0_dir, "round0_manifest.json")))
    actual = _sha(os.path.join(round0_dir, "round0_anchors.json"))
    if actual != manifest["round0_anchors.json"]:
        raise RuntimeError("round0_anchors.json does not match its manifest hash")
    commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"],
                                     text=True).strip()
    return commit, {f: _sha(os.path.join(REPO, f)) for f in TRACKED}, manifest


def build_instances(round0_dir):
    """Exactly the registered cross product, in a fixed order."""
    anchors = json.load(open(os.path.join(round0_dir, "round0_anchors.json")))
    expanded = anchors["expanded"]
    conf = set(r0.confirmation_pool())
    disc = set(r0.discovery_pool())
    out = []
    for unit in sorted(expanded, key=lambda u: r0.anchor_sha(u)):
        used = {t for site in unit["triplet"] for t in site}
        if used & conf:
            raise RuntimeError(f"a confirmation turbine appeared in {sorted(used & conf)}")
        if not used <= disc:
            raise RuntimeError("a turbine outside the discovery pool appeared")
        for n_jobs, wait_cap, bf in itertools.product(
                ig.N_JOBS, ig.WAIT_CAP_ROWS, ig.BUDGET_FRACTION):
            ax = dict(unit)
            ax.update(n_jobs=n_jobs, wait_cap=wait_cap, budget_fraction=bf,
                      runtime_set=ig.RUNTIME_ROWS_TIER1, turbines=unit["triplet"],
                      offset=unit["season_offset"])
            out.append(ax)
    if len(out) != EXPECTED_INSTANCES:
        raise RuntimeError(f"built {len(out)} instances, expected {EXPECTED_INSTANCES}")
    ids = {json.dumps({k: v for k, v in a.items() if k != "runtime_set"},
                      sort_keys=True, default=str) for a in out}
    if len(ids) != EXPECTED_INSTANCES:
        raise RuntimeError(f"only {len(ids)} of {EXPECTED_INSTANCES} instances are unique")
    return out


def _instance(ax, seed=0):
    return ig.build_instance(ax, seed=seed)


def _blinds_one(ax):
    sc, prov = _instance(ax)
    clim = prov["clim_residual_green"]
    row = {"carbon": {}, "valid": {}}
    for name, fn in cbl.BLINDS.items():
        c, a = fn(sc, clim)
        row["carbon"][name] = c
        row["valid"][name] = c is not None
    row["rho_residual"] = prov["rho_residual"]
    row["pes_share"] = prov["pes_share"]
    return row


def phase_a(instances, out_dir, provenance):
    """Run the blinds and freeze one arm by pooled carbon. No oracle is solved here."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        rows = list(ex.map(_blinds_one, instances, chunksize=8))
    names = list(cbl.BLINDS)
    valid_everywhere = [n for n in names if all(r["valid"][n] for r in rows)]
    pooled = {n: (sum(r["carbon"][n] for r in rows) / len(rows))
              if n in valid_everywhere else None for n in names}
    if not valid_everywhere:
        art = {"status": "STOP_NO_VALID_BLIND", "pooled": pooled,
               "instances": len(rows), "provenance": provenance}
        _write(os.path.join(out_dir, "round1_blind_freeze.json"), art)
        return None, art
    frozen = min(valid_everywhere, key=lambda n: pooled[n])
    art = {"status": "FROZEN", "frozen_blind": frozen, "pooled": pooled,
           "valid_everywhere": valid_everywhere, "instances": len(rows),
           "wall_seconds": round(time.time() - t0, 2), "provenance": provenance,
           "per_instance_carbon": [r["carbon"] for r in rows]}
    _write(os.path.join(out_dir, "round1_blind_freeze.json"), art)
    return frozen, art


def _oracle_one(ax):
    sc, prov = _instance(ax)
    res = solve(sc, time_limit_s=TIME_LIMIT_S)
    return {"carbon_status": res["carbon_status"], "exact": res["exact"],
            "carbon": res["carbon"], "carbon_gap": res["carbon_gap"],
            "total_wait": res["total_wait"], "wait_status": res["wait_status"],
            "n_waiting": (None if res["assign"] is None else
                          sum(1 for i, (_d, s) in res["assign"].items()
                              if s > int(sc.a[i]))),
            "n_jobs": int(sc.n), "rho_residual": prov["rho_residual"],
            "pes_share": prov["pes_share"]}


def phase_b(instances, frozen, freeze_art, out_dir, provenance):
    """Solve the exact model and score EVPI against the already frozen blind."""
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        orc = list(ex.map(_oracle_one, instances, chunksize=4))
    blind_c = [c[frozen] for c in freeze_art["per_instance_carbon"]]
    rows = []
    for ax, o, bc in zip(instances, orc, blind_c):
        evpi = None if (o["carbon"] is None or not bc) else (bc - o["carbon"]) / bc
        route_frac = None if o["n_waiting"] is None else 1.0 - o["n_waiting"] / o["n_jobs"]
        gates = {
            "optimal": bool(o["exact"]),
            "evpi_ge_15": bool(evpi is not None and evpi >= EVPI_GATE),
            "wait_and_route_both_20pc": bool(
                route_frac is not None and 0.20 <= route_frac <= 0.80),
            "pes_share_ge_25pc": bool(ax["pes_per_job"] / ig.CAP_PES_PER_SITE >= 0.25),
        }
        rows.append({"axes": {k: v for k, v in ax.items() if k != "runtime_set"},
                     "oracle": o, "blind_carbon": bc, "evpi": evpi, "gates": gates,
                     "advances": all(gates.values())})
    summary = {
        "frozen_blind": frozen, "instances": len(rows),
        "optimal": sum(1 for r in rows if r["gates"]["optimal"]),
        "unresolved": sum(1 for r in rows if not r["gates"]["optimal"]),
        "evpi_ge_15": sum(1 for r in rows if r["gates"]["evpi_ge_15"]),
        "advancing": sum(1 for r in rows if r["advances"]),
        "evpi_quantiles": _quantiles([r["evpi"] for r in rows if r["evpi"] is not None]),
        "gate_1_threshold_in_band": "UNDEFINED: prereg section 5 gate one has no "
                                    "mechanical definition; reported, not applied",
        "wall_seconds": round(time.time() - t0, 2), "provenance": provenance,
    }
    _write(os.path.join(out_dir, "round1_rows.jsonl"), rows, lines=True)
    _write(os.path.join(out_dir, "round1_summary.json"), summary)
    return summary


def _quantiles(v):
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {q: float(np.percentile(a, q)) for q in (0, 10, 25, 50, 75, 90, 100)}


def _write(path, obj, lines=False):
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        if lines:
            f.write("\n".join(json.dumps(o, sort_keys=True, default=str) for o in obj) + "\n")
        else:
            f.write(json.dumps(obj, sort_keys=True, indent=2, default=str))
    os.replace(tmp, path)


def main(round0_dir=None, out_dir=None):
    round0_dir = round0_dir or os.path.join(HERE, "round0_out")
    out_dir = out_dir or os.path.join(HERE, "round1_out")
    os.makedirs(out_dir, exist_ok=True)
    commit, shas, manifest = preflight(round0_dir)
    provenance = {"commit": commit, "file_shas": shas, "round0_manifest": manifest,
                  "expected_instances": EXPECTED_INSTANCES, "evpi_gate": EVPI_GATE,
                  "time_limit_s": TIME_LIMIT_S}
    inst = build_instances(round0_dir)
    frozen, art = phase_a(inst, out_dir, provenance)
    if frozen is None:
        _write(os.path.join(out_dir, "round1_summary.json"),
               {"status": "STOP_NO_VALID_BLIND", "provenance": provenance})
        return {"status": "STOP_NO_VALID_BLIND"}
    return phase_b(inst, frozen, art, out_dir, provenance)


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items()
                      if k not in ("provenance",)}, sort_keys=True, indent=2))
