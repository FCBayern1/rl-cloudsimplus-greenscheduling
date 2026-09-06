"""Forecast-quality ladder runner (reports/ERROR_LADDER_PLANNER_PREREG.md, frozen 24b5de60).

Per development window (scene_v2_dev.json):
  dump     one replay of the blind arm on the dev DEFER twin with the decision + observation
           dump: the simulator's own green per site and step (truth curve, exact) and the
           jobs as the policy meets them (first sighting step, PEs, MI, time to deadline)
  rungs    truth; shrink lambda in {0.75, 0.5, 0.25, 0} around the site's full-year 2021 mean;
           shuffle (seeded permutation of the window's truth curve); anti (reversed)
  solve    CP-SAT, every rung OPTIMAL (else STOP_SOLVER_RUNG_UNRESOLVED), model-settled on truth
  replay   each schedule through schedule_replay on the dev OFFSET twin; closure per rung
           (|C_sim - C_model| / C_model <= 3 %, per-job site and start, counters zero)
  gates    L1 headroom (lambda = 0 as the no-forecast reference), L2 load-bearing rung
Nothing but the truth rung is read until the truth rung closes on every window.

Usage: python ladder_run.py dump | solve | replay | judge
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "drl-manager"))
from ladder_planner import (  # noqa: E402
    HOST_FLOOR_MW, MW_PER_PE, F_BROWN, F_GREEN, Job, build_instance, preflight_factors,
    quantisation_bound_kg, runtime_steps, settle, solve)

OUT = os.path.join(HERE, "stage_a_out")
LAD = os.path.join(OUT, "ladder")
LAMBDAS = (0.75, 0.5, 0.25, 0.0)
RUNGS = ("truth", "shrink_0.75", "shrink_0.5", "shrink_0.25", "shrink_0", "shuffle", "anti")
CLOSURE_REL = 0.03
QUANT_FRAC = 0.001
C_BROWN_REF = 0.01897
HEADROOM_REL, HEADROOM_ABS = 0.15, 0.05 * C_BROWN_REF
LOSS_MIN, HEADROOM_SHARE = 0.05, 0.80
COMPRESSED_DIVISOR = 3000.0


# ── pure pieces ───────────────────────────────────────────────────────────────────────
def jobs_from_dump(rows, vm_pe_mips, cpu_util, timestep_sec=1.0):
    """Jobs as the simulator presented them: first sighting step = arrival, deadline step =
    arrival + floor(ttd / timestep) at that sighting. Pure."""
    first = {}
    for r in rows:
        cid = int(float(r.get("cloudlet_id", -1) or -1))
        if cid < 0:
            continue
        step = int(float(r["step"]))
        if cid not in first or step < first[cid]["step"]:
            ttd_raw = r.get("ttd_sec")
            if ttd_raw in (None, "", "None"):
                raise ValueError("the dump has no ttd_sec (raw seconds from the planner channel); "
                                 "time_to_deadline is the normalised policy feature and cannot be used")
            first[cid] = {"step": step, "pes": int(float(r["pes"])), "mi": float(r["mi"]),
                          "ttd": float(ttd_raw), "present": float(r.get("deadline_present", 1) or 0) >= 0.5}
    jobs = []
    for cid, f in sorted(first.items()):
        r = runtime_steps(f["mi"], vm_pe_mips, cpu_util, timestep_sec)
        deadline = f["step"] + int(np.floor(f["ttd"] / timestep_sec)) if f["present"] else 10 ** 9
        jobs.append(Job(id=cid, arrival=f["step"], runtime=r, pes=f["pes"], deadline=deadline))
    return jobs


def rung_curve(truth_w, rung, mu_w, seed_key):
    """Pure. truth_w: (sites, T) W. shrink around mu (per site), shuffle = seeded permutation
    of each site's window series, anti = reversed; all clipped at 0."""
    G = np.asarray(truth_w, dtype=np.float64)
    if rung == "truth":
        return G.copy()
    if rung.startswith("shrink_"):
        lam = float(rung.split("_")[1])
        mu = np.asarray(mu_w, dtype=np.float64).reshape(-1, 1)
        return np.maximum(0.0, mu + lam * (G - mu))
    if rung == "shuffle":
        out = np.empty_like(G)
        for d in range(G.shape[0]):
            seed = int(hashlib.sha256(f"{seed_key}:shuffle:{d}".encode()).hexdigest()[:8], 16)
            perm = np.random.default_rng(seed).permutation(G.shape[1])
            out[d] = G[d][perm]
        return np.maximum(0.0, out)
    if rung == "anti":
        return np.maximum(0.0, G[:, ::-1].copy())
    raise ValueError(rung)


def closure_check(c_model, c_sim, ledger_rows, schedule, counters):
    """Per rung and window (B2 + C5). ledger_rows: option-ledger rows of the replay
    (id, dc, t_s, route_to_start_steps, stale); schedule: {id: (site, start)}; counters: dict
    of the zero-required fields. Returns {"pass", "violations", "rel_err"}."""
    v = []
    rel = abs(c_sim - c_model) / c_model if c_model > 0 else float("inf")
    if rel > CLOSURE_REL:
        v.append(f"carbon rel err {rel:.4f} > {CLOSURE_REL}")
    seen = {}
    for r in ledger_rows:
        jid = int(float(r["id"]))
        seen[jid] = seen.get(jid, 0) + 1
        if str(r.get("stale", "")).lower() in ("true", "1"):
            v.append(f"id {jid} stale")
            continue
        site, start = schedule.get(jid, (None, None))
        if site is None:
            v.append(f"id {jid} not in the schedule")
            continue
        if int(float(r["dc"])) != site:
            v.append(f"id {jid} ran on site {r['dc']} not {site}")
        ts = r.get("t_s")
        if ts in (None, "", "None"):
            v.append(f"id {jid} no start event")
        elif abs(float(ts) - start) > 1.0:
            v.append(f"id {jid} started {float(ts):.2f} planned {start}")
    for jid, n in seen.items():
        if n != 1:
            v.append(f"id {jid} appears {n} times")
    for k, val in counters.items():
        if float(val or 0) != 0:
            v.append(f"{k} = {val} != 0")
    return {"pass": not v, "violations": v, "rel_err": rel}


def gate_l1(c_flat, c_truth):
    """Per-window headroom = C(schedule(lambda=0)) - C(schedule(truth)), simulator-settled."""
    out = {}
    for k in c_truth:
        gap = c_flat[k] - c_truth[k]
        ok = c_flat[k] > 0 and gap / c_flat[k] >= HEADROOM_REL and gap >= HEADROOM_ABS
        out[k] = {"headroom": gap, "rel": gap / c_flat[k] if c_flat[k] else None, "valid": bool(ok)}
    return out


def gate_l2(loss_by_rung, headroom, c_truth):
    """loss_by_rung: {rung: {k: loss}} on valid windows; load-bearing iff pooled loss >= 5 % of
    pooled truth carbon and >= 80 % of headroom lies in windows with loss > 0."""
    valid = [k for k, h in headroom.items() if h["valid"]]
    pooled_truth = sum(c_truth[k] for k in valid)
    total_h = sum(headroom[k]["headroom"] for k in valid)
    out = {}
    for rung, losses in loss_by_rung.items():
        pooled = sum(losses[k] for k in valid)
        share = sum(headroom[k]["headroom"] for k in valid if losses[k] > 0) / total_h if total_h > 0 else 0.0
        out[rung] = {"pooled_loss": pooled, "pooled_loss_rel": pooled / pooled_truth if pooled_truth else None,
                     "harm_headroom_share": share,
                     "load_bearing": bool(pooled_truth > 0 and pooled >= LOSS_MIN * pooled_truth and share >= HEADROOM_SHARE)}
    return out


# ── runner ────────────────────────────────────────────────────────────────────────────
def _dev():
    d = json.load(open(os.path.join(OUT, "scene_v2_dev.json")))
    assert d["status"] == "OK"
    return d["dev_offsets"]


def _mu_w(cfg_block):
    """Per-site full-year 2021 mean green in W, the simulator's conversion (kW * 1000 / divisor)."""
    split = os.path.join(REPO, "cloudsimplus-gateway", "src", "main", "resources", "windProduction", "split")
    div = float(cfg_block.get("compressed_power_divisor", COMPRESSED_DIVISOR))
    mus = []
    for d in sorted(cfg_block["datacenters"], key=lambda x: int(x["datacenter_id"])):
        tot = 0.0
        for t in d.get("turbine_ids", []) or []:
            rows = list(csv.DictReader(open(os.path.join(split, f"Turbine_{t}_2021.csv"))))
            col = "Patv" if "Patv" in rows[0] else [c for c in rows[0] if "patv" in c.lower() or "power" in c.lower()][0]
            tot += float(np.mean([float(r[col] or 0) for r in rows]))
        mus.append(tot * 1000.0 / div)
    return mus


def _evaluate(cfg, cell, k, offset, arm, out_csv, extra_env):
    env = dict(os.environ)
    env.update({"GATEWAY_LIBS": os.path.join(REPO, "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"),
                "EVAL_CONFIG_PATH": cfg, "ORACLE_EXPERIMENT": cell, "ORACLE_OFFSET_ROWS": str(offset),
                "ORACLE_WIND_DIR": os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"),
                "PLANNER_EXPECTED_CAP": "640;512;640;512;192", "PLANNER_STATIC_TOTAL_W": "0", **extra_env})
    cmd = [os.path.join(REPO, "drl-manager/.venv/bin/python"), "-m", "src.baselines.evaluate",
           "--experiment", cell, "--global", arm, "--local", "drain", "--episodes", "1",
           "--seed", "42", "--reset-skip", str(k), "--output", out_csv]
    with open(out_csv.replace(".csv", ".log"), "w") as log:
        rc = subprocess.run(cmd, cwd=os.path.join(REPO, "drl-manager"), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return rc == 0 and os.path.exists(out_csv)


def cmd_dump():
    import run_stage_a as rs
    dev = _dev()
    cfg, cell = rs.scene_dev_config("defer")
    os.makedirs(os.path.join(LAD, "dump"), exist_ok=True)
    for k, off in enumerate(dev):
        d = os.path.join(LAD, "dump", f"k{k}_decisions.csv")
        for p in (d, d.replace("_decisions.csv", "_decisions_obs.npz")):
            if os.path.exists(p):
                os.remove(p)
        ok = _evaluate(cfg, cell, k, off, "reactive_wait_planner", os.path.join(LAD, "dump", f"k{k}.csv"),
                       {"EVAL_DECISION_DUMP": d, "EVAL_DECISION_DUMP_OBS": "1"})
        print(f"dump k{k} offset {off}: {'ok' if ok else 'FAILED'}", flush=True)


def cmd_solve(rungs=RUNGS):
    """Solves the given rungs. The truth rung is solved and closed first (`--truth-only`);
    the other six are solved only after truth closure (the freeze rule: no reading of the
    other rungs' carbon before tests, the quantisation preflight and the truth closure)."""
    import yaml
    import run_stage_a as rs
    dev = _dev()
    cfg_path, cell = rs.scene_dev_config("defer")
    blk = yaml.safe_load(open(cfg_path))[cell]
    preflight_factors(blk["datacenters"])
    mu = _mu_w(blk)
    cap = [640, 512, 640, 512, 192]
    mips = float(blk["datacenters"][0].get("vm_pe_mips", 40000)); u = float(blk.get("cloudlet_cpu_utilization", 1.0))
    os.makedirs(os.path.join(LAD, "solve"), exist_ok=True)
    sp = os.path.join(LAD, "solve_summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    summary = {int(k): v for k, v in summary.items()}
    for k, off in enumerate(dev):
        rows = list(csv.DictReader(open(os.path.join(LAD, "dump", f"k{k}_decisions.csv"))))
        z = np.load(os.path.join(LAD, "dump", f"k{k}_decisions_obs.npz"))
        truth = np.asarray(z["dc_current_green_power_w"], dtype=np.float64).T      # (sites, T)
        jobs = jobs_from_dump(rows, mips, u)
        T = truth.shape[1]
        need = max(j.latest + j.runtime for j in jobs) + 1
        if need > T:                                                                  # pad with the last value
            truth = np.concatenate([truth, np.repeat(truth[:, -1:], need - T, axis=1)], axis=1)
        qb = quantisation_bound_kg(truth.shape[0], truth.shape[1])
        prev = summary.get(k, {})
        summary[k] = {"offset": off, "n_jobs": len(jobs), "T": int(truth.shape[1]), "quantisation_bound_kg": qb,
                      "quantisation_ok": qb <= QUANT_FRAC * C_BROWN_REF, "mu_w": mu, "rungs": prev.get("rungs", {})}
        inst_truth = build_instance(jobs, cap, truth)
        for rung in rungs:
            G = rung_curve(truth, rung, mu, seed_key=f"ladder:{off}")
            inst = build_instance(jobs, cap, G)
            res = solve(inst)
            rec = {"status": res["status"], "wall_s": res.get("wall_s")}
            if res["status"] == "OPTIMAL":
                st = settle(inst_truth, res["schedule"])
                rec.update({"J_on_rung": res["J_int"], "C_model_truth_kg": st["C_kg"], "J_model_truth": st["J_int"],
                            "rmse_vs_truth_w": float(np.sqrt(np.mean((G - truth) ** 2)))})
                with open(os.path.join(LAD, "solve", f"k{k}_{rung}.json"), "w") as f:
                    json.dump({"window": k, "offset": off, "rung": rung, "schedule": {str(i): list(v) for i, v in res["schedule"].items()},
                               "grid": list(range(73)), **rec}, f, indent=1)
            summary[k]["rungs"][rung] = rec
            print(f"k{k} {rung:12s} {rec['status']:10s} " + (f"C_model_truth {rec['C_model_truth_kg']:.6f}" if 'C_model_truth_kg' in rec else ""), flush=True)
    with open(sp, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)


def cmd_replay(rungs=RUNGS):
    import run_stage_a as rs
    dev = _dev()
    cfg, cell = rs.scene_dev_config("offset")
    os.makedirs(os.path.join(LAD, "replay"), exist_ok=True)
    for k, off in enumerate(dev):
        for rung in rungs:
            sj = os.path.join(LAD, "solve", f"k{k}_{rung}.json")
            if not os.path.exists(sj):
                print(f"k{k} {rung}: no OPTIMAL schedule, skipped"); continue
            out_csv = os.path.join(LAD, "replay", f"k{k}_{rung}.csv")
            ok = _evaluate(cfg, cell, k, off, "schedule_replay", out_csv, {"SCHEDULE_JSON": sj})
            print(f"replay k{k} {rung}: {'ok' if ok else 'FAILED'}", flush=True)


def cmd_judge(rungs=RUNGS):
    dev = _dev()
    summary = json.load(open(os.path.join(LAD, "solve_summary.json")))
    res = {"windows": {}, "rungs": list(rungs)}
    c_sim, c_model = {r: {} for r in rungs}, {r: {} for r in rungs}
    unresolved, closure_fail = [], []
    for k, off in enumerate(dev):
        sk = summary[str(k)]
        if not sk["quantisation_ok"]:
            res["verdict"] = "INVALID_QUANTISATION"; break
        for rung in rungs:
            rec = sk["rungs"].get(rung, {})
            if rec.get("status") != "OPTIMAL":
                unresolved.append(f"k{k}:{rung}"); continue
            p = os.path.join(LAD, "replay", f"k{k}_{rung}.csv")
            if not os.path.exists(p):
                closure_fail.append(f"k{k}:{rung}: no replay"); continue
            row = list(csv.DictReader(open(p)))[-1]
            led_p = p.replace(".csv", "_option_ledger.csv")
            led = list(csv.DictReader(open(led_p))) if os.path.exists(led_p) else []
            sched = {int(i): tuple(v) for i, v in json.load(open(os.path.join(LAD, "solve", f"k{k}_{rung}.json")))["schedule"].items()}
            counters = {c: row.get(c, 0) for c in ("deadline_forced_count", "ep_opt_hold_refused", "ep_opt_hold_masked",
                                                    "ep_opt_release_failed", "ep_opt_held_open", "ep_opt_stale")}
            counters["completion_short"] = max(0.0, 0.995 - float(row.get("completion_rate_mi", 1) or 1))
            counters["ontime_short"] = max(0.0, 0.995 - float(row.get("ontime_mi_share", 1) or 1))
            cs, cm = float(row["total_carbon_kg"]), float(rec["C_model_truth_kg"])
            chk = closure_check(cm, cs, led, sched, counters)
            c_sim[rung][k], c_model[rung][k] = cs, cm
            res["windows"].setdefault(str(k), {})[rung] = {"C_sim": cs, "C_model": cm, "closure": chk}
            if not chk["pass"]:
                closure_fail.append(f"k{k}:{rung}")
    if "verdict" in res:
        pass
    elif unresolved:
        res.update({"verdict": "STOP_SOLVER_RUNG_UNRESOLVED", "unresolved": unresolved})
    elif closure_fail:
        res.update({"verdict": "STOP_PLANNER_CLOSURE_RUNG", "closure_failures": closure_fail})
    elif set(rungs) == set(RUNGS):
        l1 = gate_l1(c_sim["shrink_0"], c_sim["truth"])
        losses = {r: {k: c_sim[r][k] - c_sim["truth"][k] for k in c_sim["truth"]} for r in rungs if r != "truth"}
        l2 = gate_l2(losses, l1, c_sim["truth"])
        res.update({"gate_l1": l1, "gate_l2": l2,
                    "n_valid_windows": sum(1 for h in l1.values() if h["valid"])})
        res["verdict"] = "STOP_LADDER_HEADROOM" if res["n_valid_windows"] < 4 else "LADDER_READ"
    else:
        res["verdict"] = "TRUTH_CLOSED" if not closure_fail else "STOP_PLANNER_CLOSURE_RUNG"
    with open(os.path.join(LAD, "ladder_verdict.json" if set(rungs) == set(RUNGS) else "truth_closure.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "windows"}, indent=1)[:3000])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "dump":
        cmd_dump()
    elif what == "solve":
        cmd_solve(("truth",) if "--truth-only" in sys.argv else tuple(r for r in RUNGS if r != "truth") if "--rest" in sys.argv else RUNGS)
    elif what == "replay":
        cmd_replay(("truth",) if "--truth-only" in sys.argv else RUNGS)
    elif what == "judge":
        cmd_judge(("truth",) if "--truth-only" in sys.argv else RUNGS)
    else:
        print(__doc__)
