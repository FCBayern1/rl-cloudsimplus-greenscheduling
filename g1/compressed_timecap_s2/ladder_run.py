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
    HOST_FLOOR_MW, MW_PER_PE, F_BROWN, F_GREEN, MODEL_VERSION, Job, Site, build_instance, preflight_factors,
    quantisation_bound_kg, runtime_steps, settle, site_from_profile, solve)

OUT = os.path.join(HERE, "stage_a_out")
# Addendum E: fresh directory, uniform HiGHS limit 3600 s, atomic per-solve records.
# LADDER_DIR overrides the directory only for reading an older stage's artefacts.
LAD = os.path.join(OUT, os.environ.get("LADDER_DIR", "ladder_v3"))
TIME_LIMIT_S = float(os.environ.get("LADDER_TIME_LIMIT_S", "3600"))


def atomic_json(path, obj):
    """Write obj to path atomically (temp file + rename) so a record survives any crash."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
LAMBDAS = (0.75, 0.5, 0.25, 0.0)
RUNGS = ("truth", "shrink_0.75", "shrink_0.5", "shrink_0.25", "shrink_0", "shuffle", "anti")
CLOSURE_REL = 0.03
# version-2 closure (prereg v2 §4): the energy terms are gated separately so nothing can hide
# under the composite carbon figure
CLOSURE_DRAW_REL = 0.001
CLOSURE_BROWN_ABS_WH = 0.002
QUANT_FRAC = 0.001
C_BROWN_REF = 0.01897
HEADROOM_REL, HEADROOM_ABS = 0.15, 0.05 * C_BROWN_REF
LOSS_MIN, HEADROOM_SHARE = 0.05, 0.80
COMPRESSED_DIVISOR = 3000.0
# Settlement diagnostic C (2026-09-06): the wind-file row behind observation step t.
#   OBS_CLOCK_LAG      the simulator clock at observation step t is t + min_time_between_events
#                      (the first clock advance lands on the event granularity; measured: step 3
#                      at clock 4.0 with 1.0 s, step 33 at clock 33.01 with 0.01 s), and the
#                      provider takes floor(clock) as the row
#   SPLINE_SKIP_ROWS   in COMPRESSED mode the provider drops the first 12 CSV rows before
#                      building its time axis (GreenEnergyProvider.loadCsvData: rowIndex < 12)
# so row = episode_offset + tz_rows + t + floor(min_time_between_events) + SPLINE_SKIP_ROWS;
# verified exactly on every step of the six development windows (1.0 s: +13).
SPLINE_SKIP_ROWS = 12
CERT_MIN_TIME_BETWEEN_EVENTS = 1.0          # the scene's value; the clock grid depends on it


def obs_clock_lag(blk):
    import math
    return int(math.floor(float(blk.get("min_time_between_events", 0.1))))


OBS_CLOCK_LAG = 1                            # for min_time_between_events = 1.0
# Settlement diagnostic B: the certification twin aligns the datacenters' cloudlet-processing
# updates to the 1 s step grid (datacenter_scheduling_interval = 1.0; the scene left CloudSim's
# default 0 = update only at estimated finishes, where a 0.01 s scheduling constant plus the
# (long) truncation of partial MI leaves a 401-MI sliver that waits a whole extra second on a
# busy site: 49 s not 48). min_time_between_events stays 1.0 so the clock grid is unchanged.
CERT_SCHEDULING_INTERVAL = 1.0
WIND_DIR = os.path.join(REPO, "cloudsimplus-gateway", "src", "main", "resources", "windProduction", "simplified")


class LadderStop(RuntimeError):
    """A preregistered STOP raised by the harness (curve out of range, curve mismatch)."""


# ── pure pieces ───────────────────────────────────────────────────────────────────────
def sites_from_config(cfg_all, blk):
    """Per-site topology and power function from the experiment block (diagnostic D): host
    profile and count from the host_count_spec_* key, VM count/PEs/MIPS from the block."""
    common = cfg_all.get("common", {}) if isinstance(cfg_all, dict) else {}
    sites = []
    for d in sorted(blk["datacenters"], key=lambda x: int(x["datacenter_id"])):
        prof = [k for k in d if k.startswith("host_count_spec_")]
        if len(prof) != 1:
            raise ValueError(f"site {d.get('name')}: expected exactly one host_count_spec_* key, got {prof}")
        profile = prof[0][len("host_count_"):].upper()
        s = site_from_profile(str(d.get("name", d["datacenter_id"])), profile, hosts=int(d[prof[0]]),
                              vms=int(d.get("initial_s_vm_count", 0)), vm_pes=int(d.get("small_vm_pes", common.get("small_vm_pes", 32))),
                              vm_mips=float(d.get("vm_pe_mips", common.get("vm_pe_mips", 40000))))
        hm = float(d.get("host_pe_mips", common.get("host_pe_mips", s.host_mips)))
        if abs(hm - s.host_mips) > 1e-9:
            s = Site(**{**s.__dict__, "host_mips": hm})
        sites.append(s)
    return sites


def wind_rows_kw(turbine_id, year, wind_dir=WIND_DIR):
    """The provider's raw row array: power_kw per file row, in file order, negatives clipped."""
    p = os.path.join(wind_dir, f"Turbine_{turbine_id}_{year}.csv")
    return np.array([max(0.0, float(r["power_kw"] or 0)) for r in csv.DictReader(open(p))], dtype=np.float64)


def truth_curve(blk, offset, T, wind_dir=WIND_DIR):
    """(G (sites, T) in W, meta) straight from the wind files for observation steps 0..T-1:
    G[d, t] = sum over the site's turbines of kW[row(t)] * 1000 / divisor with
    row(t) = offset + tz_d + t + OBS_CLOCK_LAG + SPLINE_SKIP_ROWS. No hold-last, no wrap:
    a row past the file's end is a STOP (LadderStop). meta carries the per-site row range,
    files and a signature of the exact values used."""
    div = float(blk.get("compressed_power_divisor", COMPRESSED_DIVISOR))
    year = int(blk.get("wind_csv_year", 2021))
    dcs = sorted(blk["datacenters"], key=lambda x: int(x["datacenter_id"]))
    G = np.zeros((len(dcs), int(T)), dtype=np.float64)
    meta = {"offset": int(offset), "T": int(T), "year": year, "divisor": div, "obs_clock_lag": obs_clock_lag(blk),
            "spline_skip_rows": SPLINE_SKIP_ROWS, "sites": []}
    for i, d in enumerate(dcs):
        tz = int(d.get("time_zone_offset_rows", 0))
        start = int(offset) + tz + obs_clock_lag(blk) + SPLINE_SKIP_ROWS
        end = start + int(T)
        files = []
        if d.get("green_energy_enabled", True):
            for tid in d.get("turbine_ids", []) or []:
                p = wind_rows_kw(tid, year, wind_dir)
                if end > len(p):
                    raise LadderStop(f"STOP_CURVE_OUT_OF_RANGE: site {i} turbine {tid} needs rows [{start}, {end}) "
                                     f"of {len(p)} (offset {offset}, T {T})")
                G[i] += p[start:end] * 1000.0 / div
                files.append(f"Turbine_{tid}_{year}.csv")
        meta["sites"].append({"site": i, "tz_rows": tz, "row_start": start, "row_end": end, "files": files})
    meta["signature"] = curve_signature(G)
    return G, meta


def curve_signature(G):
    """sha256 of the curve's values rounded to integer mW (the planner's quantisation)."""
    mw = np.rint(np.asarray(G, dtype=np.float64) * 1000.0).astype(np.int64)
    return hashlib.sha256(mw.tobytes() + str(mw.shape).encode()).hexdigest()[:16]


def curve_rows_match(G_planner, G_obs, tol_w=1e-3):
    """Per-row equality of the planner's curve and the simulator's observed green on the rows
    the observation covers. Returns (ok, n_rows_compared, max_abs_diff_w, first_bad_row)."""
    P = np.asarray(G_planner, dtype=np.float64); O = np.asarray(G_obs, dtype=np.float64)
    n = min(P.shape[1], O.shape[1])
    if O.shape[1] > P.shape[1]:
        return False, n, float("inf"), int(P.shape[1])
    diff = np.abs(P[:, :n] - O[:, :n])
    bad = np.where(diff.max(axis=0) > tol_w)[0]
    return (bad.size == 0), int(n), float(diff.max() if n else 0.0), (int(bad[0]) if bad.size else -1)


def cert_config(mode):
    """The certification twin: the development twin of `mode` with the simulator's
    min_time_between_events set to CERT_MIN_TIME_BETWEEN_EVENTS (diagnostic B). Written on every
    call to config_ladder_cert_{mode}.yml; the scene's own configs are untouched."""
    import yaml
    import run_stage_a as rs
    src_path, cell = rs.scene_dev_config(mode)
    cfg = yaml.safe_load(open(src_path))
    blk = cfg[cell]
    if abs(float(blk.get("min_time_between_events", 0.1)) - CERT_MIN_TIME_BETWEEN_EVENTS) > 1e-12:
        raise LadderStop(f"STOP_CLOCK_GRID: the scene's min_time_between_events is {blk.get('min_time_between_events')}, "
                         f"the certification mapping assumes {CERT_MIN_TIME_BETWEEN_EVENTS}")
    for d in blk["datacenters"]:
        d["datacenter_scheduling_interval"] = CERT_SCHEDULING_INTERVAL
    path = os.path.join(HERE, f"config_ladder_cert_{mode}.yml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return path, cell
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


def closure_check(c_model, c_sim, ledger_rows, schedule, counters, energy=None):
    """Per rung and window (B2 + C5, v2 §4). ledger_rows: option-ledger rows of the replay
    (id, dc, t_s, route_to_start_steps, stale); schedule: {id: (site, start)}; counters: dict
    of the zero-required fields; energy: optional {"draw_model", "draw_sim", "brown_model",
    "brown_sim"} in Wh gated separately. Returns {"pass", "violations", "rel_err", ...}."""
    v = []
    rel = abs(c_sim - c_model) / c_model if c_model > 0 else float("inf")
    if rel > CLOSURE_REL:
        v.append(f"carbon rel err {rel:.4f} > {CLOSURE_REL}")
    terms = {}
    if energy:
        dm, ds = float(energy["draw_model"]), float(energy["draw_sim"])
        drel = abs(ds - dm) / dm if dm > 0 else float("inf")
        babs = abs(float(energy["brown_sim"]) - float(energy["brown_model"]))
        terms = {"draw_rel_err": drel, "brown_abs_err_wh": babs}
        if drel > CLOSURE_DRAW_REL:
            v.append(f"draw rel err {drel:.5f} > {CLOSURE_DRAW_REL}")
        if babs > CLOSURE_BROWN_ABS_WH:
            v.append(f"brown abs err {babs:.5f} Wh > {CLOSURE_BROWN_ABS_WH}")
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
    return {"pass": not v, "violations": v, "rel_err": rel, **terms}


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
    dev = _dev()
    cfg, cell = cert_config("defer")
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
    dev = _dev()
    cfg_path, cell = cert_config("defer")
    cfg_all = yaml.safe_load(open(cfg_path))
    blk = cfg_all[cell]
    preflight_factors(blk["datacenters"])
    mu = _mu_w(blk)
    sites = sites_from_config(cfg_all, blk)
    mips = float(blk["datacenters"][0].get("vm_pe_mips", 40000)); u = float(blk.get("cloudlet_cpu_utilization", 1.0))
    os.makedirs(os.path.join(LAD, "solve"), exist_ok=True)
    sp = os.path.join(LAD, "solve_summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    summary = {int(k): v for k, v in summary.items() if k != "environment"}
    import platform, scipy, subprocess as _sp
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        ncpu = os.cpu_count()
    except Exception:
        cpu, ncpu = platform.processor(), os.cpu_count()
    try:
        highs = _sp.run([sys.executable, "-c", "from scipy.optimize._highspy import _core; print(getattr(_core, 'HIGHS_VERSION', '?'))"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        highs = "unknown"
    environment = {"cpu": cpu, "cpu_count": ncpu, "python": platform.python_version(), "scipy": scipy.__version__,
                   "highs": highs, "time_limit_s": TIME_LIMIT_S, "mip_rel_gap": 0.0, "one_solve_at_a_time": True,
                   "model_version": MODEL_VERSION, "min_time_between_events": blk.get("min_time_between_events"),
                   "datacenter_scheduling_interval": CERT_SCHEDULING_INTERVAL,
                   "sites": [s.__dict__ for s in sites]}
    for k, off in enumerate(dev):
        rows = list(csv.DictReader(open(os.path.join(LAD, "dump", f"k{k}_decisions.csv"))))
        z = np.load(os.path.join(LAD, "dump", f"k{k}_decisions_obs.npz"))
        obs_green = np.asarray(z["dc_current_green_power_w"], dtype=np.float64).T   # (sites, T_obs)
        jobs = jobs_from_dump(rows, mips, u)
        need = max(max(j.latest + j.runtime for j in jobs) + 1, obs_green.shape[1])
        # diagnostic C: the truth curve comes from the wind files over the whole horizon; the
        # dump's observation must agree with it row by row on the rows it covers, else STOP
        truth, curve_meta = truth_curve(blk, off, need)
        ok, n_cmp, max_diff, first_bad = curve_rows_match(truth, obs_green)
        qb = quantisation_bound_kg(truth.shape[0], truth.shape[1])
        prev = summary.get(k, {})
        summary[k] = {"offset": off, "n_jobs": len(jobs), "T": int(truth.shape[1]), "quantisation_bound_kg": qb,
                      "quantisation_ok": qb <= QUANT_FRAC * C_BROWN_REF, "mu_w": mu, "rungs": prev.get("rungs", {}),
                      "curve": {**curve_meta, "dump_rows": int(obs_green.shape[1]), "dump_rows_match": bool(ok),
                                "rows_compared": n_cmp, "max_abs_diff_w": max_diff, "first_bad_row": first_bad}}
        atomic_json(sp, {**{str(kk): vv for kk, vv in summary.items()}, "environment": environment})
        if not ok:
            print(f"k{k}: STOP_CURVE_MISMATCH: wind-file curve differs from the dump at row {first_bad} "
                  f"(max |diff| {max_diff:.4g} W); no solve", flush=True)
            return
        inst_truth = build_instance(jobs, sites, truth)
        for rung in rungs:
            G = rung_curve(truth, rung, mu, seed_key=f"ladder:{off}")
            inst = build_instance(jobs, sites, G)
            # Addendum D: HiGHS (scipy.optimize.milp, gap 0, 600 s) is the only judging solver;
            # LADDER_SOLVER=cpsat is the cross-check path and never runs in a formal stage.
            import time as _time
            t_ext = _time.time()
            if os.environ.get("LADDER_SOLVER", "milp") == "cpsat":
                res = solve(inst, time_limit_s=TIME_LIMIT_S)
            else:
                from ladder_planner import solve_milp
                res = solve_milp(inst, time_limit_s=TIME_LIMIT_S)
            res["solver"] = os.environ.get("LADDER_SOLVER", "milp")
            res["external_wall_s"] = _time.time() - t_ext
            rec = {k: res.get(k) for k in ("status", "wall_s", "external_wall_s", "solver", "bound", "fun", "mip_gap",
                                            "mip_dual_bound", "mip_node_count", "milp_status", "milp_message",
                                            "schedule_hash", "checks", "verify_violations")}
            rec["time_limit_s"] = TIME_LIMIT_S
            rec["J_int_incumbent"] = res.get("J_int")
            if res["status"] == "OPTIMAL":
                st = settle(inst_truth, res["schedule"])
                rec.update({"J_on_rung": res["J_int"], "C_model_truth_kg": st["C_kg"], "J_model_truth": st["J_int"],
                            "rmse_vs_truth_w": float(np.sqrt(np.mean((G - truth) ** 2))),
                            "draw_model_wh": float(st["draw_mw"].sum() / 3.6e6), "brown_model_wh": float(st["brown_mw"].sum() / 3.6e6),
                            "green_model_wh": float(st["green_mw"].sum() / 3.6e6)})
            # Addendum E3: every outcome is written atomically the moment the solve returns
            atomic_json(os.path.join(LAD, "solve", f"k{k}_{rung}.json"),
                        {"window": k, "offset": off, "rung": rung,
                         "schedule": ({str(i): list(v) for i, v in res["schedule"].items()} if res.get("schedule") else None),
                         "grid": list(range(73)), "curve_signature": curve_meta["signature"],
                         "rung_curve_signature": curve_signature(G), **rec})
            summary[k]["rungs"][rung] = rec
            atomic_json(sp, {**{str(kk): vv for kk, vv in summary.items()}, "environment": environment})
            if res["status"] != "OPTIMAL":
                print(f"k{k} {rung}: {res['status']} at the {TIME_LIMIT_S:.0f} s limit -> STOP_SOLVER_RUNG_UNRESOLVED; no further solve", flush=True)
                return
            print(f"k{k} {rung:12s} {rec['status']:10s} " + (f"C_model_truth {rec['C_model_truth_kg']:.6f}" if 'C_model_truth_kg' in rec else ""), flush=True)
    with open(sp, "w") as f:
        json.dump({**{str(k): v for k, v in summary.items()}, "environment": environment}, f, indent=2)


def replay_env(schedule_json, dump_csv=None):
    """Environment for one replay: the schedule and the every-step offset grid (OFFSET_GRID_DENSE=1);
    without the dense grid the executor's 12-value dyadic mask and the replay arm's 73-value
    action index disagree and every plan is silently mis-executed (found on ladder_v3 k0).
    With dump_csv the replay also records its per-step observations (the simulator's green rows,
    checked against the planner's curve at closure, diagnostic C)."""
    env = {"SCHEDULE_JSON": schedule_json, "OFFSET_GRID_DENSE": "1"}
    if dump_csv:
        env.update({"EVAL_DECISION_DUMP": dump_csv, "EVAL_DECISION_DUMP_OBS": "1"})
    return env


def cmd_replay(rungs=RUNGS):
    dev = _dev()
    cfg, cell = cert_config("offset")
    os.makedirs(os.path.join(LAD, "replay"), exist_ok=True)
    for k, off in enumerate(dev):
        for rung in rungs:
            sj = os.path.join(LAD, "solve", f"k{k}_{rung}.json")
            if not os.path.exists(sj) or json.load(open(sj)).get("status") != "OPTIMAL":
                print(f"k{k} {rung}: no OPTIMAL schedule, skipped"); continue
            out_csv = os.path.join(LAD, "replay", f"k{k}_{rung}.csv")
            dump = os.path.join(LAD, "replay", f"k{k}_{rung}_decisions.csv")
            for p in (dump, dump.replace(".csv", "_obs.npz")):
                if os.path.exists(p):
                    os.remove(p)
            # the frozen settlement path is the EVERY-STEP offset executor (prereg §2.2, A, D):
            # the simulator's grid must be 0..W so the plan's (site, start) maps 1:1 onto an action
            ok = _evaluate(cfg, cell, k, off, "schedule_replay", out_csv, replay_env(sj, dump))
            print(f"replay k{k} {rung}: {'ok' if ok else 'FAILED'}", flush=True)


def _cert_block():
    import yaml
    path, cell = cert_config("defer")
    return yaml.safe_load(open(path))[cell]


def cmd_closure(k, schedule_json, tag="closure"):
    """Engineering closure of one FIXED schedule on development window k (no solve): model
    settlement (version 2, wind-file curve) vs the certification twin's replay, the 3 % gate,
    per-job site/start, counters, curve rows, and the per-term decomposition (draw, brown,
    green, carbon) so every known term is bounded individually. Writes
    <LAD>/closure/<tag>_k<k>.json and prints the record."""
    import yaml
    dev = _dev(); off = dev[k]
    cfg_path, cell = cert_config("defer")
    cfg_all = yaml.safe_load(open(cfg_path)); blk = cfg_all[cell]
    sites = sites_from_config(cfg_all, blk)
    mips = float(blk["datacenters"][0].get("vm_pe_mips", 40000)); u = float(blk.get("cloudlet_cpu_utilization", 1.0))
    rows = list(csv.DictReader(open(os.path.join(LAD, "dump", f"k{k}_decisions.csv"))))
    jobs = jobs_from_dump(rows, mips, u)
    raw = json.load(open(schedule_json))
    sched = {int(i): tuple(v) for i, v in raw["schedule"].items()}
    need = max(s + {j.id: j for j in jobs}[i].runtime for i, (d, s) in sched.items()) + 2
    G, meta = truth_curve(blk, off, need)
    inst = build_instance(jobs, sites, G)
    from ladder_planner import verify_schedule
    viol = verify_schedule(inst, sched)
    st = settle(inst, sched)
    out_dir = os.path.join(LAD, "closure"); os.makedirs(out_dir, exist_ok=True)
    sj = os.path.join(out_dir, f"{tag}_k{k}_schedule.json")
    json.dump({"schedule": {str(i): list(v) for i, v in sched.items()}, "grid": list(range(73))}, open(sj, "w"))
    out_csv = os.path.join(out_dir, f"{tag}_k{k}.csv"); dump = os.path.join(out_dir, f"{tag}_k{k}_decisions.csv")
    for p in (dump, dump.replace(".csv", "_obs.npz"), out_csv):
        if os.path.exists(p):
            os.remove(p)
    cfg_off, cell_off = cert_config("offset")
    ok = _evaluate(cfg_off, cell_off, k, off, "schedule_replay", out_csv, replay_env(sj, dump))
    if not ok:
        raise RuntimeError(f"replay failed: {out_csv.replace('.csv', '.log')}")
    row = list(csv.DictReader(open(out_csv)))[-1]
    led_p = out_csv.replace(".csv", "_option_ledger.csv")
    led = list(csv.DictReader(open(led_p))) if os.path.exists(led_p) else []
    counters = {c: row.get(c, 0) for c in ("deadline_forced_count", "ep_opt_hold_refused", "ep_opt_hold_masked",
                                            "ep_opt_release_failed", "ep_opt_held_open", "ep_opt_stale")}
    counters["completion_short"] = max(0.0, 0.995 - float(row.get("completion_rate_mi", 1) or 1))
    z = np.load(dump.replace(".csv", "_obs.npz"))
    S = np.asarray(z["dc_current_power_w"], dtype=np.float64).T
    G_obs = np.asarray(z["dc_current_green_power_w"], dtype=np.float64).T
    okc, ncmp, mdiff, bad = curve_rows_match(G, G_obs)
    T = min(S.shape[1], G.shape[1])
    M = st["draw_mw"][:, :T] / 1000.0
    wh = lambda a: float(a.sum() / 3600.0)
    brown_m = wh(st["brown_mw"][:, :T] / 1000.0); green_m = wh(st["green_mw"][:, :T] / 1000.0)
    cs, cm = float(row["total_carbon_kg"]), float(st["C_kg"])
    chk = closure_check(cm, cs, led, sched, counters)
    if not okc:
        chk["violations"].append(f"curve rows differ from row {bad} (max |diff| {mdiff:.4g} W)"); chk["pass"] = False
    if viol:
        chk["violations"].extend(f"schedule: {v}" for v in viol); chk["pass"] = False
    if not st["premise_ok"]:
        chk["violations"].append("premise A violated: more running jobs than hosts"); chk["pass"] = False
    exec_spans = sorted(set(round(float(r["exec_s"]), 3) for r in led if r.get("exec_s") not in (None, "", "nan")))
    rec = {"window": k, "offset": off, "tag": tag, "schedule_json": schedule_json, "n_jobs": len(sched),
           "model_version": MODEL_VERSION, "min_time_between_events": blk.get("min_time_between_events"),
           "datacenter_scheduling_interval": blk["datacenters"][0].get("datacenter_scheduling_interval"),
           "curve": {**meta, "replay_rows": int(G_obs.shape[1]), "rows_match": bool(okc), "max_abs_diff_w": mdiff},
           "terms": {"draw_wh": {"model": wh(M), "sim": float(row.get("total_energy_wh", 0)), "sim_samples": wh(S[:, :T]),
                                 "max_abs_step_diff_w": float(np.abs(S[:, :T] - M).max())},
                     "brown_wh": {"model": brown_m, "sim": float(row.get("brown_used_wh", 0))},
                     "green_wh": {"model": green_m, "sim": float(row.get("green_used_wh", 0))},
                     "carbon_kg": {"model": cm, "sim": cs, "rel_err": chk["rel_err"], "abs_err": cs - cm}},
           "exec_spans_s": exec_spans, "counters": counters, "closure": chk}
    atomic_json(os.path.join(out_dir, f"{tag}_k{k}.json"), rec)
    print(json.dumps({kk: vv for kk, vv in rec.items() if kk != "curve"}, indent=1))
    return rec


def cmd_judge(rungs=RUNGS):
    dev = _dev()
    summary = json.load(open(os.path.join(LAD, "solve_summary.json")))
    res = {"windows": {}, "rungs": list(rungs), "environment": summary.get("environment")}
    c_sim, c_model = {r: {} for r in rungs}, {r: {} for r in rungs}
    unresolved, closure_fail = [], []
    for k, off in enumerate(dev):
        sk = summary.get(str(k))
        if sk is None:  # Addendum E: the solve stopped at an earlier unproven cell; never reached
            unresolved.extend(f"k{k}:{rung} (not solved: after the stop)" for rung in rungs); continue
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
            energy = None
            if rec.get("draw_model_wh") is not None:
                energy = {"draw_model": rec["draw_model_wh"], "draw_sim": row.get("total_energy_wh", 0),
                          "brown_model": rec["brown_model_wh"], "brown_sim": row.get("brown_used_wh", 0)}
            chk = closure_check(cm, cs, led, sched, counters, energy)
            # diagnostic C: the replay's observed green rows must equal the planner's truth
            # curve row by row (signatures on both sides); a replay past the curve is a failure
            curve = sk.get("curve") or {}
            obs_p = p.replace(".csv", "_decisions_obs.npz")
            if os.path.exists(obs_p) and curve:
                G_obs = np.asarray(np.load(obs_p)["dc_current_green_power_w"], dtype=np.float64).T
                G_plan, _ = truth_curve(_cert_block(), off, int(curve["T"]))
                okc, ncmp, mdiff, bad = curve_rows_match(G_plan, G_obs)
                chk["curve"] = {"rows_compared": ncmp, "max_abs_diff_w": mdiff, "first_bad_row": bad,
                                "planner_signature": curve.get("signature"), "replay_signature": curve_signature(G_obs),
                                "replay_rows": int(G_obs.shape[1]), "planner_rows": int(curve["T"]), "match": bool(okc)}
                if not okc:
                    chk["violations"].append(f"curve rows differ from row {bad} (max |diff| {mdiff:.4g} W)"); chk["pass"] = False
            else:
                chk["violations"].append("no replay observation dump or curve record: curve rows unverified"); chk["pass"] = False
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
    elif what == "closure":                      # closure <k> <schedule.json> [tag]
        cmd_closure(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "closure")
    else:
        print(__doc__)
