"""Settlement diagnostic (Codex ruling 2026-09-06): align the planner model and the simulator
per second and per DC on FIXED schedules, no optimisation, no new verdict.

Probes (all 32-PE, 48-step jobs on the dev offset twin, wind offset k0; arrival 5 s because at
t=0 the VMs are not yet created and every hold is masked, a probe artefact seen on the first run):
  L1  one job                        model 65.64 W x 48 steps on 1 host
  L2  two jobs, same site+start      model 1 host (130.28 W); does the simulator use 2 hosts?
  L3a two jobs staggered by 20 s     overlap packing + two boundaries
  L3b three jobs, third after the first two end   does a freed VM/host get reused?
  L3c three concurrent jobs          model 2 hosts; simulator?
  L4  the k0 optimal schedule        (already aligned: see settle_diag/k0_align.npz)

For every probe the simulator's per-step DC power sample (dc_current_power_w) is decomposed as
P = hosts x 1 W + jobs x 64.64 W (u = 0.4 per 32-PE job on a 64-PE RS500A host) and compared
with the model's hosts/pes arrays from `settle`. Green is taken from the SAME run so only draw
and its timing are under test. Usage: python settle_probe.py [probe ...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "drl-manager"))
import ladder_run as lr  # noqa: E402
from ladder_planner import Job, build_instance, settle  # noqa: E402
import yaml  # noqa: E402

CFG = os.path.join(HERE, "config_settle_probe.yml")
OUT = os.path.join(HERE, "stage_a_out", "settle_diag", "probes")
CAP = [640, 512, 640, 512, 192]        # version-1 legacy (attribute_k0 only)
DYN = 64.64


def probe_sites():
    """Per-site topology of the probe config (model version 2, diagnostics A and D)."""
    cfg = yaml.safe_load(open(CFG))
    blk = cfg[[k for k in cfg if k.startswith("sp_")][0]]
    return lr.sites_from_config(cfg, blk)
PROBES = {
    "L1_single": {0: (0, 20)},
    "L2_pair_same_start": {0: (0, 20), 1: (0, 20)},
    "L3a_pair_staggered": {0: (0, 20), 1: (0, 40)},
    "L3b_triple_reuse": {0: (0, 20), 1: (0, 20), 2: (0, 75)},   # 75 <= first sighting (3) + 72: the replay arm clips later starts
    "L3c_triple_fragment": {0: (0, 20), 1: (0, 20), 2: (0, 30)},
}


def decompose(P):
    """(jobs, hosts, residual W) for one power sample under the 64-PE / 0.4-utilisation model."""
    if P < 0.5:
        return 0, 0, 0.0
    best = None
    for j in range(1, 40):
        for h in range(max(1, (j + 1) // 2), j + 1):
            err = abs(P - DYN * j - h)
            if best is None or err < best[0]:
                best = (err, j, h)
    return best[1], best[2], best[0]


def run_probe(name, plan):
    os.makedirs(OUT, exist_ok=True)
    sj = os.path.join(OUT, f"{name}_schedule.json")
    json.dump({"schedule": {str(k): list(v) for k, v in plan.items()}, "grid": list(range(73))}, open(sj, "w"))
    dump = os.path.join(OUT, f"{name}_decisions.csv")
    for p in (dump, dump.replace(".csv", "_obs.npz")):
        if os.path.exists(p):
            os.remove(p)
    env = lr.replay_env(sj)
    env.update({"EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1"})
    ok = lr._evaluate(CFG, f"sp_{name}", 0, lr._dev()[0], "schedule_replay", os.path.join(OUT, f"{name}.csv"), env)
    if not ok:
        raise RuntimeError(f"{name}: replay failed, see {OUT}/{name}.log")
    z = np.load(dump.replace(".csv", "_obs.npz"))
    S = np.asarray(z["dc_current_power_w"], dtype=np.float64).T
    G = np.asarray(z["dc_current_green_power_w"], dtype=np.float64).T
    return S, G


def compare(name, plan, S, G):
    jobs = [Job(id=i, arrival=5, runtime=48, pes=32, deadline=3000) for i in sorted(plan)]
    T = max(S.shape[1], max(s for _, s in plan.values()) + 48 + 2)
    Gm = G if G.shape[1] >= T else np.concatenate([G, np.repeat(G[:, -1:], T - G.shape[1], axis=1)], axis=1)
    inst = build_instance(jobs, probe_sites(), Gm)
    st = settle(inst, plan)
    M = st["draw_mw"] / 1000.0
    Tc = min(M.shape[1], S.shape[1])
    site = 0
    rows = []
    for t in range(Tc):
        m, s = M[site, t], S[site, t]
        if m > 0.5 or s > 0.5:
            js, hs, res = decompose(s)
            rows.append((t, int(st["pes"][site, t] // 32), int(st["hosts"][site, t]), round(m, 2), js, hs, round(s, 2), round(res, 3)))
    jm = sum(r[1] for r in rows); hm = sum(r[2] for r in rows); jsim = sum(r[4] for r in rows); hsim = sum(r[5] for r in rows)
    other_sites = float(np.abs(S[[d for d in range(S.shape[0]) if d != site]]).sum())
    summary = {"probe": name, "plan": {str(k): list(v) for k, v in plan.items()},
               "model_job_steps": jm, "sim_job_steps": jsim, "model_host_steps": hm, "sim_host_steps": hsim,
               "draw_model_wh": round(M.sum() / 3600, 5), "draw_sim_wh": round(S.sum() / 3600, 5),
               "power_on_other_sites_w_sum": other_sites,
               "sim_active_rows": [r[0] for r in rows if r[6] > 0.5][:1] + [r[0] for r in rows if r[6] > 0.5][-1:],
               "model_active_rows": [r[0] for r in rows if r[3] > 0.5][:1] + [r[0] for r in rows if r[3] > 0.5][-1:],
               "model_version": st["model_version"], "premise_ok": st["premise_ok"],
               "exact": bool(np.allclose(M[:, :Tc], S[:, :Tc], atol=0.01))}
    return summary, rows


def job_counts(S, dyn=DYN):
    """Per-(site, step) job and host counts recovered from the simulator's power samples."""
    js = np.zeros_like(S)
    hs = np.zeros_like(S)
    for dc in range(S.shape[0]):
        for t in range(S.shape[1]):
            js[dc, t], hs[dc, t], _ = decompose(S[dc, t])
    return js, hs


def site_busy_at_start(sched, jid, runtime=48):
    """True when, at this job's planned start, another job on the same site started strictly
    earlier is still running or finishes at that very step (the simulator is mid-cycle)."""
    s, st = sched[jid]
    return any(so == s and sto < st <= sto + runtime for o, (so, sto) in sched.items() if o != jid)


def attribute_k0():
    """L4: split the k0 draw and brown gap into the mechanisms found by L1-L3 and write
    settle_diag/k0_attribution.json. Uses the dump-run truth curve exactly as the planner did."""
    import csv
    from ladder_run import jobs_from_dump, LAD
    diag = os.path.join(HERE, "stage_a_out", "settle_diag")
    rows = list(csv.DictReader(open(f"{LAD}/dump/k0_decisions.csv")))
    z = np.load(f"{LAD}/dump/k0_decisions_obs.npz")
    truth = np.asarray(z["dc_current_green_power_w"], dtype=np.float64).T
    jobs = jobs_from_dump(rows, 40000.0, 1.0)
    need = max(j.latest + j.runtime for j in jobs) + 1
    dump_rows = truth.shape[1]
    if need > dump_rows:
        truth = np.concatenate([truth, np.repeat(truth[:, -1:], need - dump_rows, axis=1)], axis=1)
    from ladder_planner import sites_from_caps
    inst = build_instance(jobs, sites_from_caps(CAP), truth)   # version-1 topology: this is the archived v1 attribution
    sched = {int(i): tuple(v) for i, v in json.load(open(f"{LAD}/solve/k0_truth.json"))["schedule"].items()}
    st = settle(inst, sched)
    a = np.load(os.path.join(diag, "k0_align.npz"))
    S, Gs = a["sim_w"], a["green_w"]
    T = S.shape[1]
    M, G = st["draw_mw"][:, :T] / 1000.0, truth[:, :T]
    hm, jm = st["hosts"][:, :T], st["pes"][:, :T] / 32.0
    js, hs = job_counts(S)
    pack = (hs - hm) * 1.0
    steps = (js - jm) * DYN
    other = (S - M) - pack - steps
    wh = lambda x: float(x.sum() / 3600.0)
    brown = lambda P, Gc: float(np.maximum(0.0, P - Gc).sum() / 3600.0)
    b0 = brown(M, G)
    out = {
        "draw_wh": {"model": wh(M), "sim": wh(S), "gap": wh(S - M),
                    "A_extra_hosts": wh(pack), "B_extra_job_steps": wh(steps), "residual_DC2_host_model": wh(other)},
        "extra_host_steps": float(pack.sum()), "extra_job_steps": float(steps.sum() / DYN),
        "one_host_per_job_everywhere": bool(np.all(hs == js)), "max_concurrent_jobs_per_site": js.max(1).tolist(),
        "brown_wh": {"model": b0, "sim_with_planner_curve": brown(S, G), "sim_with_its_own_curve": brown(S, Gs[:, :T]),
                     "A_only": brown(M + pack, G) - b0, "B_only": brown(M + steps, G) - b0,
                     "C_curve_tail_beyond_dump": brown(S, Gs[:, :T]) - brown(S, G)},
        "dump_rows": int(dump_rows), "planner_horizon_rows": int(need), "replay_rows": int(T),
        "rule_B": {"jobs": len(sched),
                   "busy_at_start": sorted(j for j in sched if site_busy_at_start(sched, j)),
                   "plus_one_end_sample": sorted(j for j in sched if sched[j][1] + 48 < T
                                                 and js[sched[j][0], sched[j][1] + 48] > jm[sched[j][0], sched[j][1] + 48])},
    }
    rb = out["rule_B"]
    rb["agree"] = sorted(set(rb["busy_at_start"]) & set(rb["plus_one_end_sample"]))
    rb["busy_no_plus"] = sorted(set(rb["busy_at_start"]) - set(rb["plus_one_end_sample"]))
    rb["plus_not_busy"] = sorted(set(rb["plus_one_end_sample"]) - set(rb["busy_at_start"]))
    json.dump(out, open(os.path.join(diag, "k0_attribution.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def main(names):
    if names == ["k0"]:
        attribute_k0()
        return
    results = {}
    for name in names:
        plan = PROBES[name]
        S, G = run_probe(name, plan)
        summary, rows = compare(name, plan, S, G)
        results[name] = summary
        print(f"\n== {name}  plan {summary['plan']}")
        print(f"   model job-steps {summary['model_job_steps']} host-steps {summary['model_host_steps']} draw {summary['draw_model_wh']} Wh | "
              f"sim job-steps {summary['sim_job_steps']} host-steps {summary['sim_host_steps']} draw {summary['draw_sim_wh']} Wh | "
              f"active rows model {summary['model_active_rows']} sim {summary['sim_active_rows']}")
        print("   t  jobs_m hosts_m model_W | jobs_s hosts_s sim_W resid")
        last = None
        for r in rows:
            key = r[1:]
            if key != last:
                print("   %3d  %d %d %7.2f | %d %d %7.2f %.3f" % r)
                last = key
        json.dump({"summary": summary, "rows": rows}, open(os.path.join(OUT, f"{name}_align.json"), "w"), indent=1)
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or list(PROBES))
