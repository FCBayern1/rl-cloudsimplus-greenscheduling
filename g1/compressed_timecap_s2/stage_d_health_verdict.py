"""Stage D 50k/56k health gate (STAGE_D_PREREG section 5 + Addendum B split).

Pass/fail only, no effect claim. Two failure classes:
  wiring     missing checkpoint / eval row / metric field, tier not in effect, hash mismatch
             -> FIX_AND_RERUN (append-only fix)
  substantive policy collapse, zero forecast sensitivity, delta-r without variance, gate
             pinned at a bound, reward and carbon in opposite directions, contract red
             -> STOP_STAGE_D_HEALTH (no re-tuning)

judge() is a pure function of the collected tables so it can be tested without disk:
  evals: {(line, tag, tier, cell, k): {"carbon","reward","comp","ontime","forced",
          "defer_rate","clip"}} or None; tag in {"first","last"}
  crd:   {line: {"dr_mean","dr_std","rho_routing_mean","rho_forecast_mean"}} for NE/E
  probe: {line: {"kl_clean_vs_shrink","control_sensitivity"}} for V/E (None = missing)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LINES = ("NV", "V", "NE", "E")
CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
KS = (26, 34, 42)
TIERS = {"NV": ("hollow",), "NE": ("hollow",), "V": ("godeye", "calibrated_shrink_v1", "shuffle", "anti"),
         "E": ("godeye", "calibrated_shrink_v1", "shuffle", "anti")}
CLEAN = {"NV": "hollow", "NE": "hollow", "V": "godeye", "E": "godeye"}
DEFER_LO, DEFER_HI = 0.02, 0.98
CONTRACT = {"comp": 0.995, "ontime": 0.995}


def _pooled(rows, keys, field):
    vals = [rows[k][field] for k in keys if rows.get(k)]
    return sum(vals) / len(vals) if vals else None


def judge(evals, crd, probe):
    wiring, substantive, notes = [], [], {}
    # 1. every expected row present (last checkpoint, all tiers; first checkpoint, clean tier)
    expected = [(L, "last", t, c, k) for L in LINES for t in TIERS[L] for c in CELLS for k in KS] + \
               [(L, "first", CLEAN[L], c, k) for L in LINES for c in CELLS for k in KS]
    missing = [e for e in expected if evals.get(e) is None]
    if missing:
        wiring.append(("missing_eval_rows", len(missing), missing[:5]))
    have = [e for e in expected if evals.get(e) is not None]
    # 2. contract on every last-checkpoint clean-tier row (the policy's own deployment)
    red = [e for e in have if e[1] == "last" and e[2] == CLEAN[e[0]]
           and (evals[e]["comp"] < CONTRACT["comp"] or evals[e]["ontime"] < CONTRACT["ontime"]
                or evals[e]["forced"] > 0)]
    if red:
        substantive.append(("contract_red_on_clean_deployment", len(red), red[:5]))
    # 3. policy alive: defer rate inside (2%, 98%) pooled over the clean last rows, per line
    for L in LINES:
        keys = [e for e in have if e[0] == L and e[1] == "last" and e[2] == CLEAN[L]]
        dr = _pooled(evals, keys, "defer_rate")
        notes[f"{L}_defer_rate"] = dr
        if dr is not None and not (DEFER_LO < dr < DEFER_HI):
            substantive.append(("policy_collapse_defer_rate", L, dr))
    # 4. corruption changes behaviour: V and E carbon under shrink differs from clean
    for L in ("V", "E"):
        c0 = _pooled(evals, [e for e in have if e[0] == L and e[1] == "last" and e[2] == "godeye"], "carbon")
        c1 = _pooled(evals, [e for e in have if e[0] == L and e[1] == "last" and e[2] == "calibrated_shrink_v1"], "carbon")
        notes[f"{L}_carbon_clean"] = c0
        notes[f"{L}_carbon_shrink"] = c1
        p = probe.get(L) or {}
        kl = p.get("kl_clean_vs_shrink")
        sens = p.get("control_sensitivity")
        if kl is None or sens is None:
            wiring.append(("probe_missing", L))
        else:
            notes[f"{L}_kl"] = kl
            notes[f"{L}_sensitivity"] = sens
            if kl <= 0.0 or sens == 0.0:
                substantive.append(("forecast_insensitive", L, kl, sens))
    # 5. EU-CRD internals alive
    for L in ("NE", "E"):
        s = crd.get(L)
        if not s or any(s.get(f) is None for f in ("dr_mean", "dr_std")):
            wiring.append(("crd_stats_missing", L))
            continue
        notes[f"{L}_dr_std"] = s["dr_std"]
        if s["dr_std"] <= 0.0:
            substantive.append(("delta_r_no_variance", L, s["dr_std"]))
        for f in ("rho_routing_mean", "rho_forecast_mean"):
            v = s.get(f)
            if v is not None and (v <= 0.0501 or v >= 0.9999):
                substantive.append(("gate_pinned", L, f, v))
    # 6. reward and carbon same direction between first and last checkpoint (clean tier)
    for L in ("V", "E"):
        f_keys = [e for e in have if e[0] == L and e[1] == "first" and e[2] == CLEAN[L]]
        l_keys = [e for e in have if e[0] == L and e[1] == "last" and e[2] == CLEAN[L]]
        cf, cl = _pooled(evals, f_keys, "carbon"), _pooled(evals, l_keys, "carbon")
        rf, rl = _pooled(evals, f_keys, "reward"), _pooled(evals, l_keys, "reward")
        if None in (cf, cl, rf, rl):
            continue
        notes[f"{L}_carbon_first_last"] = (cf, cl)
        notes[f"{L}_reward_first_last"] = (rf, rl)
        if (cl - cf) * (rl - rf) > 0:      # carbon down must come with reward up
            substantive.append(("reward_carbon_opposite", L, cf, cl, rf, rl))
    # 7. clip rate
    clip = [e for e in have if evals[e].get("clip", 0) > 0]
    if clip:
        wiring.append(("carbon_norm_clip_seen", len(clip)))
    verdict = ("STOP_STAGE_D_HEALTH" if substantive else
               "FIX_AND_RERUN" if wiring else "PASS_HEALTH")
    return {"verdict": verdict, "wiring": wiring, "substantive": substantive, "notes": notes,
            "rows_expected": len(expected), "rows_present": len(have)}


def load(results_dir, logs_dir, probe_dir):
    evals = {}
    for L in LINES:
        for tag in ("last", "first"):
            d = os.path.join(results_dir, f"{L}_{tag}")
            for f in glob.glob(os.path.join(d, "*.csv")):
                base = os.path.basename(f)[:-4]
                cell = "_".join(base.split("_")[:5])
                rest = base[len(cell) + 1:]
                tier, k = rest.rsplit("_k", 1)
                r = list(csv.DictReader(open(f)))
                if not r:
                    continue
                r = r[-1]
                g = lambda key, d=0.0: float(r.get(key, d) or d)  # noqa: E731
                evals[(L, tag, tier, cell, int(k))] = {
                    "carbon": g("total_carbon_kg"), "reward": g("global_reward_sum"),
                    "comp": g("completion_rate_mi"), "ontime": g("ontime_mi_share"),
                    "forced": g("deadline_forced_count"), "defer_rate": g("global_defer_action_rate"),
                    "clip": g("ep_carbon_norm_clip_count")}
    crd = {}
    for L in ("NE", "E"):
        rj = sorted(glob.glob(os.path.join(logs_dir, f"{L}_s*", "*", "result.json")))
        if not rj:
            continue
        last = None
        for line in open(rj[-1]):
            if line.strip():
                last = json.loads(line)
        if last is None:
            continue
        flat = {}

        def walk(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(v, prefix + k + "/")
                else:
                    flat[prefix + k] = v
        walk(last)
        pick = lambda suffix: next((v for k, v in flat.items() if k.endswith(suffix)), None)  # noqa: E731
        crd[L] = {"dr_mean": pick("crd/dr_mean"), "dr_std": pick("crd/dr_std"),
                  "rho_routing_mean": pick("crd/rho_routing_mean"),
                  "rho_forecast_mean": pick("crd/rho_forecast_mean")}
    probe = {}
    for L in ("V", "E"):
        p = os.path.join(probe_dir, f"probe_{L}.json")
        if os.path.exists(p):
            probe[L] = json.load(open(p))
    return evals, crd, probe


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "drl-manager/results/stage_d")
    logs_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "drl-manager/logs/stage_d")
    probe_dir = sys.argv[3] if len(sys.argv) > 3 else results_dir
    evals, crd, probe = load(results_dir, logs_dir, probe_dir)
    out = judge(evals, crd, probe)
    with open(os.path.join(results_dir, "stage_d_health_verdict.json"), "w") as fh:
        fh.write(json.dumps(out, sort_keys=True, indent=2, default=str))
    print(json.dumps(out, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
