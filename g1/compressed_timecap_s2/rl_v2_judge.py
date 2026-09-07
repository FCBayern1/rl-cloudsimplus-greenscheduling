"""RL_V2 smoke judge (reports/RL_V2_SMOKE_PREREG.md §6). Reads the readings written by rl_v2_smoke.sh:
  stage_a_out/rl_v2/init/<line>_k<k>.csv (+ decisions dump)      deterministic init checkpoint
  stage_a_out/rl_v2/ref/cover_<chan>_<tier>_k<k>.csv (+ dump)    cover_argmax reference, index ties
  stage_a_out/rl_v2/last/<line>_<tier>_k<k>.csv (+ dump)         last checkpoint, stochastic decode
  stage_a_out/rl_v2/expert_k<k>.csv, flat: from F_FITS_V2 test readings / labels (val + test windows)
Usage: python rl_v2_judge.py [init|all]
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage_a_out", "rl_v2")
FV2 = os.path.join(HERE, "stage_a_out", "f_v2")
LINES = ("NV", "V", "NE", "E")
CHAN = {"NV": "none", "V": "full", "NE": "none", "E": "full"}
TIERS_FULL = ("godeye", "shrink75", "shrink50", "shrink25", "shrink0", "shuffle", "anti")
CONTRACT = {"completion_rate_mi": 0.995, "ontime_mi_share": 0.995}
READ_K = list(range(12, 18))            # windows k12..k17 of the F_FITS_V2 index (val 2 + test 4)


def _row(p):
    return list(csv.DictReader(open(p)))[-1] if os.path.exists(p) else None


def _contract(row):
    if row is None:
        return ["no run"]
    v = [f"{key} {row.get(key)} < {lo}" for key, lo in CONTRACT.items() if float(row.get(key, 0) or 0) < lo]
    v += [f"{key} = {row.get(key)}" for key in ("deadline_forced_count", "ep_opt_stale") if float(row.get(key, 0) or 0) != 0]
    return v


def _decisions(p):
    """{(step, slot): action} of a decision dump, first sighting per job."""
    d = {}
    seen = set()
    for r in csv.DictReader(open(p)):
        cid = int(r["cloudlet_id"])
        if cid < 0 or cid in seen:
            continue
        seen.add(cid); d[(int(r["step"]), int(r["slot"]))] = int(r["action"])
    return d


def references():
    """Per reading window: C_flat (offline flat) and C_causal from the F_FITS_V2 pass (test) or labels (val)."""
    ref = {}
    test = json.load(open(os.path.join(FV2, "test_readings.json")))
    for k in READ_K:
        expert = _row(os.path.join(FV2, "labels", f"k{k}.csv"))
        c_causal = float(expert["total_carbon_kg"]) if expert else None
        c_flat = float(test[str(k)]["shrink_0"]["C_sim"]) if str(k) in test else None
        if c_flat is None:
            fp = os.path.join(OUT, "flat", f"k{k}.csv")            # computed by the smoke for the validation windows
            r = _row(fp); c_flat = float(r["total_carbon_kg"]) if r else None
        ref[k] = {"C_flat": c_flat, "C_causal": c_causal}
    return ref


def init_check():
    out = {"lines": {}, "pass": True}
    for L in LINES:
        chan = CHAN[L]; rec = {"windows": {}}
        for k in READ_K:
            pi = os.path.join(OUT, "init", f"{L}_k{k}.csv"); pr = os.path.join(OUT, "ref", f"cover_{chan}_godeye_k{k}.csv")
            ri, rr = _row(pi), _row(pr)
            w = {"init_ok": ri is not None, "ref_ok": rr is not None}
            if ri and rr:
                di = _decisions(pi.replace(".csv", "_decisions.csv")); dr = _decisions(pr.replace(".csv", "_decisions.csv"))
                keys = sorted(set(di) | set(dr))
                w["decisions"] = len(keys); w["mismatches"] = sum(1 for key in keys if di.get(key) != dr.get(key))
                w["carbon_init"] = float(ri["total_carbon_kg"]); w["carbon_ref"] = float(rr["total_carbon_kg"])
                w["carbon_equal"] = abs(w["carbon_init"] - w["carbon_ref"]) < 1e-12
                w["pass"] = w["mismatches"] == 0 and w["carbon_equal"]
            else:
                w["pass"] = False
            rec["windows"][k] = w; out["pass"] = out["pass"] and w["pass"]
        out["lines"][L] = rec
    out["verdict"] = "INIT_OK" if out["pass"] else "STOP_INIT_MISMATCH"
    json.dump(out, open(os.path.join(OUT, "init_check.json"), "w"), indent=1)
    print(json.dumps({L: {k: (w.get("mismatches"), w.get("carbon_equal")) for k, w in r["windows"].items()} for L, r in out["lines"].items()}, indent=1)); print(out["verdict"])
    return out


def judge():
    ref = references()
    out = {"init": init_check(), "readings": {}, "gates": {}}
    C = {}; cap = {}; contracts_ok = True
    for L in LINES:
        chan = CHAN[L]
        for tier in (TIERS_FULL if chan == "full" else ("godeye",)):
            tot = 0.0; num = den = 0.0; n = 0
            for k in READ_K:
                r = _row(os.path.join(OUT, "last", f"{L}_{tier}_k{k}.csv"))
                if r is None:
                    continue
                if _contract(r):
                    contracts_ok = False
                c = float(r["total_carbon_kg"]); tot += c; n += 1
                if ref[k]["C_flat"] is not None and ref[k]["C_causal"] is not None:
                    num += ref[k]["C_flat"] - c; den += ref[k]["C_flat"] - ref[k]["C_causal"]
            C[(L, tier)] = tot if n == len(READ_K) else None
            cap[(L, tier)] = (num / den) if (den and n == len(READ_K)) else None
    R = {}; Rcap = {}
    for chan in ("full", "none"):
        for tier in (TIERS_FULL if chan == "full" else ("godeye",)):
            tot = 0.0; num = den = 0.0; n = 0
            for k in READ_K:
                r = _row(os.path.join(OUT, "ref", f"cover_{chan}_{tier}_k{k}.csv"))
                if r is None:
                    continue
                c = float(r["total_carbon_kg"]); tot += c; n += 1
                if ref[k]["C_flat"] is not None and ref[k]["C_causal"] is not None:
                    num += ref[k]["C_flat"] - c; den += ref[k]["C_flat"] - ref[k]["C_causal"]
            R[(chan, tier)] = tot if n == len(READ_K) else None; Rcap[(chan, tier)] = (num / den) if (den and n == len(READ_K)) else None
    g = {}
    g["1_init"] = out["init"]["pass"]
    g["2_trained_and_contracts"] = bool(all(C.get((L, "godeye")) is not None for L in LINES) and contracts_ok)
    cv, cr = cap.get(("V", "godeye")), Rcap.get(("full", "godeye"))
    g["3_prior_preserved"] = bool(cv is not None and cr and cv >= 0.80 * cr)
    lv = (C[("V", "shrink75")] / C[("V", "godeye")] - 1) if C.get(("V", "godeye")) and C.get(("V", "shrink75")) else None
    lr_ = (R[("full", "shrink75")] / R[("full", "godeye")] - 1) if R.get(("full", "godeye")) and R.get(("full", "shrink75")) else None
    g["4_shrink_hurts"] = bool(lv is not None and lv >= 0.05 and lr_ is not None and lr_ > 0)
    le = (C[("E", "shrink75")] / C[("E", "godeye")] - 1) if C.get(("E", "godeye")) and C.get(("E", "shrink75")) else None
    ce = cap.get(("E", "godeye"))
    g["5_eucrd_keeps_more"] = bool(le is not None and lv is not None and le <= 0.5 * lv and ce is not None and cv is not None and ce >= 0.80 * cv)
    g["6_not_ignoring"] = bool(C.get(("E", "godeye")) and C.get(("NE", "godeye")) and (C[("NE", "godeye")] - C[("E", "godeye")]) / C[("NE", "godeye")] >= 0.05)
    kl = None
    try:
        a = np.concatenate([[v for v in _decisions(os.path.join(OUT, "last", f"E_godeye_k{k}_decisions.csv")).values()] for k in READ_K])
        b = np.concatenate([[v for v in _decisions(os.path.join(OUT, "last", f"E_shrink75_k{k}_decisions.csv")).values()] for k in READ_K])
        K = 73; ha = np.bincount(a % K, minlength=K) + 0.5; hb = np.bincount(b % K, minlength=K) + 0.5
        pa, pb = ha / ha.sum(), hb / hb.sum(); kl = float((pa * np.log(pa / pb)).sum())
    except Exception as e:  # noqa: BLE001
        kl = f"unavailable: {e}"
    g["6_kl_action_marginals_E"] = kl
    g["6_not_ignoring"] = bool(g["6_not_ignoring"] and isinstance(kl, float) and kl > 0)
    g["7_crd_internals"] = "reported from training logs by rl_v2_smoke.sh (see crd_stats.json)"
    out["gates"] = g
    out["readings"] = {"policy_carbon": {f"{L}_{t}": v for (L, t), v in C.items()}, "policy_capture": {f"{L}_{t}": v for (L, t), v in cap.items()},
                       "reference_carbon": {f"cover_{c}_{t}": v for (c, t), v in R.items()}, "reference_capture": {f"cover_{c}_{t}": v for (c, t), v in Rcap.items()},
                       "loss_shrink75": {"V": lv, "E": le, "cover": lr_}, "references": ref}
    hard = ("1_init", "2_trained_and_contracts", "3_prior_preserved", "4_shrink_hurts", "5_eucrd_keeps_more", "6_not_ignoring")
    out["verdict"] = "PASS_SMOKE" if all(g[h] for h in hard) else "FAIL_SMOKE:" + ",".join(h for h in hard if not g[h])
    json.dump(out, open(os.path.join(OUT, "smoke_verdict.json"), "w"), indent=1)
    print(json.dumps({"gates": g, "verdict": out["verdict"], "readings": out["readings"]}, indent=1)[:6000])
    return out


if __name__ == "__main__":
    (init_check if (len(sys.argv) > 1 and sys.argv[1] == "init") else judge)()
