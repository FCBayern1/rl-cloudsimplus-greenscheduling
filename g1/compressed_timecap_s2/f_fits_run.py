"""F1-F3 on the causal expert's decisions (reports/F_FITS_PREREG.md).

  corpus     build the three corpora: F1 = the expert's own offset-twin dumps; F2 / F3 = the
             expert's schedule replayed with the observation dump on the certification interface
             twin (candidate key from truth / from TimeCAP)
  fit F      fit + score one interface (F1|F2|F3) with option_bc's frozen recipe
  execute F  deploy the fit as the option_bc arm on its twin, held-out windows k4, k5
  judge      capture gate, classification report, verdict -> f_verdict.json
Usage: python f_fits_run.py corpus | fit F1 | execute F1 | judge
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_run as lr  # noqa: E402

CAUSAL = os.path.join(HERE, "stage_a_out", "causal_v1")
OUT = os.path.join(HERE, "stage_a_out", "f_fits")
LADDER_VERDICT = os.path.join(lr.REPO, "reports", "manifests", "ladder_v4", "run1", "ladder_verdict.json")
PY = os.path.join(lr.REPO, "drl-manager", ".venv", "bin", "python")
TRAIN_K, HELD_K = (0, 1, 2, 3), (4, 5)
CAPTURE_MIN = 0.50
CONTRACT = {"completion_rate_mi": 0.995, "ontime_mi_share": 0.995}


def twin(F):
    """(config path, block) of the twin an interface is observed and executed on."""
    if F == "F1":
        return lr.cert_config("offset")
    p, cell = lr.cert_config("interface")
    if F == "F2":
        return p, cell
    cfg = yaml.safe_load(open(p))
    cfg[cell]["green_oracle_mode"] = "timecap"
    cfg[cell].pop("perturb_tier", None)
    p3 = os.path.join(HERE, "config_ladder_cert_interface_timecap.yml")
    with open(p3, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return p3, cell


def expert_schedule(k):
    """{id: (site, start)} of the causal expert on window k from its option ledger (s_f)."""
    led = list(csv.DictReader(open(os.path.join(CAUSAL, f"causal_truth_k{k}_option_ledger.csv"))))
    return {int(r["id"]): (int(r["dc"]), int(float(r["s_f"]))) for r in led}


def build_corpus():
    dev = lr._dev()
    for F in ("F1", "F2", "F3"):
        cfg, cell = twin(F)
        d = os.path.join(OUT, "corpus", F); os.makedirs(d, exist_ok=True)
        for k, off in enumerate(dev):
            dst = os.path.join(d, f"{cell}_k{k}_decisions.csv")
            if F == "F1":
                src = os.path.join(CAUSAL, f"causal_truth_k{k}_decisions.csv")
                shutil.copy(src, dst); shutil.copy(src.replace(".csv", "_obs.npz"), dst.replace(".csv", "_obs.npz"))
                print(f"F1 k{k}: copied", flush=True)
                continue
            sched = expert_schedule(k)
            sj = os.path.join(d, f"k{k}_schedule.json")
            json.dump({"schedule": {str(i): list(v) for i, v in sched.items()}, "grid": list(range(73))}, open(sj, "w"))
            for p in (dst, dst.replace(".csv", "_obs.npz")):
                if os.path.exists(p):
                    os.remove(p)
            ok = lr._evaluate(cfg, cell, k, off, "schedule_replay", os.path.join(d, f"replay_k{k}.csv"), lr.replay_env(sj, dst))
            row = list(csv.DictReader(open(os.path.join(d, f"replay_k{k}.csv"))))[-1] if ok else {}
            ref = list(csv.DictReader(open(os.path.join(CAUSAL, f"causal_truth_k{k}.csv"))))[-1]
            print(f"{F} k{k}: {'ok' if ok else 'FAILED'} replay carbon {row.get('total_carbon_kg')} vs expert {ref['total_carbon_kg']} "
                  f"masked {row.get('ep_opt_hold_masked')}", flush=True)


def fit(F):
    cfg, cell = twin(F)
    # the module is built on the dense every-step grid (the corpus masks have 5 x 73 columns);
    # without OFFSET_GRID_DENSE the block's dyadic 9-value grid is assumed and the mask sizes clash
    env = dict(os.environ, OPTION_BC_CORPUS=os.path.join(OUT, "corpus", F), OPTION_BC_OUT=os.path.join(OUT, "fit", F),
               OPTION_BC_CONFIG=cfg, OPTION_BC_BLOCK=cell, OPTION_BC_HOLD_MIN_KAPPA="2", OFFSET_GRID_DENSE="1")
    log = os.path.join(OUT, f"fit_{F}.log"); os.makedirs(OUT, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run([PY, "option_bc.py", "all", "--offset"], cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"fit {F}: rc {rc}, log {log}", flush=True)


def execute(F):
    cfg, cell = twin(F)
    dev = lr._dev()
    d = os.path.join(OUT, "exec", F); os.makedirs(d, exist_ok=True)
    for k in HELD_K:
        out_csv = os.path.join(d, f"k{k}.csv")
        env = {"OFFSET_GRID_DENSE": "1", "OPTION_BC_MODEL": os.path.join(OUT, "fit", F), "OPTION_BC_CONFIG": cfg, "OPTION_BC_BLOCK": cell}
        ok = lr._evaluate(cfg, cell, k, dev[k], "option_bc", out_csv, env)
        row = list(csv.DictReader(open(out_csv)))[-1] if ok and os.path.exists(out_csv) else {}
        print(f"execute {F} k{k}: {'ok' if ok else 'FAILED'} carbon {row.get('total_carbon_kg')} ontime {row.get('ontime_mi_share')} forced {row.get('deadline_forced_count')}", flush=True)


def judge():
    lad = json.load(open(LADDER_VERDICT))
    flat = {int(k): w["shrink_0"]["C_sim"] for k, w in lad["windows"].items()}
    res = {"interfaces": {}, "held_windows": list(HELD_K)}
    for F in ("F1", "F2", "F3"):
        rec = {"windows": {}}
        num = den = 0.0; contract_ok = True; complete = True
        for k in HELD_K:
            p = os.path.join(OUT, "exec", F, f"k{k}.csv")
            ref = list(csv.DictReader(open(os.path.join(CAUSAL, f"causal_truth_k{k}.csv"))))[-1]
            c_exp = float(ref["total_carbon_kg"]); head = flat[k] - c_exp
            if not os.path.exists(p):
                complete = False; continue
            row = list(csv.DictReader(open(p)))[-1]; c = float(row["total_carbon_kg"])
            viol = [f"{key} {row.get(key)} < {lo}" for key, lo in CONTRACT.items() if float(row.get(key, 0) or 0) < lo]
            viol += [f"{key} = {row.get(key)}" for key in ("deadline_forced_count", "ep_opt_stale") if float(row.get(key, 0) or 0) != 0]
            rec["windows"][k] = {"C_fit": c, "C_causal_truth": c_exp, "C_flat": flat[k], "capture": (flat[k] - c) / head if head > 0 else None,
                                 "contract_violations": viol}
            num += flat[k] - c; den += head; contract_ok = contract_ok and not viol
        rec["pooled_capture"] = num / den if den else None
        sp = os.path.join(OUT, "fit", F, "score.json")
        if os.path.exists(sp):
            sc = json.load(open(sp))
            rec["classification"] = {"corpus_check": sc.get("corpus_check"), "main_gate_raw": sc.get("main_gate_raw"),
                                     "supporting": sc.get("supporting"), "valid": bool((sc.get("corpus_check") or {}).get("valid"))}
        rec["capture_pass"] = bool(complete and den and num / den >= CAPTURE_MIN and contract_ok)
        rec["complete"] = complete
        res["interfaces"][F] = rec
    f3 = res["interfaces"]["F3"]; f2 = res["interfaces"]["F2"]; f1 = res["interfaces"]["F1"]
    if not all(res["interfaces"][F]["complete"] for F in ("F1", "F2", "F3")):
        res["verdict"] = "INCOMPLETE"
    elif f3["capture_pass"]:
        res["verdict"] = "F3_PASS_RL_PREREG_MAY_PROCEED"
    elif f2["capture_pass"]:
        res["verdict"] = "F2_PASS_F3_FAIL_TIMECAP_QUALITY"
    elif f1["capture_pass"]:
        res["verdict"] = "F1_PASS_ONLY"
    else:
        res["verdict"] = "ALL_FAIL_OPEN"
    json.dump(res, open(os.path.join(OUT, "f_verdict.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "corpus":
        build_corpus()
    elif what == "fit":
        fit(sys.argv[2])
    elif what == "execute":
        execute(sys.argv[2])
    elif what == "judge":
        judge()
    else:
        print(__doc__)
