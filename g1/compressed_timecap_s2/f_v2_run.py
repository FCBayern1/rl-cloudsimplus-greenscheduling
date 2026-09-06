"""F_FITS_V2 runner (reports/F_FITS_V2_PREREG.md, frozen 80b693fb + Addendum A).

  sentinel   one instrumented cover_argmax episode per window on the F2 and F3 twins; checks §1
  labels     causal expert on the certification offset twin, all 18 windows, true candidate costs
  corpus     the expert's schedule replayed on the F2 / F3 twins with the observation dump
  fit        features + targets -> validation-selected residual for F2 and F3, freeze record
  test       the one-pass reading of the four test windows: offline ladder (truth, flat) for
             validity, causal expert, cover_argmax, cover_residual; verdict
Usage: python f_v2_run.py sentinel | labels | corpus | fit | test | judge
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "drl-manager"))
import ladder_run as lr  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out", "f_v2")
WINDOWS = os.path.join(OUT, "windows.json")
K = 73
TOL_COVER_W = 1e-2
CONTRACT = {"completion_rate_mi": 0.995, "ontime_mi_share": 0.995}
CAPTURE_MIN, CAPTURE_TOL, MIN_VALID = 0.50, 0.02, 3


def windows():
    w = json.load(open(WINDOWS))
    dev = lr._dev()
    roles = [("dev", o) for o in dev] + [("train", o) for o in w["train"]] + [("val", o) for o in w["val"]] + [("test", o) for o in w["test"]]
    return roles


def allowlist():
    return [o for _, o in windows()]


def twin(F):
    """(config, block) with the F_FITS_V2 allowlist; F1 = offset twin, F2 = interface truth key, F3 = interface TimeCAP key."""
    al = allowlist()
    if F == "F1":
        return lr.cert_config("offset", allowlist=al, tag="fv2")
    if F == "defer":
        return lr.cert_config("defer", allowlist=al, tag="fv2")
    p, cell = lr.cert_config("interface", allowlist=al, tag="fv2")
    if F == "F2":
        return p, cell
    cfg = yaml.safe_load(open(p)); cfg[cell]["green_oracle_mode"] = "timecap"; cfg[cell].pop("perturb_tier", None)
    p3 = os.path.join(HERE, "config_ladder_cert_interface_fv2_timecap.yml")
    with open(p3, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    return p3, cell


def blk_of(F):
    p, cell = twin(F)
    return yaml.safe_load(open(p))[cell]


# ── sentinel ─────────────────────────────────────────────────────────────────────────
def sentinel():
    from gym_cloudsimplus.envs.option_executor import cand_green_cover
    rec = {"windows": {}, "pass": True}
    blk = blk_of("F2")
    mips = float(blk["datacenters"][0].get("vm_pe_mips", 40000)); u = float(blk.get("cloudlet_cpu_utilization", 1.0))
    d = os.path.join(OUT, "sentinel"); os.makedirs(d, exist_ok=True)
    for F in ("F2", "F3"):
        cfg, cell = twin(F)
        for k, (role, off) in enumerate(windows()):
            dump = os.path.join(d, f"{F}_k{k}_decisions.csv")
            for p in (dump, dump.replace(".csv", "_obs.npz")):
                if os.path.exists(p):
                    os.remove(p)
            ok = lr._evaluate(cfg, cell, k, off, "cover_argmax", os.path.join(d, f"{F}_k{k}.csv"),
                              {"OFFSET_GRID_DENSE": "1", "EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1", "FORECAST_ALIGN_DUMP": "1"})
            r = {"ok": ok}
            if ok:
                z = np.load(dump.replace(".csv", "_obs.npz"))
                G_obs = np.asarray(z["dc_current_green_power_w"], dtype=np.float64)          # (T, n)
                fgs = np.asarray(z["future_green_series"], dtype=np.float64)                  # (T, n, H)
                T, n, H = fgs.shape
                G, _ = lr.truth_curve(blk, off, T + H + 2)
                # 1: series index h at step t == simulator green at step t + h (F2 exact; F3: index 0 only)
                err1 = 0.0
                n_nan = 0
                for t in range(T):
                    if np.isnan(fgs[t]).any():                        # the reset observation carries no series
                        n_nan += 1; continue
                    hmax = H if F == "F2" else 1                     # TimeCAP: only the present row is truth
                    err1 = max(err1, float(np.abs(fgs[t, :, :hmax] - G[:, t:t + hmax]).max()))
                r["series_rows_without_channel"] = n_nan
                r["series_max_abs_err_w"] = err1
                r["obs_rows_match"] = bool(lr.curve_rows_match(G, G_obs.T)[0])
                # 3: the key recomputed from the dumped series and committed grid == published key
                cov = np.asarray(z["cand_green_cover"], dtype=np.float64); occ = np.asarray(z["committed_pes"], dtype=np.float64)
                st = np.asarray(z["committed_static_w"], dtype=np.float64)
                lag_arr = np.asarray(z["committed_lag"], dtype=np.float64).reshape(-1)
                lag = int(lag_arr[np.isfinite(lag_arr)][0])
                pes = np.asarray(z["batch_cloudlet_pes"]); mi = np.asarray(z["batch_cloudlet_mi"])
                rows = list(csv.DictReader(open(dump)))
                ids_by_t = {}
                for x in rows:
                    ids_by_t.setdefault(int(x["step"]), {})[int(x["slot"])] = int(x["cloudlet_id"])
                mask = np.asarray(z["batch_cloudlet_offset_allowed"], dtype=np.float64)
                err3 = 0.0; checked = 0
                for t, slots in ids_by_t.items():
                    if np.isnan(fgs[t]).any():
                        continue
                    ids = np.full(cov.shape[1], -1); pe = np.zeros(cov.shape[1]); mm = np.zeros(cov.shape[1])
                    for s_, cid in slots.items():
                        ids[s_] = cid; pe[s_] = pes[t, s_]; mm[s_] = mi[t, s_]
                    rec_cov = cand_green_cover(fgs[t], occ[t], pe, mm, ids, t, list(range(K)), mips, u, static_w=st[t], lag=lag) * mask[t]
                    for s_ in slots:
                        err3 = max(err3, float(np.abs(rec_cov[s_] - cov[t, s_]).max())); checked += 1
                r["key_max_abs_err"] = err3; r["key_decisions_checked"] = checked
                r["pass"] = bool(ok and r["obs_rows_match"] and err1 <= TOL_COVER_W and err3 <= 1e-4)
            else:
                r["pass"] = False
            rec["windows"][f"{F}_k{k}"] = r
            rec["pass"] = rec["pass"] and r["pass"]
            print(f"sentinel {F} k{k} ({role} {off}): {'PASS' if r['pass'] else 'FAIL'} {r}", flush=True)
    rec["verdict"] = "SENTINEL_PASS" if rec["pass"] else "STOP_ROW_ALIGNMENT"
    json.dump(rec, open(os.path.join(OUT, "sentinel.json"), "w"), indent=1)
    print(rec["verdict"])


# ── labels ────────────────────────────────────────────────────────────────────────────
def labels():
    cfg, cell = twin("F1")
    d = os.path.join(OUT, "labels"); os.makedirs(d, exist_ok=True)
    for k, (role, off) in enumerate(windows()):
        out_csv = os.path.join(d, f"k{k}.csv"); dump = os.path.join(d, f"k{k}_decisions.csv"); lab = os.path.join(d, f"k{k}_costs.npz")
        for p in (out_csv, dump, dump.replace(".csv", "_obs.npz"), lab):
            if os.path.exists(p):
                os.remove(p)
        env = {"OFFSET_GRID_DENSE": "1", "CAUSAL_RUNG": "truth", "EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1", "CAUSAL_LABEL_COSTS": lab}
        ok = lr._evaluate(cfg, cell, k, off, "causal_expert", out_csv, env)
        row = list(csv.DictReader(open(out_csv)))[-1] if ok else {}
        print(f"labels k{k} ({role} {off}): {'ok' if ok else 'FAILED'} labels {row.get('causal_labels')} excluded {row.get('causal_label_excluded')} "
              f"inconsistent {row.get('causal_label_inconsistent')} unsolved {row.get('causal_unsolved')} ontime {row.get('ontime_mi_share')}", flush=True)


def expert_schedule(k):
    led = list(csv.DictReader(open(os.path.join(OUT, "labels", f"k{k}_option_ledger.csv"))))
    return {int(r["id"]): (int(r["dc"]), int(float(r["s_f"]))) for r in led}


def corpus():
    for F in ("F2", "F3"):
        cfg, cell = twin(F)
        d = os.path.join(OUT, "corpus", F); os.makedirs(d, exist_ok=True)
        for k, (role, off) in enumerate(windows()):
            sched = expert_schedule(k)
            sj = os.path.join(d, f"k{k}_schedule.json")
            json.dump({"schedule": {str(i): list(v) for i, v in sched.items()}, "grid": list(range(K))}, open(sj, "w"))
            dst = os.path.join(d, f"k{k}_decisions.csv")
            for p in (dst, dst.replace(".csv", "_obs.npz")):
                if os.path.exists(p):
                    os.remove(p)
            ok = lr._evaluate(cfg, cell, k, off, "schedule_replay", os.path.join(d, f"replay_k{k}.csv"), lr.replay_env(sj, dst))
            row = list(csv.DictReader(open(os.path.join(d, f"replay_k{k}.csv"))))[-1] if ok else {}
            ref = list(csv.DictReader(open(os.path.join(OUT, "labels", f"k{k}.csv"))))[-1]
            same = abs(float(row.get("total_carbon_kg", "nan")) - float(ref["total_carbon_kg"])) < 1e-12 if ok else False
            print(f"corpus {F} k{k} ({role}): {'ok' if ok else 'FAILED'} reproduces expert {same} masked {row.get('ep_opt_hold_masked')}", flush=True)


# ── fit ───────────────────────────────────────────────────────────────────────────────
def _decisions(F, k):
    """(X, legal, target, meta) per decision of window k on twin F, from the corpus dump and the cost labels."""
    from src.baselines.cover_residual import features_from_obs
    d = os.path.join(OUT, "corpus", F)
    z = np.load(os.path.join(d, f"k{k}_decisions_obs.npz"))
    rows = list(csv.DictReader(open(os.path.join(d, f"k{k}_decisions.csv"))))
    lab = np.load(os.path.join(OUT, "labels", f"k{k}_costs.npz"))
    costs = {int(i): (c, bool(e)) for i, c, e in zip(lab["ids"], lab["costs"], lab["excluded"])}
    n = int(z["dc_current_green_power_w"].shape[1])
    seen = set(); X, L, T, meta = [], [], [], []
    for r in rows:
        cid = int(r["cloudlet_id"])
        if cid < 0 or cid in seen or cid not in costs:
            continue
        seen.add(cid); t, s = int(r["step"]), int(r["slot"])
        c, excluded = costs[cid]
        if excluded:
            meta.append({"id": cid, "excluded": True}); continue
        obs = {key: z[key][t] for key in ("cand_green_cover", "batch_cloudlet_offset_allowed", "dc_current_green_power_w",
                                          "dc_future_short_mean", "dc_future_long_mean", "dc_utilizations")}
        planner = {"batch_cloudlet_pes": z["batch_cloudlet_pes"][t], "batch_cloudlet_mi": z["batch_cloudlet_mi"][t],
                   "batch_cloudlet_time_to_deadline": np.full(z["batch_cloudlet_pes"].shape[1], float(r["ttd_sec"] or 0))}
        Xi, cov, legal = features_from_obs(obs, planner, s, n, K)
        finite = np.isfinite(c) & (legal >= 0.5)
        if not finite.any():
            meta.append({"id": cid, "excluded": True, "reason": "no legal finite cost"}); continue
        cmin = c[finite].min()
        tgt = (finite & (c <= cmin + 0.5)).astype(np.float64)
        X.append(Xi); L.append(legal); T.append(tgt); meta.append({"id": cid, "t": t, "n_target": int(tgt.sum())})
    return np.stack(X), np.stack(L), np.stack(T), meta


def fit():
    from src.baselines.cover_residual import select, save
    roles = windows()
    idx = {role: [k for k, (rr, _) in enumerate(roles) if rr == role] for role in ("dev", "train", "val", "test")}
    rec = {}
    for F in ("F2", "F3"):
        parts = {}
        for role in ("dev", "train", "val"):
            Xs, Ls, Ts, ex = [], [], [], 0
            for k in idx[role]:
                X, L, T, meta = _decisions(F, k)
                Xs.append(X); Ls.append(L); Ts.append(T); ex += sum(1 for m in meta if m.get("excluded"))
            parts[role] = (np.concatenate(Xs), np.concatenate(Ls), np.concatenate(Ts), ex)
        Xtr = np.concatenate([parts["dev"][0], parts["train"][0]]); Ltr = np.concatenate([parts["dev"][1], parts["train"][1]]); Ttr = np.concatenate([parts["dev"][2], parts["train"][2]])
        Xv, Lv, Tv = parts["val"][0], parts["val"][1], parts["val"][2]
        model, table = select(Xtr, Ltr, Ttr, Xv, Lv, Tv)
        path = os.path.join(OUT, "fit", F); save(model, path, meta={"twin": F, "n_train": int(Xtr.shape[0]), "n_val": int(Xv.shape[0]),
                                                                   "excluded_train": parts["dev"][3] + parts["train"][3], "excluded_val": parts["val"][3], "selection": table})
        h = hashlib.sha256(open(os.path.join(path, "residual.pt"), "rb").read()).hexdigest()[:16]
        rec[F] = {"n_train": int(Xtr.shape[0]), "n_val": int(Xv.shape[0]), "selection": table["selected"], "grid": table["grid"], "model_sha256": h}
        print(f"fit {F}: train {Xtr.shape[0]} val {Xv.shape[0]} selected {table['selected']} model {h}", flush=True)
    json.dump(rec, open(os.path.join(OUT, "fit_record.json"), "w"), indent=1)


# ── test pass ─────────────────────────────────────────────────────────────────────────
def _contract(row):
    v = [f"{key} {row.get(key)} < {lo}" for key, lo in CONTRACT.items() if float(row.get(key, 0) or 0) < lo]
    v += [f"{key} = {row.get(key)}" for key in ("deadline_forced_count", "ep_opt_stale", "ep_opt_hold_masked") if float(row.get(key, 0) or 0) != 0]
    return v


def test_pass():
    """Offline ladder truth + flat (validity), then cover_argmax and cover_residual on both twins, on the four test windows."""
    from ladder_planner import build_instance, solve_milp, settle
    roles = windows(); tests = [(k, off) for k, (r, off) in enumerate(roles) if r == "test"]
    d = os.path.join(OUT, "test"); os.makedirs(d, exist_ok=True)
    cfg_def, cell_def = twin("defer"); cfg_off, cell_off = twin("F1")
    cfg_all = yaml.safe_load(open(cfg_def)); blk = cfg_all[cell_def]; sites = lr.sites_from_config(cfg_all, blk)
    mips = float(blk["datacenters"][0].get("vm_pe_mips", 40000)); u = float(blk.get("cloudlet_cpu_utilization", 1.0))
    rec = {}
    for k, off in tests:
        w = {}
        dump = os.path.join(d, f"k{k}_dump_decisions.csv")
        for p in (dump, dump.replace(".csv", "_obs.npz")):
            if os.path.exists(p):
                os.remove(p)
        lr._evaluate(cfg_def, cell_def, k, off, "reactive_wait_planner", os.path.join(d, f"k{k}_dump.csv"), {"EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1"})
        rows = list(csv.DictReader(open(dump))); jobs = lr.jobs_from_dump(rows, mips, u)
        obs_green = np.asarray(np.load(dump.replace(".csv", "_obs.npz"))["dc_current_green_power_w"], dtype=np.float64).T
        need = max(max(j.latest + j.runtime for j in jobs) + 1, obs_green.shape[1])
        truth, meta = lr.truth_curve(blk, off, need)
        w["curve_rows_match"] = bool(lr.curve_rows_match(truth, obs_green)[0])
        mu = lr._mu_w(blk)
        for rung in ("truth", "shrink_0"):
            G = lr.rung_curve(truth, rung, mu, seed_key=f"ladder:{off}")
            res = solve_milp(build_instance(jobs, sites, G), time_limit_s=3600)
            sj = os.path.join(d, f"k{k}_{rung}_schedule.json")
            json.dump({"schedule": {str(i): list(v) for i, v in res.get("schedule", {}).items()}, "grid": list(range(K))}, open(sj, "w"))
            ok = res.get("status") == "OPTIMAL" and lr._evaluate(cfg_off, cell_off, k, off, "schedule_replay", os.path.join(d, f"k{k}_{rung}.csv"), lr.replay_env(sj))
            row = list(csv.DictReader(open(os.path.join(d, f"k{k}_{rung}.csv"))))[-1] if ok else {}
            cm = settle(build_instance(jobs, sites, truth), res["schedule"])["C_kg"] if res.get("status") == "OPTIMAL" else None
            w[rung] = {"status": res.get("status"), "C_model": cm, "C_sim": (float(row["total_carbon_kg"]) if row else None),
                       "closure_rel": (abs(float(row["total_carbon_kg"]) - cm) / cm if row and cm else None), "contract": _contract(row) if row else ["no replay"]}
        expert = list(csv.DictReader(open(os.path.join(OUT, "labels", f"k{k}.csv"))))[-1]
        w["causal"] = {"C_sim": float(expert["total_carbon_kg"]), "contract": _contract(expert)}
        for F in ("F2", "F3"):
            cfg, cell = twin(F)
            for arm, env in (("cover_argmax", {}), ("cover_residual", {"COVER_RESIDUAL_MODEL": os.path.join(OUT, "fit", F)})):
                out_csv = os.path.join(d, f"k{k}_{F}_{arm}.csv")
                ok = lr._evaluate(cfg, cell, k, off, arm, out_csv, {"OFFSET_GRID_DENSE": "1", **env})
                row = list(csv.DictReader(open(out_csv)))[-1] if ok else {}
                w[f"{F}_{arm}"] = {"C_sim": (float(row["total_carbon_kg"]) if row else None), "contract": _contract(row) if row else ["no run"]}
        rec[str(k)] = {"offset": off, **w}
        print(f"test k{k} ({off}): truth {w['truth']['C_sim']} flat {w['shrink_0']['C_sim']} causal {w['causal']['C_sim']} "
              f"F2 argmax {w['F2_cover_argmax']['C_sim']} F2 fit {w['F2_cover_residual']['C_sim']} F3 argmax {w['F3_cover_argmax']['C_sim']} F3 fit {w['F3_cover_residual']['C_sim']}", flush=True)
        json.dump(rec, open(os.path.join(OUT, "test_readings.json"), "w"), indent=1)


def judge():
    rec = json.load(open(os.path.join(OUT, "test_readings.json")))
    out = {"windows": {}, "F2": {}, "F3": {}}
    num = {"F2": 0.0, "F3": 0.0}; den = 0.0; n_valid = 0; per_ok = {"F2": True, "F3": True}; contract_ok = {"F2": True, "F3": True}
    for k, w in rec.items():
        ct, cf = w["truth"]["C_sim"], w["shrink_0"]["C_sim"]
        closed = (w["truth"]["status"] == "OPTIMAL" and w["shrink_0"]["status"] == "OPTIMAL" and w["truth"]["closure_rel"] is not None
                  and w["truth"]["closure_rel"] <= lr.CLOSURE_REL and w["shrink_0"]["closure_rel"] <= lr.CLOSURE_REL and not w["truth"]["contract"] and not w["shrink_0"]["contract"])
        head = (cf - ct) if (ct is not None and cf is not None) else None
        valid = bool(closed and head is not None and cf > 0 and head / cf >= lr.HEADROOM_REL and head >= lr.HEADROOM_ABS)
        ww = {"valid": valid, "headroom": head, "C_truth": ct, "C_flat": cf, "C_causal": w["causal"]["C_sim"]}
        if valid:
            n_valid += 1; hc = cf - w["causal"]["C_sim"]; den += hc
            for F in ("F2", "F3"):
                ca = w[f"{F}_cover_argmax"]["C_sim"]; cr = w[f"{F}_cover_residual"]["C_sim"]
                cap_a = (cf - ca) / hc if hc > 0 else None; cap_r = (cf - cr) / hc if hc > 0 else None
                ww[F] = {"capture_argmax": cap_a, "capture_residual": cap_r, "contract_residual": w[f"{F}_cover_residual"]["contract"]}
                num[F] += (cf - cr) if cr is not None else 0.0
                if cap_r is None or cap_a is None or cap_r < cap_a - CAPTURE_TOL:
                    per_ok[F] = False
                if w[f"{F}_cover_residual"]["contract"]:
                    contract_ok[F] = False
        out["windows"][k] = ww
    for F in ("F2", "F3"):
        pooled = num[F] / den if den else None
        out[F] = {"pooled_capture": pooled, "per_window_within_tol": per_ok[F], "contract_ok": contract_ok[F],
                  "pass": bool(n_valid >= MIN_VALID and pooled is not None and pooled >= CAPTURE_MIN and per_ok[F] and contract_ok[F])}
    out["n_valid"] = n_valid
    out["verdict"] = ("STOP_TEST_HEADROOM" if n_valid < MIN_VALID else ("F2_PASS" if out["F2"]["pass"] else "STOP_F2_LEARNER"))
    json.dump(out, open(os.path.join(OUT, "f_v2_verdict.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    {"sentinel": sentinel, "labels": labels, "corpus": corpus, "fit": fit, "test": test_pass, "judge": judge}.get(what, lambda: print(__doc__))()
