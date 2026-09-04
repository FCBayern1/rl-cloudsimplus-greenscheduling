"""Stage D long-run verdict (STAGE_D_LONGRUN_PREREG §5 + §8), mechanical and testable.

Per seed, per line, final checkpoint, six cells x six judgement windows:
    C_NV, C_V0 (clean), C_V1 (calibrated_shrink_v1), C_NE, C_E0, C_E1, plus shuffle/anti readings
    C = pooled carbon intensity = sum carbon / sum completed MI over the 36-run grid.

Gates per seed:
  1  (C_NV - C_V0)/C_NV >= 0.05
  2  (C_V1 - C_V0)/C_V0 >= 0.05 and (C_V1 - C_V0) >= 0.5 (C_NV - C_V0)
  3  (C_NE - C_E0)/C_NE >= 0.05
  4  (C_E1 - C_E0)/C_E0 <= 0.5 (C_V1 - C_V0)/C_V0
  5  C_E1 < C_V1; C_E0 <= 1.05 C_V0; E's corrupted deployment contract-green (a corrupted E
     contract failure fails the robustness gate); CRD liveness (spread > 0, shares not pinned)
G0 (per seed): every expected row present with the registered offset and tier (else
INVALID_DATA); every CLEAN deployment contract-green (else STOP_STAGE_D_CONTRACT); reward and
ledger carbon co-directional from checkpoint_init to final on every line.
Direction: gates 1-4 each hold in >= 4/5 seeds. No (cell, window) is ever voided; V's
corrupted-deployment contract failures are reported as additional harm, not gated.

Verdicts: PASS_STAGE_D | STOP_STAGE_D_CONTRACT | STOP_STAGE_D_STEP2 | STOP_STAGE_D_STEP3 | INVALID_DATA
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

LINES = ("NV", "V", "NE", "E")
CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
TIERS = {"NV": ("hollow",), "NE": ("hollow",),
         "V": ("godeye", "calibrated_shrink_v1", "shuffle", "anti"),
         "E": ("godeye", "calibrated_shrink_v1", "shuffle", "anti")}
CLEAN = {"NV": "hollow", "NE": "hollow", "V": "godeye", "E": "godeye"}
G1 = G2 = G3 = 0.05
SEEDS_NEEDED = 4
CONTRACT = {"comp": 0.995, "ontime": 0.995}


def contract_ok(r):
    return (r["comp"] >= CONTRACT["comp"] and r["ontime"] >= CONTRACT["ontime"]
            and r["forced"] <= 0 and r.get("cap", 0) <= 0)


def pooled(rows, keys):
    c = sum(rows[k]["carbon"] for k in keys)
    mi = sum(rows[k]["mi"] for k in keys)
    return c / mi if mi > 0 else None


def judge_seed(rows, offsets, crd=None):
    """rows: {(line, tag, tier, cell, k): {"carbon","mi","comp","ontime","forced","cap",
    "reward","offset","tier_effective"}} for tag in {"final","init"}; offsets: registered
    list indexed by k. Returns the per-seed record."""
    ks = list(range(len(offsets)))
    expected = [("final", L, t, c, k) for L in LINES for t in TIERS[L] for c in CELLS for k in ks]
    expected += [("init", L, CLEAN[L], c, k) for L in LINES for c in CELLS for k in ks]
    invalid = []
    for tag, L, t, c, k in expected:
        r = rows.get((L, tag, t, c, k))
        if r is None:
            invalid.append(("missing", L, tag, t, c, k))
        elif int(r.get("offset", -1)) != int(offsets[k]):
            invalid.append(("offset", L, tag, t, c, k, r.get("offset")))
        elif r.get("tier_effective") not in (None, t):
            invalid.append(("tier", L, tag, t, c, k, r.get("tier_effective")))
    if invalid:
        return {"verdict": "INVALID_DATA", "invalid": invalid[:20], "n_invalid": len(invalid)}

    grid = lambda L, tag, t: [(L, tag, t, c, k) for c in CELLS for k in ks]  # noqa: E731
    C = {"NV": pooled(rows, grid("NV", "final", "hollow")), "NE": pooled(rows, grid("NE", "final", "hollow")),
         "V0": pooled(rows, grid("V", "final", "godeye")), "V1": pooled(rows, grid("V", "final", "calibrated_shrink_v1")),
         "E0": pooled(rows, grid("E", "final", "godeye")), "E1": pooled(rows, grid("E", "final", "calibrated_shrink_v1")),
         "V_shuffle": pooled(rows, grid("V", "final", "shuffle")), "V_anti": pooled(rows, grid("V", "final", "anti")),
         "E_shuffle": pooled(rows, grid("E", "final", "shuffle")), "E_anti": pooled(rows, grid("E", "final", "anti"))}
    clean_bad = [(L, c, k) for L in LINES for c in CELLS for k in ks if not contract_ok(rows[(L, "final", CLEAN[L], c, k)])]
    v_corrupt_bad = [(c, k) for c in CELLS for k in ks if not contract_ok(rows[("V", "final", "calibrated_shrink_v1", c, k)])]
    e_corrupt_bad = [(c, k) for c in CELLS for k in ks if not contract_ok(rows[("E", "final", "calibrated_shrink_v1", c, k)])]
    codir = {}
    for L in LINES:
        ci, cf = pooled(rows, grid(L, "init", CLEAN[L])), pooled(rows, grid(L, "final", CLEAN[L]))
        ri = sum(rows[x]["reward"] for x in grid(L, "init", CLEAN[L])) / len(ks) / len(CELLS)
        rf = sum(rows[x]["reward"] for x in grid(L, "final", CLEAN[L])) / len(ks) / len(CELLS)
        codir[L] = {"carbon": (ci, cf), "reward": (ri, rf), "ok": (cf - ci) * (rf - ri) <= 0}
    g = {
        "g1": (C["NV"] - C["V0"]) / C["NV"] >= G1,
        "g2": (C["V1"] - C["V0"]) / C["V0"] >= G2 and (C["V1"] - C["V0"]) >= 0.5 * (C["NV"] - C["V0"]),
        "g3": (C["NE"] - C["E0"]) / C["NE"] >= G3,
        "g4": (C["E1"] - C["E0"]) / C["E0"] <= 0.5 * (C["V1"] - C["V0"]) / C["V0"],
    }
    crd_ok = True
    crd_notes = {}
    if crd:
        for L in ("NE", "E"):
            s = crd.get(L) or {}
            spread = s.get("dr_std")
            if spread is None:
                spread = max(float(s.get("rho_routing_std") or 0), float(s.get("reweight_w_std") or 0))
            pinned = any(v is not None and (v <= 0.0501 or v >= 0.9999)
                         for v in (s.get("rho_routing_mean"), s.get("rho_forecast_mean")))
            crd_notes[L] = {"spread": spread, "pinned": pinned}
            crd_ok = crd_ok and spread > 0 and not pinned
    g["g5"] = (C["E1"] < C["V1"] and C["E0"] <= 1.05 * C["V0"] and not e_corrupt_bad and crd_ok)
    g["g0_clean_contract"] = not clean_bad
    g["g0_codirectional"] = all(v["ok"] for v in codir.values())
    return {"verdict": None, "C": C, "gates": g, "clean_contract_bad": clean_bad,
            "v_corrupted_contract_bad": v_corrupt_bad, "e_corrupted_contract_bad": e_corrupt_bad,
            "codirectional": codir, "crd": crd_notes,
            "effects": {"vanilla_forecast_value": (C["NV"] - C["V0"]) / C["NV"],
                        "vanilla_harm": (C["V1"] - C["V0"]) / C["V0"],
                        "eucrd_forecast_value": (C["NE"] - C["E0"]) / C["NE"],
                        "eucrd_harm": (C["E1"] - C["E0"]) / C["E0"]}}


def judge(seed_records):
    """seed_records: {seed: judge_seed(...)} -> overall verdict."""
    if any(r["verdict"] == "INVALID_DATA" for r in seed_records.values()):
        return {"verdict": "INVALID_DATA",
                "invalid_seeds": [s for s, r in seed_records.items() if r["verdict"] == "INVALID_DATA"]}
    n = len(seed_records)
    contract = [s for s, r in seed_records.items() if not r["gates"]["g0_clean_contract"]]
    codir = [s for s, r in seed_records.items() if not r["gates"]["g0_codirectional"]]
    counts = {k: sum(1 for r in seed_records.values() if r["gates"][k]) for k in ("g1", "g2", "g3", "g4", "g5")}
    out = {"n_seeds": n, "counts": counts, "seeds_needed": SEEDS_NEEDED,
           "contract_failed_seeds": contract, "codirection_failed_seeds": codir,
           "per_seed_effects": {s: r["effects"] for s, r in seed_records.items()}}
    if contract or codir:
        out["verdict"] = "STOP_STAGE_D_CONTRACT" if contract else "STOP_STAGE_D_STEP2"
        return out
    step2 = counts["g1"] >= SEEDS_NEEDED and counts["g2"] >= SEEDS_NEEDED
    step3 = counts["g3"] >= SEEDS_NEEDED and counts["g4"] >= SEEDS_NEEDED and counts["g5"] >= SEEDS_NEEDED
    out["verdict"] = "PASS_STAGE_D" if (step2 and step3) else ("STOP_STAGE_D_STEP3" if step2 else "STOP_STAGE_D_STEP2")
    return out


def load_seed(results_dir, seed, offsets, mi_per_job):
    rows = {}
    base = os.path.join(results_dir, f"seed_{seed}")
    for L in LINES:
        for tag in ("final", "init"):
            for f in glob.glob(os.path.join(base, f"{L}_{tag}", "*.csv")):
                b = os.path.basename(f)[:-4]
                cell = "_".join(b.split("_")[:5]); rest = b[len(cell) + 1:]; tier, k = rest.rsplit("_k", 1)
                r = list(csv.DictReader(open(f)))
                if not r:
                    continue
                r = r[-1]
                g = lambda key, d=0.0: float(r.get(key, d) or d)  # noqa: E731
                rows[(L, tag, tier, cell, int(k))] = {
                    "carbon": g("total_carbon_kg"), "mi": g("total_finished_cloudlets") * mi_per_job[cell],
                    "comp": g("completion_rate_mi"), "ontime": g("ontime_mi_share"),
                    "forced": g("deadline_forced_count"), "cap": g("ep_global_carbon_cap_count"),
                    "reward": g("global_reward_sum"), "offset": int(g("green_episode_offset_rows", -1)),
                    "tier_effective": None}
    return rows


def main():
    import ladder_v2_verdict as lv
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "drl-manager/results/stage_d_longrun")
    win = json.load(open(os.path.join(HERE, "stage_a_out", "stage_d_windows.json")))
    offsets = [int(w["offset"]) for w in win["eval_windows"]]
    mi = lv._mi_per_job()
    seeds = sorted(int(d.split("_")[1]) for d in os.listdir(results_dir) if d.startswith("seed_"))
    recs = {}
    for s in seeds:
        crd_path = os.path.join(results_dir, f"seed_{s}", "crd_stats.json")
        crd = json.load(open(crd_path)) if os.path.exists(crd_path) else None
        recs[s] = judge_seed(load_seed(results_dir, s, offsets, mi), offsets, crd)
    out = judge(recs)
    out["seeds"] = {s: {k: v for k, v in r.items() if k != "invalid"} for s, r in recs.items()}
    with open(os.path.join(results_dir, "stage_d_longrun_verdict.json"), "w") as fh:
        fh.write(json.dumps(out, sort_keys=True, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
