"""Stage D' development-smoke verdict (STAGE_D_PRIME_DESIGN §10, Codex 2026-09-05).

Six criteria that must hold at once on ONE old seed and the old (certified) windows:

    contract_clean_all_lines   every clean deployment of every line: completion >= 0.995,
                               on-time >= 0.995
    no_forced                  Java deadline_forced_count == 0 on every clean deployment
    timing_selectivity         V: lift >= 0.10 and balanced AUC >= 0.60 on the frozen corpus
    defer_not_collapsed        V's clean-deployment defer rate within [DEFER_MIN, DEFER_MAX]
    guard_no_mass_erasure      E's last checkpoint: P(w_guarded < 0.2 | DEFER) <= ERASE_MAX
    reward_carbon_codirection  every line: init -> final reward up and carbon down

Any failure is STOP_DPRIME_SMOKE; no second guard strength, no margin retune. judge() is
pure so it can be tested without disk. Inputs are the health reader's rows, the timing
selectivity JSON and the credit-audit JSON of E's last checkpoint.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_MIN, DEFER_MIN, DEFER_MAX, ERASE_MAX = 0.995, 0.02, 0.90, 0.05
LIFT_MIN, AUC_MIN = 0.10, 0.60


def judge(clean_rows, selectivity, audit_e_last, codirection):
    """clean_rows: list of {line, completion, ontime, forced, defer_rate} over clean deployments;
    selectivity: {lift, auc}; audit_e_last: the audit JSON (uses warmed.DEFER.lower_tail_suppression
    computed on the guarded weight); codirection: {line: bool}. Pure."""
    gates = {}
    gates["contract_clean_all_lines"] = bool(clean_rows) and all(
        r["completion"] >= CONTRACT_MIN and r["ontime"] >= CONTRACT_MIN for r in clean_rows)
    gates["no_forced"] = bool(clean_rows) and all(r["forced"] == 0 for r in clean_rows)
    # §16 Q2: the gate is the RAW (pre-mask), job-paired, recurrent selectivity; the deployed
    # (post-mask) figure is reported as a safety diagnostic and never substitutes for it.
    main = selectivity.get("main_gate_raw_paired", selectivity)
    lift, auc = main.get("lift"), main.get("auc")
    gates["timing_selectivity"] = lift is not None and auc is not None and lift >= LIFT_MIN and auc >= AUC_MIN
    v_rates = [r["defer_rate"] for r in clean_rows if r["line"] == "V"]
    gates["defer_not_collapsed"] = bool(v_rates) and all(DEFER_MIN <= x <= DEFER_MAX for x in v_rates)
    s = (audit_e_last or {}).get("warmed") or {}
    d = s.get("DEFER", {})
    # wiring sentinel (near-tautological under eta = 0.5, kept as such per §16 Q1)
    # It must read the APPLIED (guarded) weight: under eta = 0.5 that weight is >= 0.5 by
    # construction, so any lower tail there means the guard is not wired. The raw weight's
    # tail is the substantive quantity and belongs to guard_no_mass_erasure, not here.
    gates["guard_wiring_sentinel"] = bool(d.get("n")) and d.get("lower_tail_suppression_guarded", 1.0) <= ERASE_MAX
    # substantive guard gate from the cross statistics of E's last checkpoint (§16 Q1)
    gg = ((audit_e_last or {}).get("cross") or {}).get("guard_gate") or {}
    gates["guard_no_mass_erasure"] = bool(gg.get("pass"))
    gates["reward_carbon_codirection"] = bool(codirection) and all(codirection.get(L, False) for L in ("NV", "V", "NE", "E"))
    verdict = "PASS_DPRIME_SMOKE" if all(gates.values()) else "STOP_DPRIME_SMOKE"
    return {"verdict": verdict, "gates": gates,
            "detail": {"n_clean_rows": len(clean_rows), "v_defer_rates": v_rates,
                       "lift": lift, "auc": auc,
                       "e_defer_lower_tail": d.get("lower_tail_suppression"),
                       "e_defer_n": d.get("n")}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--health", required=True, help="stage_d_health_verdict output JSON")
    ap.add_argument("--selectivity", required=True, help="timing_selectivity.py output JSON")
    ap.add_argument("--audit", required=True, help="stage_d_credit_audit.py JSON of E's last checkpoint")
    ap.add_argument("--out", default=os.path.join(HERE, "stage_a_out", "dprime_smoke_verdict.json"))
    ap.add_argument("--results", default=None, help="smoke results dir (derives clean rows and co-direction from the evaluation rows)")
    ap.add_argument("--logs", default=None)
    ap.add_argument("--cross", default=None, help="cross_statistics.json of E's last checkpoint (guard_gate, §16 Q1)")
    a = ap.parse_args()
    h = json.load(open(a.health))
    audit = json.load(open(a.audit))
    if a.cross:
        cs = json.load(open(a.cross))
        # the cross file is keyed by audit name; take the single E entry (or the first)
        entry = next((v for k, v in cs.items() if k.startswith("audit_E") or k.startswith("E_")), None) or next(iter(cs.values()), {})
        audit["cross"] = entry
    clean_rows, codir = h.get("clean_rows", []), {L: bool(v.get("ok")) for L, v in (h.get("codirectional") or {}).items()}
    if a.results:
        # derive from the evaluation rows themselves, the same loader the health verdict uses
        sys.path.insert(0, HERE)
        import stage_d_health_verdict as hv
        evals, _crd, _probe = hv.load(a.results, a.logs or a.results.replace("results", "logs"), a.results)
        CLEAN = {"NV": "hollow", "NE": "hollow", "V": "godeye", "E": "godeye"}
        clean_rows = [{"line": L, "completion": v["comp"], "ontime": v["ontime"], "forced": v["forced"],
                       "defer_rate": v.get("defer_rate", 0.0)}
                      for (L, tag, tier, _c, _k), v in evals.items() if tag == "last" and tier == CLEAN[L]]
        codir = {}
        for L in CLEAN:
            last = [v for (l, tag, tier, _c, _k), v in evals.items() if l == L and tag == "last" and tier == CLEAN[L]]
            first = [v for (l, tag, tier, _c, _k), v in evals.items() if l == L and tag == "first" and tier == CLEAN[L]]
            if last and first:
                r1, r0 = sum(v["reward"] for v in last), sum(v["reward"] for v in first)
                c1, c0 = sum(v["carbon"] for v in last), sum(v["carbon"] for v in first)
                codir[L] = bool(r1 > r0 and c1 < c0)
    out = judge(clean_rows, json.load(open(a.selectivity)), audit, codir)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
