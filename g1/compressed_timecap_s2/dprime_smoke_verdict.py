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
    lift, auc = selectivity.get("lift"), selectivity.get("auc")
    gates["timing_selectivity"] = lift is not None and auc is not None and lift >= LIFT_MIN and auc >= AUC_MIN
    v_rates = [r["defer_rate"] for r in clean_rows if r["line"] == "V"]
    gates["defer_not_collapsed"] = bool(v_rates) and all(DEFER_MIN <= x <= DEFER_MAX for x in v_rates)
    s = (audit_e_last or {}).get("warmed") or {}
    d = s.get("DEFER", {})
    gates["guard_no_mass_erasure"] = bool(d.get("n")) and d.get("lower_tail_suppression", 1.0) <= ERASE_MAX
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
    a = ap.parse_args()
    h = json.load(open(a.health))
    clean_rows = h.get("clean_rows", [])
    codir = {L: bool(v.get("ok")) for L, v in (h.get("codirectional") or {}).items()}
    out = judge(clean_rows, json.load(open(a.selectivity)), json.load(open(a.audit)), codir)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
