#!/usr/bin/env python3
"""V3.2B certification gates (Codex second review, 2026-08-16 09:10).

Two hard gates, both machine-checkable so no chain script can quietly soften
them again (the first BCv2 chain shipped a delta>0 triage gate where the
pre-registered Gate 2 demands >= +0.05 - that class of drift ends here):

  probe_gate        - the six-condition certified probe gate.
  teacher_data_gate - per-offset PAIRED completion gate for teacher data.
                      Absolute >=99.5% at every offset is physically
                      unreachable at horizon 7200: the no-defer control
                      itself scores 99.47%/99.38% at offsets 0/2018 (audit
                      r0_s1_fixed.json). The reachable contract is
                      min(0.995, control_at_same_offset - eps).

Both return {"pass": bool, "reasons": [...]}; empty reasons == clean pass.
CLI: v32b_gates.py probe <probe.json> | teacher <teacher.json> <control.json>
"""
import json
import sys

DELTA_MIN = 0.05          # pre-registered Gate 2 threshold, never softened
MONO_MIN = 0.75           # A4: >=75% adjacent pairs
SATURATION_MAX = 0.90     # defer baseline must stay below this
CONTRACT = 0.995
PAIR_EPS = 0.005


def probe_gate(probe: dict) -> dict:
    """Certified probe gate: ALL six pre-registered conditions."""
    reasons = []
    jt = probe.get("job_temporal", {}) or {}
    mono = probe.get("monotone", {}) or {}
    delta = float(jt.get("delta", float("-inf")))
    lo = float(jt.get("p_defer_not_worth", 1.0))
    hi = float(jt.get("p_defer_worth_waiting", 0.0))
    if not (delta >= DELTA_MIN):
        reasons.append(f"delta {delta:+.4f} < +{DELTA_MIN}")
    if not (float(mono.get("monotone_frac_gain", 0.0)) >= MONO_MIN):
        reasons.append(f"gain monotonicity {mono.get('monotone_frac_gain')} < {MONO_MIN}")
    slack_frac = mono.get("monotone_frac_slack", mono.get("monotone_frac_ttd", 0.0))
    if not (float(slack_frac) >= MONO_MIN):
        reasons.append(f"slack monotonicity {slack_frac} < {MONO_MIN}")
    if not (lo < SATURATION_MAX):
        reasons.append(f"defer baseline saturated: P(defer|not-worth) {lo:.4f} >= {SATURATION_MAX}")
    if not (hi > lo):
        reasons.append(f"P(defer|worth) {hi:.4f} <= P(defer|not-worth) {lo:.4f}")
    if not bool(jt.get("judgeable", False)):
        reasons.append("channel not judgeable vs inert null")
    return {"pass": not reasons, "reasons": reasons}


def teacher_data_gate(teacher_records: list, control_by_offset: dict,
                      contract: float = CONTRACT,
                      eps: float = PAIR_EPS) -> dict:
    """Paired completion gate over EVERY teacher episode.

    Each episode must reach min(contract, control_at_same_offset - eps).
    Episodes whose offset has no control reference FAIL (no unpaired waivers).
    """
    reasons = []
    for r in teacher_records:
        off = r.get("green_offset")
        compl = float(r.get("completion_rate_mi", 0.0))
        ctrl = control_by_offset.get(off, control_by_offset.get(str(off)))
        if ctrl is None:
            reasons.append(f"offset {off}: no control reference")
            continue
        floor = min(contract, float(ctrl) - eps)
        if compl < floor:
            reasons.append(
                f"offset {off}: teacher {compl:.4f} < floor {floor:.4f} "
                f"(control {float(ctrl):.4f})")
    return {"pass": not reasons, "reasons": reasons}


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "probe":
        verdict = probe_gate(json.loads(open(sys.argv[2]).read()))
    elif len(sys.argv) >= 4 and sys.argv[1] == "teacher":
        teacher = json.loads(open(sys.argv[2]).read())
        control = json.loads(open(sys.argv[3]).read())
        ctrl_map = {r["green_offset"]: r["completion_rate_mi"]
                    for r in control["records"] if r["arm"] == "control"}
        verdict = teacher_data_gate(teacher["records"], ctrl_map)
    else:
        sys.exit("usage: v32b_gates.py probe <probe.json> | "
                 "teacher <teacher.json> <control.json>")
    print(json.dumps(verdict, ensure_ascii=False))
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
