"""M5 cross-statistics (STAGE_D_PRIME_DESIGN §10): does the low weight fall on the DEFER
transitions whose advantage is negative, i.e. on the corrective signal?

From the per-transition arrays saved by stage_d_credit_audit.py --save-raw:

    E[w | DEFER, A<0]  vs  E[w | DEFER, A>=0]
    P(w<0.2 | DEFER, A<0)  vs  P(w<0.2 | DEFER, A>=0)
    retained negative mass  = sum(|A| w) / sum(|A|) over DEFER & A<0, for w_raw and for the
                              counterfactual guard w' = 1 + eta (w_raw - 1), eta = 0.5
    the same for ROUTE, as the within-checkpoint comparison

Usage: python stage_d_credit_cross.py [results_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DIR = os.path.join(REPO, "drl-manager", "results", "stage_d_credit_audit")
ETA = 0.5
LOW = 0.2


def cross(w, adv, share, eta=ETA, w_guarded=None):
    """Pure. w, adv, share: 1-D arrays of the same length (w = the raw weight before the guard).
    w_guarded: the weight the learner actually applied, when recorded; then the report also
    carries the bitwise check against 1 + eta (w - 1) and the retained mass under it."""
    w = np.asarray(w, float); adv = np.asarray(adv, float); share = np.asarray(share, float)
    wg = 1.0 + eta * (w - 1.0)
    out = {"eta": eta}
    if w_guarded is not None:
        wga = np.asarray(w_guarded, float)
        out["bitwise_max_abs_err"] = float(np.max(np.abs(wga - wg))) if wga.size else None
        wg = wga
    for cls, sel in (("DEFER", share >= 0.5), ("ROUTE", share < 0.5)):
        neg, pos = sel & (adv < 0), sel & (adv >= 0)
        rec = {"n": int(sel.sum()), "n_neg": int(neg.sum()), "n_pos": int(pos.sum())}
        rec["E_w_neg"] = float(w[neg].mean()) if neg.any() else None
        rec["E_w_pos"] = float(w[pos].mean()) if pos.any() else None
        rec["P_low_neg"] = float((w[neg] < LOW).mean()) if neg.any() else None
        rec["P_low_pos"] = float((w[pos] < LOW).mean()) if pos.any() else None
        mass = np.abs(adv[neg]).sum()
        rec["neg_mass_retained_raw"] = float((np.abs(adv[neg]) * w[neg]).sum() / mass) if mass > 0 else None
        rec["neg_mass_retained_guarded"] = float((np.abs(adv[neg]) * wg[neg]).sum() / mass) if mass > 0 else None
        out[cls] = rec
    d = out["DEFER"]
    out["low_weight_falls_on_negative"] = (d["P_low_neg"] is not None and d["P_low_pos"] is not None
                                           and d["P_low_neg"] > d["P_low_pos"])
    out["guard_recovers_negative_mass"] = (d["neg_mass_retained_guarded"] is not None
                                           and d["neg_mass_retained_raw"] is not None
                                           and d["neg_mass_retained_guarded"] - d["neg_mass_retained_raw"] >= 0.05)
    out["guard_gate"] = guard_gate(d, out.get("bitwise_max_abs_err"))
    return out


GUARD_MIN_N, GUARD_R_MIN, GUARD_EPS, GUARD_BIT_TOL = 100, 0.90, 0.01, 1e-6


def guard_gate(d, bitwise_err):
    """Substantive guard gate (design §16 Q1) on the DEFER class of E's last checkpoint:
    n(DEFER, A<0) >= 100; R_guarded >= 0.90; if R_raw < 0.95 then
    R_guarded - R_raw >= 0.5 (1 - R_raw) - eps; and w_guarded = 1 + 0.5 (w_raw - 1) bitwise
    (within GUARD_BIT_TOL) when the applied weight was recorded."""
    r_raw, r_g = d.get("neg_mass_retained_raw"), d.get("neg_mass_retained_guarded")
    g = {"n_neg_ok": int(d.get("n_neg", 0)) >= GUARD_MIN_N,
         "retained_ok": r_g is not None and r_g >= GUARD_R_MIN,
         "recovery_ok": (r_raw is None or r_g is None) and False or
                        (r_raw >= 0.95 or (r_g - r_raw) >= 0.5 * (1.0 - r_raw) - GUARD_EPS),
         "bitwise_ok": bitwise_err is None or bitwise_err <= GUARD_BIT_TOL}
    g["pass"] = all(g.values())
    return g


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(glob.glob(os.path.join(d, "*_raw.npz")))
    if not files:
        print("no raw npz files; run stage_d_credit_audit.py --save-raw first")
        return
    report = {}
    print(f"{'file':28s} {'cls':5s} {'n':>6s} {'E[w|A<0]':>9s} {'E[w|A>=0]':>9s} {'P(low|A<0)':>10s} {'P(low|A>=0)':>11s} {'ret_raw':>8s} {'ret_eta':>8s}")
    for f in files:
        z = np.load(f)
        r = cross(z["w"], z["adv_pre"], z["share"], w_guarded=(z["w_guarded"] if "w_guarded" in z.files else None))
        name = os.path.basename(f).replace("_raw.npz", "")
        report[name] = r
        for cls in ("DEFER", "ROUTE"):
            c = r[cls]
            g = lambda k: ("%9.3f" % c[k]) if c.get(k) is not None else "        -"  # noqa: E731
            print(f"{name:28s} {cls:5s} {c['n']:6d} {g('E_w_neg')} {g('E_w_pos')} {g('P_low_neg'):>10s} {g('P_low_pos'):>11s} "
                  f"{g('neg_mass_retained_raw'):>8s} {g('neg_mass_retained_guarded'):>8s}")
    with open(os.path.join(d, "cross_statistics.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
