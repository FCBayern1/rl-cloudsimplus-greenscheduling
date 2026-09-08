"""Judge for the switch comparison (reports/EUCRD_SWITCH_AB_PREREG.md §2). Reads the per-call
dump written by the learner's A/B diagnostic and applies the frozen criteria. No tuning.

  W1 the weights differ      : mean |w_A - w_B| > 1e-3 and max > 1e-2
  W2 the change is targeted  : mean |dw| on firing cells >= 2x the mean on non-firing cells
  W3 the gradient differs    : cosine(grad_A, grad_B) < 0.9999, clamped to [-1, 1]
  W4 reported, not gated     : the change split by the sign of the advantage

The relative L2 difference is a SEPARATE diagnostic and never substitutes for W3: a gradient
that is merely rescaled has a non-zero L2 difference and a cosine of exactly 1, so L2 answers
"did the magnitude change" while only the cosine answers "did the direction change" (ruling
2026-09-08). A zero gradient in either variant leaves the cosine undefined; such calls are
counted and excluded rather than scored as 0 or 1.

Usage: python switch_ab_judge.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SWITCH_AB_OUT", "").strip() or os.path.join(HERE, "stage_a_out", "eucrd_switch_ab")
DUMP = os.path.join(OUT, "switch_ab.jsonl")
W_DELTA_MEAN_MIN, W_DELTA_MAX_MIN, TARGET_RATIO_MIN, GRAD_COS_MAX = 1e-3, 1e-2, 2.0, 0.9999


def rows():
    if not os.path.exists(DUMP):
        return []
    out = []
    for line in open(DUMP):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _pool(rs, key):
    v = [r[key] for r in rs if r.get(key) is not None]
    return (float(np.mean(v)) if v else None), (float(np.max(v)) if v else None), len(v)


def judge():
    rs = rows()
    res = {"calls": len(rs), "dump": DUMP,
           "thresholds": {"w_delta_mean_min": W_DELTA_MEAN_MIN, "w_delta_max_min": W_DELTA_MAX_MIN,
                          "target_ratio_min": TARGET_RATIO_MIN, "grad_cos_max": GRAD_COS_MAX}}
    if not rs:
        res["gates"] = {k: False for k in ("W1_weights_differ", "W2_change_is_targeted",
                                           "W3_gradient_differs")}
        res["verdict"] = "STOP_SWITCH_AB:no_calls_recorded"
        _write(res)
        return res

    dm, dmax, _ = _pool(rs, "w_delta_mean")
    _, dmx, _ = _pool(rs, "w_delta_max")
    f_mean, _, n_f = _pool(rs, "w_delta_mean_firing")
    nf_mean, _, _ = _pool(rs, "w_delta_mean_nonfiring")
    # cosine clamped into [-1, 1]: nearly parallel gradients round outside the range
    gcos = [min(1.0, max(-1.0, float(r["grad_cosine"])))
            for r in rs if r.get("grad_cosine") is not None]
    n_undefined = sum(1 for r in rs if r.get("grad_cosine") is None
                      or not np.isfinite(r.get("grad_cosine", np.nan))
                      or (r.get("grad_norm_a") in (0, 0.0) or r.get("grad_norm_b") in (0, 0.0)))
    ratio = (f_mean / nf_mean) if (f_mean and nf_mean and nf_mean > 0) else None

    g = {}
    g["W1_weights_differ"] = bool(dm is not None and dm > W_DELTA_MEAN_MIN
                                  and dmx is not None and dmx > W_DELTA_MAX_MIN)
    g["W2_change_is_targeted"] = bool(ratio is not None and ratio >= TARGET_RATIO_MIN)
    g["W3_gradient_differs"] = bool(gcos and float(np.max(gcos)) < GRAD_COS_MAX)

    res["pooled"] = {
        "w_delta_mean": dm, "w_delta_max": dmx,
        "w_delta_mean_firing": f_mean, "w_delta_mean_nonfiring": nf_mean,
        "targeting_ratio": ratio, "calls_with_firing": n_f,
        "grad_cosine_mean": (float(np.mean(gcos)) if gcos else None),
        "grad_cosine_max": (float(np.max(gcos)) if gcos else None),
        "grad_cosine_min": (float(np.min(gcos)) if gcos else None),
        "calls_with_undefined_cosine": int(n_undefined),
        "grad_norm_ratio_mean": (float(np.mean([r["grad_norm_b"] / r["grad_norm_a"]
                                                for r in rs
                                                if r.get("grad_norm_a")]))
                                 if any(r.get("grad_norm_a") for r in rs) else None),
        "grad_rel_l2_mean": _pool(rs, "grad_rel_l2")[0],
        # null baseline: the same advantages scored twice. The A-vs-B numbers are only
        # meaningful above it (added after the first run measured a non-zero harness noise).
        "grad_cosine_null_mean": _pool(rs, "grad_cosine_null")[0],
        "grad_rel_l2_null_mean": _pool(rs, "grad_rel_l2_null")[0],
        "adv_cosine_mean": _pool(rs, "adv_cosine")[0],
        "n_valid_mean": _pool(rs, "n_valid")[0], "n_firing_mean": _pool(rs, "n_firing")[0],
    }
    # W4: reported only
    res["advantage_sign"] = {
        "frac_neg_damped_mean": _pool(rs, "frac_neg_damped")[0],
        "frac_pos_damped_mean": _pool(rs, "frac_pos_damped")[0],
        "w_ratio_neg_mean": _pool(rs, "w_ratio_neg_mean")[0],
        "w_ratio_pos_mean": _pool(rs, "w_ratio_pos_mean")[0],
        "n_neg_adv_mean": _pool(rs, "n_neg_adv")[0], "n_pos_adv_mean": _pool(rs, "n_pos_adv")[0],
    }
    # Paired against its OWN null, call by call (ruling 2026-09-08): the A/B difference is
    # only meaningful where it exceeds the difference the same advantages produce when scored
    # twice. A pooled mean cosine cannot establish that, and a low cosine means a LARGER
    # difference, so the comparison is made on the deviation from 1 per call.
    paired = [(1.0 - r["grad_cosine"], 1.0 - r["grad_cosine_null"], r.get("grad_rel_l2"),
               r.get("grad_rel_l2_null"))
              for r in rs if r.get("grad_cosine") is not None and r.get("grad_cosine_null") is not None]
    if paired:
        # The pairing is read on the RELATIVE L2 difference, not on 1 - cosine: with nearly
        # parallel gradients the cosine rounds above 1, so 1 - cos goes negative and every
        # ratio or inequality built on it is meaningless. rel_l2 stays non-negative and
        # well-conditioned. The cosine numbers are still reported (W3 is registered on them).
        ab = np.array([p[2] for p in paired if p[2] is not None])
        nu = np.array([p[3] for p in paired if p[3] is not None])
        cos_ab = np.clip(np.array([1.0 - p[0] for p in paired]), -1.0, 1.0)
        cos_nu = np.clip(np.array([1.0 - p[1] for p in paired]), -1.0, 1.0)
        res["paired_null"] = {
            "calls": int(min(ab.size, nu.size)),
            "measure": "relative L2 difference of the two gradients, per call "
                       "(a magnitude diagnostic; W3 stays on the cosine)",
            "frac_calls_ab_exceeds_own_null": float(np.mean(ab > nu)) if ab.size == nu.size else None,
            "frac_calls_ab_exceeds_2x_own_null": (float(np.mean(ab > 2 * nu))
                                                  if ab.size == nu.size else None),
            "rel_l2_ab_median": float(np.median(ab)) if ab.size else None,
            "rel_l2_null_median": float(np.median(nu)) if nu.size else None,
            "rel_l2_ratio_median": (float(np.median(ab / np.maximum(nu, 1e-12)))
                                    if ab.size == nu.size and nu.size else None),
            "rel_l2_ab_max": float(np.max(ab)) if ab.size else None,
            "rel_l2_null_max": float(np.max(nu)) if nu.size else None,
            "cos_ab_min": float(np.min(cos_ab)), "cos_null_min": float(np.min(cos_nu)),
            "resolvable": bool(ab.size == nu.size and float(np.median(ab)) > float(np.median(nu))),
        }

    # Term decomposition (prereg Addendum A): the surrogate gradient on its own, A vs B, with
    # its own null and the norms of the terms it shares the total with. Diagnostic only; it
    # does not re-judge W3 and the first call (the estimators' own warm-up) is excluded.
    dec = [r for r in rs[1:] if r.get("pi_cosine") is not None]
    if dec:
        pi_cos = np.clip(np.array([r["pi_cosine"] for r in dec]), -1.0, 1.0)
        pi_rel = np.array([r["pi_rel_l2"] for r in dec])
        pi_nul = np.array([r["pi_rel_l2_null"] for r in dec])
        tot_rel = np.array([r["grad_rel_l2"] for r in dec if r.get("grad_rel_l2") is not None])
        res["decomposition"] = {
            "calls": len(dec),
            "pi_cosine_min": float(pi_cos.min()), "pi_cosine_mean": float(pi_cos.mean()),
            "pi_rel_l2_median": float(np.median(pi_rel)), "pi_rel_l2_max": float(pi_rel.max()),
            "pi_rel_l2_null_max": float(pi_nul.max()),
            "pi_resolvable": bool(np.median(pi_rel) > np.median(pi_nul)),
            "total_rel_l2_median": (float(np.median(tot_rel)) if tot_rel.size else None),
            "dilution_ratio_median": (float(np.median(pi_rel) / max(1e-12, np.median(tot_rel)))
                                      if tot_rel.size else None),
            "pi_norm_median": float(np.median([r["pi_norm_a"] for r in dec])),
            "vf_norm_median": float(np.median([r["vf_norm_a"] for r in dec])),
            "entkl_norm_median": float(np.median([r["entkl_norm_a"] for r in dec])),
            "pi_share_of_total_norm_median": float(np.median([r["pi_share_of_total_norm"] for r in dec])),
        }

    top = [t for r in rs for t in (r.get("top_changed") or [])]
    if top:
        res["top_changed"] = {
            "n": len(top),
            "frac_firing": float(np.mean([1.0 if t["firing"] else 0.0 for t in top])),
            "frac_negative_advantage": float(np.mean([1.0 if t["adv_sign"] < 0 else 0.0 for t in top])),
            "delta_mean": float(np.mean([t["delta"] for t in top])),
        }
    res["gates"] = g
    keys = ("W1_weights_differ", "W2_change_is_targeted", "W3_gradient_differs")
    res["verdict"] = ("SWITCH_AB_PASS" if all(g[k] for k in keys)
                      else "STOP_SWITCH_AB:" + ",".join(k for k in keys if not g[k]))
    _write(res)
    return res


def _write(res):
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "switch_ab.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("calls", "gates", "verdict", "pooled", "paired_null",
                                          "decomposition", "advantage_sign", "top_changed")
                      if k in res}, indent=1))


if __name__ == "__main__":
    judge()
