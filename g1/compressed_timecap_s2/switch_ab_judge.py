"""Judge for the switch comparison (reports/EUCRD_SWITCH_AB_PREREG.md §2). Reads the per-call
dump written by the learner's A/B diagnostic and applies the frozen criteria. No tuning.

  W1 the weights differ      : mean |w_A - w_B| > 1e-3 and max > 1e-2
  W2 the change is targeted  : mean |dw| on firing cells >= 2x the mean on non-firing cells
  W3 the gradient differs    : cosine(grad_A, grad_B) < 0.9999
  W4 reported, not gated     : the change split by the sign of the advantage

Usage: python switch_ab_judge.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage_a_out", "eucrd_switch_ab")
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
    gcos = [r["grad_cosine"] for r in rs if r.get("grad_cosine") is not None]
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
        ab = np.array([p[0] for p in paired]); nu = np.array([p[1] for p in paired])
        rel = np.array([p[2] for p in paired if p[2] is not None])
        rel_n = np.array([p[3] for p in paired if p[3] is not None])
        res["paired_null"] = {
            "calls": len(paired),
            "frac_calls_ab_exceeds_own_null": float(np.mean(ab > nu)),
            "frac_calls_ab_exceeds_2x_own_null": float(np.mean(ab > 2 * nu)),
            "one_minus_cos_ab_median": float(np.median(ab)),
            "one_minus_cos_null_median": float(np.median(nu)),
            "one_minus_cos_ratio_median": (float(np.median(ab / np.maximum(nu, 1e-12)))
                                           if len(nu) else None),
            "rel_l2_ab_median": (float(np.median(rel)) if rel.size else None),
            "rel_l2_null_median": (float(np.median(rel_n)) if rel_n.size else None),
            "worst_call_ab": float(np.max(ab)), "worst_call_null": float(np.max(nu)),
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
                                          "advantage_sign", "top_changed") if k in res}, indent=1))


if __name__ == "__main__":
    judge()
