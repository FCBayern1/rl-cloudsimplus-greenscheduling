import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_d_credit_audit import defer_share, summarize  # noqa: E402


def test_defer_share_counts_only_valid_slots():
    acts = np.array([[5, 5, 0, 1], [0, 1, 2, 3]])
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
    s = defer_share(acts, defer_index=5, mask=mask)
    assert abs(s[0] - 2 / 3) < 1e-12 and s[1] == 0.0
    assert abs(defer_share(acts, 5)[0] - 0.5) < 1e-12          # no mask: all four slots


def _rec(n_defer, n_route, w_defer, w_route):
    n = n_defer + n_route
    share = np.array([1.0] * n_defer + [0.0] * n_route)
    w = np.array([w_defer] * n_defer + [w_route] * n_route)
    adv = np.array([-0.5] * n_defer + [0.4] * n_route)
    return {"rho": w * 0.86, "w": w, "adv_pre": adv, "adv_post": adv * w, "dq": np.zeros(n) + 0.1,
            "dr": np.zeros(n), "c_t": np.zeros(n) + 0.5, "share": share, "tau": np.array([1.5])}


def test_summary_reports_tails_and_class_difference():
    out = summarize(_rec(30, 70, w_defer=0.06, w_route=1.16))
    assert out["DEFER"]["n"] == 30 and out["ROUTE"]["n"] == 70
    assert out["DEFER"]["lower_tail_suppression"] == 1.0 and out["DEFER"]["upper_tail_amplification"] == 0.0
    assert out["ROUTE"]["upper_tail_amplification"] == 1.0
    assert abs(out["w_defer_minus_route"] - (0.06 - 1.16)) < 1e-12
    assert out["DEFER"]["adv_positive_frac"] == 0.0 and out["ROUTE"]["adv_positive_frac"] == 1.0
    assert out["DEFER"]["adv_abs_mean_post"] < out["DEFER"]["adv_abs_mean_pre"]     # erased on the way out


def test_summary_reports_the_applied_weight_tail_when_recorded():
    rec = _rec(30, 70, w_defer=0.06, w_route=1.16)
    assert "lower_tail_suppression_guarded" not in summarize(rec)["DEFER"]      # old captures: absent
    rec["w_guarded"] = 1.0 + 0.5 * (rec["w"] - 1.0)                              # eta = 0.5 shrink
    out = summarize(rec)
    assert out["DEFER"]["lower_tail_suppression"] == 1.0                          # raw tail unchanged
    assert out["DEFER"]["lower_tail_suppression_guarded"] == 0.0
    assert abs(out["DEFER"]["w_guarded_min"] - 0.53) < 1e-12


def test_summary_handles_an_empty_class():
    out = summarize(_rec(0, 10, w_defer=1.0, w_route=1.0))
    assert out["DEFER"]["n"] == 0 and "w_defer_minus_route" not in out
