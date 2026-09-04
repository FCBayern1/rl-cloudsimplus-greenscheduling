import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_d_credit_audit_summary import ckpt_order, rows_from, verdict  # noqa: E402


def _res(line, ck, w_def, w_rou, low_def=0.1, low_rou=0.1):
    cls = lambda w, low: {"n": 10, "w": {"mean": w}, "rho": {"mean": 0.8}, "upper_tail_amplification": 0.8,  # noqa: E731
                          "lower_tail_suppression": low, "adv_positive_frac": 0.5, "adv_abs_mean_pre": 0.7,
                          "dq": {"mean": 0.0}, "c_t": {"mean": 0.3}}
    return {"line": line, "checkpoint": f"/x/{ck}",
            "warmed": {"defer_share_mean": 0.4, "tau": 0.5, "DEFER": cls(w_def, low_def), "ROUTE": cls(w_rou, low_rou),
                       "w_defer_minus_route": w_def - w_rou}}


def test_checkpoints_sort_with_init_first():
    assert ckpt_order("checkpoint_init") == -1 and ckpt_order("checkpoint_000009") == 9
    rows = rows_from([_res("E", "checkpoint_000003", 1, 1), _res("E", "checkpoint_init", 1, 1)])
    assert [r["ckpt"] for r in rows["E"]] == ["checkpoint_init", "checkpoint_000003"]


def test_verdict_reports_consistent_sign_after_warmup_only():
    res = [_res("E", f"checkpoint_00000{i}", 0.95, 1.03, 0.14, 0.10) for i in range(10)]
    res[2] = _res("E", "checkpoint_000002", 1.2, 1.0)      # early, before warm-up: ignored
    res += [_res("NE", f"checkpoint_00000{i}", 1.0, 1.0) for i in range(10)]
    v = verdict(rows_from(res), after=5)
    assert v["E"]["n_late"] == 5 and v["E"]["all_negative"] and not v["E"]["all_positive"]
    assert v["NE"]["all_negative"] is False and v["NE"]["all_positive"] is False
    assert all(abs(x - 0.04) < 1e-9 for x in v["E"]["low_tail_defer_minus_route"][3:])
