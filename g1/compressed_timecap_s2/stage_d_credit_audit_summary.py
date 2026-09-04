"""Aggregate the M5 credit-audit JSONs (stage_d_credit_audit.py) into one table per line,
ordered by checkpoint, with the quantities Q3 names (STAGE_D_PRIME_DESIGN §2, §5).

Usage: python stage_d_credit_audit_summary.py [results_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_DIR = os.path.join(REPO, "drl-manager", "results", "stage_d_credit_audit")


def ckpt_order(name):
    return -1 if name.endswith("checkpoint_init") else int(name.rsplit("_", 1)[1])


def rows_from(results):
    """results: list of audit dicts -> {line: [row, ...]} sorted by checkpoint. Pure."""
    out = {}
    for r in results:
        s = r.get("warmed") or r.get("first_batch")
        if not s:
            continue
        ck = os.path.basename(r["checkpoint"])
        d, ro = s.get("DEFER", {}), s.get("ROUTE", {})
        row = {"ckpt": ck, "order": ckpt_order(ck), "defer_share": s.get("defer_share_mean"),
               "n_defer": d.get("n", 0), "n_route": ro.get("n", 0), "tau": s.get("tau")}
        for cls, c in (("defer", d), ("route", ro)):
            if c.get("n"):
                row[f"w_{cls}"] = c["w"]["mean"]
                row[f"rho_{cls}"] = c["rho"]["mean"]
                row[f"up_{cls}"] = c["upper_tail_amplification"]
                row[f"low_{cls}"] = c["lower_tail_suppression"]
                row[f"advpos_{cls}"] = c["adv_positive_frac"]
                row[f"adv_{cls}"] = c["adv_abs_mean_pre"]
                row[f"dq_{cls}"] = c["dq"]["mean"]
                row[f"ct_{cls}"] = c["c_t"]["mean"]
        row["w_diff"] = s.get("w_defer_minus_route")
        out.setdefault(r["line"], []).append(row)
    for line in out:
        out[line].sort(key=lambda x: x["order"])
    return out


def verdict(rows_by_line, after=5):
    """Consistent-sign test of w(DEFER) - w(ROUTE) on checkpoints >= `after` for E, and
    the same statistic on N_E as the control. Pure; reports, does not gate."""
    rep = {}
    for line, rows in rows_by_line.items():
        late = [r["w_diff"] for r in rows if r["order"] >= after and r.get("w_diff") is not None]
        rep[line] = {"n_late": len(late),
                     "w_diff_mean": (sum(late) / len(late)) if late else None,
                     "all_negative": bool(late) and all(x < 0 for x in late),
                     "all_positive": bool(late) and all(x > 0 for x in late),
                     "low_tail_defer_minus_route": [round(r["low_defer"] - r["low_route"], 3)
                                                    for r in rows if "low_defer" in r and "low_route" in r]}
    return rep


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    results = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*.json")))]
    by = rows_from(results)
    for line, rows in by.items():
        print(f"=== {line} ({len(rows)} checkpoints) ===")
        print(f"{'ckpt':18s} {'defer%':>6s} {'w_def':>6s} {'w_rou':>6s} {'w_diff':>7s} {'low_def':>7s} {'low_rou':>7s} "
              f"{'up_def':>6s} {'up_rou':>6s} {'adv+d':>6s} {'adv+r':>6s} {'|adv|d':>6s} {'|adv|r':>6s} {'ct_d':>5s} {'ct_r':>5s}")
        for r in rows:
            g = lambda k, f="{:6.3f}": (f.format(r[k]) if r.get(k) is not None else " " * 6)  # noqa: E731
            print(f"{r['ckpt']:18s} {100 * (r['defer_share'] or 0):6.1f} {g('w_defer')} {g('w_route')} "
                  f"{g('w_diff', '{:+7.3f}')} {g('low_defer', '{:7.3f}')} {g('low_route', '{:7.3f}')} "
                  f"{g('up_defer')} {g('up_route')} {g('advpos_defer')} {g('advpos_route')} "
                  f"{g('adv_defer')} {g('adv_route')} {g('ct_defer', '{:5.2f}')} {g('ct_route', '{:5.2f}')}")
    v = verdict(by)
    print(json.dumps(v, indent=1))
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({"rows": by, "verdict": v}, f, indent=2)


if __name__ == "__main__":
    main()
