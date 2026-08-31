"""Mechanical validity check for the planner gate. No judgement, only the frozen contract.

reports/PLANNER_GATE_PREREG.md fixes what every cell must satisfy before a carbon number
from it may be compared. This reads the result rows and reports pass or fail per cell,
then the pooled terminal carbon per arm across the windows in the set.
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

# Per-id closure against the simulator's own execution events (Codex 2026-08-30,
# Addendum B). The retired criterion compared planner_occ with cap - dc_available_pes;
# that field is a VM allocation counter which never recovers once a cloudlet finishes, so
# it can neither budget a drain nor audit a ledger and is no longer a contract term.
CONTRACT = [
    ("deadline_forced_count", lambda v: float(v) == 0.0, "forced == 0"),
    ("completion_rate_mi", lambda v: float(v) >= 0.995, "terminal completion_mi >= 99.5%"),
    ("ontime_mi_share", lambda v: float(v) >= 0.995, "terminal ontime_mi >= 99.5%"),
    ("planner_n_stale_dropped", lambda v: float(v) == 0.0, "stale == 0"),
    ("planner_n_unplanned_start", lambda v: float(v) == 0.0, "unplanned start == 0"),
    ("planner_n_wrong_dc", lambda v: float(v) == 0.0, "started on the committed site"),
    ("planner_n_dispatched_never_started", lambda v: float(v) == 0.0,
     "every dispatch started"),
    ("planner_n_running_unknown", lambda v: float(v) == 0.0, "no unaccounted execution"),
    ("planner_running_pes_over_cap", lambda v: float(v) <= 1e-9, "no capacity overrun"),
    ("planner_occ_max_over_cap", lambda v: float(v) <= 1e-9, "plan within capacity"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--margin", required=True)
    ap.add_argument("--workload", type=int, default=8000)
    args = ap.parse_args()

    cells = sorted(glob.glob(os.path.join(args.outdir, f"*_m{args.margin}.csv")))
    if not cells:
        print(f"no cells for margin {args.margin} in {args.outdir}")
        return 2

    per_arm = defaultdict(dict)
    all_ok = True
    print(f"{'cell':<44} {'carbon/MI':>11} {'comp_mi':>8} {'ontime':>7} {'forced':>7} "
          f"{'stale':>6} {'unpl':>6} {'unkn':>7}  verdict")
    for path in cells:
        row = list(csv.DictReader(open(path)))[0]
        base = os.path.basename(path)[:-4]
        arm, window = base.rsplit("_m", 1)[0].rsplit("_", 1)
        fails = []
        for key, ok, label in CONTRACT:
            if key not in row:
                fails.append(f"{label} (column missing)")
            elif not ok(row[key]):
                fails.append(f"{label} (got {row[key]})")
        if int(float(row.get("total_cloudlets", 0))) != args.workload:
            fails.append(f"workload {row.get('total_cloudlets')} != {args.workload}")
        verdict = "PASS" if not fails else "FAIL"
        all_ok &= not fails
        print(f"{base:<44} {float(row['carbon_per_completion_mi']):>11.6f} "
              f"{float(row['completion_rate_mi']):>8.4f} {float(row['ontime_mi_share']):>7.4f} "
              f"{row.get('deadline_forced_count','?'):>7} {row.get('planner_n_stale_dropped','?'):>6} "
              f"{row.get('planner_n_unplanned_start','?'):>6} "
              f"{row.get('planner_n_running_unknown','?'):>7}  {verdict}")
        for f in fails:
            print(f"{'':<44}   - {f}")
        per_arm[arm][window] = float(row["carbon_per_completion_mi"])

    print(f"\npooled terminal carbon/MI across windows (margin {args.margin})")
    for arm in sorted(per_arm, key=lambda a: sum(per_arm[a].values()) / len(per_arm[a])):
        w = per_arm[arm]
        pooled = sum(w.values()) / len(w)
        print(f"  {arm:<26} n={len(w)}  pooled={pooled:.6f}  " +
              "  ".join(f"{k}={v:.6f}" for k, v in sorted(w.items())))

    print(f"\ncontract: {'ALL CELLS PASS' if all_ok else 'AT LEAST ONE CELL FAILS'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
