"""Stage D P0 reward truth table (STAGE_D_PREREG Addendum B, hard blocker 2).

Replayed arms on the V training block over the frozen training windows:
blind (reactive_wait_planner), clean (godeye), shrink (calibrated_shrink_v1), always_defer.

Checks, per window and pooled (sums over windows):
  order    reward ordering equals carbon ordering for (clean, blind), (shrink, clean),
           (always_defer, blind): lower carbon <=> higher global reward
  clean    clean beats blind on both carbon and reward
  shrink   shrink loses to clean on both carbon and reward
  clip     carbon-normalisation clip rate <= 5% of samples, zero cap hits
  defer    always_defer's reward strictly below the blind's (no defer arbitrage)
  contract completion/ontime/forced/ledger contract on every run

judge() is a pure function of a row table so it can be tested without disk.
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_stage_a as ra      # noqa: E402
import stage_a_verdict as sv  # noqa: E402

OUT = ra.OUT
ARMS = ("blind", "clean", "shrink", "always_defer")
DIRS = {"blind": "p0_reactive_wait_planner", "clean": "p0_godeye",
        "shrink": "p0_calibrated_shrink_v1", "always_defer": "p0_always_defer"}
CLIP_MAX = 0.05
PAIRS = (("clean", "blind"), ("shrink", "clean"), ("always_defer", "blind"))


def _consistent(a, b):
    """carbon(a) < carbon(b) must come with reward(a) > reward(b), and vice versa."""
    dc = a["carbon"] - b["carbon"]
    dr = a["reward"] - b["reward"]
    return (dc < 0 and dr > 0) or (dc > 0 and dr < 0) or (dc == 0 and dr == 0)


def judge(rows, windows):
    """rows: {(arm, k): {"carbon","reward","clip","samples","cap","contract_ok"}} or None."""
    missing = [(a, k) for a in ARMS for k in windows if rows.get((a, k)) is None]
    if missing:
        return {"verdict": "INVALID_INCOMPLETE_DATA", "missing": missing}
    contract_bad = [(a, k) for a in ARMS for k in windows if not rows[(a, k)]["contract_ok"]]
    per_window = {}
    for k in windows:
        r = {a: rows[(a, k)] for a in ARMS}
        per_window[k] = {f"{x}_vs_{y}": _consistent(r[x], r[y]) for x, y in PAIRS}
        per_window[k]["clean_better_both"] = r["clean"]["carbon"] < r["blind"]["carbon"] and r["clean"]["reward"] > r["blind"]["reward"]
        per_window[k]["shrink_worse_both"] = r["shrink"]["carbon"] > r["clean"]["carbon"] and r["shrink"]["reward"] < r["clean"]["reward"]
        per_window[k]["defer_reward_below_blind"] = r["always_defer"]["reward"] < r["blind"]["reward"]
    pooled = {a: {"carbon": sum(rows[(a, k)]["carbon"] for k in windows),
                  "reward": sum(rows[(a, k)]["reward"] for k in windows)} for a in ARMS}
    clip_rate = {a: (sum(rows[(a, k)]["clip"] for k in windows)
                     / max(1.0, sum(rows[(a, k)]["samples"] for k in windows))) for a in ARMS}
    cap_hits = {a: sum(rows[(a, k)]["cap"] for k in windows) for a in ARMS}
    wins = {key: sum(1 for k in windows if per_window[k][key]) for key in per_window[windows[0]]}
    n = len(windows)
    gates = {
        "order_pooled": all(_consistent(pooled[x], pooled[y]) for x, y in PAIRS),
        "order_per_window_majority": all(wins[f"{x}_vs_{y}"] * 2 > n for x, y in PAIRS),
        "clean_better_both_pooled": pooled["clean"]["carbon"] < pooled["blind"]["carbon"]
                                    and pooled["clean"]["reward"] > pooled["blind"]["reward"],
        "shrink_worse_both_pooled": pooled["shrink"]["carbon"] > pooled["clean"]["carbon"]
                                    and pooled["shrink"]["reward"] < pooled["clean"]["reward"],
        "clip_rate_le_5pc": all(v <= CLIP_MAX for v in clip_rate.values()),
        "no_cap_hits": all(v == 0 for v in cap_hits.values()),
        "defer_no_arbitrage": pooled["always_defer"]["reward"] < pooled["blind"]["reward"],
        "contract_green": not contract_bad,
    }
    return {"verdict": "PASS_P0" if all(gates.values()) else "STOP_P0",
            "gates": gates, "pooled": pooled, "clip_rate": clip_rate, "cap_hits": cap_hits,
            "per_window_wins": wins, "windows": list(windows), "contract_bad": contract_bad}


def load_rows(variant=""):
    suffix = f"_{variant}" if variant else ""
    man = json.load(open(os.path.join(HERE, f"stage_d_manifest{suffix}.json")))
    windows = list(range(len(man["train_windows"])))
    cell = "sd_V_s2_r48_w72_c3_n35"
    rows = {}
    for arm, d in DIRS.items():
        for k in windows:
            p = os.path.join(OUT, d.replace("p0_", f"p0{suffix}_", 1), f"{cell}_k{k}.csv")
            if not os.path.exists(p):
                rows[(arm, k)] = None
                continue
            r = list(csv.DictReader(open(p)))[-1]
            f = lambda key, default=0.0: float(r.get(key, default) or default)  # noqa: E731
            rows[(arm, k)] = {"carbon": f("total_carbon_kg"), "reward": f("global_reward_sum"),
                              "clip": f("ep_carbon_norm_clip_count"),
                              "samples": f("ep_carbon_norm_sample_count"),
                              "cap": f("ep_global_carbon_cap_count"),
                              "contract_ok": bool(sv._contract_ok(r))}
    return rows, windows


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    rows, windows = load_rows(variant)
    out = judge(rows, windows)
    out["reward_variant"] = variant or "legacy"
    with open(os.path.join(OUT, f"p0_verdict{'_' + variant if variant else ''}.json"), "w") as fh:
        fh.write(json.dumps(out, sort_keys=True, indent=2, default=str))
    print(json.dumps(out, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
