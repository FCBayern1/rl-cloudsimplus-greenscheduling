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
POLICY_ARMS = ("blind", "clean", "shrink")
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
    # The contract gate covers the policy arms. always_defer is the arbitrage probe: it is
    # expected to miss deadlines, and its contract outcome is reported, not gated
    # (STAGE_D_PREREG Addendum D, amended after the first physical-variant reading).
    contract_bad = [(a, k) for a in POLICY_ARMS for k in windows if not rows[(a, k)]["contract_ok"]]
    probe_contract_bad = [k for k in windows if not rows[("always_defer", k)]["contract_ok"]]
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
            "per_window_wins": wins, "windows": list(windows), "contract_bad": contract_bad,
            "probe_always_defer_contract_bad_windows": probe_contract_bad}


DPRIME_ARMS = ("blind", "clean", "shrink", "always_defer", "nodefer")


def contract_ok_dprime(r):
    """P0' contract. Under the deadline-safe DEFER mask the env may route a planner arm's
    DEFER itself at the last safe step; the planner's ledger then counts that job as an
    unplanned start. Those starts are the mask doing its job, so
    planner_n_unplanned_start may equal, but not exceed, ep_mask_route_count. Every other
    zero-field and the completion / on-time / forced terms are unchanged."""
    f = lambda k: float(r.get(k, 0) or 0)  # noqa: E731
    if f("completion_rate_mi") < ra.CONTRACT["completion_rate_mi"] or f("ontime_mi_share") < ra.CONTRACT["ontime_mi_share"]:
        return False
    for z in ra.ZERO_FIELDS:
        if z == "planner_n_unplanned_start":
            if f(z) > f("ep_mask_route_count"):
                return False
            if f(z) > 0 and not unplanned_ids_are_mask_routed(r):
                return False
        elif f(z) != 0.0:
            return False
    return True


def _ids(s):
    return {int(x) for x in str(s or "").split(";") if x.strip() not in ("", "None")}


def unplanned_ids_are_mask_routed(r):
    """Per-id closure (design §16): every unplanned start must be a job the mask routed.
    Requires both id lists to be present and the mask's ids fully known."""
    if float(r.get("ep_mask_routed_ids_unknown", 0) or 0) > 0:
        return False
    if "planner_unplanned_start_ids" not in r or "ep_mask_routed_ids" not in r:
        return False
    return _ids(r["planner_unplanned_start_ids"]) <= _ids(r["ep_mask_routed_ids"])
DPRIME_DIRS = dict(DIRS, nodefer="p0_godeye_nodefer")
ONTIME_MIN, FORCED_MAX = 0.995, 0


def judge_dprime(rows, windows):
    """P0' (STAGE_D_PRIME_DESIGN §4 step 3, Codex Q2). On top of the P0 gates, the actual
    PPO objective (discounted return) must order the three behaviours that differ only in
    timing:  clean (best window, on time)  >  nodefer (start now)  >  always_defer (wait
    until the mask routes it);  clean must also beat nodefer on carbon; and always_defer
    must be routed legally by the mask: on-time >= 0.995 and Java forced == 0.
    rows: as load_rows(dprime=True) -> adds "reward_disc", "ontime", "forced"."""
    missing = [(a, k) for a in DPRIME_ARMS for k in windows if rows.get((a, k)) is None]
    if missing:
        return {"verdict": "INVALID_INCOMPLETE_DATA", "missing": missing}
    base = judge({key: v for key, v in rows.items() if key[0] in ARMS}, windows)
    if base["verdict"].startswith("INVALID"):
        return base
    order_ok = {}
    for k in windows:
        c, s, d = rows[("clean", k)], rows[("nodefer", k)], rows[("always_defer", k)]
        order_ok[k] = {"clean_gt_nodefer_disc": c["reward_disc"] > s["reward_disc"],
                       "nodefer_gt_defer_disc": s["reward_disc"] > d["reward_disc"],
                       "clean_lt_nodefer_carbon": c["carbon"] < s["carbon"]}
    pooled = {a: {"reward_disc": sum(rows[(a, k)]["reward_disc"] for k in windows),
                  "carbon": sum(rows[(a, k)]["carbon"] for k in windows)} for a in DPRIME_ARMS}
    n = len(windows)
    wins = {key: sum(1 for k in windows if order_ok[k][key]) for key in order_ok[windows[0]]}
    defer_legal = [k for k in windows
                   if rows[("always_defer", k)]["ontime"] >= ONTIME_MIN
                   and rows[("always_defer", k)]["forced"] <= FORCED_MAX]
    gates = dict(base["gates"])
    gates.update({
        "disc_order_pooled": pooled["clean"]["reward_disc"] > pooled["nodefer"]["reward_disc"]
                             > pooled["always_defer"]["reward_disc"],
        "disc_order_per_window_majority": all(wins[key] * 2 > n for key in
                                              ("clean_gt_nodefer_disc", "nodefer_gt_defer_disc")),
        "clean_beats_nodefer_carbon_pooled": pooled["clean"]["carbon"] < pooled["nodefer"]["carbon"],
        "always_defer_routed_legally_by_mask": len(defer_legal) == n,
    })
    out = dict(base)
    out.update({"verdict": "PASS_P0_PRIME" if all(gates.values()) else "STOP_P0_PRIME",
                "gates": gates, "pooled_dprime": pooled, "per_window_wins_dprime": wins,
                "always_defer_legal_windows": defer_legal})
    return out


def load_rows(variant="", dprime=False):
    suffix = f"_{variant}" if variant else ""
    man = json.load(open(os.path.join(HERE, f"stage_d_manifest{suffix}.json")))
    windows = list(range(len(man["train_windows"])))
    cell = "sd_V_s2_r48_w72_c3_n35"
    rows = {}
    for arm, d in (DPRIME_DIRS if dprime else DIRS).items():
        for k in windows:
            p = os.path.join(OUT, d.replace("p0_", f"p0{suffix}_", 1), f"{cell}_k{k}.csv")
            if not os.path.exists(p):
                rows[(arm, k)] = None
                continue
            r = list(csv.DictReader(open(p)))[-1]
            f = lambda key, default=0.0: float(r.get(key, default) or default)  # noqa: E731
            rows[(arm, k)] = {"carbon": f("total_carbon_kg"), "reward": f("global_reward_sum"),
                              "reward_disc": f("global_reward_discounted_sum"),
                              "ontime": f("ontime_mi_share", 1.0), "forced": f("deadline_forced_count"),
                              "clip": f("ep_carbon_norm_clip_count"),
                              "samples": f("ep_carbon_norm_sample_count"),
                              "cap": f("ep_global_carbon_cap_count"),
                              "mask_routed": f("ep_mask_route_count"),
                              "unplanned": f("planner_n_unplanned_start"),
                              "contract_ok": bool(contract_ok_dprime(r) if dprime else sv._contract_ok(r))}
    return rows, windows


SCENE_DIRS = {"blind": "p0_scene_v1_reactive_wait_planner", "clean": "p0_scene_v1_godeye",
              "shrink": "p0_scene_v1_calibrated_shrink_hz_v2", "always_defer": "p0_scene_v1_always_defer",
              "nodefer": "p0_scene_v1_godeye_nodefer"}


def load_rows_scene():
    """Scene v1 P0' rows (SCENE_INTERFACE_DESIGN §2, step 2c): the development windows of
    scene_v1_cert.json, the defer-mode block, dirs p0_scene_v1_<arm>."""
    man = json.load(open(os.path.join(HERE, "stage_a_out", "scene_v1_manifest.json")))
    cert = json.load(open(os.path.join(HERE, "stage_a_out", "scene_v1_cert.json")))
    cell = man["configs"]["defer"]["block"]
    windows = list(range(len(cert["development"]["dev_offsets"])))
    rows = {}
    for arm, d in SCENE_DIRS.items():
        for k in windows:
            p = os.path.join(OUT, d, f"{cell}_k{k}.csv")
            if not os.path.exists(p):
                rows[(arm, k)] = None
                continue
            r = list(csv.DictReader(open(p)))[-1]
            f = lambda key, default=0.0: float(r.get(key, default) or default)  # noqa: E731
            rows[(arm, k)] = {"carbon": f("total_carbon_kg"), "reward": f("global_reward_sum"),
                              "reward_disc": f("global_reward_discounted_sum"),
                              "ontime": f("ontime_mi_share", 1.0), "forced": f("deadline_forced_count"),
                              "clip": f("ep_carbon_norm_clip_count"), "samples": f("ep_carbon_norm_sample_count"),
                              "cap": f("ep_global_carbon_cap_count"), "mask_routed": f("ep_mask_route_count"),
                              "unplanned": f("planner_n_unplanned_start"), "contract_ok": bool(contract_ok_dprime(r))}
    return rows, windows


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else ""
    if variant == "scene":
        rows, windows = load_rows_scene()
        out = judge_dprime(rows, windows)
        out["reward_variant"] = "scene_v1"
        with open(os.path.join(OUT, "p0_verdict_scene_v1.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print(json.dumps(out, indent=2))
        return
    if variant == "dprime":
        rows, windows = load_rows(variant, dprime=True)
        out = judge_dprime(rows, windows)
        out["reward_variant"] = variant
        with open(os.path.join(OUT, "p0_verdict_dprime.json"), "w") as fh:
            fh.write(json.dumps(out, sort_keys=True, indent=2, default=str))
        print(json.dumps(out, sort_keys=True, indent=2, default=str))
        return
    rows, windows = load_rows(variant)
    out = judge(rows, windows)
    out["reward_variant"] = variant or "legacy"
    with open(os.path.join(OUT, f"p0_verdict{'_' + variant if variant else ''}.json"), "w") as fh:
        fh.write(json.dumps(out, sort_keys=True, indent=2, default=str))
    print(json.dumps(out, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
