"""Stage D training blocks: four lines derived from the HZ x2 scene by a whitelisted diff.

    N_V   vanilla PPO, future forecast hollowed (forecast_mode: none)
    V     vanilla PPO, clean truth-informed forecast
    N_E   EU-CRD,      future forecast hollowed
    E     EU-CRD,      clean forecast

Everything physical (zero-floor twins, 32-PE VMs, no splitting, brown 0.5, divisor 3000,
capacity, reward keys) is the HZ x2 block of cell c3_n50 verbatim. The training trace is
the generator's c3_n35 cell at 32 PEs (not one of the six evaluation cells). The EU-CRD
subtree is copied from the frozen v5.2 block of config_rl_step2_pilot.yml with only
`enabled` flipped, and its canonical SHA256 is recorded. Codex R-m / R-o, 2026-09-03.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess

import yaml

import gen_s2 as g

HERE = os.path.dirname(os.path.abspath(__file__))
HZ_CONFIG = os.path.join(HERE, "config_s2hz_m2.yml")
RL_PILOT_CONFIG = os.path.join(HERE, "config_rl_step2_pilot.yml")
WINDOWS = os.path.join(HERE, "stage_a_out", "stage_d_windows.json")
HZ_CELL = "s2_r48_w72_c3_n50"
TRAIN_CELL = "s2_r48_w72_c3_n35"
LINES = {"NV": {"crd": False, "hollow": True},
         "V": {"crd": False, "hollow": False},
         "NE": {"crd": True, "hollow": True},
         "E": {"crd": True, "hollow": False}}
# Credit-assignment baseline (Mesnard et al. 2021, hindsight baseline) as its own two
# lines, so it is compared under the same protocol without touching the frozen four.
# The learner routes to the CRD learner's `_apply_cca` hook on a vanilla backbone when
# crd.enabled is false, no risk objective is set, and crd.cca.enabled is true.
CCA_CFG = {"enabled": True, "horizon": 12, "hidden": 64, "lr": 0.001, "train_iters": 1}
CCA_LINES = {"NC": {"crd": False, "hollow": True, "cca": True},
             "C": {"crd": False, "hollow": False, "cca": True}}
# Risk-sensitive comparison set, frozen at the values the earlier campaign used; never
# retuned for this scene. Same vanilla backbone and observation as V, so they share V's
# matched no-forecast line (N_V) rather than each carrying its own.
RISK_CFG = {"RCV": {"kind": "cvar", "alpha": 0.2, "lam": 0.5},
            "RRS": {"kind": "risk_sensitive", "beta": 1.0},
            "RMV": {"kind": "mean_variance", "lam": 1.0},
            "RDC": {"kind": "dist_cvar", "alpha": 0.1, "lam": 0.7}}
RISK_LINES = {k: {"crd": False, "hollow": False, "risk": k} for k in RISK_CFG}
# Keys a line may differ in from the HZ block. Anything else is a generator bug.
# Stage D' overlay (STAGE_D_PRIME_DESIGN §2–3, Codex 2026-09-05): raw per-job timing state,
# no best-start hint, the contract-aligned on-time SLA through the existing Lagrangian
# channel, and the deadline-safe DEFER mask. The margin is a placeholder until the
# development smoke fixes it; the D' preregistration freezes the final value.
DPRIME_OVERLAY = {"obs_v31_features": True, "obs_v32_job_forecast": False,
                  "defer_deadline_mask": True, "defer_deadline_mask_margin_sec": 2.0,
                  "sla_mode": "ontime_mi", "sla_target": 0.995}
# D' EU-CRD guard (§10, frozen): symmetric shrink of the responsibility weight toward 1,
# eta = 0.5, key responsibility_shrink_strength. Merged into crd.responsibility of every
# line (inert where crd.enabled is false). The frozen v5.2 subtree is otherwise untouched;
# the manifest records both the source subtree SHA and this overlay.
DPRIME_CRD_OVERLAY = {"responsibility": {"responsibility_shrink_strength": 0.5}}
# Option action mode (reports/OPTION_ACTION_DESIGN.md §2, Addenda A/B): the D' block plus
# ROUTE_NOW(d) | HOLD_FOR_GREEN(d) through the shared env executor; eps = the frozen
# latest-start margin in steps. Used by the four zero-training gates; no RL runs on it
# until a separate preregistration is ruled.
OPTION_OVERLAY = {**DPRIME_OVERLAY, "global_action_mode": "option_v1", "option_eps_steps": 2}
# (DC, dispatch-offset) fallback (OPTION_ACTION_DESIGN §8, Addenda A5, C): the grid K(W)
# with W = the scene's wait cap in steps (72 on HZ), fixed-start reservations, no green read.
OFFSET_OVERLAY = {**DPRIME_OVERLAY, "global_action_mode": "offset_v1", "option_eps_steps": 2,
                  "offset_wait_cap_steps": 72}
WHITELIST = {"experiment_name", "simulation_name", "cloudlet_trace_file", "green_oracle_mode",
             "perturb_tier", "forecast_mode", "crd", "training", "wandb",
             "green_episode_offset_allowlist", *OPTION_OVERLAY, *OFFSET_OVERLAY, "wind_csv_year"}
BETWEEN_LINES = {"experiment_name", "simulation_name", "forecast_mode", "crd"}


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def crd_subtree():
    cfg = yaml.safe_load(open(RL_PILOT_CONFIG))
    blk = cfg[[k for k in cfg if k != "common"][0]]
    sub = copy.deepcopy(blk["crd"])
    src_sha = canonical_sha(sub)
    return sub, src_sha


def train_trace(trace_dir=None):
    """32-PE version of the c3_n35 trace, same arrivals/runtime/deadline as the RL pilot."""
    trace_dir = trace_dir or g.TRACE_DIR
    cell = next(c for c in g.cells() if g.cell_name(c) == TRAIN_CELL)
    base = g.base_block()
    rows, _ = g.trace(cell, float(base["datacenters"][0]["vm_pe_mips"]))
    rows32 = [(i, a, mi, g.H_PES, fs, os_, dl) for (i, a, mi, _p, fs, os_, dl) in rows]
    text = g.trace_text(rows32)
    path = os.path.join(trace_dir, f"{TRAIN_CELL}_pes{g.H_PES}.csv")
    with open(path, "w") as f:
        f.write(text)
    return f"traces/s2/{TRAIN_CELL}_pes{g.H_PES}.csv", g.content_sha(text)


# Reward variants (STAGE_D_PREREG Addendum C). "legacy" = the C-regime reward the HZ block
# inherited: flat defer cost charged at every sighting plus an instant per-action carbon
# price, which P0 showed dominates and inverts the carbon ordering. "physical" keeps only
# the physical carbon term (beta * normalised ledger carbon, FIXED 5e-05) and the
# completion shaping: reward ordering equals carbon ordering by construction.
REWARD_VARIANTS = {
    "legacy": {},
    # Registered name (Codex R-q, 2026-09-03): "ledger-aligned reward". It removes the
    # repeatedly-charged waiting proxy and the dispatch-instant carbon proxy and keeps the
    # real ledger carbon plus completion shaping; it is not a "pure physical" reward.
    # The file suffix stays "physical" for continuity of the generated artefacts.
    "physical": {"defer_base_cost": 0.0, "defer_urgency_weight": 0.0,
                 "per_action_carbon_weight": 0.0},
}
REWARD_VARIANTS["ledger_aligned"] = REWARD_VARIANTS["physical"]
REWARD_KEYS = {"defer_base_cost", "defer_urgency_weight", "per_action_carbon_weight"}


def build(total_timesteps, out_dir=None, trace_dir=None, reward_variant="legacy",
          checkpoint_freq=8000, out_name=None, lines=None, overlay=None):
    out_dir = out_dir or HERE
    hz = yaml.safe_load(open(HZ_CONFIG))
    base = hz[HZ_CELL]
    common = hz.get("common", {})
    crd, crd_sha = crd_subtree()
    trace_rel, trace_sha = train_trace(trace_dir)
    win = json.load(open(WINDOWS))
    if win.get("status") != "OK":
        raise RuntimeError("window preflight is not OK; no training block may be built")
    allow = ";".join(str(w["offset"]) for w in win["train_windows"])
    overrides = REWARD_VARIANTS[reward_variant]
    if reward_variant == "ledger_aligned":
        reward_variant = "physical"            # same overrides, same artefact names
    suffix = "" if reward_variant == "legacy" else f"_{reward_variant}"
    blocks = {}
    for line, spec in (lines or LINES).items():
        b = copy.deepcopy(base)
        b.update(copy.deepcopy(overrides))
        if overlay:
            b.update(copy.deepcopy(overlay))
        name = f"sd_{line}_{TRAIN_CELL}"
        b["experiment_name"] = name
        b["simulation_name"] = f"STAGED_{name}"
        b["cloudlet_trace_file"] = trace_rel
        b["green_oracle_mode"] = "perturbed_godeye"
        b["perturb_tier"] = "godeye"                 # training is always clean
        b["forecast_mode"] = "none" if spec["hollow"] else "full"
        b["crd"] = dict(copy.deepcopy(crd), enabled=bool(spec["crd"]))
        if overlay is DPRIME_OVERLAY or (overlay and overlay.get("defer_deadline_mask")):
            for sub, vals in DPRIME_CRD_OVERLAY.items():
                b["crd"][sub] = dict(b["crd"].get(sub, {}), **vals)
        if spec.get("cca"):
            b["crd"]["cca"] = copy.deepcopy(CCA_CFG)
        if spec.get("risk"):
            b["crd"]["risk"] = copy.deepcopy(RISK_CFG[spec["risk"]])
        # One checkpoint per PPO iteration (train_batch_size 8000) so the health gate can
        # read the first and the last checkpoint; total_timesteps should be a multiple.
        # Codex smoke rulings: a true init checkpoint before the first SGD step, every
        # iteration's checkpoint kept (num_to_keep 0 = keep all), never pruned by score.
        b["training"] = dict(b.get("training", {}), total_timesteps=int(total_timesteps),
                             checkpoint_freq_timesteps=int(checkpoint_freq), checkpoint_num_to_keep=0,
                             save_init_checkpoint=True)
        b["wandb"] = dict(b.get("wandb", {}), enabled=False)
        b["green_episode_offset_allowlist"] = allow
        blocks[name] = b
    text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True, default_flow_style=False)
    cfg_name = out_name or f"config_stage_d{suffix}.yml"
    if out_name:
        suffix = "_" + out_name.replace("config_stage_d_", "").replace(".yml", "")
    path = os.path.join(out_dir, cfg_name)
    with open(path, "w") as f:
        f.write(text)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    manifest = {"config": cfg_name, "reward_variant": reward_variant, "reward_overrides": overrides,
                "overlay": overlay or {},
                "crd_overlay": (DPRIME_CRD_OVERLAY if (overlay and overlay.get("defer_deadline_mask")) else {}),
                "checkpoint_freq_timesteps": int(checkpoint_freq),
                "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "hz_source": {"file": "config_s2hz_m2.yml", "cell": HZ_CELL},
                "crd_subtree_sha256": crd_sha, "crd_source": {"file": "config_rl_step2_pilot.yml",
                                                              "commit_at_build": commit},
                "train_trace": {"file": trace_rel, "sha": trace_sha},
                "train_windows": win["train_windows"], "eval_windows": win["eval_windows"],
                "windows_sha256": win.get("sha256"), "total_timesteps": int(total_timesteps),
                "lines": {n: {"crd_enabled": b["crd"]["enabled"], "forecast_mode": b["forecast_mode"]}
                          for n, b in blocks.items()}}
    with open(os.path.join(out_dir, f"stage_d_manifest{suffix}.json"), "w") as f:
        f.write(json.dumps(manifest, sort_keys=True, indent=2))
    return blocks, manifest


def diff_keys(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


EVAL_CELLS = [f"s2_r48_w72_c{c}_n{n}" for c in (1, 3, 5) for n in (20, 50)]
EVAL_TIERS = ("godeye", "calibrated_shrink_v1", "shuffle", "anti")
EVAL_WHITELIST = {"experiment_name", "simulation_name", "green_oracle_mode", "perturb_tier",
                  "forecast_mode", "training", "wandb", "perturb_error_params",
                  "green_episode_offset_allowlist", *DPRIME_OVERLAY, "wind_csv_year"} | REWARD_KEYS
AUDIT_JSON = os.path.join(HERE, "timecap_error_audit.json")


def judgement_offsets():
    win = json.load(open(WINDOWS))
    if win.get("status") != "OK":
        raise RuntimeError("window preflight is not OK")
    return [int(w["offset"]) for w in win["eval_windows"]]


def build_eval(out_dir=None, reward_variant="physical", windows="certified", overlay=None, out_name=None,
               allowlist_override=None):
    """Deployment blocks: six HZ cells x four provider tiers x {full, hollow} forecast.

    Each block is the HZ x2 cell block with the RL keys and the registered reward variant;
    the window is chosen at evaluation time by --reset-skip on the simulator schedule
    (certified windows k=26/34/42 for the health smoke), never by an allowlist. Hollow
    blocks serve the N_V / N_E lines, whose observation has no future forecast.
    """
    out_dir = out_dir or HERE
    hz = yaml.safe_load(open(HZ_CONFIG))
    common = hz.get("common", {})
    overrides = REWARD_VARIANTS[reward_variant]
    # windows="judgement": the six unread offsets of the preflight become the block's
    # allowlist and --reset-skip 0..5 selects them (long-run main verdict, Codex R-v).
    # windows="certified": the simulator schedule, --reset-skip 26/34/42 (health smoke,
    # secondary "certified benchmark evaluation").
    if allowlist_override:
        allow = allowlist_override
    elif windows == "judgement":
        allow = ";".join(str(o) for o in judgement_offsets())
    elif windows == "dev":
        # development windows = the Stage D training offsets, selected by --reset-skip 0..5
        # (six-cell saturated probe, design §16 Q3)
        allow = ";".join(str(w["offset"]) for w in json.load(open(WINDOWS))["train_windows"])
    else:
        allow = None
    blocks = {}
    for cell in EVAL_CELLS:
        for tier in EVAL_TIERS:
            for mode in ("full", "none"):
                if mode == "none" and tier != "godeye":
                    continue                      # a hollowed observation has no tier
                b = copy.deepcopy(hz[cell])
                b.update(copy.deepcopy(overrides))
                if overlay:
                    # D': the deployment observation must carry the same timing keys and
                    # mask the policy was trained with; the SLA keys are training-side and
                    # inert here but kept so one overlay describes both sides.
                    b.update(copy.deepcopy(overlay))
                name = f"sde_{cell}_{tier if mode == 'full' else 'hollow'}"
                b["experiment_name"] = name
                b["simulation_name"] = f"STAGED_EVAL_{name}"
                b["green_oracle_mode"] = "perturbed_godeye"
                b["perturb_tier"] = tier
                b["forecast_mode"] = mode
                if tier == "calibrated_shrink_v1":
                    # The RL-side provider reads the audited error parameters from this
                    # file (primary_error_params block), the same audit the planner arms
                    # received through PLANNER_PERTURB_CAL.
                    b["perturb_error_params"] = AUDIT_JSON
                b["training"] = dict(b.get("training", {}))
                b["wandb"] = dict(b.get("wandb", {}), enabled=False)
                if allow:
                    b["green_episode_offset_allowlist"] = allow
                blocks[name] = b
    text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True, default_flow_style=False)
    cfg_name = out_name or ("config_stage_d_eval.yml" if windows == "certified" else "config_stage_d_eval_judgement.yml")
    path = os.path.join(out_dir, cfg_name)
    with open(path, "w") as f:
        f.write(text)
    return blocks, {"config": cfg_name, "blocks": len(blocks), "windows": windows,
                    "allowlist": allow, "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "reward_variant": reward_variant, "overlay": overlay or {}}


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    variant = sys.argv[2] if len(sys.argv) > 2 else "legacy"
    if variant == "eval":
        _, m = build_eval()
        print(json.dumps(m, indent=1))
        raise SystemExit(0)
    if variant == "eval_judgement":
        _, m = build_eval(windows="judgement")
        print(json.dumps(m, indent=1))
        raise SystemExit(0)
    if variant == "eval_dprime_2020":
        # Formal D' judgement blocks (design §20): the 2020 series of the same turbines,
        # allowlist = the six hash-selected 2020 windows, wind_csv_year 2020 everywhere,
        # audit year 2020. Fail-fast on every one of those.
        w20 = json.load(open(os.path.join(HERE, "stage_a_out", "stage_d_prime_windows_2020.json")))
        if w20.get("status") != "OK" or len(w20.get("windows", [])) != 6:
            raise SystemExit(f"2020 windows not OK: {w20.get('status')}")
        aud = json.load(open(AUDIT_JSON))
        if int(aud.get("year", 0)) != 2020:
            raise SystemExit(f"audit year is {aud.get('year')}, must be 2020")
        overlay = dict(DPRIME_OVERLAY, wind_csv_year=2020)
        blocks, man = build_eval(windows="judgement", overlay=overlay,
                                 out_name="config_stage_d_eval_dprime_2020.yml",
                                 allowlist_override=";".join(str(o) for o in w20["windows"]))
        bad = [n for n, b in blocks.items() if int(b.get("wind_csv_year", 0)) != 2020]
        if bad:
            raise SystemExit(f"blocks without wind_csv_year 2020: {bad[:3]}")
        man.update({"year": 2020, "windows_2020": w20["windows"], "read_file_sha256": w20.get("read_file_sha256"),
                    "audit_year": aud.get("year"), "audit_turbines": aud.get("dc_turbines")})
        with open(os.path.join(HERE, "stage_d_manifest_eval_dprime_2020.json"), "w") as f:
            json.dump(man, f, indent=2)
        print(json.dumps({k: man[k] for k in ("config", "blocks", "allowlist", "config_sha256", "year", "windows_2020")}, indent=1))
        raise SystemExit(0)
    if variant == "eval_dprime_dev":
        blocks, man = build_eval(windows="dev", overlay=DPRIME_OVERLAY,
                                 out_name="config_stage_d_eval_dprime_dev.yml")
        print(json.dumps(man, indent=1))
        raise SystemExit(0)
    if variant == "eval_dprime":
        # development-smoke deployment blocks: certified windows (already read), D' overlay
        blocks, man = build_eval(windows="certified", overlay=DPRIME_OVERLAY,
                                 out_name="config_stage_d_eval_dprime.yml")
        print(json.dumps(man, indent=1))
        raise SystemExit(0)
    if variant == "dprime_offset":
        blocks, man = build(steps, reward_variant="physical", checkpoint_freq=40000,
                            out_name="config_stage_d_dprime_offset.yml", overlay=OFFSET_OVERLAY)
        print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
        raise SystemExit(0)
    if variant == "dprime_option":
        blocks, man = build(steps, reward_variant="physical", checkpoint_freq=40000,
                            out_name="config_stage_d_dprime_option.yml", overlay=OPTION_OVERLAY)
        print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
        raise SystemExit(0)
    if variant == "dprime":
        blocks, man = build(steps, reward_variant="physical", checkpoint_freq=40000,
                            out_name="config_stage_d_dprime.yml", overlay=DPRIME_OVERLAY)
        print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
        raise SystemExit(0)
    if variant == "risk":
        blocks, man = build(steps, reward_variant="physical", checkpoint_freq=40000,
                            out_name="config_stage_d_risk.yml", lines=RISK_LINES)
        print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
        raise SystemExit(0)
    if variant == "cca":
        blocks, man = build(steps, reward_variant="physical", checkpoint_freq=40000,
                            out_name="config_stage_d_cca.yml", lines=CCA_LINES)
        print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
        raise SystemExit(0)
    if variant == "longrun":
        blocks, man = build(steps, reward_variant="physical", checkpoint_freq=40000,
                            out_name="config_stage_d_longrun.yml")
        print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
        raise SystemExit(0)
    blocks, man = build(steps, reward_variant=variant)
    print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
