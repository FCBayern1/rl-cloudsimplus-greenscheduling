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
# Keys a line may differ in from the HZ block. Anything else is a generator bug.
WHITELIST = {"experiment_name", "simulation_name", "cloudlet_trace_file", "green_oracle_mode",
             "perturb_tier", "forecast_mode", "crd", "training", "wandb",
             "green_episode_offset_allowlist"}
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


def build(total_timesteps, out_dir=None, trace_dir=None, reward_variant="legacy"):
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
    for line, spec in LINES.items():
        b = copy.deepcopy(base)
        b.update(copy.deepcopy(overrides))
        name = f"sd_{line}_{TRAIN_CELL}"
        b["experiment_name"] = name
        b["simulation_name"] = f"STAGED_{name}"
        b["cloudlet_trace_file"] = trace_rel
        b["green_oracle_mode"] = "perturbed_godeye"
        b["perturb_tier"] = "godeye"                 # training is always clean
        b["forecast_mode"] = "none" if spec["hollow"] else "full"
        b["crd"] = dict(copy.deepcopy(crd), enabled=bool(spec["crd"]))
        # One checkpoint per PPO iteration (train_batch_size 8000) so the health gate can
        # read the first and the last checkpoint; total_timesteps should be a multiple.
        # Codex smoke rulings: a true init checkpoint before the first SGD step, every
        # iteration's checkpoint kept (num_to_keep 0 = keep all), never pruned by score.
        b["training"] = dict(b.get("training", {}), total_timesteps=int(total_timesteps),
                             checkpoint_freq_timesteps=8000, checkpoint_num_to_keep=0,
                             save_init_checkpoint=True)
        b["wandb"] = dict(b.get("wandb", {}), enabled=False)
        b["green_episode_offset_allowlist"] = allow
        blocks[name] = b
    text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True, default_flow_style=False)
    cfg_name = f"config_stage_d{suffix}.yml"
    path = os.path.join(out_dir, cfg_name)
    with open(path, "w") as f:
        f.write(text)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    manifest = {"config": cfg_name, "reward_variant": reward_variant, "reward_overrides": overrides,
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
                  "forecast_mode", "training", "wandb", "perturb_error_params"} | REWARD_KEYS
AUDIT_JSON = os.path.join(HERE, "timecap_error_audit.json")


def build_eval(out_dir=None, reward_variant="physical"):
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
    blocks = {}
    for cell in EVAL_CELLS:
        for tier in EVAL_TIERS:
            for mode in ("full", "none"):
                if mode == "none" and tier != "godeye":
                    continue                      # a hollowed observation has no tier
                b = copy.deepcopy(hz[cell])
                b.update(copy.deepcopy(overrides))
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
                blocks[name] = b
    text = yaml.safe_dump({"common": common, **blocks}, sort_keys=True, default_flow_style=False)
    path = os.path.join(out_dir, "config_stage_d_eval.yml")
    with open(path, "w") as f:
        f.write(text)
    return blocks, {"config": "config_stage_d_eval.yml", "blocks": len(blocks),
                    "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "reward_variant": reward_variant}


if __name__ == "__main__":
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
    variant = sys.argv[2] if len(sys.argv) > 2 else "legacy"
    if variant == "eval":
        _, m = build_eval()
        print(json.dumps(m, indent=1))
        raise SystemExit(0)
    blocks, man = build(steps, reward_variant=variant)
    print(json.dumps({k: v for k, v in man.items() if k not in ("train_windows", "eval_windows")}, indent=1))
