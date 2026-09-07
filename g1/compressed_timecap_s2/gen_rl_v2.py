"""RL_V2 smoke blocks (reports/RL_V2_SMOKE_PREREG.md §3-§4): four lines derived from the
certification interface twin by a whitelisted diff, eval blocks per tier, manifest of hashes.

    rl2_NV  vanilla PPO, forecast hollowed (persistence cover)
    rl2_V   vanilla PPO, clean truth forecast
    rl2_NE  EU-CRD,      forecast hollowed
    rl2_E   EU-CRD,      clean truth forecast

Usage: python gen_rl_v2.py [--steps 56000]
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_run as lr  # noqa: E402

OUT_CFG = os.path.join(HERE, "config_rl_v2.yml")
OUT_EVAL = os.path.join(HERE, "config_rl_v2_eval.yml")
MANIFEST = os.path.join(HERE, "stage_a_out", "rl_v2", "manifest.json")
LINES = {"NV": {"hollow": True, "crd": False}, "V": {"hollow": False, "crd": False},
         "NE": {"hollow": True, "crd": True}, "E": {"hollow": False, "crd": True}}
TIERS = ("godeye", "shrink75", "shrink50", "shrink25", "shrink0", "shuffle", "anti")
SEED = 20260907
# RL_V2 Addendum A2: the smallest gain of the frozen grid whose mean sampled-cover ratio on the
# twelve TRAINING windows reaches 0.95 (measured 0.478 / 0.769 / 0.887 / 0.952 for 1 / 5 / 10 / 20)
COVER_GAIN = 20.0
# keys a line block may differ in from the certification interface twin
WHITELIST = {"experiment_name", "simulation_name", "forecast_mode", "crd", "training", "wandb",
             "green_episode_offset_allowlist", "gtrxl", "perturb_tier"}
BETWEEN_LINES = {"experiment_name", "simulation_name", "forecast_mode", "crd"}


def windows():
    w = json.load(open(os.path.join(HERE, "stage_a_out", "f_v2", "windows.json")))
    dev = lr._dev()
    return {"train": list(dev) + list(w["train"]), "read": list(w["val"]) + list(w["test"])}


def build(total_timesteps=56000, checkpoint_freq=8000):
    win = windows()
    src_path, cell = lr.cert_config("interface", allowlist=win["train"], tag="rl2")
    cfg = yaml.safe_load(open(src_path)); base = cfg[cell]
    blocks = {}
    for line, spec in LINES.items():
        b = copy.deepcopy(base)
        name = f"rl2_{line}_s2_r48_w72_c3_n35"
        b["experiment_name"] = name
        b["simulation_name"] = f"RLV2_{name}"
        b["perturb_tier"] = "godeye"                     # training is always clean
        b["forecast_mode"] = "none" if spec["hollow"] else "full"
        b["crd"] = dict(copy.deepcopy(b.get("crd", {})), enabled=bool(spec["crd"]))
        b["gtrxl"] = dict(copy.deepcopy(b.get("gtrxl", {})), cover_prior_fixed=True, cover_prior_gain=COVER_GAIN)
        b["training"] = dict(b.get("training", {}), total_timesteps=int(total_timesteps),
                             checkpoint_freq_timesteps=int(checkpoint_freq), checkpoint_num_to_keep=0,
                             save_init_checkpoint=True)
        b["wandb"] = dict(b.get("wandb", {}), enabled=False)
        blocks[name] = b
        extra = sorted(k for k in set(b) | set(base) if b.get(k) != base.get(k))
        bad = [k for k in extra if k not in WHITELIST]
        if bad:
            raise RuntimeError(f"{name}: keys outside the whitelist changed: {bad}")
    names = list(blocks)
    for a in names:
        for c in names:
            d = [k for k in set(blocks[a]) | set(blocks[c]) if blocks[a].get(k) != blocks[c].get(k)]
            if any(k not in BETWEEN_LINES for k in d):
                raise RuntimeError(f"{a} vs {c}: lines differ outside the named keys: {d}")
    text = yaml.safe_dump({"common": cfg["common"], **blocks}, sort_keys=True, default_flow_style=False)
    with open(OUT_CFG, "w") as f:
        f.write(text)
    # eval blocks: the reading windows as the allowlist, one block per (forecast channel, tier)
    read_path, read_cell = lr.cert_config("interface", allowlist=win["read"], tag="rl2read")
    rcfg = yaml.safe_load(open(read_path)); rbase = rcfg[read_cell]
    eblocks = {}
    for chan in ("full", "none"):
        for tier in (TIERS if chan == "full" else ("godeye",)):
            e = copy.deepcopy(rbase)
            name = f"rl2e_{chan}_{tier}"
            e["experiment_name"] = name; e["simulation_name"] = f"RLV2E_{name}"
            e["forecast_mode"] = chan; e["perturb_tier"] = tier
            e["gtrxl"] = dict(copy.deepcopy(e.get("gtrxl", {})), cover_prior_fixed=True, cover_prior_gain=COVER_GAIN)
            e["wandb"] = dict(e.get("wandb", {}), enabled=False)
            eblocks[name] = e
    etext = yaml.safe_dump({"common": rcfg["common"], **eblocks}, sort_keys=True, default_flow_style=False)
    with open(OUT_EVAL, "w") as f:
        f.write(etext)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    jar = os.path.join(lr.REPO, "cloudsimplus-gateway", "build", "install", "cloudsimplus-gateway", "lib", "cloudsimplus-gateway.jar")
    man = {"config": os.path.basename(OUT_CFG), "config_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
           "eval_config": os.path.basename(OUT_EVAL), "eval_config_sha256": hashlib.sha256(etext.encode()).hexdigest()[:16],
           "source": {"file": os.path.basename(src_path), "cell": cell},
           "crd_subtree_sha256": hashlib.sha256(json.dumps(base.get("crd", {}), sort_keys=True).encode()).hexdigest()[:16],
           "jar_sha256": hashlib.sha256(open(jar, "rb").read()).hexdigest()[:16] if os.path.exists(jar) else None,
           "windows": win, "seed": SEED, "cover_prior_gain": COVER_GAIN, "total_timesteps": int(total_timesteps), "checkpoint_freq": int(checkpoint_freq),
           "tiers": list(TIERS), "commit_at_build": commit,
           "lines": {n: {"crd_enabled": b["crd"]["enabled"], "forecast_mode": b["forecast_mode"]} for n, b in blocks.items()}}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(man, open(MANIFEST, "w"), indent=1, sort_keys=True)
    return blocks, eblocks, man


if __name__ == "__main__":
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 56000
    b, e, m = build(steps)
    print(json.dumps({"lines": m["lines"], "config_sha256": m["config_sha256"], "eval_blocks": list(e), "train_windows": m["windows"]["train"], "read_windows": m["windows"]["read"]}, indent=1))
