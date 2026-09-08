"""Small-step update trial arms (reports/EUCRD_SMALL_STEP_PREREG.md §1): from the G5-err block,
one PPO iteration, warmup 0, three arms differing only in the forecast source / reweighting.

Usage: python gen_small_step.py   -> config_small_step.yml, manifest, prints the checkpoint
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "config_eucrd_wiring.yml")
OUT = os.path.join(HERE, "config_small_step.yml")
MANIFEST = os.path.join(HERE, "stage_a_out", "small_step", "manifest.json")
STEPS = 8000
ARMS = {
    "ss_on":   {"source": "candidate_carbon_regret", "reweight": True},
    "ss_off":  {"source": "instantaneous_carbon_cf", "reweight": True},
    "ss_norw": {"source": "candidate_carbon_regret", "reweight": False},
}


def _diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def build():
    cfg = yaml.safe_load(open(SRC))
    base = cfg["g5_err"]
    assert base["perturb_tier"] == "shrink75"
    out = {"common": cfg["common"]}
    blocks = {}
    for name, spec in ARMS.items():
        b = copy.deepcopy(base)
        b["experiment_name"] = name; b["simulation_name"] = name.upper()
        crd = copy.deepcopy(b["crd"]); fc = dict(crd["forecast"]); fc["source"] = spec["source"]
        crd["forecast"] = fc
        resp = dict(crd.get("responsibility", {}))
        resp["reweight_warmup_calls"] = 0              # one iteration sits inside 450 (prereg §1)
        resp["reweight_advantages"] = bool(spec["reweight"])
        crd["responsibility"] = resp; crd["enabled"] = True
        b["crd"] = crd
        b["training"] = dict(b["training"], total_timesteps=STEPS, checkpoint_freq_timesteps=STEPS,
                             checkpoint_num_to_keep=0, save_init_checkpoint=False)
        blocks[name] = b
        out[name] = b
    on, off, norw = blocks["ss_on"], blocks["ss_off"], blocks["ss_norw"]
    assert _diff(on, off) == ["crd", "experiment_name", "simulation_name"]
    assert _diff(on["crd"], off["crd"]) == ["forecast"]
    assert _diff(on["crd"]["forecast"], off["crd"]["forecast"]) == ["source"]
    assert _diff(on["crd"], norw["crd"]) == ["responsibility"]
    assert _diff(on["crd"]["responsibility"], norw["crd"]["responsibility"]) == ["reweight_advantages"]
    text = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
    with open(OUT, "w") as f:
        f.write(text)
    from gen_switch_ab_decomp import checkpoint
    ck = checkpoint()
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    man = {"config": os.path.basename(OUT), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
           "steps": STEPS, "arms": ARMS, "checkpoint": ck, "seed": 20260910,
           "warmup_override": 0, "base_block": "g5_err"}
    json.dump(man, open(MANIFEST, "w"), indent=1)
    print(json.dumps(man, indent=1))
    return out


if __name__ == "__main__":
    build()
