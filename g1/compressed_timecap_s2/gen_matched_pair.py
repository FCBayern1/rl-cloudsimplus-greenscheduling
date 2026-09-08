"""Matched imperfect-forecast pair (reports/EUCRD_MATCHED_PAIR_PREREG.md §1-2): V_err and E_err
from the RL_V2 CPU-learner twin, trained on shrink75 for 120 000 steps, three paired seeds.
The identity of the two lines is asserted, not assumed.

Usage: python gen_matched_pair.py   -> config_matched_pair.yml + stage_a_out/matched_pair/manifest.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "config_rl_v2_cpu.yml")
OUT = os.path.join(HERE, "config_matched_pair.yml")
MANIFEST = os.path.join(HERE, "stage_a_out", "matched_pair", "manifest.json")
STEPS = 120000
TIER = "shrink75"
SEEDS = (20260911, 20260912, 20260913)
SOURCE = "candidate_carbon_regret"
BASES = {"V": "rl2_V_s2_r48_w72_c3_n35", "E": "rl2_E_s2_r48_w72_c3_n35"}


def _diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def build():
    cfg = yaml.safe_load(open(SRC))
    v0, e0 = cfg[BASES["V"]], cfg[BASES["E"]]
    assert _diff(v0, e0) == ["crd", "experiment_name", "simulation_name"], _diff(v0, e0)
    out = {"common": cfg["common"]}
    blocks = {}
    for line, base in BASES.items():
        b = copy.deepcopy(cfg[base])
        b["perturb_tier"] = TIER
        crd = copy.deepcopy(b.get("crd", {})); fc = dict(crd.get("forecast", {}))
        fc["source"] = SOURCE; crd["forecast"] = fc
        crd["enabled"] = (line == "E")
        b["crd"] = crd
        b["training"] = dict(b.get("training", {}), total_timesteps=STEPS,
                             checkpoint_freq_timesteps=8000, checkpoint_num_to_keep=0,
                             save_init_checkpoint=False)
        blocks[line] = b
    v, e = blocks["V"], blocks["E"]
    # identity of the pair: only crd and identity differ, and inside crd only `enabled`
    assert _diff(v, e) == ["crd", "experiment_name", "simulation_name"], _diff(v, e)
    assert _diff(v["crd"], e["crd"]) == ["enabled"], _diff(v["crd"], e["crd"])
    assert e["crd"]["forecast"]["source"] == SOURCE and v["crd"]["enabled"] is False
    for k in ("global_model", "local_model"):
        assert v[k].get("max_grad_norm") == e[k].get("max_grad_norm"), k
    assert v["global_model"]["max_grad_norm"] == 20.0 and v["local_model"]["max_grad_norm"] == 0.5
    assert v["green_episode_offset_allowlist"] == e["green_episode_offset_allowlist"]
    for line, b in blocks.items():
        b["experiment_name"] = f"mp_{line}_err"; b["simulation_name"] = f"MP_{line}_ERR"
        out[f"mp_{line}_err"] = b
    text = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
    with open(OUT, "w") as f:
        f.write(text)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    man = {"config": os.path.basename(OUT), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
           "source": os.path.basename(SRC), "bases": BASES, "steps": STEPS, "tier": TIER,
           "seeds": list(SEEDS), "forecast_source": SOURCE,
           "grad_clip": {"global": v["global_model"]["max_grad_norm"], "local": v["local_model"]["max_grad_norm"]},
           "between_line_diff": _diff(v, e), "crd_diff": _diff(v["crd"], e["crd"])}
    json.dump(man, open(MANIFEST, "w"), indent=1)
    print(json.dumps(man, indent=1))
    return out


if __name__ == "__main__":
    build()
