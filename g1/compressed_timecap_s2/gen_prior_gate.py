"""Prior-preservation gate run (RL_POLICY_DEGRADATION_INVESTIGATION §5 step 4): one vanilla
line, one seed, 24 000 steps, everything else identical to the matched pair's V block. The
question is only whether training still walks away from the prior after the two train/rollout
fixes of 2026-09-10.

Usage: python gen_prior_gate.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "config_matched_pair.yml")
OUT = os.path.join(HERE, "config_prior_gate.yml")
MANIFEST = os.path.join(HERE, "stage_a_out", "prior_gate", "manifest.json")
STEPS = 24000
SEED = 20260914          # a seed not used by the matched pair


def build():
    cfg = yaml.safe_load(open(SRC))
    b = copy.deepcopy(cfg["mp_V_err"])
    b["experiment_name"] = "pg_V"; b["simulation_name"] = "PG_V"
    b["training"] = dict(b["training"], total_timesteps=STEPS,
                         checkpoint_freq_timesteps=4000,      # finer: 6 checkpoints
                         checkpoint_num_to_keep=0, save_init_checkpoint=True)
    diff = sorted(k for k in set(b) | set(cfg["mp_V_err"]) if b.get(k) != cfg["mp_V_err"].get(k))
    assert diff == ["experiment_name", "simulation_name", "training"], diff
    out = {"common": cfg["common"], "pg_V": b}
    text = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
    open(OUT, "w").write(text)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump({"config": os.path.basename(OUT),
               "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
               "source": os.path.basename(SRC), "steps": STEPS, "seed": SEED,
               "checkpoint_every": 4000, "diff_vs_mp_V_err": diff},
              open(MANIFEST, "w"), indent=1)
    print(json.dumps({"config": OUT, "steps": STEPS, "seed": SEED, "diff": diff}, indent=1))
    return out


if __name__ == "__main__":
    build()
