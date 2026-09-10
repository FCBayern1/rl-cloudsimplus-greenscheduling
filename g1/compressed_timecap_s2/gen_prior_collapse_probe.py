"""Generate the first-epoch prior-collapse probe configuration.

This is a diagnostic fork of the frozen prior-gate configuration.  It keeps the
same seed, model, environment and 8,000 sampled steps, but performs one PPO
epoch instead of five.  It does not alter the frozen gate or any production
configuration.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "config_prior_gate.yml"
OUTPUT = HERE / "config_prior_collapse_probe.yml"
MANIFEST = HERE / "stage_a_out" / "prior_gate" / "collapse_probe_manifest.json"


def build() -> dict:
    src = yaml.safe_load(SOURCE.read_text())
    block = copy.deepcopy(src["pg_V"])
    block["experiment_name"] = "prior_collapse_probe_sgd1"
    block["simulation_name"] = "PRIOR_COLLAPSE_PROBE_SGD1"
    training = dict(block["training"])
    training.update(
        total_timesteps=8000,
        num_sgd_iter=1,
        checkpoint_freq_timesteps=8000,
        checkpoint_num_to_keep=0,
        save_init_checkpoint=True,
    )
    block["training"] = training
    out = {"common": src["common"], "probe_sgd1": block}
    text = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
    OUTPUT.write_text(text)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "seed": 20260914,
        "total_timesteps": 8000,
        "num_sgd_iter": 1,
        "offset_grid_dense": True,
        "offset_grid": list(range(73)),
        "global_action_choices": 5 * 73,
        "purpose": "distinguish one PPO epoch from five-epoch accumulation",
    }, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(json.dumps(build(), indent=2)[:2000])
