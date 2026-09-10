"""Generate the frozen anchored-gated-residual warm-up and gate run.

The run is one continuous 48k-step vanilla line: the first 24k steps train
only the critic (60 loss calls = 3 iterations × 5 epochs × 4 minibatches),
then the actor is released for the registered 24k gate interval.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE / "config_prior_gate.yml"
OUTPUT = HERE / "config_gated_residual_gate.yml"
MANIFEST = HERE / "stage_a_out" / "gated_residual_gate" / "generation_manifest.json"
SEED = 20260915
TOTAL_STEPS = 48_000
FREEZE_ACTOR_LOSS_CALLS = 60
ANCHORED = {
    "enabled": True,
    "anchor_margin": 0.1,
    "gate_init": 0.01,
    "gate_max": 0.5,
    "regularizer_lambda": 100.0,
}


def build() -> dict:
    src = yaml.safe_load(SOURCE.read_text())
    original = src["pg_V"]
    block = copy.deepcopy(original)
    block["experiment_name"] = "gated_residual_V"
    block["simulation_name"] = "GATED_RESIDUAL_V"
    block["gtrxl"] = dict(block["gtrxl"], anchored_gated_residual=dict(ANCHORED))
    block["training"] = dict(
        block["training"],
        total_timesteps=TOTAL_STEPS,
        checkpoint_freq_timesteps=8_000,
        checkpoint_num_to_keep=0,
        save_init_checkpoint=True,
    )

    changed = sorted(k for k in set(block) | set(original) if block.get(k) != original.get(k))
    assert changed == ["experiment_name", "gtrxl", "simulation_name", "training"], changed
    assert block["gtrxl"]["cover_prior_fixed"] is True
    assert block["gtrxl"]["cover_prior_gain"] == 20.0
    assert block["gtrxl"]["anchored_gated_residual"] == ANCHORED
    assert block["per_slot_credit"]["enabled"] is True
    assert block["training"]["train_batch_size"] == 8_000
    assert block["training"]["sgd_minibatch_size"] == 2_048
    assert block["training"]["num_sgd_iter"] == 5
    assert FREEZE_ACTOR_LOSS_CALLS == 3 * 5 * 4

    out = {"common": src["common"], "gated_residual_V": block}
    text = yaml.safe_dump(out, sort_keys=True, default_flow_style=False)
    OUTPUT.write_text(text)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "prereg_commit": "e82701e8",
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output": str(OUTPUT.relative_to(REPO)),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "critic_warmup_steps": 24_000,
        "gate_steps": 24_000,
        "freeze_actor_loss_calls": FREEZE_ACTOR_LOSS_CALLS,
        "required_environment": {
            "FREEZE_ACTOR_ITERS": str(FREEZE_ACTOR_LOSS_CALLS),
            "OFFSET_GRID_DENSE": "1",
            "PYTHONHASHSEED": "0",
        },
        "anchored_gated_residual": ANCHORED,
        "changed_top_level_keys_vs_prior_gate": changed,
    }, indent=2, sort_keys=True) + "\n")
    return out


if __name__ == "__main__":
    generated = build()
    print(json.dumps({
        "config": str(OUTPUT),
        "experiment": "gated_residual_V",
        "seed": SEED,
        "freeze_actor_loss_calls": FREEZE_ACTOR_LOSS_CALLS,
    }, indent=2))
