"""Diagnostic block for the gradient decomposition (EUCRD_SWITCH_AB_PREREG Addendum A): the
G5-err configuration with the learning rate at zero (no optimisation update) and one sampling
iteration, run with the G5-err final global module loaded through the strict warm-start path.

Usage: python gen_switch_ab_decomp.py   -> config_switch_ab_decomp.yml, prints the checkpoint
"""
from __future__ import annotations

import copy
import glob
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DRL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "drl-manager")
SRC = os.path.join(HERE, "config_eucrd_wiring.yml")
OUT = os.path.join(HERE, "config_switch_ab_decomp.yml")
STEPS = 8000


def checkpoint():
    """The G5-err final global module as a plain tensor state dict. RLlib 2.40 writes the
    module state as a pickle of numpy arrays, which the strict warm-start loader (torch.load
    with weights_only=True) refuses; it is converted once, key for key, with nothing dropped."""
    import pickle
    import torch
    ps = sorted(glob.glob(os.path.join(
        DRL, "logs", "eucrd_wiring", "err", "*", "PPO_*", "checkpoint_*",
        "learner_group", "learner", "rl_module", "global_policy", "module_state.pt")))
    if not ps:
        raise SystemExit("no G5-err global module state found")
    out = os.path.join(HERE, "stage_a_out", "eucrd_switch_ab_decomp", "g5_err_global_state.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(ps[-1], "rb") as f:
        raw = pickle.load(f)
    state = {k: torch.as_tensor(v) for k, v in raw.items()}
    torch.save(state, out)
    back = torch.load(out, map_location="cpu", weights_only=True)
    assert list(back) == list(raw) and all(torch.is_tensor(v) for v in back.values()), "conversion changed the keys"
    return out


def build():
    cfg = yaml.safe_load(open(SRC))
    b = copy.deepcopy(cfg["g5_err"])
    b["experiment_name"] = "switch_ab_decomp"; b["simulation_name"] = "SWITCH_AB_DECOMP"
    b["training"] = dict(b.get("training", {}), total_timesteps=STEPS,
                         checkpoint_freq_timesteps=0, checkpoint_num_to_keep=0,
                         save_init_checkpoint=False)
    # the optimiser learning rate is read per policy from the model blocks
    # (train_rlmodule_gtrxl: global_model.learning_rate -> lr override, local_model.learning_rate)
    for k in ("global_model", "local_model"):
        if isinstance(b.get(k), dict):
            b[k] = dict(b[k], learning_rate=0.0)
    diff = sorted(k for k in set(b) | set(cfg["g5_err"]) if b.get(k) != cfg["g5_err"].get(k))
    assert diff == ["experiment_name", "global_model", "local_model", "simulation_name", "training"], diff
    out = {"common": cfg["common"], "switch_ab_decomp": b}
    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=True)
    ck = checkpoint()
    print(json.dumps({"config": OUT, "checkpoint": ck, "steps": STEPS, "diff_vs_g5_err": diff,
                      "n_state_entries": len(__import__("torch").load(ck, weights_only=True))}, indent=1))
    return ck


if __name__ == "__main__":
    build()
