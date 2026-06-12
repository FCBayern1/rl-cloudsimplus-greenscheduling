"""
Root-cause regression test for the dead critic (2026-06-12 discovery).

The 2026-05-12 OOM fix wrapped every RLModule's `compute_values` in
`torch.inference_mode()` — correct for its motivating caller (the GAE
connector's full-batch V(s) pass needs no gradients), but FATAL for the
other caller: `PPOTorchLearner.compute_loss_for_module` builds the vf loss
from `module.compute_values(batch, embeddings=fwd_out.get(EMBEDDINGS))`.
With a grad-free tensor, the vf loss is a constant w.r.t. parameters and
BOTH critics (global + local) received exactly zero gradient from 2026-05-12
onward — no vf_clip / vf_coef / loss-normalization knob could matter.

The fix: `_forward_train` already computes grad-carrying values in the same
forward pass — emit them as `Columns.EMBEDDINGS`, and `compute_values`
returns them directly when provided. Callers without embeddings (GAE) keep
the inference_mode re-forward, preserving the OOM protection.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_compute_values_grad.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import tree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ray.rllib.core.columns import Columns

from src.training.train_rlmodule_gtrxl import create_rlmodule_config, load_config

CONFIG_YML = REPO_ROOT.parent / "config.yml"
EXPERIMENT = "experiment_multi_5dc_carbon_v2"


@pytest.fixture(scope="module")
def specs():
    cfg = load_config(str(CONFIG_YML))[EXPERIMENT]
    c = create_rlmodule_config(
        cfg, cfg.get("global_model", {}), cfg.get("local_model", {}),
        cfg.get("training", {}),
    )
    return c.rl_module_spec.rl_module_specs


def _make_batch(spec):
    obs = spec.observation_space.sample()
    return {
        Columns.OBS: tree.map_structure(
            lambda x: torch.as_tensor(np.asarray(x)[None, None]).float(), obs
        )
    }


def _learner_style_values(module, batch):
    """Exactly what PPOTorchLearner.compute_loss_for_module does."""
    fwd_out = module._forward_train(dict(batch))
    return fwd_out, module.compute_values(
        dict(batch), embeddings=fwd_out.get(Columns.EMBEDDINGS)
    )


@pytest.mark.parametrize("module_id", ["global_policy", "shared_local_policy"])
def test_learner_path_values_carry_grad(specs, module_id):
    module = specs[module_id].build()
    module.train()
    _, values = _learner_style_values(module, _make_batch(specs[module_id]))
    assert values.requires_grad, (
        f"{module_id}: compute_values output is grad-free on the LEARNER path "
        "— the vf loss is a constant and the critic receives zero gradient "
        "(the 2026-05-12 inference_mode OOM fix leaked into the loss path)."
    )


@pytest.mark.parametrize("module_id", ["global_policy", "shared_local_policy"])
def test_vf_loss_gradient_reaches_value_head(specs, module_id):
    module = specs[module_id].build()
    module.train()
    _, values = _learner_style_values(module, _make_batch(specs[module_id]))
    loss = (values - 1000.0).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in module.value_head.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        f"{module_id}: vf loss backward left value_head untouched."
    )


def test_vf_gradient_respects_actor_isolation(specs):
    """Dual-trunk invariant: with critic_separate_trunk, the vf gradient must
    reach the critic trunk but never the actor's encoders/trunk."""
    module = specs["global_policy"].build()
    if not getattr(module, "_critic_separate_trunk", False):
        pytest.skip("critic_separate_trunk disabled in this config")
    module.train()
    _, values = _learner_style_values(module, _make_batch(specs["global_policy"]))
    (values - 1000.0).pow(2).mean().backward()
    crit_grads = [p.grad for p in module.critic_gtrxl.parameters()]
    assert any(g is not None and torch.any(g != 0) for g in crit_grads), (
        "vf gradient never reached the critic GTrXL trunk."
    )
    actor_grads = [
        p.grad for p in module.gtrxl.parameters()
        if p.grad is not None and torch.any(p.grad != 0)
    ]
    assert not actor_grads, "vf gradient leaked into the ACTOR trunk."


@pytest.mark.parametrize("module_id", ["global_policy", "shared_local_policy"])
def test_gae_path_stays_grad_free(specs, module_id):
    """The GAE connector calls compute_values WITHOUT embeddings on the full
    rollout batch — that path must stay inference_mode (the 10+GB OOM fix)."""
    module = specs[module_id].build()
    module.train()
    values = module.compute_values(_make_batch(specs[module_id]))
    assert not values.requires_grad


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
