"""Frozen implementation contract for GATED_RESIDUAL_PREREG (2026-09-10)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.learners.per_slot_credit_loss import anchored_gate_regularizer  # noqa: E402
from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule  # noqa: E402

TINY = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32,
        "dropout": 0.0, "mem_len": 4, "max_seq_len": 16}
N, NB, K = 3, 4, 9
ANCHOR = {"enabled": True, "anchor_margin": 0.1, "gate_init": 0.01,
          "gate_max": 0.5, "regularizer_lambda": 100.0}


def _space():
    inner = {
        "dc_current_green_power_w": spaces.Box(0.0, 5e6, (N,), np.float32),
        "dc_green_ratio": spaces.Box(0.0, 1.0, (N,), np.float32),
        "dc_held_count": spaces.Box(0.0, 1.0, (N,), np.float32),
        "batch_cloudlet_pes": spaces.Box(0, 100, (NB,), np.int32),
        "batch_cloudlet_mi": spaces.Box(0, 2_000_000, (NB,), np.int64),
        "batch_cloudlet_defer_allowed": spaces.Box(0.0, 1.0, (NB,), np.float32),
        "batch_cloudlet_offset_allowed": spaces.Box(0.0, 1.0, (NB, N * K), np.float32),
        "cand_green_cover": spaces.Box(0.0, 1.0, (NB, N * K), np.float32),
        "upcoming_cloudlets_count": spaces.Box(0, 100_000, (1,), np.int32),
    }
    return spaces.Dict({"observation": spaces.Dict(inner),
                        "action_mask": spaces.Box(0.0, 1.0, (NB,), np.float32)})


def _build(seed=0, anchor=True, explicit_disabled=False):
    torch.manual_seed(seed)
    cfg = {**TINY, "cover_prior_fixed": True, "cover_prior_gain": 20.0}
    if anchor:
        cfg["anchored_gated_residual"] = dict(ANCHOR)
    elif explicit_disabled:
        cfg["anchored_gated_residual"] = {"enabled": False}
    mod = RLModuleSpec(
        module_class=GTrXLScoreBasedGlobalRLModule,
        observation_space=_space(), action_space=spaces.MultiDiscrete([N * K] * NB),
        model_config=cfg).build()
    mod.eval()
    return mod


def _batch(B=2, T=None, seed=0, exact_tie=False):
    torch.manual_seed(seed)
    lead = (B,) if T is None else (B, T)
    cover = torch.rand(*lead, NB, N * K)
    if exact_tie:
        cover[..., 0, :] = 0.5
    legal = torch.rand(*lead, NB, N * K) > 0.2
    legal[..., 0] = True
    obs = {
        "dc_current_green_power_w": torch.rand(*lead, N) * 1e5,
        "dc_green_ratio": torch.rand(*lead, N),
        "dc_held_count": torch.rand(*lead, N),
        "batch_cloudlet_pes": torch.randint(1, 16, (*lead, NB)).int(),
        "batch_cloudlet_mi": torch.randint(1, 1_000_000, (*lead, NB)).long(),
        "batch_cloudlet_defer_allowed": torch.ones(*lead, NB),
        "batch_cloudlet_offset_allowed": legal.float(),
        "cand_green_cover": cover,
        "upcoming_cloudlets_count": torch.randint(0, 100, (*lead, 1)).int(),
    }
    return {Columns.OBS: {"observation": obs, "action_mask": torch.ones(*lead, NB)}}


def _logits(mod, batch):
    return mod._forward_train(batch)[Columns.ACTION_DIST_INPUTS]


def _slot_logits(logits):
    return logits.reshape(*logits.shape[:-1], NB, N * K)


def _rule(batch):
    obs = batch[Columns.OBS]["observation"]
    cover = obs["cand_green_cover"]
    legal = obs["batch_cloudlet_offset_allowed"] >= 0.5
    return torch.where(legal, cover, torch.full_like(cover, -1e9)).argmax(-1)


def test_default_off_is_bit_identical():
    absent = _build(seed=4, anchor=False)
    disabled = _build(seed=4, anchor=False, explicit_disabled=True)
    assert absent.state_dict().keys() == disabled.state_dict().keys()
    for key in absent.state_dict():
        torch.testing.assert_close(absent.state_dict()[key], disabled.state_dict()[key], rtol=0, atol=0)
    batch = _batch(seed=5)
    with torch.no_grad():
        torch.testing.assert_close(_logits(absent, batch), _logits(disabled, batch), rtol=0, atol=0)


def test_config_mapping_survives_the_trainer_merge_whitelist():
    from src.training.train_rlmodule_gtrxl import _merged_gtrxl_model_settings

    env = {"gtrxl": {"cover_prior_fixed": True,
                     "anchored_gated_residual": dict(ANCHOR)}}
    merged = _merged_gtrxl_model_settings({}, env)
    assert merged["anchored_gated_residual"] == ANCHOR
    # The merge must copy rather than alias the user configuration.
    merged["anchored_gated_residual"]["gate_init"] = 0.2
    assert env["gtrxl"]["anchored_gated_residual"]["gate_init"] == 0.01


def test_generated_gate_config_reaches_global_module_and_custom_learner():
    from src.learners.per_slot_credit_loss import PerSlotCreditPPOTorchLearner
    from src.training.train_rlmodule_gtrxl import create_rlmodule_config, load_config

    path = REPO_ROOT.parent / "g1/compressed_timecap_s2/config_gated_residual_gate.yml"
    exp = load_config(str(path))["gated_residual_V"]
    cfg = create_rlmodule_config(
        exp, exp.get("global_model", {}), exp.get("local_model", {}), exp.get("training", {}))
    assert cfg.learner_class is PerSlotCreditPPOTorchLearner
    model_cfg = cfg.rl_module_spec.rl_module_specs["global_policy"].model_config
    assert model_cfg["anchored_gated_residual"] == ANCHOR
    assert model_cfg["per_slot_credit"] == {"enabled": True, "mask_padding": True}


def test_initial_greedy_equals_rule_and_anchor_margin_including_ties():
    mod = _build(seed=6)
    batch = _batch(seed=7, exact_tie=True)
    with torch.no_grad():
        logits = _slot_logits(_logits(mod, batch))[:, 0]
    assert torch.equal(logits.argmax(-1), _rule(batch))
    rule = _rule(batch)
    chosen = logits.gather(-1, rule.unsqueeze(-1)).squeeze(-1)
    competitors = logits.clone()
    competitors.scatter_(-1, rule.unsqueeze(-1), -torch.inf)
    margin = chosen - competitors.max(-1).values
    assert torch.all(margin >= 0.1 - 1e-6)
    assert math.isclose(float(mod.residual_gate_value().detach()), 0.01, rel_tol=0, abs_tol=1e-7)


def test_gate_floor_prevents_arbitrarily_large_raw_residual_from_flipping_rule():
    mod = _build(seed=8)
    batch = _batch(seed=9, exact_tie=True)
    # Saturate the bounded residual in different directions. Its largest
    # candidate-to-candidate effect is still 2*g0 = 0.02 < anchor 0.1.
    with torch.no_grad():
        for layer in (mod.dc_encoder, mod.ctx_to_dc, mod.offset_head):
            layer.weight.uniform_(-1e6, 1e6)
            layer.bias.uniform_(-1e6, 1e6)
        logits = _slot_logits(_logits(mod, batch))[:, 0]
    assert torch.equal(logits.argmax(-1), _rule(batch))


def test_regularizer_uses_legal_competitors_and_real_jobs_only():
    class FakeModule:
        model_config = {"anchored_gated_residual": {"enabled": True}}
        num_batch_slots = 3
        num_action_choices = 4
        cover_gain = torch.tensor([1.0])
        residual_anchor_margin = torch.tensor(0.1)
        residual_regularizer_lambda = torch.tensor(100.0)
        residual_gate_value = lambda self: torch.tensor(0.01)

    module = FakeModule()
    cover = torch.tensor([[[0.8, 0.2, 0.1, 0.0],
                           [0.8, 0.2, 0.1, 0.0],
                           [0.8, 0.2, 0.1, 0.0]]])
    legal = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]]]).float()
    mi = torch.tensor([[10, 0, 10]])
    rule = torch.zeros_like(cover); rule[..., 0] = 0.1
    effective = torch.zeros_like(cover)
    effective[0, 0, 1] = 0.2       # real + legal: d=0.2
    effective[0, 1, 1] = 1.0       # padding: ignored
    effective[0, 2, 1] = 2.0       # illegal: ignored
    logits = (cover + rule + effective).reshape(1, -1).requires_grad_()
    batch = {Columns.OBS: {"observation": {
        "cand_green_cover": cover,
        "batch_cloudlet_offset_allowed": legal,
        "batch_cloudlet_mi": mi,
    }}}
    loss, metrics = anchored_gate_regularizer(module, batch, logits)
    # 100 * mean([0.2^2, 0^2]) = 2.0
    torch.testing.assert_close(loss, torch.tensor(2.0), rtol=0, atol=1e-6)
    assert metrics["anchored_gate/real_decisions"] == 2
    assert math.isclose(metrics["anchored_gate/d_nonzero_frac"], 0.5)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0

    zero_logits = (cover + rule).reshape(1, -1)
    zero, _ = anchored_gate_regularizer(module, batch, zero_logits)
    torch.testing.assert_close(zero, torch.tensor(0.0), rtol=0, atol=0)


def test_anchored_train_and_rollout_paths_are_identical():
    mod = _build(seed=10)
    batch = _batch(B=2, T=5, seed=11, exact_tie=True)
    with torch.no_grad():
        sequence = _logits(mod, batch)
        pieces, state = [], None
        for t in range(5):
            obs = batch[Columns.OBS]
            step = {Columns.OBS: {
                "observation": {k: v[:, t:t + 1] for k, v in obs["observation"].items()},
                "action_mask": obs["action_mask"][:, t:t + 1],
            }}
            if state is not None:
                step[Columns.STATE_IN] = state
            out = mod._forward_train(step)
            pieces.append(out[Columns.ACTION_DIST_INPUTS])
            state = out.get(Columns.STATE_OUT)
        rollout = torch.cat(pieces, dim=1)

    torch.testing.assert_close(sequence, rollout, rtol=1e-5, atol=1e-6)
    seq_p = torch.softmax(_slot_logits(sequence), -1)
    roll_p = torch.softmax(_slot_logits(rollout), -1)
    torch.testing.assert_close(seq_p, roll_p, rtol=1e-5, atol=1e-6)
    actions = seq_p.argmax(-1)
    assert torch.equal(actions, roll_p.argmax(-1))
    seq_lp = torch.log_softmax(_slot_logits(sequence), -1).gather(
        -1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    roll_lp = torch.log_softmax(_slot_logits(rollout), -1).gather(
        -1, actions.unsqueeze(-1)).squeeze(-1).sum(-1)
    torch.testing.assert_close(seq_lp, roll_lp, rtol=1e-5, atol=1e-6)
