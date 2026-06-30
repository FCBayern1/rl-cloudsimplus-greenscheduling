"""Core-math tests for Tier-1 per-slot credit (no RLlib learner / gateway needed).

Verifies the per-slot log-prob and entropy used by PerSlotCreditPPOTorchLearner sum (over ALL
slots) to exactly RLlib's TorchMultiCategorical joint values. If that holds, masking = summing
over a SUBSET of slots is correct by construction, and gate-off (all slots valid) is bit-identical.
"""
import torch
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical


def _per_slot_logp(logits, actions, n_slots, n_choices):
    lg = logits.reshape(*logits.shape[:-1], n_slots, n_choices)
    logsm = torch.log_softmax(lg, dim=-1)
    a = actions.reshape(*lg.shape[:-1]).long().unsqueeze(-1)
    return logsm.gather(-1, a).squeeze(-1)            # (..., n_slots)


def _per_slot_entropy(logits, n_slots, n_choices):
    lg = logits.reshape(*logits.shape[:-1], n_slots, n_choices)
    logsm = torch.log_softmax(lg, dim=-1)
    return -(logsm.exp() * logsm).sum(-1)            # (..., n_slots)


def _dist(logits, n_slots, n_choices):
    return TorchMultiCategorical.from_logits(logits, input_lens=[n_choices] * n_slots)


def test_per_slot_logp_sums_to_joint():
    torch.manual_seed(0)
    N, S, C = 5, 8, 6
    logits = torch.randn(N, S * C)
    dist = _dist(logits, S, C)
    actions = dist.sample()                          # (N, S)
    joint = dist.logp(actions)                       # (N,)
    ps = _per_slot_logp(logits, actions, S, C)       # (N, S)
    assert torch.allclose(ps.sum(-1), joint, atol=1e-5), (ps.sum(-1), joint)


def test_per_slot_entropy_sums_to_joint():
    torch.manual_seed(1)
    N, S, C = 5, 8, 6
    logits = torch.randn(N, S * C)
    dist = _dist(logits, S, C)
    joint_ent = dist.entropy()                       # (N,)
    ps_ent = _per_slot_entropy(logits, S, C)         # (N, S)
    assert torch.allclose(ps_ent.sum(-1), joint_ent, atol=1e-5), (ps_ent.sum(-1), joint_ent)


def test_masking_drops_padding_slots():
    # With a valid mask, the masked logp = joint logp over ONLY the valid slots.
    torch.manual_seed(2)
    N, S, C = 3, 10, 6
    logits = torch.randn(N, S * C)
    dist = _dist(logits, S, C)
    actions = dist.sample()
    ps = _per_slot_logp(logits, actions, S, C)       # (N, S)
    valid = torch.zeros(N, S)
    valid[:, :4] = 1.0                                # first 4 slots real, rest padding
    masked = (ps * valid).sum(-1)
    assert torch.allclose(masked, ps[:, :4].sum(-1), atol=1e-6)
    # and it must differ from the full joint (padding actually dropped)
    assert not torch.allclose(masked, ps.sum(-1))
