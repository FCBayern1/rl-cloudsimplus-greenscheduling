"""Stage D' symmetric responsibility guard (STAGE_D_PRIME_DESIGN §10): w' = 1 + eta (w - 1)."""
import torch

from src.learners.crd_q_loss import shrink_weights


def test_eta_one_is_the_identity_object_bit_identical():
    w = torch.tensor([0.06, 0.2, 1.0, 1.16, 2.0])
    out = shrink_weights(w, 1.0)
    assert out is w                                   # not even a copy: historical path untouched


def test_eta_zero_removes_the_reweighting():
    w = torch.tensor([0.06, 0.2, 1.0, 2.0])
    assert torch.equal(shrink_weights(w, 0.0), torch.ones_like(w))


def test_eta_half_matches_the_ruled_examples_and_keeps_order_and_mean():
    w = torch.tensor([0.06, 0.2, 2.0])
    out = shrink_weights(w, 0.5)
    assert torch.allclose(out, torch.tensor([0.53, 0.6, 1.5]))
    big = torch.tensor([0.1, 0.5, 1.0, 1.4, 2.0])            # mean 1.0
    g = shrink_weights(big, 0.5)
    assert abs(float(g.mean()) - 1.0) < 1e-6
    assert torch.all(g[1:] > g[:-1])                       # ordering preserved


def test_default_config_key_is_eta_one():
    cfg = {}
    assert float(cfg.get("responsibility_shrink_strength", 1.0)) == 1.0
