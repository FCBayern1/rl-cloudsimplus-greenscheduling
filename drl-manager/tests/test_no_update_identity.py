"""No-update identity: a training step that changes no parameter must leave the deployed
policy bit-identical.

This is the guard for the whole class of train/rollout inconsistencies found on 2026-09-10
(the evaluation ran without the recurrent memory and with dropout active; the positional
encoding was indexed by position inside the training chunk). Each of those made the policy
that acts differ from the policy that is scored, without anything crashing.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.networks.gtrxl import GTrXL


def _model(seed=0):
    torch.manual_seed(seed)
    m = GTrXL(input_dim=6, d_model=16, num_layers=2, nhead=2, dim_feedforward=32,
              mem_len=4, max_seq_len=32, dropout=0.1)
    with torch.no_grad():
        m.pos_encoder.copy_(torch.randn_like(m.pos_encoder))
    return m


def _rollout(m, x):
    """Deploy the way evaluation deploys: eval mode, one step at a time, carrying state."""
    m.eval()
    out, state = [], None
    with torch.no_grad():
        for t in range(x.shape[1]):
            o, state = m(x[:, t:t + 1, :], state=state)
            out.append(o)
    return torch.cat(out, dim=1)


def test_zero_learning_rate_update_leaves_the_deployed_policy_identical():
    m = _model()
    torch.manual_seed(1)
    x = torch.randn(2, 6, 6)

    before = _rollout(m, x)

    opt = torch.optim.Adam(m.parameters(), lr=0.0)      # a real step, zero size
    m.train()
    loss = m(x)[0].pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

    after = _rollout(m, x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_deployment_is_deterministic_across_repeats():
    """Dropout left on at evaluation made repeated deployments differ; it must not."""
    m = _model(seed=2)
    torch.manual_seed(3)
    x = torch.randn(2, 5, 6)
    a, b = _rollout(m, x), _rollout(m, x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_train_mode_with_dropout_is_not_what_deployment_uses():
    """A guard on the guard: with dropout active the same input does vary, so the identity
    above is evidence that deployment really is in eval mode, not that dropout is absent."""
    m = _model(seed=4)
    torch.manual_seed(5)
    x = torch.randn(2, 5, 6)
    m.train()
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x)
    assert not torch.allclose(a, b, rtol=1e-6, atol=1e-7)


def test_a_real_update_does_change_the_deployed_policy():
    """The identity test must not pass by measuring nothing."""
    m = _model(seed=6)
    torch.manual_seed(7)
    x = torch.randn(2, 5, 6)
    before = _rollout(m, x)

    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    loss = m(x)[0].pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

    after = _rollout(m, x)
    assert not torch.allclose(before, after, rtol=1e-5, atol=1e-6)
