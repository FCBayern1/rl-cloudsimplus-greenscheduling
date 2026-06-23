"""Regression test for RLlibNewAPIGlobalScheduler._sample_multidiscrete.

Bug: the method hardcoded `self.num_datacenters` as the per-slot choice count, so an
arch-B checkpoint with a global DEFER head (num_datacenters + 1 logits per slot) failed
with `shape '[1, 128, 5]' is invalid for input of size 768` (768 = 128 * 6).

Fix: derive the choice count from the actual tensor size (last_dim // batch_size), so
both the plain (5 choices) and defer (6 choices) action heads reshape correctly.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.baselines.global_schedulers import RLlibNewAPIGlobalScheduler


def _make_scheduler(batch_size, stochastic=False):
    # Bypass the heavy __init__ (it loads an RLlib checkpoint); we only exercise the
    # pure reshape/argmax logic, which depends solely on self.batch_size.
    sched = object.__new__(RLlibNewAPIGlobalScheduler)
    sched.batch_size = batch_size
    sched.stochastic = stochastic
    return sched


def test_sample_multidiscrete_no_defer_5_choices():
    sched = _make_scheduler(batch_size=128)
    num_choices = 5
    dist_inputs = torch.zeros(1, 128 * num_choices)
    actions = sched._sample_multidiscrete(dist_inputs)
    assert tuple(actions.shape) == (1, 128)
    assert int(actions.max()) <= num_choices - 1


def test_sample_multidiscrete_with_defer_6_choices():
    """The arch-B defer head adds a 6th choice; must not raise and must be reachable."""
    sched = _make_scheduler(batch_size=128)
    num_choices = 6  # 5 DCs + 1 defer
    dist_inputs = torch.full((1, 128 * num_choices), -10.0)
    # Make the defer logit (index 5) the argmax for every slot.
    logits = dist_inputs.reshape(1, 128, num_choices)
    logits[:, :, 5] = 10.0
    dist_inputs = logits.reshape(1, 128 * num_choices)

    actions = sched._sample_multidiscrete(dist_inputs)
    assert tuple(actions.shape) == (1, 128)
    # Every slot should pick the defer action (index 5).
    assert int(actions.min()) == 5 and int(actions.max()) == 5


def test_sample_multidiscrete_argmax_picks_per_slot_max():
    sched = _make_scheduler(batch_size=4)
    num_choices = 6
    logits = torch.full((1, 4, num_choices), -1.0)
    want = [0, 3, 5, 2]
    for slot, choice in enumerate(want):
        logits[0, slot, choice] = 5.0
    dist_inputs = logits.reshape(1, 4 * num_choices)
    actions = sched._sample_multidiscrete(dist_inputs)
    assert actions[0].tolist() == want


def test_stochastic_spreads_where_argmax_collapses():
    """With identical near-uniform logits on every slot, greedy argmax collapses all slots
    to one DC, while stochastic sampling spreads them across DCs (the whole point of the flag)."""
    num_choices = 5
    batch = 128
    # Identical logits per slot, very slight preference for DC1 — argmax must pick DC1 everywhere.
    base = torch.tensor([0.0, 0.01, 0.0, 0.0, 0.0])
    logits = base.repeat(1, batch, 1)
    dist_inputs = logits.reshape(1, batch * num_choices)

    greedy = _make_scheduler(batch, stochastic=False)._sample_multidiscrete(dist_inputs)
    assert len(set(greedy[0].tolist())) == 1  # collapsed onto a single DC

    torch.manual_seed(0)
    stoch = _make_scheduler(batch, stochastic=True)._sample_multidiscrete(dist_inputs)
    assert len(set(stoch[0].tolist())) >= 3  # spread across multiple DCs
    assert tuple(stoch.shape) == (1, batch)
