"""crd.baseline.kind: configurable counterfactual baseline (2026-07-25).

Motivation: GreenQueueBalanced outperforms the learned router on some regimes
(C-regime GQB 0.136 vs RL 0.152), making dQ = Q(action)-Q(baseline)
systematically negative and degenerating the routing credit. A weaker
same-information baseline restores a signed learning signal.
"""
import pytest
from src.learners.crd_q_loss import CRDPPOTorchLearner
from src.baselines.global_schedulers import (
    GreenQueueBalancedGlobalScheduler, GreenAwareGlobalScheduler,
    MinQueueGlobalScheduler, RoundRobinGlobalScheduler,
)

build = CRDPPOTorchLearner._build_baseline_by_kind

@pytest.mark.parametrize("kind,cls", [
    ("green_queue_balanced", GreenQueueBalancedGlobalScheduler),
    ("green_aware",          GreenAwareGlobalScheduler),
    ("min_queue",            MinQueueGlobalScheduler),
    ("round_robin",          RoundRobinGlobalScheduler),
])
def test_kind_builds_expected_scheduler(kind, cls):
    s = build(kind, num_dc=5, batch_size=128, green_weight=0.6)
    assert isinstance(s, cls)
    assert s.num_datacenters == 5 and s.batch_size == 128

def test_default_kind_is_historical_gqb():
    # absent kind -> caller passes "green_queue_balanced"; verify it keeps green_weight
    s = build("green_queue_balanced", num_dc=5, batch_size=8, green_weight=0.42)
    assert isinstance(s, GreenQueueBalancedGlobalScheduler)
    assert s.green_weight == pytest.approx(0.42)

def test_unknown_kind_falls_back_not_crash():
    s = build("nonexistent_scheduler", num_dc=5, batch_size=8, green_weight=0.6)
    assert isinstance(s, GreenQueueBalancedGlobalScheduler)

def test_weak_baselines_are_green_blind():
    """min_queue/round_robin must ignore green so dQ can be positive for a
    green-aware policy (the whole point of the swap)."""
    obs = {"dc_queue_sizes": [0, 0, 0, 0, 0], "dc_green_ratio": [0.0, 0.0, 0.9, 0.0, 0.0],
           "dc_available_pes": [10]*5}
    mq = build("min_queue", 5, 4, 0.6).schedule(obs)
    rr = build("round_robin", 5, 4, 0.6).schedule(obs)
    # neither should send the whole batch to the green DC (index 2)
    assert mq.count(2) < 4 and rr.count(2) < 4
    ga = build("green_aware", 5, 4, 0.6).schedule(obs)
    assert ga.count(2) >= 1   # green_aware DOES prefer the green DC


# ---------------- policy_self (COMA-style) ----------------
import torch
from ray.rllib.core.columns import Columns
from src.learners.crd_q_loss import _PolicySelfBaselineMarker, COL_CRD_BASELINE_ACTION


def test_policy_self_returns_marker_not_scheduler():
    s = build("policy_self", num_dc=5, batch_size=128, green_weight=0.6)
    assert isinstance(s, _PolicySelfBaselineMarker)
    assert s.num_datacenters == 5 and s.batch_size == 128


def _learner_stub():
    obj = object.__new__(CRDPPOTorchLearner)
    obj._crd_dq_align_warned = {}
    return obj


def test_policy_self_samples_valid_actions():
    """Baseline action must be per-slot indices in [0, n_choices) with the same
    shape as the taken action — the layout every downstream stage expects."""
    lnr = _learner_stub()
    B, T, n_slots, n_choices = 2, 3, 4, 6
    batch = {
        Columns.ACTIONS: torch.zeros(B, T, n_slots, dtype=torch.long),
        Columns.ACTION_DIST_INPUTS: torch.randn(B, T, n_slots * n_choices),
    }
    lnr._compute_policy_self_baseline_action(module_id="m", batch=batch)
    ba = batch[COL_CRD_BASELINE_ACTION]
    assert ba.shape == (B, T, n_slots)
    assert ba.min() >= 0 and ba.max() < n_choices


def test_policy_self_follows_the_policy_distribution():
    """A near-deterministic policy must produce that same action as baseline —
    this is what centres dQ at zero instead of letting a strong heuristic
    baseline dominate."""
    lnr = _learner_stub()
    n_slots, n_choices = 3, 5
    logits = torch.full((1, 1, n_slots, n_choices), -20.0)
    logits[..., 2] = 20.0                      # all mass on choice 2
    batch = {
        Columns.ACTIONS: torch.zeros(1, 1, n_slots, dtype=torch.long),
        Columns.ACTION_DIST_INPUTS: logits.reshape(1, 1, n_slots * n_choices),
    }
    lnr._compute_policy_self_baseline_action(module_id="m", batch=batch)
    assert torch.all(batch[COL_CRD_BASELINE_ACTION] == 2)


def test_policy_self_skips_on_shape_mismatch():
    lnr = _learner_stub()
    batch = {
        Columns.ACTIONS: torch.zeros(1, 1, 4, dtype=torch.long),
        Columns.ACTION_DIST_INPUTS: torch.randn(1, 1, 7),   # 7 % 4 != 0
    }
    lnr._compute_policy_self_baseline_action(module_id="m", batch=batch)
    assert COL_CRD_BASELINE_ACTION not in batch    # skipped, no crash
