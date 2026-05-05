"""
M0.3 verification: confirm baseline schedulers' signatures and output shapes
are compatible with what the CRD callback (M2) will need.

Run from repo root:
    cd drl-manager && python -m pytest tests/test_crd_baseline_schedulers.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.global_schedulers import GreenQueueBalancedGlobalScheduler
from src.baselines.local_schedulers import BestFitLocalScheduler


def test_green_queue_balanced_signature_and_shape():
    """schedule(global_obs) -> List[int] of length batch_size, values in [0, num_dc)."""
    num_dc = 3
    batch_size = 5
    sched = GreenQueueBalancedGlobalScheduler(num_datacenters=num_dc, batch_size=batch_size)

    global_obs = {
        "dc_green_ratio": np.array([0.8, 0.2, 0.5], dtype=np.float32),
        "dc_queue_sizes": np.array([2, 1, 0], dtype=np.int32),
    }

    actions = sched.schedule(global_obs)

    assert isinstance(actions, list), "schedule must return a list"
    assert len(actions) == batch_size, f"len={len(actions)} != batch_size={batch_size}"
    assert all(isinstance(a, int) for a in actions), "all actions must be int"
    assert all(0 <= a < num_dc for a in actions), f"actions out of range: {actions}"


def test_best_fit_local_signature_and_shape():
    """schedule(local_obs, action_mask) -> int in [0, len(mask))."""
    sched = BestFitLocalScheduler(num_vms=4)

    num_vms = 4
    # action_mask[0] is "no-op" / "wait", indices 1..num_vms map to VMs.
    action_mask = np.array([1, 1, 1, 0, 1], dtype=np.int8)
    local_obs = {
        "vm_available_pes": np.array([4, 8, 2, 1], dtype=np.int32),
        "next_cloudlet_pes": 2,
    }

    action = sched.schedule(local_obs, action_mask)

    assert isinstance(action, int), f"action must be int, got {type(action)}"
    assert 0 <= action < len(action_mask), f"action {action} out of range"


def test_baseline_outputs_are_deterministic():
    """Repeated calls with the same obs must produce the same action (no RNG state)."""
    sched = GreenQueueBalancedGlobalScheduler(num_datacenters=3, batch_size=4)
    obs = {
        "dc_green_ratio": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "dc_queue_sizes": np.array([1, 1, 1], dtype=np.int32),
    }
    a1 = sched.schedule(obs)
    a2 = sched.schedule(obs)
    assert a1 == a2, "GreenQueueBalanced must be deterministic for fixed obs"

    local = BestFitLocalScheduler(num_vms=3)
    mask = np.array([1, 1, 1, 1], dtype=np.int8)
    lobs = {"vm_available_pes": [4, 8, 2], "next_cloudlet_pes": 2}
    assert local.schedule(lobs, mask) == local.schedule(lobs, mask)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
