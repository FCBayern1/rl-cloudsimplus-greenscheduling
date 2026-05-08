"""Tests for the per-decision latency instrumentation in evaluate.py.

Covers:
- `_summarize_decision_latency` reduction (mean/p50/p95/p99) and edge cases.
- `run_evaluation` end-to-end emits the timing fields (with a mocked env and
  schedulers that sleep a known duration), and the recorded latency dominates
  the injected sleep.
"""

import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.evaluate import _summarize_decision_latency, run_evaluation


def test_summarize_empty_returns_zero_count():
    out = _summarize_decision_latency([], "x")
    assert out["x_count"] == 0
    assert out["x_us_mean"] == 0.0
    assert out["x_us_p99"] == 0.0


def test_summarize_units_and_ordering():
    # 10 samples in nanoseconds: 1us, 2us, ..., 10us
    samples_ns = [i * 1000 for i in range(1, 11)]
    out = _summarize_decision_latency(samples_ns, "g")
    assert out["g_count"] == 10
    assert out["g_us_mean"] == 5.5
    assert out["g_us_p50"] <= out["g_us_p95"] <= out["g_us_p99"]
    assert 9.5 <= out["g_us_p99"] <= 10.0


def test_run_evaluation_emits_timing_fields():
    """End-to-end: env stops after 2 steps, schedulers sleep ~50us per call.

    We mock the env class itself (HierarchicalMultiDCEnv) so no Java gateway is
    needed. We also stub out the scheduler factories via the GLOBAL_SCHEDULERS /
    LOCAL_SCHEDULERS dicts the function reads.
    """
    sleep_s = 50e-6  # 50us — well above clock resolution on Linux
    num_dcs = 3

    # --- Fake env -----------------------------------------------------------
    env = MagicMock()
    env.num_datacenters = num_dcs
    env.global_routing_batch_size = 5
    env.max_vms = 4
    env.get_local_action_masks.return_value = np.ones(4, dtype=np.float32)

    obs = {"global": {}, "local": {i: {} for i in range(num_dcs)}}
    env.reset.return_value = (obs, {})

    step_counter = {"n": 0}

    def fake_step(action):
        step_counter["n"] += 1
        terminated = step_counter["n"] >= 2
        return obs, {}, terminated, False, {"any": "info"}

    env.step.side_effect = fake_step
    env.close.return_value = None

    # --- Fake schedulers ----------------------------------------------------
    class SleepyGlobal:
        def __init__(self, num_dcs, batch_size):
            pass
        def reset(self):
            pass
        def schedule(self, obs):
            time.sleep(sleep_s)
            return np.zeros(5, dtype=np.int64)

    class SleepyLocal:
        def __init__(self, max_vms):
            pass
        def reset(self):
            pass
        def schedule(self, obs, mask):
            time.sleep(sleep_s)
            return 0

    with patch("src.baselines.evaluate.HierarchicalMultiDCEnv", return_value=env), \
         patch("src.baselines.evaluate.GLOBAL_SCHEDULERS", {"sleepy": SleepyGlobal}), \
         patch("src.baselines.evaluate.LOCAL_SCHEDULERS",  {"sleepy": SleepyLocal}), \
         patch("src.baselines.evaluate.collect_metrics", return_value={"routed_rate": 1.0}):
        results = run_evaluation(
            global_scheduler_name="sleepy",
            local_scheduler_name="sleepy",
            config={"env_id": "HierarchicalMultiDC-v0"},
            num_episodes=1,
            verbose=False,
        )

    assert len(results) == 1
    r = results[0]
    # 2 env steps -> 2 global decisions, 2*num_dcs local decisions
    assert r["global_decision_count"] == 2
    assert r["local_decision_count"] == 2 * num_dcs
    # Recorded latency should be >= the injected sleep (50us); allow generous
    # upper bound for CI jitter.
    assert r["global_decision_us_mean"] >= 40.0
    assert r["local_decision_us_mean"]  >= 40.0
    # p99 >= p50 sanity
    assert r["global_decision_us_p99"] >= r["global_decision_us_p50"]
    assert r["local_decision_us_p99"]  >= r["local_decision_us_p50"]


def test_force_full_episode_ignores_terminated():
    """When force_full_episode=True, env returning terminated=True early must
    NOT end the episode — only truncated should. This is the fairness
    mechanism for cross-algorithm carbon comparison."""
    num_dcs = 2
    cap_steps = 5  # truncated fires at this step

    env = MagicMock()
    env.num_datacenters = num_dcs
    env.global_routing_batch_size = 3
    env.max_vms = 4
    env.get_local_action_masks.return_value = np.ones(4, dtype=np.float32)
    obs = {"global": {}, "local": {i: {} for i in range(num_dcs)}}
    env.reset.return_value = (obs, {})

    counter = {"n": 0}

    def fake_step(action):
        counter["n"] += 1
        # Env reports terminated=True from step 2 onwards (workload drained)
        terminated = counter["n"] >= 2
        # Truncated only fires at the cap
        truncated = counter["n"] >= cap_steps
        return obs, {}, terminated, truncated, {}

    env.step.side_effect = fake_step
    env.close.return_value = None

    class _Sched:
        def __init__(self, *a, **k): pass
        def reset(self): pass
        def schedule(self, *a, **k): return 0

    with patch("src.baselines.evaluate.HierarchicalMultiDCEnv", return_value=env), \
         patch("src.baselines.evaluate.GLOBAL_SCHEDULERS", {"s": _Sched}), \
         patch("src.baselines.evaluate.LOCAL_SCHEDULERS",  {"s": _Sched}), \
         patch("src.baselines.evaluate.collect_metrics", return_value={"routed_rate": 1.0}):
        results = run_evaluation(
            global_scheduler_name="s", local_scheduler_name="s",
            config={"env_id": "HierarchicalMultiDC-v0"},
            num_episodes=1, verbose=False,
            force_full_episode=True,
        )

    # Episode must run to truncation (5 steps), not terminate at step 2.
    assert results[0]["episode_length"] == cap_steps


def test_default_behavior_respects_terminated():
    """Inverse case: with the flag off (default), terminated=True should
    end the episode immediately."""
    num_dcs = 2
    env = MagicMock()
    env.num_datacenters = num_dcs
    env.global_routing_batch_size = 3
    env.max_vms = 4
    env.get_local_action_masks.return_value = np.ones(4, dtype=np.float32)
    obs = {"global": {}, "local": {i: {} for i in range(num_dcs)}}
    env.reset.return_value = (obs, {})

    counter = {"n": 0}

    def fake_step(action):
        counter["n"] += 1
        return obs, {}, counter["n"] >= 2, False, {}

    env.step.side_effect = fake_step
    env.close.return_value = None

    class _Sched:
        def __init__(self, *a, **k): pass
        def reset(self): pass
        def schedule(self, *a, **k): return 0

    with patch("src.baselines.evaluate.HierarchicalMultiDCEnv", return_value=env), \
         patch("src.baselines.evaluate.GLOBAL_SCHEDULERS", {"s": _Sched}), \
         patch("src.baselines.evaluate.LOCAL_SCHEDULERS",  {"s": _Sched}), \
         patch("src.baselines.evaluate.collect_metrics", return_value={"routed_rate": 1.0}):
        results = run_evaluation(
            global_scheduler_name="s", local_scheduler_name="s",
            config={"env_id": "HierarchicalMultiDC-v0"},
            num_episodes=1, verbose=False,
        )
    assert results[0]["episode_length"] == 2


if __name__ == "__main__":
    test_summarize_empty_returns_zero_count()
    test_summarize_units_and_ordering()
    test_run_evaluation_emits_timing_fields()
    test_force_full_episode_ignores_terminated()
    test_default_behavior_respects_terminated()
    print("OK")
