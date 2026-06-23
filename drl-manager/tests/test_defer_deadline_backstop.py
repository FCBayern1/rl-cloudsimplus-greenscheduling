"""Integration test for Fix A — the defer deadline backstop.

Bug: a global DEFER action only requeued the cloudlet to the tail with NO deadline
enforcement, so a deterministic 'always-defer' policy could defer work forever →
starvation (godeye collapsed to 22% completion on the het trace).

Fix A: when a deferred cloudlet's deadline is within defer_deadline_slack_sec of the
clock, force-route it to the greenest available DC instead of deferring again.

This test drives an ALWAYS-DEFER global policy (with always-drain local agents) on a
tiny short-deadline trace and asserts:
  - force ON  → the backstop fires (deadline_forced_count > 0) and work completes;
  - force OFF → nothing is forced and (almost) nothing completes (all deferred forever).

Requires the Java gateway (gradlew build/launch) so it is marked slow/integration.
"""
import os
import sys

import numpy as np
import pytest
import yaml

DRL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if DRL not in sys.path:
    sys.path.insert(0, DRL)

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv

CONFIG = os.path.join(DRL, "..", "config.yml")
EXPERIMENT = "experiment_multi_5dc_carbon_v2_deferrable_gdpd"
N_STEPS = 260  # first force-route fires ~step (deadline 120 - slack 25); 260 is comfortably past it


def _make_env(force_enabled: bool):
    cfg = yaml.safe_load(open(CONFIG))[EXPERIMENT]
    cfg = dict(cfg)
    cfg["cloudlet_trace_file"] = "traces/test_backstop_tiny.csv"
    cfg["max_cloudlets_to_create_from_workload_file"] = 300
    cfg["defer_deadline_force_enabled"] = force_enabled
    cfg["defer_deadline_slack_sec"] = 25.0
    cfg.pop("py4j_port", None)
    os.makedirs("/tmp/backstop_gateway", exist_ok=True)
    cfg.setdefault("gateway_log_dir", "/tmp/backstop_gateway")
    cfg.setdefault("output_dir", "/tmp/backstop_gateway")
    return HierarchicalMultiDCParallelEnv(config=cfg)


def _run_all_defer(env):
    """Step an always-defer (global) / always-drain (local) policy; return last info."""
    base = env.base_env
    defer_idx = base.num_datacenters          # DEFER action index
    batch = base.global_routing_batch_size
    obs, info = env.reset(seed=42)
    last_global_info = {}
    for _ in range(N_STEPS):
        actions = {}
        for agent in env.agents:
            if agent == "global_agent":
                actions[agent] = np.full(batch, defer_idx, dtype=np.int64)
            else:
                actions[agent] = 64  # drain: dispatch as much as possible
        obs, rewards, terms, truncs, infos = env.step(actions)
        gi = infos.get("global_agent", {}) if isinstance(infos, dict) else {}
        # The Java per-step stats live in the nested `global_energy_stats` dict.
        stats = gi.get("global_energy_stats") if isinstance(gi, dict) else None
        if stats:
            last_global_info = stats
        if (isinstance(terms, dict) and all(terms.values())) or \
           (isinstance(truncs, dict) and all(truncs.values())):
            break
    return last_global_info


def _get(info, key, default=0.0):
    v = info.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


@pytest.mark.slow
def test_backstop_on_forces_and_completes():
    env = _make_env(force_enabled=True)
    try:
        info = _run_all_defer(env)
    finally:
        env.close()
    forced = _get(info, "deadline_forced_count")
    finished = _get(info, "total_finished_cloudlets")
    assert forced > 0, f"backstop should force-route deferred work, got deadline_forced_count={forced}"
    assert finished > 0, f"forced work should complete, got total_finished_cloudlets={finished}"


@pytest.mark.slow
def test_backstop_off_defers_forever():
    env = _make_env(force_enabled=False)
    try:
        info = _run_all_defer(env)
    finally:
        env.close()
    forced = _get(info, "deadline_forced_count")
    finished = _get(info, "total_finished_cloudlets")
    assert forced == 0, f"with backstop OFF nothing should be forced, got {forced}"
    # all work stays deferred in the global queue → essentially nothing finishes
    assert finished < 5, f"with backstop OFF an always-defer policy should starve, got finished={finished}"
