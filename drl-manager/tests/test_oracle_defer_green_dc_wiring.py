"""The two defer oracles must route/gate on the derived green DCs, not [0, 1, 2].

Drives ``oracle_diurnal_defer.run`` and ``oracle_baselines_defer.run`` against a
fake env (no Java gateway) built on the 8-DC topology, and asserts that DC5 —
the non-contiguous wind site — is actually used.

Run from repo root:
    cd drl-manager && python -m pytest tests/test_oracle_defer_green_dc_wiring.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

oracle_diurnal_defer = pytest.importorskip("oracle_diurnal_defer")
oracle_baselines_defer = pytest.importorskip("oracle_baselines_defer")

NUM_DC = 8
GREEN_COLS = [0, 1, 2, 5]        # the 8-DC topology's wind sites
BATCH = 4
STEPS = 6


def dc(name, turbines, enabled=True):
    return {"name": name, "turbine_ids": turbines, "green_energy_enabled": enabled}


DC8_CONFIGS = [
    dc("DC_Nordic", [7012, 7036]),
    dc("DC_Germany", [7095, 7091]),
    dc("DC_US_East", [7096]),
    dc("DC_US_West", [], enabled=False),
    dc("DC_APAC", [], enabled=False),
    dc("DC_Nordic2", [7101, 7103]),
    dc("DC_EU_South", [], enabled=False),
    dc("DC_SEA", [], enabled=False),
]


class FakeEnv:
    """Minimal stand-in for HierarchicalMultiDCEnv.

    Only DC5 has green power, so any policy that gates or routes on a hardcoded
    [0, 1, 2] sees a permanently dark world and never touches the real wind site.
    """

    def __init__(self, green_dc, dc_configs=DC8_CONFIGS, num_dc=NUM_DC):
        self.dc_configs = dc_configs
        self.num_datacenters = num_dc
        self.global_routing_batch_size = BATCH
        self.green_dc = green_dc
        self.global_actions = []
        self.local_actions = []
        self._t = 0

    def _obs(self):
        green = np.zeros(self.num_datacenters)
        green[self.green_dc] = 1000.0
        return {
            "global": {
                "dc_current_green_power_w": green,
                "dc_current_power_w": np.full(self.num_datacenters, 100.0),
                "dc_available_pes": np.arange(self.num_datacenters, dtype=float),
            }
        }

    def reset(self, seed=None):
        self._t = 0
        self.global_actions = []
        self.local_actions = []
        return self._obs(), {}

    def step(self, action):
        self.global_actions.append(list(action["global"]))
        self.local_actions.append(dict(action["local"]))
        self._t += 1
        return self._obs(), 0.0, self._t >= STEPS, False, {}

    def close(self):
        pass


@pytest.fixture(autouse=True)
def stub_collect_metrics(monkeypatch):
    """Both oracles end with collect_metrics(info, num_dc); info is empty here."""
    stub = lambda info, num_dc: {
        "total_carbon_kg": 0.0,
        "green_used_wh": 0.0,
        "green_waste_wh": 0.0,
        "waste_ratio": 0.0,
        "completion_rate_mi": 0.0,
    }
    monkeypatch.setattr(oracle_diurnal_defer, "collect_metrics", stub)
    monkeypatch.setattr(oracle_baselines_defer, "collect_metrics", stub)


# --- oracle_diurnal_defer ----------------------------------------------------

def test_diurnal_roundrobin_routes_to_the_non_contiguous_green_dc():
    env = FakeEnv(green_dc=5)
    oracle_diurnal_defer.run(env, defer=False, lever="global", seed=0, drain=8,
                             route_thresh_w=1.0, routing="roundrobin")
    routed = {d for step in env.global_actions for d in step}
    assert routed == {0, 1, 2, 5}, f"round-robin used DCs {sorted(routed)}"


def test_diurnal_leastloaded_can_pick_dc5():
    """dc_available_pes rises with index, so the greenest-and-freest pick is DC5."""
    env = FakeEnv(green_dc=5)
    oracle_diurnal_defer.run(env, defer=False, lever="global", seed=0, drain=8,
                             route_thresh_w=1.0, routing="leastloaded")
    routed = {d for step in env.global_actions for d in step}
    assert routed == {5}, f"least-loaded green pick was {sorted(routed)}, expected DC5"


def test_diurnal_light_gate_sees_green_at_dc5_and_does_not_defer():
    """With green only at DC5, a [0,1,2] sum reads 0 W and would DEFER every step."""
    env = FakeEnv(green_dc=5)
    m = oracle_diurnal_defer.run(env, defer=True, lever="global", seed=0, drain=8,
                                 route_thresh_w=500.0, routing="roundrobin")
    assert m["_hold_decisions"] == 0
    assert all(env.num_datacenters not in step for step in env.global_actions), \
        "DEFER action emitted despite DC5 being windy"


def test_diurnal_defers_when_the_only_green_dc_is_dark():
    """Control: below threshold everywhere -> the gate must still fire."""
    env = FakeEnv(green_dc=5)
    m = oracle_diurnal_defer.run(env, defer=True, lever="global", seed=0, drain=8,
                                 route_thresh_w=5000.0, routing="roundrobin")
    assert m["_hold_decisions"] == STEPS * BATCH
    assert all(step == [env.num_datacenters] * BATCH for step in env.global_actions)


def test_diurnal_explicit_green_dcs_argument_wins():
    env = FakeEnv(green_dc=5)
    oracle_diurnal_defer.run(env, defer=False, lever="global", seed=0, drain=8,
                             route_thresh_w=1.0, routing="roundrobin", green_dcs=[6, 7])
    routed = {d for step in env.global_actions for d in step}
    assert routed == {6, 7}


def test_diurnal_five_dc_topology_behaviour_is_unchanged():
    env = FakeEnv(green_dc=0, dc_configs=DC8_CONFIGS[:5], num_dc=5)
    oracle_diurnal_defer.run(env, defer=False, lever="global", seed=0, drain=8,
                             route_thresh_w=1.0, routing="roundrobin")
    routed = {d for step in env.global_actions for d in step}
    assert routed == {0, 1, 2}


# --- oracle_baselines_defer --------------------------------------------------

def test_baselines_light_gate_sees_green_at_dc5():
    env = FakeEnv(green_dc=5)
    m = oracle_baselines_defer.run(env, "round_robin", defer=True, seed=0, drain=8,
                                   thresh=500.0)
    assert m["_defer"] == 0, "defer fired although the wind site DC5 was above threshold"


def test_baselines_defers_when_dc5_is_below_threshold():
    env = FakeEnv(green_dc=5)
    m = oracle_baselines_defer.run(env, "round_robin", defer=True, seed=0, drain=8,
                                   thresh=5000.0)
    assert m["_defer"] == STEPS * BATCH


def test_baselines_explicit_green_dcs_argument_wins():
    """Restricting the gate to brown DCs makes the world read as permanently dark."""
    env = FakeEnv(green_dc=5)
    m = oracle_baselines_defer.run(env, "round_robin", defer=True, seed=0, drain=8,
                                   thresh=500.0, green_dcs=[3, 4])
    assert m["_defer"] == STEPS * BATCH


def test_baselines_day_night_split_tracks_the_derived_green_dcs():
    env = FakeEnv(green_dc=5)
    m = oracle_baselines_defer.run(env, "round_robin", defer=False, seed=0, drain=8,
                                   thresh=500.0)
    assert m["_day_W"] > 0, "every step should be classified as 'day' (DC5 windy)"
    assert m["_night_W"] == 0
