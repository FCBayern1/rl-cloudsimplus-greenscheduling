import pickle

import numpy as np
import pytest

import gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo as pz_module
from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import (
    HierarchicalMultiDCParallelEnv,
    _validate_fixed_local_scheduler,
)
from src.baselines.evaluate import assert_fixed_drain_evaluation_compatible
from src.training.train_rlmodule_gtrxl import select_policies_to_train


class _MockBaseEnv:
    def __init__(self):
        self.config = {"local_dispatch_mode": "dispatch_rate"}
        self.last_action = None

    def get_local_action_masks(self, dc_index):
        # The largest legal dispatch is intentionally not always the final index.
        return (
            np.array([1, 1, 0, 1], dtype=bool)
            if dc_index == 0
            else np.array([1, 0, 1, 0], dtype=bool)
        )

    def step(self, action):
        self.last_action = action
        return (
            {"global": {}, "local": {}},
            {"global": 1.0, "local": {0: 2.0, 1: 3.0}},
            False,
            False,
            {},
        )


def _mock_wrapper(mode):
    env = object.__new__(HierarchicalMultiDCParallelEnv)
    env.config = {
        "fixed_local_scheduler": mode,
        "local_dispatch_mode": "dispatch_rate",
    }
    env.fixed_local_scheduler = mode
    env.base_env = _MockBaseEnv()
    env.num_datacenters = 2
    env.agents = ["global_agent", "local_agent_0", "local_agent_1"]
    env._lagrangian_enabled = False
    env._lagrangian_cfg = {}
    env._ep_c_step_running_sum = 0.0
    env._ep_c_step_running_count = 0
    env._ep_lagrangian_penalty_sum = 0.0
    env._hierarchical_to_flat_observations = (
        lambda observation, crd_info=None: {agent: {} for agent in env.agents}
    )
    return env


def test_drain_overrides_every_local_action_with_mask_maximum():
    env = _mock_wrapper("drain")
    env.step({
        "global_agent": [0, 1],
        "local_agent_0": 0,
        "local_agent_1": 3,
    })
    assert env.base_env.last_action["local"] == {0: 3, 1: 2}


def test_none_mode_preserves_local_actions_exactly():
    env = _mock_wrapper("none")
    original = {
        "global": [0, 1],
        "local": {0: 1, 1: 3},
    }
    assert env._apply_fixed_local_scheduler(original) is original

    env.step({
        "global_agent": [0, 1],
        "local_agent_0": 1,
        "local_agent_1": 3,
    })
    assert env.base_env.last_action["local"] == {0: 1, 1: 3}


def test_drain_vm_placement_fails_before_base_env_construction(monkeypatch):
    def must_not_construct(*args, **kwargs):
        raise AssertionError("base env/Java construction must not be reached")

    monkeypatch.setattr(pz_module, "HierarchicalMultiDCEnv", must_not_construct)
    with pytest.raises(ValueError, match="requires.*dispatch_rate"):
        HierarchicalMultiDCParallelEnv({
            "fixed_local_scheduler": "drain",
            "local_dispatch_mode": "vm_placement",
        })


def test_drain_trains_only_global_policy_and_default_trains_all():
    policies = {"global_policy", "shared_local_policy"}
    assert select_policies_to_train(policies, "drain") == ["global_policy"]
    assert set(select_policies_to_train(policies, "none")) == policies


def test_fixed_drain_checkpoint_rejects_rllib_local_but_accepts_drain(tmp_path):
    checkpoint = tmp_path / "checkpoint_000001"
    checkpoint.mkdir()
    payload = {
        "ctor_args_and_kwargs": (({
            "env_config": {"fixed_local_scheduler": "drain"}
        },), {})
    }
    with (checkpoint / "class_and_ctor_args.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="--local drain"):
        assert_fixed_drain_evaluation_compatible(
            str(checkpoint), "rllib", config={}
        )
    # Legal certification path: global remains RLlib while local_override is
    # drain. Both the checkpoint and requested experiment say fixed-drain; the
    # guard must return before considering requested_mode.
    assert_fixed_drain_evaluation_compatible(
        str(checkpoint), "drain",
        config={"fixed_local_scheduler": "drain"},
    )

    legacy_checkpoint = tmp_path / "checkpoint_legacy"
    legacy_checkpoint.mkdir()
    assert_fixed_drain_evaluation_compatible(
        str(legacy_checkpoint), "rllib", config={}
    )


def test_mock_drain_runs_100_steps_without_gateway():
    env = _mock_wrapper("drain")
    for _ in range(100):
        observations, rewards, terminated, truncated, infos = env.step({
            "global_agent": [0, 1],
            "local_agent_0": 0,
            "local_agent_1": 0,
        })
        assert not any(terminated.values())
        assert not any(truncated.values())
        assert env.base_env.last_action["local"] == {0: 3, 1: 2}
        assert set(observations) == set(rewards) == set(infos)


def test_validation_defaults_to_none():
    assert _validate_fixed_local_scheduler({}) == "none"
