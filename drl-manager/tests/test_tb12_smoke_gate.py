"""tb12_smoke_gate 纯判定函数测试(Codex 四门口径)+ cap 硬止损单测。"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tb12_smoke_gate import (gate1_reward_physics, gate2_sla, gate3_collapse,
                             gate4_information)


def test_gate1_reward_up_kg_down_passes():
    ok, d = gate1_reward_physics({"reward": -100, "kg": 1.0},
                                 {"reward": -80, "kg": 0.9}, [0, 0])
    assert ok and d["reward_up"] and d["kg_down"]


def test_gate1_reward_up_kg_up_is_the_v1_accident_and_stops():
    # v1 事故指纹:回报改善但物理 kg 不降 -> STOP
    ok, _ = gate1_reward_physics({"reward": -100, "kg": 1.0},
                                 {"reward": -80, "kg": 1.0}, [0, 0])
    assert not ok


def test_gate1_any_cap_hit_fails_even_if_direction_ok():
    ok, d = gate1_reward_physics({"reward": -100, "kg": 1.0},
                                 {"reward": -80, "kg": 0.9}, [0, 1])
    assert not ok and not d["cap_all_zero"]


def test_gate1_reward_flat_or_down_does_not_require_kg_down():
    ok, _ = gate1_reward_physics({"reward": -100, "kg": 1.0},
                                 {"reward": -120, "kg": 1.1}, [0])
    assert ok  # 门只约束"改善必须真实",不要求 50k 一定改善


def test_gate2_both_arms_must_clear_target():
    assert gate2_sla(0.996, 0.999)[0]
    assert not gate2_sla(0.996, 0.99)[0]


def test_gate3_flags_full_defer_collapse():
    good = {"offset": 1, "arm": "fc", "deadline_forced": 0,
            "active_releases": 5, "defer_frac": 0.3}
    collapsed = {"offset": 2, "arm": "fc", "deadline_forced": 5,
                 "active_releases": 0, "defer_frac": 1.0}
    ok, d = gate3_collapse([good, collapsed])
    assert not ok and len(d["violations"]) == 1
    assert gate3_collapse([good])[0]


def test_gate4_requires_measurable_difference_only():
    assert gate4_information({1: 1.0, 2: 2.0}, {1: 1.0, 2: 2.1})[0]
    assert not gate4_information({1: 1.0}, {1: 1.0})[0]


def test_cap_hard_stop_raises_only_when_enabled():
    from src.callbacks.rllib_green_energy_logger import GreenEnergyLoggerCallback
    stats = {"global_carbon_cap_count": 2, "global_carbon_max_ratio": 4.2}
    with pytest.raises(RuntimeError, match="CARBON CAP HIT"):
        GreenEnergyLoggerCallback.check_carbon_cap(stats, True)
    # 未启用 -> 只读数不抛(legacy 行为不变)
    assert GreenEnergyLoggerCallback.check_carbon_cap(stats, False) == (2, 4.2)
    assert GreenEnergyLoggerCallback.check_carbon_cap({}, True) == (0, 0.0)
