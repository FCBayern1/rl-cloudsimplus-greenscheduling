"""T3 重做全知臂的守卫(Codex 批准 2026-08-27)。

匹配对纪律:green_forecast_queue_balanced 必须与 green_queue_balanced
**逐行同逻辑**,唯一差异是绿电信号(当前 vs 未来)。
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.baselines.global_schedulers import (GLOBAL_SCHEDULERS,
                                             GreenQueueBalancedGlobalScheduler)

NAME = "green_forecast_queue_balanced"
ND, B = 5, 16


def obs(cur, fut=None, queues=None):
    o = {"dc_green_ratio": list(cur),
         "dc_queue_sizes": list(queues if queues is not None else [0] * ND),
         "dc_available_pes": [64] * ND}
    if fut is not None:
        o["dc_future_long_mean"] = list(fut)
    return o


def test_registered_and_subclasses_the_proven_capacity_logic():
    cls = GLOBAL_SCHEDULERS[NAME]
    assert issubclass(cls, GreenQueueBalancedGlobalScheduler)


def test_uses_future_signal_not_current():
    """当前绿电与未来绿电指向不同 DC 时,必须跟随未来。"""
    cur = [1.0, 0.0, 0.0, 0.0, 0.0]
    fut = [0.0, 0.0, 0.0, 0.0, 1.0]
    a = GLOBAL_SCHEDULERS[NAME](ND, B).schedule(obs(cur, fut))
    b = GreenQueueBalancedGlobalScheduler(ND, B).schedule(obs(cur))
    assert a.count(4) > a.count(0), "未来最绿的 DC4 应拿到更多"
    assert b.count(0) > b.count(4), "盲态臂应跟随当前最绿的 DC0"


def test_matches_blind_arm_exactly_when_future_equals_current():
    """信号相同 ⇒ 两臂动作**逐位相同**(证明只差信号,不差逻辑)。"""
    cur = [0.9, 0.4, 0.1, 0.0, 0.7]
    a = GLOBAL_SCHEDULERS[NAME](ND, B).schedule(obs(cur, cur))
    b = GreenQueueBalancedGlobalScheduler(ND, B).schedule(obs(cur))
    assert a == b


def test_falls_back_to_current_when_no_forecast_present():
    cur = [0.9, 0.4, 0.1, 0.0, 0.7]
    a = GLOBAL_SCHEDULERS[NAME](ND, B).schedule(obs(cur))       # 无 future 键
    b = GreenQueueBalancedGlobalScheduler(ND, B).schedule(obs(cur))
    assert a == b


def test_inherits_blind_arms_collapse_behaviour_exactly():
    """诚实记录继承性质:green_queue_balanced 的评分是软加权
    (green 0.6 vs queue 0.4),**极端单热信号下它本身也会整批坍缩**。
    匹配对要求逐行同逻辑,所以新臂必须**同样**坍缩 —— 不是缺陷,
    是"唯一差异是信号"的必然结果。此测试锁住这一等价性。"""
    one_hot = [1.0, 0.0, 0.0, 0.0, 0.0]
    a = GLOBAL_SCHEDULERS[NAME](ND, B).schedule(obs([0.2] * ND, one_hot))
    b = GreenQueueBalancedGlobalScheduler(ND, B).schedule(obs(one_hot))
    assert a == b == [0] * B          # 两臂同样坍缩


def test_spreads_under_realistic_mixed_signal():
    """现实中绿电比是混合的,队列项才起作用 —— 两臂都应分散。"""
    mixed = [0.9, 0.8, 0.7, 0.6, 0.5]
    a = GLOBAL_SCHEDULERS[NAME](ND, B).schedule(obs([0.2] * ND, mixed))
    assert len(set(a)) >= 3, f"混合信号下仍不分散: {a}"
    assert max(a.count(d) for d in range(ND)) < B


def test_does_not_mutate_caller_obs():
    o = obs([0.5] * ND, [0.1, 0.2, 0.3, 0.4, 0.5])
    before = dict(o)
    GLOBAL_SCHEDULERS[NAME](ND, B).schedule(o)
    assert o["dc_green_ratio"] == before["dc_green_ratio"]


def test_masks_the_no_green_sentinel_05():
    """实测 bug:无绿电 DC 的 dc_future_long_mean 恒为 0.5(Java 无数据默认值),
    高于真实读数(均值 0.242)⇒ 会把作业吸到零绿电且最脏的 DC。遮罩必须挡住。"""
    sched = GLOBAL_SCHEDULERS[NAME](ND, B)
    o = {"dc_green_ratio": [0.2] * ND,
         "dc_future_long_mean": [0.25, 0.23, 0.24, 0.50, 0.50],   # DC3/4 是哨兵值
         "dc_current_green_power_w": [400.0, 320.0, 120.0, 0.0, 0.0],
         "dc_queue_sizes": [0] * ND, "dc_available_pes": [64] * ND}
    a = sched.schedule(o)
    assert a.count(3) + a.count(4) == 0, f"仍被 0.5 哨兵吸走: {a}"
    assert set(a) <= {0, 1, 2}


def test_mask_is_learned_across_steps_and_cleared_on_reset():
    sched = GLOBAL_SCHEDULERS[NAME](ND, B)
    o = {"dc_green_ratio": [0.2] * ND,
         "dc_future_long_mean": [0.25, 0.23, 0.24, 0.50, 0.50],
         "dc_current_green_power_w": [0.0, 320.0, 0.0, 0.0, 0.0],   # 本步只有 DC1 有绿
         "dc_queue_sizes": [0] * ND, "dc_available_pes": [64] * ND}
    sched.schedule(o)
    o2 = dict(o); o2["dc_current_green_power_w"] = [400.0, 0.0, 0.0, 0.0, 0.0]
    sched.schedule(o2)                       # DC0 也被记为有绿电能力
    assert list(sched._green_capable) == [True, True, False, False, False]
    sched.reset()
    assert sched._green_capable is None
