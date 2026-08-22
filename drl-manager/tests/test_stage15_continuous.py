"""Stage 1.5 流体积分与协调界的不变量(Codex P0 复核指出此前无单测)。"""
import numpy as np
import pytest

from stage15_continuous import (C_BROWN, C_GREEN, ROW_S, W_PER_PE,
                                coordinated_bound, fluid_carbon)


def _job(mi, pes, rt, arr, slack):
    return (mi, pes, rt, arr, slack, 0)


class TestFluidCarbon:
    def test_all_green_when_supply_covers_demand(self):
        jobs = [_job(1e6, 1, 600.0, 0.0, 0.0)]
        watts = np.full(10, 100.0)               # G >> D
        c, brown, green = fluid_carbon(jobs, [0.0], watts, 1.0)
        assert brown == 0.0
        assert green == pytest.approx(W_PER_PE * 600.0)
        assert c == pytest.approx(C_GREEN * green)

    def test_all_brown_when_no_green(self):
        jobs = [_job(1e6, 2, 1200.0, 0.0, 0.0)]
        watts = np.zeros(10)
        c, brown, green = fluid_carbon(jobs, [0.0], watts, 1.0)
        assert green == 0.0
        assert brown == pytest.approx(2 * W_PER_PE * 1200.0)

    def test_shared_green_saturates(self):
        """两个作业同时跑,绿电只够一个 -> 一半能量是棕的(自拥挤被计入)。"""
        jobs = [_job(1e6, 1, 600.0, 0.0, 0.0), _job(1e6, 1, 600.0, 0.0, 0.0)]
        watts = np.full(10, W_PER_PE)            # 恰好一份
        _, brown, green = fluid_carbon(jobs, [0.0, 0.0], watts, 1.0)
        assert green == pytest.approx(W_PER_PE * 600.0)
        assert brown == pytest.approx(W_PER_PE * 600.0)

    def test_partial_row_coverage(self):
        jobs = [_job(1e6, 1, 300.0, 150.0, 0.0)]  # 覆盖 0.25+0.25 行
        watts = np.zeros(10)
        _, brown, _ = fluid_carbon(jobs, [150.0], watts, 1.0)
        assert brown == pytest.approx(W_PER_PE * 300.0)


class TestCoordinatedBound:
    def test_bound_never_exceeds_any_schedule(self):
        """协调界的碳 <= 任何具体释放方案的碳(它是下界)。"""
        rng = np.random.default_rng(7)
        jobs = [_job(1e6, 1, 1800.0, float(rng.uniform(0, 3000)), 6000.0)
                for _ in range(20)]
        watts = rng.uniform(0, 5 * W_PER_PE, 40)
        c_bound, _ = coordinated_bound(jobs, watts, 1.0)
        for trial in range(5):
            rel = [j[3] + float(rng.uniform(0, j[4])) for j in jobs]
            c, _, _ = fluid_carbon(jobs, rel, watts, 1.0)
            assert c_bound <= c + 1e-6

    def test_absorption_capped_by_rate(self):
        """单作业速率封顶:绿电再多,吸收也不超过 pes x W x rt。"""
        jobs = [_job(1e6, 1, 600.0, 0.0, 6000.0)]
        watts = np.full(20, 1000.0)
        _, gfrac = coordinated_bound(jobs, watts, 1.0)
        assert gfrac == pytest.approx(1.0)

    def test_window_constraint_respected(self):
        """绿电只出现在窗口之外 -> 吸收为 0。"""
        jobs = [_job(1e6, 1, 600.0, 0.0, 600.0)]   # 窗 [0, 1800]
        watts = np.zeros(20); watts[10:] = 1000.0  # 绿电从 6000s 起
        _, gfrac = coordinated_bound(jobs, watts, 1.0)
        assert gfrac == 0.0
