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


class TestCoordinatedContiguous:
    def _mk(self, n=6):
        import numpy as np
        rng = np.random.default_rng(3)
        return [(1e6, 1, 1200.0, float(rng.uniform(0, 2000)), 8000.0, 0)
                for _ in range(n)]

    def test_blind_is_causal_by_construction(self):
        """改动 cutoff 之后的 G 不得以任何方向改变 cutoff 之前的决定。

        Codex 复核指出旧版只查"原轨迹已在截止点前释放"的作业 —— 若改未来
        让某个原本晚释放的作业反向跑到截止点之前,旧断言抓不到。现在对称:
        "是否在截止点前释放"这个谓词本身必须逐作业一致,一致者时刻逐位相同。
        双向各测一次(未来变富 / 变穷)。"""
        import numpy as np
        from stage15_continuous import coordinated_blind_contig
        jobs = self._mk()
        cutoff = 20 * 600
        w1 = np.full(40, 3.0)
        for future in (100.0, 0.0):
            w2 = w1.copy(); w2[20:] = future
            r1 = coordinated_blind_contig(jobs, w1, 1.0)
            r2 = coordinated_blind_contig(jobs, w2, 1.0)
            for a, b in zip(r1, r2):
                assert (a < cutoff) == (b < cutoff),                     "未来 G 改变了'是否在截止点前释放'"
                if a < cutoff:
                    assert a == b, "未来 G 泄漏进早期释放时刻"

    def test_no_job_released_before_arrival_or_after_latest(self):
        import numpy as np
        from stage15_continuous import (coordinated_blind_contig,
                                        coordinated_clair_contig)
        jobs = self._mk()
        w = np.full(40, 2.0)
        for f in (coordinated_blind_contig, coordinated_clair_contig):
            for j, rel in zip(jobs, f(jobs, w, 1.0)):
                assert j[3] - 1e-6 <= rel <= j[3] + j[4] + 600 + 1e-6

    def test_clair_skips_the_decoy_window(self):
        """确定性用例:短窗诱饵 + 稍后的大窗。因果盲(绿电跟随)会在第一个
        有空余绿电的行放行,掉进 2 行的短窗;clair 看得到 10 行后的大窗,
        整块绿电跑完。这是"信息价值 = 连续性下的窗口选择"的最小实例。

        (白噪声玩具风上不做断言:贪心插入 clair 在无时间结构的供给上
        packing 可能差于绿电跟随,实测发生 —— 这也意味着真实数据上量到的
        信息价值可能被低估,方向已在简报声明。)"""
        import numpy as np
        from stage15_continuous import (coordinated_blind_contig,
                                        coordinated_clair_contig,
                                        fluid_carbon)
        from stage15_continuous import W_PER_PE
        rt = 6000.0                                # 10 行
        jobs = [(1e6, 1, rt, 0.0, 12000.0, 0)]
        w = np.zeros(40)
        w[0:2] = 2 * W_PER_PE                      # 诱饵:只够 2 行
        w[10:22] = 2 * W_PER_PE                    # 大窗:装得下整个作业
        rb = coordinated_blind_contig(jobs, w, 1.0)
        rc = coordinated_clair_contig(jobs, w, 1.0)
        assert rb[0] < 600.0, "因果盲应掉进诱饵窗"
        assert 5400.0 <= rc[0] <= 6600.0, "clair 应等大窗"
        cb = fluid_carbon(jobs, rb, w, 1.0)[0]
        cc = fluid_carbon(jobs, rc, w, 1.0)[0]
        assert cc < cb

    def test_relaxed_bound_dominates_both(self):
        import numpy as np
        from stage15_continuous import (coordinated_blind_contig,
                                        coordinated_clair_contig,
                                        coordinated_bound, fluid_carbon)
        jobs = self._mk(10)
        rng = np.random.default_rng(13)
        w = rng.uniform(0, 6.0, 60)
        bound = coordinated_bound(jobs, w, 1.0)[0]
        for f in (coordinated_blind_contig, coordinated_clair_contig):
            assert bound <= fluid_carbon(jobs, f(jobs, w, 1.0), w, 1.0)[0] + 1e-6


class TestOnlineArrivalClair:
    """P0-1:clairvoyant 可读未来风,不可读未来作业。"""

    def test_future_job_cannot_change_earlier_commitments(self):
        import numpy as np
        from stage15_continuous import coordinated_clair_contig
        rng = np.random.default_rng(21)
        w = rng.uniform(0, 8.0, 80)
        early = [(1e6, 1, 1800.0, float(a), 6000.0, 0)
                 for a in (0.0, 2000.0, 5000.0)]
        late = early + [(1e6, 1, 1800.0, 20000.0, 6000.0, 0)]
        r1 = coordinated_clair_contig(early, w, 1.0)
        r2 = coordinated_clair_contig(late, w, 1.0)
        assert r1 == r2[:3], "晚到作业改变了更早作业的已承诺释放"

    def test_wind_clairvoyance_is_allowed(self):
        """对照:改未来风【应当】能改变决策(这是被允许的那一半)。"""
        import numpy as np
        from stage15_continuous import coordinated_clair_contig, W_PER_PE
        jobs = [(1e6, 1, 6000.0, 0.0, 12000.0, 0)]
        w1 = np.zeros(40); w1[0:2] = 2 * W_PER_PE
        w2 = w1.copy(); w2[10:22] = 2 * W_PER_PE
        assert (coordinated_clair_contig(jobs, w1, 1.0)
                != coordinated_clair_contig(jobs, w2, 1.0))
