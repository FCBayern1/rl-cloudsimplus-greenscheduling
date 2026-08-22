"""逐状态 DP 盲的不变量。"""
import numpy as np
import pytest

from dp_blind import MAX_AGE, dp_blind_releases, fit_tables, row_states, solve_dp


@pytest.fixture(scope="module")
def tabs():
    rng = np.random.default_rng(5)
    w = rng.uniform(0, 10.0, 2000)
    return fit_tables(w, 5.0, 3600.0)


class TestDP:
    def test_budget_zero_forces_release(self, tabs):
        c, h = tabs
        V, rel = solve_dp(c, h, 10)
        assert rel[0].all(), "b=0 必须无条件释放(latest-start 兜底)"

    def test_value_never_exceeds_immediate_cost(self, tabs):
        c, h = tabs
        V, _ = solve_dp(c, h, 20)
        for b in range(21):
            assert (V[b] <= c + 1e-12).all(), "等待选项只能降低期望成本"

    def test_value_monotone_in_budget(self, tabs):
        c, h = tabs
        V, _ = solve_dp(c, h, 20)
        for b in range(20):
            assert (V[b + 1] <= V[b] + 1e-12).all(), "预算多不会更差"

    def test_policy_is_causal(self):
        """改 cutoff 之后的评测序列不得改变之前的释放(对称谓词)。"""
        rng = np.random.default_rng(9)
        w20 = rng.uniform(0, 10.0, 2000)
        c, h = fit_tables(w20, 5.0, 3600.0)
        _, pol = solve_dp(c, h, 30)
        jobs = [(1e6, 2, 3600.0, float(a), 30 * 600.0, 0)
                for a in rng.uniform(0, 5000, 8)]
        w1 = rng.uniform(0, 10.0, 60)
        cutoff = 30 * 600
        for future in (0.0, 10.0):
            w2 = w1.copy(); w2[30:] = future
            r1 = dp_blind_releases(jobs, w1, 5.0, pol)
            r2 = dp_blind_releases(jobs, w2, 5.0, pol)
            for a, b in zip(r1, r2):
                assert (a < cutoff) == (b < cutoff)
                if a < cutoff:
                    assert a == b

    def test_row_states_age_resets_on_flip(self):
        on = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
        ph, age = row_states(on)
        assert list(age) == [0, 1, 0, 1, 2, 0]
