"""决策价值探针的自检 —— p* 的推导与两个边界。"""
import numpy as np
import pytest

from probe_decision_value import entropy, p_star


class TestPStar:
    """盲策略等待 <=> p*(L_expire - C_green) > L_expire - C_brown。"""

    def test_no_expiry_penalty_means_always_wait(self):
        """L = C_brown（backstop 兜底，前十考场）-> p* = 0 -> 永远等占优。

        这是"等待免费"的精确表述，也是 naive 十次不可战胜的根因。
        """
        assert p_star(0.55, 0.01, 0.55) == 0.0

    def test_hard_contract_means_never_wait(self):
        """L -> inf（丢作业不可接受）-> p* -> 1 -> 永不等占优。"""
        assert p_star(0.55, 0.01, 1e12) == pytest.approx(1.0, abs=1e-9)

    def test_interior_requires_brown_below_expire(self):
        """只有 C_brown < L < inf 时 p* 才在 (0,1) 内 —— M1 的全部意义。"""
        for L in (0.8, 1.1, 2.0, 5.5):
            assert 0.0 < p_star(0.55, 0.01, L) < 1.0

    def test_p_star_rises_with_expiry_cost(self):
        ps = [p_star(0.55, 0.01, L) for L in (0.6, 1.1, 5.5, 55.0)]
        assert all(b > a for a, b in zip(ps, ps[1:]))

    def test_matches_the_indifference_equation(self):
        """p*·C_green + (1-p*)·L == C_brown 处恰好无差异。"""
        cb, cg, L = 0.55, 0.01, 1.1
        p = p_star(cb, cg, L)
        assert p * cg + (1 - p) * L == pytest.approx(cb)


class TestEntropy:
    def test_certain_outcome_is_zero(self):
        assert entropy(0.0) == pytest.approx(0.0, abs=1e-9)
        assert entropy(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_coin_flip_is_one_bit(self):
        assert entropy(0.5) == pytest.approx(1.0)

    def test_symmetric(self):
        assert entropy(0.3) == pytest.approx(entropy(0.7))
