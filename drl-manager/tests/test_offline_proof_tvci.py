"""第十考场离线证明的正确性锁定。

这个脚本的结论会决定要不要动 Java，所以它自身的实现细节必须被测住。
本轮开发中已经踩到两类 bug，各配一条回归：
  - 阈值定义域与比较对象不一致（θ 定在含 0 的 cost 分布上却拿去比恒正的 c）
  - 补丁没打全（best_blind 的候选集合有两处，只改了一处 → 报出错误的最强盲臂）

Run from repo root:
    cd drl-manager && python -m pytest tests/test_offline_proof_tvci.py -v
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

tv = pytest.importorskip("offline_proof_tvci3")

ARMS = ("nowait", "naive_green", "naive_carbon", "green_age",
        "onset_wait", "clock_carbon", "clock_onset", "combo")


# --- 回归 1：best_blind 的候选集合必须在所有出现处一致 -----------------------

def test_best_blind_candidate_sets_are_identical_everywhere():
    """踩过的坑：`bb = min((...))` 有两处，只改一处会报出错误的最强盲臂。"""
    src = (REPO / "offline_proof_tvci3.py").read_text()
    sets = re.findall(r"bb = min\(\(([^)]*)\)", src)
    assert len(sets) >= 2, "预期至少两处 best_blind 选择"
    normalised = {tuple(sorted(s.replace('"', "").split(","))) for s in sets}
    assert len(normalised) == 1, f"候选集合不一致：{normalised}"


def test_arms_tuple_covers_every_blind_arm():
    src = (REPO / "offline_proof_tvci3.py").read_text()
    arms = re.search(r"ARMS = \(([^)]*)\)", src).group(1)
    for a in ARMS:
        assert f'"{a}"' in arms, f"{a} 不在 ARMS 里，汇总表会漏掉它"


# --- 回归 2：阈值定义域 -------------------------------------------------------

def test_carbon_threshold_lives_on_the_intensity_distribution():
    """踩过的坑：θ 定义在 cost（含绿电的 0）上却用来比 c（恒 > 0）→ 规则从不触发。"""
    src = (REPO / "offline_proof_tvci3.py").read_text()
    line = next(l for l in src.splitlines() if "theta = " in l)
    assert "np.quantile(cr" in line, f"θ 必须定义在 c 分布上，实际：{line.strip()}"


# --- 模型不变量 ---------------------------------------------------------------

def test_green_makes_cost_zero_and_brown_does_not():
    g = np.array([1, 1, 0, 0, 1], dtype=np.int8)
    c = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    cost = np.where(g == 1, 0.0, c)
    assert cost.tolist() == [0.0, 0.0, 4.0, 5.0, 0.0]


def test_integrated_carbon_is_runtime_weighted_not_point_value():
    """碳按执行窗积分：跨绿/棕边界的作业只为棕电部分付费。"""
    cost = np.array([0.0, 0.0, 4.0, 4.0])
    csum = np.concatenate([[0.0], np.cumsum(cost)])
    s = np.array([0]); L = np.array([4]); mi = np.array([4.0])
    got = tv.carbon_of(cost, s, L, mi, csum)
    assert got[0] == pytest.approx(8.0)          # (0+0+4+4) * (4/4)


def test_fifo_never_exceeds_capacity():
    rng = np.random.default_rng(0)
    r = np.sort(rng.integers(0, 500, size=60))
    L = rng.integers(10, 60, size=60)
    K = 5
    s = tv.fifo_start(r, L, K)
    assert (s >= r).all(), "实际开始不得早于释放时刻"
    T = int((s + L).max()) + 2
    occ = np.zeros(T + 2)
    np.add.at(occ, s, 1); np.add.at(occ, s + L, -1)
    assert np.cumsum(occ).max() <= K


def test_infinite_capacity_leaves_release_times_untouched():
    r = np.array([0, 5, 10]); L = np.array([3, 3, 3])
    assert tv.fifo_start(r, L, 10_000).tolist() == r.tolist()


def test_reservation_values_are_non_increasing():
    """v_k = E[min(cost, v_{k-1})]：预算越多，保留值越低（越挑剔）。

    最优停时臂只存在于 v2（offline_proof_tvci2）。v3 起把它去掉了，因为 v2 实测
    它与 naive_green 逐位相同——绿电免费且占 57% 的步时保留值递推趋近 0，
    规则退化为"只在 cost=0 时停"。这条测试锁住那个数学性质，以防将来重新引入时写错。
    """
    tv2 = pytest.importorskip("offline_proof_tvci2")
    samples = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    v = tv2.reservation_values(samples, 6)
    assert all(v[k] <= v[k-1] + 1e-12 for k in range(1, len(v)))


def test_green_schedule_alternates_and_is_binary():
    g = tv.build_green(20000, np.random.default_rng(3))
    assert set(np.unique(g)).issubset({0, 1})
    assert 0.3 < g.mean() < 0.9, "绿电占比应在合理区间"


def test_ci_curve_amplitude_zero_is_constant():
    g = tv.build_green(9000, np.random.default_rng(1))
    c = tv.build_ci(9000, 3000, 0.0, g, rho=0.0)
    assert np.allclose(c, c[0]), "amp=0 必须给出恒定碳强度（归因对照的基准）"


def test_rho_one_makes_ci_track_green_negatively():
    g = tv.build_green(9000, np.random.default_rng(2))
    c = tv.build_ci(9000, 600, 0.6, g, rho=1.0)
    sm = np.convolve(g.astype(float), np.ones(600)/600, mode="same")
    assert np.corrcoef(c, sm)[0, 1] < -0.5, "ρ=1 时碳强度应与绿电反相关"
