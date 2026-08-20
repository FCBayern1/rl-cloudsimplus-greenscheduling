"""绿窗碳上界复算脚本的自检。

这个脚本存在的意义是"出数的人不做唯一核对人"——它必须用 gate_flags 那条
clairvoyant 判据的同一定义,否则复核就退化成了复述。
"""
import pathlib

import pytest

from verify_gwo1_green_bound import ANCHORS10, analyse

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent
_TRACE = _REPO / "cloudsimplus-gateway/src/main/resources/traces/gwo1_n1200_x130.csv"


@pytest.fixture(scope="module")
def ten():
    return analyse("calib/gwo1_schedule.json", str(_TRACE), ANCHORS10, 180000)


class TestBound:
    def test_act_share_matches_5080(self, ten):
        """5080 报 10.6%;独立复算须落在同一位小数附近。"""
        assert 0.100 <= ten["act_share"] <= 0.112

    def test_brown_share_matches_5080(self, ten):
        """5080 报 6.09%(MI 口径)。注意这【不是】验收线,见下面那条。"""
        assert 0.055 <= ten["brown_share"] <= 0.067

    def test_carbon_bound_is_the_acceptance_line(self, ten):
        """④ 的验收线 = 可搬走的棕电 MI / nowait 的棕电 MI = 12.95%。

        分母是 nowait 的【棕电】MI,不是全部 MI。按全部 MI 算得 4.14%,
        那是把可搬走的量摊到 3 倍大的分母上,线会过紧,可能把真实机制
        效应误判成 bug。碳口径的线比 MI 口径【宽】,方向别记反。
        """
        assert 0.12 <= ten["carbon_bound"] <= 0.14
        assert ten["carbon_bound"] > ten["brown_share"], "碳口径必须比 MI 口径宽"

    def test_nowait_brown_share_of_total_mi(self, ten):
        """5080 报 32.0%;碳 ~ 棕电 MI,所以这是碳差分母的物理基础。"""
        assert 0.30 <= ten["nowait_brown_mi"] / ten["total_mi"] <= 0.34

    def test_brown_is_a_strict_subset_of_acted(self, ten):
        """落棕电的只能是动手作业的一部分:前半段本来就在绿窗里跑。"""
        assert ten["brown_share"] < ten["act_share"]

    def test_registered_anchor_count(self, ten):
        assert len(ANCHORS10) == 10
        assert ten["n_green"] > 0 and ten["n_act"] > 0

    def test_bound_is_in_green_decision_units_not_episode_units(self, ten):
        """分母是绿窗决策集 MI,不是 episode 总 MI —— 直接和碳差比会放宽约 1.5x。

        这条把口径写死在测试里,免得后来者拿 6.09% 直接当碳差阈值。
        """
        import csv
        from sqt2_prescreen import HORIZON_S
        rows = list(csv.DictReader(open(_TRACE)))
        ep_mi = sum(float(r["length"]) for r in rows
                    if float(r["arrival_time"]) < HORIZON_S)
        per_ep_green = ten["green_decision_mi"] / len(ANCHORS10)
        assert per_ep_green < ep_mi, "绿窗决策集必须是 episode 的真子集"
        assert 0.55 < per_ep_green / ep_mi < 0.75
