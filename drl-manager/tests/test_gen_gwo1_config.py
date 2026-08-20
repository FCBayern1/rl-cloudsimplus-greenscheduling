"""gwo1 四个 experiment block 的克隆规则。

关键不变量:gwo1 相对 SQT2 只允许在 trace / 涡轮块 / 名称 / preflight profile
上不同。任何别的键漂移都会让 gwo1 的结论无法归因给决策域和 trace。
"""
from pathlib import Path

import pytest
import yaml

from gen_gwo1_config import PLAN, PROFILE, SCALE_TAG, TURBINE_OFFSET, build, derive

_REPO = Path(__file__).resolve().parent.parent.parent
_WIND = _REPO / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
_TRACES = _REPO / "cloudsimplus-gateway/src/main/resources/traces"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(_REPO / "config_C.yml"))


@pytest.fixture(scope="module")
def blocks(cfg):
    return build(cfg)


class TestDerive:
    def test_renames_trace_experiment_and_simulation(self, cfg):
        b = derive(cfg["experiment_sqt2_oracle"], "experiment_gwo1_oracle", "gwo1")
        assert b["cloudlet_trace_file"] == f"traces/gwo1_n1200_{SCALE_TAG}.csv"
        assert b["experiment_name"] == "gwo1_oracle"
        assert b["simulation_name"] == "GWO1_gwo1_oracle"
        assert b["preflight_temporal_profile"] == PROFILE

    def test_turbine_ids_shift_and_empties_stay_empty(self, cfg):
        src = cfg["experiment_sqt2_oracle"]
        b = derive(src, "experiment_gwo1_oracle", "gwo1")
        for old, new in zip(src["datacenters"], b["datacenters"]):
            assert new["turbine_ids"] == [t + TURBINE_OFFSET
                                          for t in old["turbine_ids"] or []]
            if not old["turbine_ids"]:
                assert new["turbine_ids"] == [], "无绿电的 DC 不能凭空长出涡轮"

    def test_source_block_not_mutated(self, cfg):
        before = yaml.safe_dump(cfg["experiment_sqt2_oracle"], sort_keys=True)
        derive(cfg["experiment_sqt2_oracle"], "experiment_gwo1_oracle", "gwo1")
        assert yaml.safe_dump(cfg["experiment_sqt2_oracle"], sort_keys=True) == before

    def test_missing_source_block_is_loud(self):
        with pytest.raises(KeyError):
            build({})


class TestInheritedContract:
    def test_full_cpu_utilization_is_inherited(self, blocks):
        """0.5 会把 runtime 拉长约 2.5 倍并作废每一个预算检查。"""
        for k, b in blocks.items():
            assert float(b["cloudlet_cpu_utilization"]) == 1.0, k

    def test_backstop_and_obs_margin_inherited(self, blocks):
        for k, b in blocks.items():
            assert b["defer_deadline_force_mode"] == "latest_start", k
            assert float(b["defer_deadline_slack_sec"]) == 120.0, k
            assert float(b["obs_v32_deadline_margin_sec"]) == 120.0, k

    def test_arms_differ_only_as_intended(self, blocks):
        """preflight 的 'arms differ only as intended' 门,提前在生成期锁住。"""
        allowed = {"forecast_mode", "green_oracle_mode",
                   "experiment_name", "simulation_name"}
        for cal_ho in ("gwo1", "gwo1ho"):
            o = blocks[f"experiment_{cal_ho}_oracle"]
            n = blocks[f"experiment_{cal_ho}_noforecast"]
            diff = {k for k in set(o) | set(n) if o.get(k) != n.get(k)}
            assert diff <= allowed, f"{cal_ho}: 多余差异 {sorted(diff - allowed)}"

    def test_cal_and_ho_share_everything_but_data(self, blocks):
        """cal 与 ho 只能在数据(trace/涡轮)和名称上不同。"""
        allowed = {"cloudlet_trace_file", "datacenters",
                   "experiment_name", "simulation_name"}
        c = blocks["experiment_gwo1_noforecast"]
        h = blocks["experiment_gwo1ho_noforecast"]
        diff = {k for k in set(c) | set(h) if c.get(k) != h.get(k)}
        assert diff <= allowed, f"cal/ho 多余差异 {sorted(diff - allowed)}"


class TestArtifactsOnDisk:
    """缺一个涡轮 CSV 不会崩,只会静默变成零绿电 —— 必须在生成期挡住。"""

    def test_every_turbine_csv_exists(self, blocks):
        missing = []
        for k, b in blocks.items():
            year = b.get("csv_year", 2021)
            for dc in b["datacenters"]:
                for t in dc["turbine_ids"] or []:
                    if not (_WIND / f"Turbine_{t}_{year}.csv").is_file():
                        missing.append((k, t))
        assert not missing, f"涡轮 CSV 缺失(会静默零绿电): {missing}"

    def test_every_trace_exists(self, blocks):
        for k, b in blocks.items():
            assert (_TRACES / Path(b["cloudlet_trace_file"]).name).is_file(), k

    def test_plan_covers_four_blocks(self):
        assert len(PLAN) == 4
        assert len({d for _, d, _ in PLAN}) == 4


class TestAnchorModulusInvariant:
    """注册锚点 <=178 时,暴露标定的 %rows 与 env 的 %offset_range 必须同值。"""

    def test_registered_anchors_do_not_wrap(self, blocks):
        from gen_gwo1_trace import ANCHORS
        rows = 200_000                       # 绿电序列长度
        for b in blocks.values():
            rng = int(b["green_episode_offset_range"])
            for k in ANCHORS:
                assert (1009 * k) % rng == (1009 * k) % rows, \
                    f"锚点 {k} 在 %{rng} 与 %{rows} 下分叉"

    def test_anchor_179_would_wrap(self, blocks):
        rng = int(blocks["experiment_gwo1_noforecast"]["green_episode_offset_range"])
        assert (1009 * 179) % rng != (1009 * 179) % 200_000
