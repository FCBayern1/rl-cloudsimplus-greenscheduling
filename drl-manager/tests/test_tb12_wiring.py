"""2026-08-23 接线事故的回归测试:CLI turbines 必须进入环境配置,
且深拷贝不得污染原配置。事故指纹:T112+113 终判 JSON 里
artifact.turbines == [100,101] —— 规划与执行读了两条不同的风。"""
import copy

import pytest


def _mk_cfg():
    return {"green_episode_offset_range": 180000,
            "datacenters": [
                {"datacenter_id": 0, "turbine_ids": [100, 101],
                 "time_zone_offset_rows": 0},
                {"datacenter_id": 3, "turbine_ids": [],   # 棕电 DC
                 "time_zone_offset_rows": 0}]}


def _wire(cfg, offset_rows, turbines):
    """复制 run_episode 的配置构造段(与 tb12_run.run_episode 保持一致)。"""
    out = copy.deepcopy(cfg)
    out["green_episode_offset_range"] = 0
    for dc in out["datacenters"]:
        dc["time_zone_offset_rows"] = int(offset_rows)
        if dc.get("turbine_ids"):
            dc["turbine_ids"] = list(turbines)
    return out


class TestTurbineWiring:
    def test_cli_turbines_reach_the_environment(self):
        cfg = _wire(_mk_cfg(), 500, (114, 115))
        assert cfg["datacenters"][0]["turbine_ids"] == [114, 115], \
            "环境涡轮必须等于 CLI 涡轮 —— 这正是 T112+113 终判作废的原因"

    def test_brown_dc_stays_turbineless(self):
        cfg = _wire(_mk_cfg(), 500, (114, 115))
        assert cfg["datacenters"][1]["turbine_ids"] == []

    def test_deepcopy_does_not_pollute_the_source(self):
        src = _mk_cfg()
        _wire(src, 500, (114, 115))
        assert src["datacenters"][0]["turbine_ids"] == [100, 101], \
            "浅拷贝会让第二个 episode 继承上一个的涡轮"
        assert src["green_episode_offset_range"] == 180000
        assert src["datacenters"][0]["time_zone_offset_rows"] == 0

    def test_run_episode_signature_requires_turbines(self):
        """turbines 是位置参数,忘传直接 TypeError —— 结构性防呆。"""
        import inspect
        from tb12_run import run_episode
        params = list(inspect.signature(run_episode).parameters)
        assert params[4] == "turbines"
        assert inspect.signature(run_episode).parameters["turbines"].default \
            is inspect.Parameter.empty


class TestVerdictTripleCheck:
    def test_prereg_artifact_env_must_agree(self):
        """判决脚本三方核对:prereg == artifact.planning == artifact.environment。"""
        from tb12_run import check_turbine_consistency
        check_turbine_consistency([114, 115], [114, 115], [114, 115])  # 不抛
        with pytest.raises(SystemExit):
            check_turbine_consistency([114, 115], [112, 113], [100, 101])
