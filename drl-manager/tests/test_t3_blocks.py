"""T3 天花板实验的配置守卫(工单 1f2f20a)。

全知臂必须与盲态基准块**只差 green_oracle_mode + 身份字段**;
否则"看得见未来"这一条就混进了别的变量。
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault(
    "EVAL_CONFIG_PATH",
    str(pathlib.Path(__file__).resolve().parents[2] / "config_C.yml"))

from src.baselines.evaluate import load_config  # noqa: E402

IDENTITY = {"experiment_name", "simulation_name"}
BASE = "experiment_g1eval_matchedvan"
GODEYE = "experiment_t3_godeye"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff_keys(a, b):
    fa, fb = flatten(a), flatten(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def test_godeye_block_differs_only_by_oracle_mode_and_identity():
    d = diff_keys(load_config(BASE), load_config(GODEYE))
    assert d == IDENTITY | {"green_oracle_mode"}, f"T3 全知臂含未注册变更: {sorted(d)}"


def test_godeye_value_is_godeye_and_base_is_timecap():
    assert load_config(GODEYE)["green_oracle_mode"] == "godeye"
    assert load_config(BASE)["green_oracle_mode"] == "timecap"


def test_offset_range_and_trace_unchanged():
    a, b = load_config(BASE), load_config(GODEYE)
    assert a["green_episode_offset_range"] == b["green_episode_offset_range"] == 44950
    assert a["cloudlet_trace_file"] == b["cloudlet_trace_file"]


def test_dc_topology_matches_workorder_constants():
    """DC/涡轮/时区/棕电强度必须与工单给的常数一致 —— 跨机对照的前提。"""
    want = [("DC_Nordic", [12, 36], 0, 0.08), ("DC_Germany", [95, 91], 18, 0.35),
            ("DC_US_East", [96], 54, 0.55), ("DC_US_West", [], 72, 0.75),
            ("DC_APAC", [], 108, 0.92)]
    for cfgname in (BASE, GODEYE):
        got = [(d.get("name"), list(d.get("turbine_ids") or []),
                d.get("time_zone_offset_rows"), d.get("brown_carbon_factor"))
               for d in load_config(cfgname)["datacenters"]]
        assert got == want, f"{cfgname} 拓扑与工单常数不符: {got}"


def test_registered_windows_match_frozen_json():
    import json
    p = pathlib.Path(__file__).resolve().parents[1] / "calib/p0c_green_windows.json"
    w = json.loads(p.read_text())["windows"]
    got = {x["stratum"]: (x["episode_index_k"], x["offset_rows"]) for x in w}
    assert got == {"low": (19, 19171), "mid": (56, 11554), "high": (34, 34306)}
