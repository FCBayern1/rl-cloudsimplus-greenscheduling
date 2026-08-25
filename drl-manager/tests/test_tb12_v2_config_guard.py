"""TB12 v2 配置守卫(Codex 硬阻塞#5):差分必须精确等于白名单。

- v1 → v2:五项修复 + 两个执行开关 + 身份字段,一个键不许多、不许少。
- fc_v2 → nofc_v2:只差 forecast_mode 与身份字段。
- v2 → v2s50k:只差步数/存档频率与身份字段。
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault(
    "EVAL_CONFIG_PATH",
    str(pathlib.Path(__file__).resolve().parents[2] / "config_C.yml"))

from src.baselines.evaluate import load_config  # noqa: E402


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


IDENTITY = {"experiment_name", "simulation_name"}

V1_TO_V2_ALLOWED = IDENTITY | {
    "carbon_normalization_fixed_max",        # P1 冻结标定
    "sla_mode", "sla_target",                # ontime_mi 合同(target 可能同值,允许出现)
    "defer_deadline_slack_sec",              # 720 覆盖 600s 量化
    "global_model.gamma", "global_model.gae_lambda",  # γ=λ=1
    "carbon_cap_hard_stop",                  # 硬阻塞#1
    "save_initial_checkpoint",               # 硬阻塞#3
}

V2_TO_SMOKE_ALLOWED = IDENTITY | {
    "training.total_timesteps",
    "training.checkpoint_freq_timesteps",
}


def test_v1_to_v2_diff_is_exactly_the_repair_set():
    d = diff_keys(load_config("experiment_tb12_rl_fc"),
                  load_config("experiment_tb12_rl_fc_v2"))
    assert d <= V1_TO_V2_ALLOWED, f"v2 含未注册变更: {sorted(d - V1_TO_V2_ALLOWED)}"
    # 五项修复本体必须真的在差分里(不许静默丢失)
    required = V1_TO_V2_ALLOWED - IDENTITY - {"sla_target"}
    assert required <= d, f"v2 缺少注册变更: {sorted(required - d)}"


def test_v1_to_v2_nofc_same_repair_set():
    d = diff_keys(load_config("experiment_tb12_rl_nofc"),
                  load_config("experiment_tb12_rl_nofc_v2"))
    assert d <= V1_TO_V2_ALLOWED, f"nofc_v2 含未注册变更: {sorted(d - V1_TO_V2_ALLOWED)}"


def test_fc_v2_vs_nofc_v2_single_variable():
    d = diff_keys(load_config("experiment_tb12_rl_fc_v2"),
                  load_config("experiment_tb12_rl_nofc_v2"))
    assert d == IDENTITY | {"forecast_mode"}, f"配对不是单变量: {sorted(d)}"


def test_fc_smoke_vs_nofc_smoke_single_variable():
    d = diff_keys(load_config("experiment_tb12_rl_fc_v2s50k"),
                  load_config("experiment_tb12_rl_nofc_v2s50k"))
    assert d == IDENTITY | {"forecast_mode"}, f"smoke 配对不是单变量: {sorted(d)}"


def test_v2_to_smoke_diff_is_steps_only():
    for arm in ("fc", "nofc"):
        d = diff_keys(load_config(f"experiment_tb12_rl_{arm}_v2"),
                      load_config(f"experiment_tb12_rl_{arm}_v2s50k"))
        assert d == V2_TO_SMOKE_ALLOWED, f"{arm} smoke 差分异常: {sorted(d)}"


def test_v2_smoke_values_pinned():
    c = load_config("experiment_tb12_rl_fc_v2s50k")
    assert c["training"]["total_timesteps"] == 50000
    assert c["training"]["checkpoint_freq_timesteps"] == 50000
    assert c["carbon_cap_hard_stop"] is True
    assert c["save_initial_checkpoint"] is True
    assert abs(c["carbon_normalization_fixed_max"] - 6.637006600838674e-03) < 1e-18
    assert c["sla_mode"] == "ontime_mi" and c["sla_target"] == 0.995
    assert c["defer_deadline_slack_sec"] == 720.0
    assert c["global_model"]["gamma"] == 1.0 and c["global_model"]["gae_lambda"] == 1.0
    assert c["csv_year"] == 2021


# ---- v3 守卫(Codex Step 2, 2026-08-26):v2→v3 仅身份字段 ----

def test_v2_to_v3_identity_only():
    for pair in (("experiment_tb12_rl_fc_v2", "experiment_tb12_rl_fc_v3"),
                 ("experiment_tb12_rl_nofc_v2", "experiment_tb12_rl_nofc_v3"),
                 ("experiment_tb12_rl_fc_v2s50k", "experiment_tb12_rl_fc_v3s50k"),
                 ("experiment_tb12_rl_nofc_v2s50k", "experiment_tb12_rl_nofc_v3s50k")):
        d = diff_keys(load_config(pair[0]), load_config(pair[1]))
        assert d == IDENTITY, f"{pair}: v3 含非身份变更 {sorted(d)}"


def test_v3_pair_single_variable():
    d = diff_keys(load_config("experiment_tb12_rl_fc_v3s50k"),
                  load_config("experiment_tb12_rl_nofc_v3s50k"))
    assert d == IDENTITY | {"forecast_mode"}, f"v3 配对不是单变量: {sorted(d)}"
    d = diff_keys(load_config("experiment_tb12_rl_fc_v3"),
                  load_config("experiment_tb12_rl_nofc_v3"))
    assert d == IDENTITY | {"forecast_mode"}, f"v3 600k 配对不是单变量: {sorted(d)}"


def test_v3_defer_base_cost_present():
    # 修复使其生效的键必须在(值从 v1 起未变,防止静默丢失)
    c = load_config("experiment_tb12_rl_fc_v3s50k")
    assert c["defer_base_cost"] == 0.5
    assert c["defer_cost_mode"] == "incremental_urgency"
