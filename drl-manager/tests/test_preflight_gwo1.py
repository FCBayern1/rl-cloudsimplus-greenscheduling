"""gwo1 认证门 + preflight 家族化重构。

最重要的一条:重构不得改变 SQT2 的认证输出 —— SQT2 已冻结,它的 30 项
PASS 是判决的前提。这里用子进程跑真 preflight 并逐字节比对两族输出。
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_DRL = Path(__file__).resolve().parent.parent
_REPO = _DRL.parent
_CFG = _REPO / "config_C.yml"


def run_preflight(oracle, blind, *extra):
    env = dict(os.environ, EVAL_CONFIG_PATH=str(_CFG))
    r = subprocess.run([sys.executable, str(_DRL / "preflight_scenario.py"),
                        oracle, blind, *extra],
                       capture_output=True, text=True, cwd=str(_DRL), env=env)
    return r.returncode, r.stdout


_LINE = re.compile(r"\[(\*\*FAIL\*\*|PASS| N/A )\] (.*)")


def gate(out, name):
    """chk() 用 {name:30s} 补齐，超长名后只有一个空格 —— 只能前缀匹配。"""
    for line in out.splitlines():
        m = _LINE.match(line)
        if m and m.group(2).startswith(name):
            return m.group(1).strip("* ")
    return None


def n_pass(out):
    return sum(1 for ln in out.splitlines() if ln.startswith("[PASS]"))


def failures(out):
    return [ln for ln in out.splitlines() if ln.startswith("[**FAIL**]")]


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(_CFG))


class TestGwo1Certification:
    @pytest.mark.parametrize("variant", ["gwo1", "gwo1ho"])
    def test_audit_passes(self, variant):
        rc, out = run_preflight(f"experiment_{variant}_oracle",
                                f"experiment_{variant}_noforecast")
        assert rc == 0, out
        assert "AUDIT PASSED" in out
        assert not failures(out)

    @pytest.mark.parametrize("variant", ["gwo1", "gwo1ho"])
    def test_gwo1_specific_gates_present_and_green(self, variant):
        _, out = run_preflight(f"experiment_{variant}_oracle",
                               f"experiment_{variant}_noforecast")
        for g in ("gwo1v1: decision exposure (MAIN)",
                  "gwo1v1: no dead anchor",
                  "gwo1v1: registered runtime scale",
                  "gwo1v1: source trace is t60",
                  "gwo1v1: anchor modulus invariant",
                  "gwo1: cashability",
                  "gwo1: full CPU utilization"):
            assert gate(out, g) == "PASS", f"{variant}: {g} -> {gate(out, g)}"

    def test_sqt2_tight_class_gates_do_not_leak_into_gwo1(self):
        """SQT2 的 tight/loose 类条件门是另一套机制,不能出现在 gwo1 上。"""
        _, out = run_preflight("experiment_gwo1_oracle",
                               "experiment_gwo1_noforecast")
        leaked = [ln for ln in out.splitlines() if "] sqt2v2:" in ln]
        assert not leaked, f"SQT2 专属门泄漏到 gwo1: {leaked}"

    def test_obs_bound_covers_scaled_mi(self, cfg):
        """×1.30 把 max MI 顶到 52e6;上界必须跟着放大,否则观测被截断。"""
        import csv
        for v in ("gwo1", "gwo1ho"):
            b = cfg[f"experiment_{v}_noforecast"]
            tr = list(csv.DictReader(open(
                _REPO / "cloudsimplus-gateway/src/main/resources"
                / b["cloudlet_trace_file"])))
            assert b["obs_cloudlet_mi_high"] >= max(float(r["length"]) for r in tr)


class TestSqt2UnchangedByRefactor:
    @pytest.mark.parametrize("variant", ["sqt2", "sqt2ho"])
    def test_sqt2_still_passes_all_30(self, variant):
        rc, out = run_preflight(f"experiment_{variant}_oracle",
                                f"experiment_{variant}_noforecast",
                                "--sqt2-cert")
        assert rc == 0, out
        assert n_pass(out) == 30, out
        assert not failures(out)

    @pytest.mark.parametrize("variant", ["sqt2", "sqt2ho"])
    def test_sqt2_labels_keep_the_sqt2_prefix(self, variant):
        """家族化后标签仍须是 'sqt2: ' / 'sqt2v2: ',否则历史存档无法比对。"""
        _, out = run_preflight(f"experiment_{variant}_oracle",
                               f"experiment_{variant}_noforecast",
                               "--sqt2-cert")
        assert gate(out, "sqt2: cashability") == "PASS"
        assert gate(out, "sqt2v2: decision exposure (MAIN)") == "PASS"
        assert not [ln for ln in out.splitlines() if "] gwo1" in ln]


class TestPrescreenWiring:
    def test_gwo1_schedules_registered(self):
        from sqt2_prescreen import SCHEDULES
        assert SCHEDULES["gwo1"] == ("experiment_gwo1_noforecast",
                                     "calib/gwo1_schedule.json")
        assert SCHEDULES["gwo1ho"] == ("experiment_gwo1ho_noforecast",
                                       "calib/gwo1ho_schedule.json")

    def test_every_schedule_artifact_exists(self):
        from sqt2_prescreen import SCHEDULES
        for k, (_, art) in SCHEDULES.items():
            assert (_DRL / art).is_file(), k

    def test_every_schedule_experiment_exists(self, cfg):
        from sqt2_prescreen import SCHEDULES
        for k, (exp, _) in SCHEDULES.items():
            assert exp in cfg, k

    def test_gwo1_reuses_the_sqt2_hazard_freeze(self):
        """绿电序列同律 -> 危险率同分布 -> 冻结门可复用(见 SCHEDULES 注释)。"""
        import json
        a = json.loads((_DRL / "calib/sqt2_schedule.json").read_text())
        b = json.loads((_DRL / "calib/gwo1_schedule.json").read_text())
        for k in ("seed", "on_range", "off_short", "off_long", "p_short"):
            assert a[k] == b[k], f"{k} 不同 -> 不能复用 sqt2_hazard_freeze.json"
