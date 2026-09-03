"""The planner subtracts a fleet static draw from green before pricing a job.

The 332 W default is the measured C-regime fleet draw and must stay the default so the
frozen arms price as before. PLANNER_STATIC_TOTAL_W overrides it for fleets where the
constant is wrong (zero-floor host twins, idle power-down).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("yaml")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def planner_cls():
    os.environ.setdefault("EVAL_CONFIG_PATH", os.path.join(REPO, "config_C.yml"))
    os.environ.setdefault("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan")
    from src.baselines.global_schedulers import CurveInformedPlannerGlobalScheduler
    return CurveInformedPlannerGlobalScheduler


def test_default_static_is_the_measured_332_w_spread_by_host_count(planner_cls, monkeypatch):
    monkeypatch.delenv("PLANNER_STATIC_TOTAL_W", raising=False)
    p = planner_cls(5, 8)
    assert p.static.shape == (5,)
    assert abs(float(p.static.sum()) - 332.0) < 1e-9
    assert (p.static >= 0).all()


def test_env_override_zeroes_the_static_floor(planner_cls, monkeypatch):
    monkeypatch.setenv("PLANNER_STATIC_TOTAL_W", "0")
    p = planner_cls(5, 8)
    assert np.allclose(p.static, 0.0)


def test_env_override_keeps_the_host_count_spread(planner_cls, monkeypatch):
    monkeypatch.delenv("PLANNER_STATIC_TOTAL_W", raising=False)
    base = planner_cls(5, 8).static.copy()
    monkeypatch.setenv("PLANNER_STATIC_TOTAL_W", "166")
    half = planner_cls(5, 8).static
    assert np.allclose(half, base / 2.0)
