"""DeferringGlobalScheduler threshold fix (2026-07-24): env-configurable
thresholds + relative mode + defer counter. The historical forecast_thresh=0.3
was above the forecast scale (~[0,0.05]) so defer was inert."""
import os, numpy as np, pytest
from src.baselines.global_schedulers import DeferringGlobalScheduler

class _Inner:
    def __init__(self, a): self.a=a
    def schedule(self, obs): return list(self.a)

def _obs(gn, gr, fut):
    return {"dc_current_green_power_w": np.array(gn,float),
            "dc_green_ratio": np.array(gr,float),
            "dc_future_short_mean": np.array(fut,float)}

def test_env_threshold_lets_defer_fire(monkeypatch):
    # DC0 seen-green, not green now (gn=0,gr=0), forecast 0.04 > env thresh 0.02
    monkeypatch.setenv("DEFER_FORECAST_THRESH", "0.02")
    d=DeferringGlobalScheduler(_Inner([0]), 3, 1)
    d._seen_green=np.array([True,False,False])
    d.schedule(_obs([0,0,0],[0,0,0],[0.04,0,0]))
    assert d._n_defers==1

def test_high_thresh_stays_inert():
    d=DeferringGlobalScheduler(_Inner([0]), 3, 1)  # default 0.3
    d._seen_green=np.array([True,False,False])
    d.schedule(_obs([0,0,0],[0,0,0],[0.04,0,0]))
    assert d._n_defers==0   # 0.04 < 0.3 → inert (the bug we fixed)

def test_relative_mode_scale_free(monkeypatch):
    monkeypatch.setenv("DEFER_RELATIVE","1")
    d=DeferringGlobalScheduler(_Inner([0]), 3, 1)
    d._seen_green=np.array([True,False,False])
    # forecast 0.04 > current green ratio 0.0 → greener coming → defer
    d.schedule(_obs([0,0,0],[0,0,0],[0.04,0,0]))
    assert d._n_defers==1

def test_relative_no_defer_when_green_now(monkeypatch):
    monkeypatch.setenv("DEFER_RELATIVE","1")
    d=DeferringGlobalScheduler(_Inner([0]), 3, 1)
    d._seen_green=np.array([True,False,False])
    # green now (gr=0.16>green_now_thresh 0.05) → never defer
    d.schedule(_obs([40,0,0],[0.16,0,0],[0.04,0,0]))
    assert d._n_defers==0

def test_brown_placeholder_never_defers(monkeypatch):
    monkeypatch.setenv("DEFER_FORECAST_THRESH","0.02")
    d=DeferringGlobalScheduler(_Inner([2]), 3, 1)  # route to brown DC2
    d._seen_green=np.array([True,True,False])       # DC2 never green
    d.schedule(_obs([0,0,0],[0,0,0],[0,0,0.5]))     # DC2 forecast=0.5 placeholder
    assert d._n_defers==0
