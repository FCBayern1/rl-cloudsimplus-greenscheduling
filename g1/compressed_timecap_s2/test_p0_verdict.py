import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("yaml")
from p0_verdict import ARMS, judge  # noqa: E402

W = [0, 1, 2]


def table(carbon, reward, clip=0.0, samples=1000.0, cap=0.0, contract=True):
    rows = {}
    for a in ARMS:
        for k in W:
            rows[(a, k)] = {"carbon": carbon[a], "reward": reward[a], "clip": clip,
                            "samples": samples, "cap": cap, "contract_ok": contract}
    return rows


GOOD_C = {"blind": 10.0, "clean": 6.0, "shrink": 12.0, "always_defer": 11.0}
GOOD_R = {"blind": -10.0, "clean": -6.0, "shrink": -12.0, "always_defer": -14.0}


def test_consistent_table_passes():
    out = judge(table(GOOD_C, GOOD_R), W)
    assert out["verdict"] == "PASS_P0" and all(out["gates"].values())


def test_reward_that_rewards_the_bad_forecast_stops():
    bad_r = dict(GOOD_R, shrink=-5.0)   # shrink emits more carbon but scores higher
    out = judge(table(GOOD_C, bad_r), W)
    assert out["gates"]["shrink_worse_both_pooled"] is False and out["verdict"] == "STOP_P0"


def test_defer_arbitrage_stops():
    r = dict(GOOD_R, always_defer=-9.0)  # deferring everything scores above the blind
    out = judge(table(GOOD_C, r), W)
    assert out["gates"]["defer_no_arbitrage"] is False and out["verdict"] == "STOP_P0"


def test_clip_rate_and_cap_hits_stop():
    assert judge(table(GOOD_C, GOOD_R, clip=100.0, samples=1000.0), W)["gates"]["clip_rate_le_5pc"] is False
    assert judge(table(GOOD_C, GOOD_R, cap=1.0), W)["gates"]["no_cap_hits"] is False


def test_probe_contract_failure_is_reported_not_gated():
    rows = table(GOOD_C, GOOD_R)
    for k in W:
        rows[("always_defer", k)]["contract_ok"] = False
    out = judge(rows, W)
    assert out["verdict"] == "PASS_P0" and out["gates"]["contract_green"] is True
    assert out["probe_always_defer_contract_bad_windows"] == W


def test_policy_arm_contract_failure_still_stops():
    rows = table(GOOD_C, GOOD_R)
    rows[("clean", 1)]["contract_ok"] = False
    out = judge(rows, W)
    assert out["gates"]["contract_green"] is False and out["verdict"] == "STOP_P0"


def test_missing_run_is_invalid():
    rows = table(GOOD_C, GOOD_R)
    rows[("clean", 1)] = None
    assert judge(rows, W)["verdict"] == "INVALID_INCOMPLETE_DATA"
