import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p0_verdict as pv  # noqa: E402

W = [0, 1, 2]


def _row(carbon, reward, reward_disc, ontime=1.0, forced=0.0, contract_ok=True):
    return {"carbon": carbon, "reward": reward, "reward_disc": reward_disc, "ontime": ontime,
            "forced": forced, "clip": 0.0, "samples": 100.0, "cap": 0.0, "contract_ok": contract_ok}


def _good():
    rows = {}
    for k in W:
        rows[("blind", k)] = _row(1.0, -10.0, -8.0)
        rows[("clean", k)] = _row(0.6, -6.0, -5.0)
        rows[("shrink", k)] = _row(0.9, -9.0, -7.5)
        rows[("nodefer", k)] = _row(0.95, -9.5, -7.0)
        rows[("always_defer", k)] = _row(1.1, -12.0, -9.0, ontime=1.0, forced=0.0)
    return rows


def test_prime_passes_when_discounted_return_orders_timing_and_mask_routes_legally():
    out = pv.judge_dprime(_good(), W)
    assert out["verdict"] == "PASS_P0_PRIME", out["gates"]
    assert out["gates"]["disc_order_pooled"] and out["gates"]["always_defer_routed_legally_by_mask"]
    assert out["always_defer_legal_windows"] == W


def test_prime_stops_when_starting_now_outscores_the_best_window_on_the_discounted_return():
    rows = _good()
    for k in W:
        rows[("nodefer", k)] = _row(0.95, -9.5, -4.0)        # discounted return above clean
    out = pv.judge_dprime(rows, W)
    assert out["verdict"] == "STOP_P0_PRIME" and not out["gates"]["disc_order_pooled"]


def test_prime_stops_when_the_mask_lets_always_defer_be_late_or_forced():
    rows = _good()
    rows[("always_defer", 1)] = _row(1.1, -12.0, -9.0, ontime=0.95, forced=3.0)
    out = pv.judge_dprime(rows, W)
    assert out["verdict"] == "STOP_P0_PRIME"
    assert out["gates"]["always_defer_routed_legally_by_mask"] is False and out["always_defer_legal_windows"] == [0, 2]


def test_prime_is_invalid_without_the_nodefer_arm():
    rows = {k: v for k, v in _good().items() if k[0] != "nodefer"}
    assert pv.judge_dprime(rows, W)["verdict"] == "INVALID_INCOMPLETE_DATA"


def test_prime_keeps_the_legacy_p0_gates():
    rows = _good()
    for k in W:
        rows[("shrink", k)] = _row(0.5, -5.0, -4.5)          # shrink better than clean: legacy P0 fails
    out = pv.judge_dprime(rows, W)
    assert out["verdict"] == "STOP_P0_PRIME" and out["gates"]["shrink_worse_both_pooled"] is False
