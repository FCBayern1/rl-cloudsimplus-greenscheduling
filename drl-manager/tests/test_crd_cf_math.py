"""
M2.0 unit tests: pure-Python CF math helpers.

Mirrors the Java JUnit fixtures in
    cloudsimplus-gateway/src/test/java/.../EnergyMetricsDeltaTest.java
to lock in numerical parity between Python and Java implementations. Any
formula drift on either side breaks these tests.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_crd_cf_math.py -v
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.crd.cf_math import (
    compute_carbon_kg,
    compute_green_used_wasted_wh,
    compute_waste_ratio,
    carbon_kg_aggregated,
    waste_ratio_aggregated,
    forecast_cf_per_step,
)


EPS = 1e-9


# ---------------------------------------------------------------------------
# Per-DC helpers — direct mirrors of the Java JUnit fixtures.
# ---------------------------------------------------------------------------


def test_carbon_typical_case():
    """Java fixture: carbonMatchesInlineFormula_typicalCase."""
    # 800 kW green, 1500 kW demand, 60s, gf=0.0, bf=0.5
    got = compute_carbon_kg(800_000.0, 1_500_000.0, 60.0 / 3600.0, 0.0, 0.5)
    # demandWh = 1500_000 * (60/3600) = 25000 Wh
    # greenWh  = min(800_000*60/3600, demandWh) = min(13333.33..., 25000) = 13333.33...
    # brownWh  = 25000 - 13333.33 = 11666.66...
    # carbon   = (13.333... * 0.0) + (11.666... * 0.5) = 5.833333...
    assert got == pytest.approx(5.833333333, abs=EPS)


def test_carbon_green_exceeds_demand():
    """Java fixture: carbonMatchesInlineFormula_greenExceedsDemand.

    green=2MW, demand=0.5MW, dt=0.5h, gf=0.04, bf=0.5
    No brown used → carbon = greenKWh * gf = (500_000*0.5/1000) * 0.04 = 250 * 0.04 = 10.0
    """
    got = compute_carbon_kg(2_000_000.0, 500_000.0, 0.5, 0.04, 0.5)
    assert got == pytest.approx(10.0, abs=EPS)


def test_carbon_zero_green():
    """Java fixture: carbonMatchesInlineFormula_zeroGreen.

    green=0, demand=1MW, dt=1h, gf=0, bf=0.5
    demandWh = 1_000_000 = 1000 kWh, all brown → 1000 * 0.5 = 500 kg
    """
    got = compute_carbon_kg(0.0, 1_000_000.0, 1.0, 0.0, 0.5)
    assert got == pytest.approx(500.0, abs=EPS)


def test_carbon_zero_demand():
    """Java fixture: carbonMatchesInlineFormula_zeroDemand."""
    got = compute_carbon_kg(1_000_000.0, 0.0, 1.0, 0.0, 0.5)
    assert got == pytest.approx(0.0, abs=EPS)


def test_green_used_wasted_typical():
    """Java fixture: greenUsedWastedSplit_typicalCase.

    green=800kW, demand=1500kW, dt=1h → greenAvailableWh=800k, demandWh=1500k
    used = min(1500k, 800k) = 800k; wasted = 800k - 800k = 0
    """
    used, wasted = compute_green_used_wasted_wh(800_000.0, 1_500_000.0, 1.0)
    assert used == pytest.approx(800_000.0, abs=EPS)
    assert wasted == pytest.approx(0.0, abs=EPS)


def test_green_used_wasted_excess_green():
    """Java fixture: greenUsedWastedSplit_greenExceedsDemand.

    green=2MW, demand=0.5MW, dt=1h → used=500k, wasted=1500k
    """
    used, wasted = compute_green_used_wasted_wh(2_000_000.0, 500_000.0, 1.0)
    assert used == pytest.approx(500_000.0, abs=EPS)
    assert wasted == pytest.approx(1_500_000.0, abs=EPS)


def test_waste_ratio_zero_when_no_green():
    """Java fixture: wasteRatio_zeroWhenNoGreen."""
    assert compute_waste_ratio(0.0, 1_000.0, 1.0) == pytest.approx(0.0, abs=EPS)


def test_waste_ratio_typical():
    """Java fixture: wasteRatio_typical.

    green=2000W, demand=500W, dt=1h
    used=500, wasted=1500 → ratio = 1500/2000 = 0.75
    """
    assert compute_waste_ratio(2000.0, 500.0, 1.0) == pytest.approx(0.75, abs=EPS)


def test_negative_inputs_clamped():
    """Java fixture: negativeInputs_clampedToZero."""
    got = compute_carbon_kg(-100.0, -200.0, 1.0, 0.0, 0.5)
    assert got == pytest.approx(0.0, abs=EPS)


# ---------------------------------------------------------------------------
# Aggregated helpers — these have no Java counterpart; verify summation logic.
# ---------------------------------------------------------------------------


def test_carbon_kg_aggregated_sums_per_dc():
    """Aggregated carbon should equal Σ per-DC carbon."""
    green = [800_000.0, 0.0, 2_000_000.0]
    demand = [1_500_000.0, 1_000_000.0, 500_000.0]
    dt = 1.0
    gf = [0.0, 0.0, 0.04]
    bf = [0.5, 0.5, 0.5]
    expected = sum(
        compute_carbon_kg(g, p, dt, gf_i, bf_i)
        for g, p, gf_i, bf_i in zip(green, demand, gf, bf)
    )
    got = carbon_kg_aggregated(green, demand, dt, gf, bf)
    assert got == pytest.approx(expected, abs=EPS)


def test_carbon_kg_aggregated_array_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        carbon_kg_aggregated([1, 2], [1, 2, 3], 1.0, [0, 0], [0.5, 0.5])


def test_waste_ratio_aggregated_sums_then_ratios():
    """Aggregated waste = total_wasted / (total_used + total_wasted), NOT mean of ratios.

    DC0: green=2000W, demand=500W, dt=1h → used=500, wasted=1500
    DC1: green=1000W, demand=1000W, dt=1h → used=1000, wasted=0
    Total: used=1500, wasted=1500 → ratio = 1500/3000 = 0.5
    Mean of per-DC ratios would be (0.75 + 0.0) / 2 = 0.375 — different!
    """
    got = waste_ratio_aggregated([2000.0, 1000.0], [500.0, 1000.0], 1.0)
    assert got == pytest.approx(0.5, abs=EPS)


def test_waste_ratio_aggregated_zero_when_no_green():
    got = waste_ratio_aggregated([0.0, 0.0], [1000.0, 2000.0], 1.0)
    assert got == pytest.approx(0.0, abs=EPS)


# ---------------------------------------------------------------------------
# forecast_cf_per_step — high-level helper used by the learner hook.
# ---------------------------------------------------------------------------


def _build_crd_info(actual=None):
    """Reusable CRD info skeleton matching what M0 writes into info["crd"]."""
    return {
        "actual_wind_w": actual or [800_000.0, 0.0, 2_000_000.0],
        "p_total_w": [1_500_000.0, 1_000_000.0, 500_000.0],
        "timestep_hours": 1.0,
        "green_carbon_factor": [0.0, 0.0, 0.04],
        "brown_carbon_factor": [0.5, 0.5, 0.5],
        "running_max_carbon": 1.0,
    }


def test_forecast_cf_zero_when_pred_equals_actual():
    """Perfect forecast → R_forecast = 0 by construction."""
    crd = _build_crd_info()
    got = forecast_cf_per_step(crd, predicted_wind_w=crd["actual_wind_w"], beta=0.5, gamma=0.3)
    assert got == pytest.approx(0.0, abs=EPS)


def test_forecast_cf_returns_zero_on_missing_predicted():
    crd = _build_crd_info()
    assert forecast_cf_per_step(crd, predicted_wind_w=None, beta=0.5, gamma=0.3) == 0.0


def test_forecast_cf_returns_zero_on_missing_crd_field():
    crd = _build_crd_info()
    del crd["timestep_hours"]
    assert forecast_cf_per_step(crd, predicted_wind_w=[0, 0, 0], beta=1, gamma=1) == 0.0


def test_forecast_cf_tracks_injected_bias_sign():
    """
    If predicted < actual (under-forecasts available wind), the agent saw
    less green than was actually available → carbon under "perfect forecast"
    would have been LOWER → R_forecast = β·(carbon_actual - carbon_pred).

    With predicted=0 (no wind in prediction), more brown is "predicted-used"
    → carbon_pred > carbon_actual → R_forecast < 0 (forecast was pessimistic).

    When predicted > actual, carbon_pred < carbon_actual → R_forecast > 0
    (forecast was optimistic).
    """
    crd = _build_crd_info()
    actual = crd["actual_wind_w"]
    # Optimistic forecast (predicts MORE wind than realized) → R_forecast > 0
    optimistic = [a * 2.0 for a in actual]
    r_opt = forecast_cf_per_step(crd, optimistic, beta=1.0, gamma=0.0)
    # Pessimistic forecast (predicts LESS wind) → R_forecast < 0
    pessimistic = [a * 0.5 for a in actual]
    r_pes = forecast_cf_per_step(crd, pessimistic, beta=1.0, gamma=0.0)
    assert r_opt > 0.0, f"optimistic forecast should give R_forecast > 0, got {r_opt}"
    assert r_pes < 0.0, f"pessimistic forecast should give R_forecast < 0, got {r_pes}"


def test_forecast_cf_correlates_with_bias_magnitude():
    """
    Linear test: sweep predicted = actual * scale (scale ∈ [0.1, 2.0]) and
    verify R_forecast monotonically tracks scale (correlation > 0.95).
    """
    import numpy as np
    crd = _build_crd_info()
    actual = crd["actual_wind_w"]
    scales = np.linspace(0.1, 2.0, 20)
    rs = [
        forecast_cf_per_step(crd, [a * s for a in actual], beta=1.0, gamma=1.0)
        for s in scales
    ]
    corr = np.corrcoef(scales, rs)[0, 1]
    assert corr > 0.95, (
        f"R_forecast does not track bias magnitude; correlation={corr:.4f}"
    )


def test_forecast_cf_beta_gamma_weights_apply_correctly():
    """β=0,γ=0 → R_forecast = 0 regardless of bias."""
    crd = _build_crd_info()
    pred = [a * 0.1 for a in crd["actual_wind_w"]]
    assert forecast_cf_per_step(crd, pred, beta=0.0, gamma=0.0) == pytest.approx(0.0, abs=EPS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
