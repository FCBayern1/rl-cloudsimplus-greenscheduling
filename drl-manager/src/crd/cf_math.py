"""
M2.0 — Pure-Python counterfactual math helpers for the EU-CRD framework.

These mirror the analytical helpers in
`cloudsimplus-gateway/.../EnergyMetricsDelta.java` exactly. The learner side
of training does not have access to the Java gateway (it runs on Ray
workers / GPU processes), but M0 ensures every transition's `info["crd"]`
carries the snapshot data needed to recompute carbon and waste in pure
Python:

    info["crd"] = {
        "actual_wind_w":         [W per DC at end of step],
        "predicted_wind_w":      [Ŵ per DC, only when WindPredictionWrapper
                                  is in use; otherwise the key is absent],
        "p_total_w":             [P_total per DC],
        "running_max_carbon":    float,
        "timestep_hours":        float,
        "green_carbon_factor":   [kgCO2/kWh per DC],
        "brown_carbon_factor":   [kgCO2/kWh per DC],
    }

Two-tier API:
- low-level per-DC formulas (`compute_carbon_kg`, `compute_green_used_wasted_wh`,
  `compute_waste_ratio`) — bit-exact mirrors of the Java statics.
- high-level per-step helpers (`carbon_kg_aggregated`, `waste_ratio_aggregated`,
  `forecast_cf_per_step`) — what the learner-side hook will actually call.

Numerical contract: given the same scalar inputs, results must match the Java
helpers to within float64 tolerance (the tests in test_crd_cf_math.py mirror
the Java JUnit fixtures exactly).
"""

from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# Low-level per-DC helpers — bit-exact mirrors of EnergyMetricsDelta statics.
# ---------------------------------------------------------------------------


def compute_green_used_wasted_wh(
    available_green_w: float, demand_w: float, duration_hours: float
) -> tuple[float, float]:
    """
    Split a single (green, demand) pair over a step into (used_wh, wasted_wh).

    Mirrors EnergyMetricsDelta.computeGreenUsedWastedWh:
      - greenAvailableWh = max(0, green) * dt
      - demandWh         = max(0, demand) * dt
      - greenUsed        = min(demandWh, greenAvailableWh)
      - greenWasted      = greenAvailableWh - greenUsed
    """
    demand_wh = max(0.0, demand_w) * duration_hours
    green_avail_wh = max(0.0, available_green_w) * duration_hours
    green_used_wh = min(demand_wh, green_avail_wh)
    green_wasted_wh = green_avail_wh - green_used_wh
    return green_used_wh, green_wasted_wh


def compute_carbon_kg(
    available_green_w: float,
    demand_w: float,
    duration_hours: float,
    green_factor: float,
    brown_factor: float,
) -> float:
    """
    Carbon emission (kg CO2) for one (green, demand) pair over one step.

    Mirrors EnergyMetricsDelta.computeCarbonKg:
        carbon_kg = greenKWh * green_factor + brownKWh * brown_factor
    where green/brown allocation matches `compute_green_used_wasted_wh`.
    """
    green_used_wh, _wasted = compute_green_used_wasted_wh(
        available_green_w, demand_w, duration_hours
    )
    demand_wh = max(0.0, demand_w) * duration_hours
    brown_used_wh = demand_wh - green_used_wh
    green_kwh = green_used_wh / 1000.0
    brown_kwh = brown_used_wh / 1000.0
    return green_kwh * green_factor + brown_kwh * brown_factor


def compute_waste_ratio(
    available_green_w: float, demand_w: float, duration_hours: float
) -> float:
    """
    Per-step waste ratio for one DC: wasted / (used + wasted), 0 when no green.
    """
    used, wasted = compute_green_used_wasted_wh(
        available_green_w, demand_w, duration_hours
    )
    total = used + wasted
    return wasted / total if total > 0.0 else 0.0


# ---------------------------------------------------------------------------
# High-level per-step helpers used by the learner hook.
# ---------------------------------------------------------------------------


def carbon_kg_aggregated(
    green_per_dc: Sequence[float],
    demand_per_dc: Sequence[float],
    duration_hours: float,
    green_factor_per_dc: Sequence[float],
    brown_factor_per_dc: Sequence[float],
) -> float:
    """Total carbon (kg) summed across all datacenters in one step."""
    if not (
        len(green_per_dc)
        == len(demand_per_dc)
        == len(green_factor_per_dc)
        == len(brown_factor_per_dc)
    ):
        raise ValueError(
            "carbon_kg_aggregated: per-DC arrays must all have the same length; "
            f"got {len(green_per_dc)}, {len(demand_per_dc)}, "
            f"{len(green_factor_per_dc)}, {len(brown_factor_per_dc)}"
        )
    total = 0.0
    for g, p, gf, bf in zip(
        green_per_dc, demand_per_dc, green_factor_per_dc, brown_factor_per_dc
    ):
        total += compute_carbon_kg(g, p, duration_hours, gf, bf)
    return total


def waste_ratio_aggregated(
    green_per_dc: Sequence[float],
    demand_per_dc: Sequence[float],
    duration_hours: float,
) -> float:
    """
    Aggregated waste ratio across DCs: total_wasted / (total_used + total_wasted).

    Mirrors MultiDatacenterSimulationCore.calculateGreenWasteRatio (which sums
    per-DC numerators and denominators before taking the ratio — NOT the mean
    of per-DC ratios).
    """
    if len(green_per_dc) != len(demand_per_dc):
        raise ValueError(
            "waste_ratio_aggregated: per-DC arrays must have same length"
        )
    total_used = 0.0
    total_wasted = 0.0
    for g, p in zip(green_per_dc, demand_per_dc):
        used, wasted = compute_green_used_wasted_wh(g, p, duration_hours)
        total_used += used
        total_wasted += wasted
    total = total_used + total_wasted
    return total_wasted / total if total > 0.0 else 0.0


def forecast_cf_per_step(
    crd_info: Mapping,
    predicted_wind_w: Sequence[float],
    beta: float,
    gamma: float,
    carbon_norm: bool = False,
    magnitude: bool = False,
) -> float:
    """
    R_forecast for a single timestep, given the CRD info snapshot and the
    predicted wind that fed the agent's decision at this step.

    Formula:
        R_forecast = β · [Ĉ(W_actual) - Ĉ(W_predicted)]
                   + γ · [R_w(W_actual) - R_w(W_predicted)]

    where Ĉ is *raw* carbon kg (no normalization here — M5 / loss
    weighting decides if and how to normalize).

    Returns 0.0 if any required snapshot field is missing; callers should
    log a warn-once if they care.
    """
    actual = crd_info.get("actual_wind_w")
    p_total = crd_info.get("p_total_w")
    dt = crd_info.get("timestep_hours")
    gf = crd_info.get("green_carbon_factor")
    bf = crd_info.get("brown_carbon_factor")
    if actual is None or p_total is None or dt is None or gf is None or bf is None:
        return 0.0
    if predicted_wind_w is None:
        return 0.0

    carbon_actual = carbon_kg_aggregated(actual, p_total, dt, gf, bf)
    carbon_pred = carbon_kg_aggregated(predicted_wind_w, p_total, dt, gf, bf)
    waste_actual = waste_ratio_aggregated(actual, p_total, dt)
    waste_pred = waste_ratio_aggregated(predicted_wind_w, p_total, dt)
    d_carbon = carbon_actual - carbon_pred
    d_waste = waste_actual - waste_pred
    # v5.2 (carbon_norm): the raw carbon diff is dt-scaled kg (~1e-3 at a 1 s
    # timestep) while the waste diff is dimensionless (~0.3) — without this
    # the beta term is numerically dead and R_forecast measures forecast error
    # through the waste proxy only. Normalise by the worst-case (all-brown)
    # step carbon so both terms are dimensionless and beta/gamma weight what
    # they claim to weight.
    if carbon_norm:
        try:
            c_max = sum(
                float(p) * float(dt) / 1000.0 * float(b)
                for p, b in zip(p_total, bf)
            )
        except TypeError:
            c_max = float(p_total) * float(dt) / 1000.0 * float(bf)
        if c_max > 1e-12:
            d_carbon = d_carbon / c_max
    # v5.2 (magnitude): the beta and gamma terms carry OPPOSITE signs for the
    # same forecast error and partially cancel — |R_f| would dip through ~0 at
    # the cancellation point, under-attributing a genuinely large error. The
    # attribution pipeline consumes |R_f| anyway, so summing magnitudes gives
    # a monotone forecast-error signal.
    if magnitude:
        return beta * abs(d_carbon) + gamma * abs(d_waste)
    return beta * d_carbon + gamma * d_waste
