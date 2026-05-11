"""
M2.2 step-2 fix: verify TimeCAPGodEyeProvider.get_predicted_wind_w_per_dc()
returns horizon-0 per-DC predicted wind power in W, used to populate
info["crd"]["predicted_wind_w"] for the CRD forecast counterfactual.

Avoids loading the real TimeCAP checkpoint by stubbing the provider's
state directly; the accessor is a pure function over `_last_per_t_pred`
and `dc_assignments` so we only need those two fields.

Run from drl-manager/ :
    .venv/bin/python -m pytest tests/test_timecap_predicted_wind_accessor.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.prediction.timecap_godeye_provider import TimeCAPGodEyeProvider


def _make_stub_provider(dc_assignments=None, per_t_pred=None):
    """Build a TimeCAPGodEyeProvider WITHOUT going through __init__ (which
    would require a real TimeCAP checkpoint file). We only populate the
    fields the accessor reads."""
    obj = TimeCAPGodEyeProvider.__new__(TimeCAPGodEyeProvider)
    obj.dc_assignments = dc_assignments or {0: [1, 2], 1: [3]}
    obj.dc_ids = sorted(obj.dc_assignments.keys())
    obj._last_per_t_pred = per_t_pred
    return obj


def test_returns_none_when_no_forecast_yet():
    p = _make_stub_provider(per_t_pred=None)
    assert p.get_predicted_wind_w_per_dc() is None


def test_returns_per_dc_sum_at_horizon_0():
    """Per-DC sum across turbines at horizon 0, kW → W."""
    pred_t1 = np.array([2.0, 5.0, 7.0], dtype=np.float32)   # kW
    pred_t2 = np.array([3.0, 8.0, 10.0], dtype=np.float32)
    pred_t3 = np.array([1.5, 4.0, 6.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [1, 2], 1: [3]},
        per_t_pred={1: pred_t1, 2: pred_t2, 3: pred_t3},
    )
    out = p.get_predicted_wind_w_per_dc(horizon=0)
    assert out is not None
    assert len(out) == 2
    # DC 0: t1[0] + t2[0] = 2.0 + 3.0 = 5.0 kW = 5000 W
    assert out[0] == pytest.approx(5000.0)
    # DC 1: t3[0] = 1.5 kW = 1500 W
    assert out[1] == pytest.approx(1500.0)


def test_horizon_param_picks_right_index():
    pred = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    p = _make_stub_provider(
        dc_assignments={0: [42]},
        per_t_pred={42: pred},
    )
    assert p.get_predicted_wind_w_per_dc(horizon=0)[0] == pytest.approx(1000.0)
    assert p.get_predicted_wind_w_per_dc(horizon=1)[0] == pytest.approx(2000.0)
    assert p.get_predicted_wind_w_per_dc(horizon=3)[0] == pytest.approx(4000.0)


def test_missing_turbine_in_pred_dict_is_skipped():
    """If a DC's turbine isn't in per_t_pred, that turbine contributes 0."""
    p = _make_stub_provider(
        dc_assignments={0: [1, 99]},  # turbine 99 isn't in per_t_pred
        per_t_pred={1: np.array([3.0, 5.0], dtype=np.float32)},
    )
    # Only turbine 1 contributes: 3.0 kW * 1000 = 3000 W
    assert p.get_predicted_wind_w_per_dc(horizon=0)[0] == pytest.approx(3000.0)


def test_horizon_beyond_pred_length_skips_silently():
    """Asking for a horizon longer than the forecast doesn't crash."""
    p = _make_stub_provider(
        dc_assignments={0: [1]},
        per_t_pred={1: np.array([5.0, 7.0], dtype=np.float32)},  # pred_len=2
    )
    out = p.get_predicted_wind_w_per_dc(horizon=10)
    # Turbine pred too short → contribution is 0 → DC sum is 0
    assert out == [0.0]


def test_dc_ordering_matches_dc_ids():
    """Output order must match self.dc_ids ordering, NOT dict insertion."""
    p = _make_stub_provider(
        dc_assignments={5: [50], 1: [10]},  # weird dc_id values
        per_t_pred={
            10: np.array([1.0], dtype=np.float32),
            50: np.array([2.0], dtype=np.float32),
        },
    )
    # dc_ids after stub init: sorted([5, 1]) = [1, 5]
    out = p.get_predicted_wind_w_per_dc(horizon=0)
    assert out[0] == pytest.approx(1000.0)  # DC 1 (turbine 10): 1.0 kW
    assert out[1] == pytest.approx(2000.0)  # DC 5 (turbine 50): 2.0 kW


def test_units_match_actual_wind_w_field():
    """
    Sanity: actual_wind_w in info["crd"] is in WATTS (per Java
    DatacenterInstance.getCurrentGreenPowerW). Our predicted_wind_w must
    use the same unit so M2.2's compute_carbon_kg sees apples-to-apples.

    Convention: TimeCAP outputs kW → multiply by 1000.0 → W.
    """
    pred = np.array([0.5], dtype=np.float32)  # 0.5 kW = 500 W
    p = _make_stub_provider(
        dc_assignments={0: [7]},
        per_t_pred={7: pred},
    )
    assert p.get_predicted_wind_w_per_dc(horizon=0) == [500.0]


def test_provider_returns_shorter_list_when_subset_of_dcs_have_turbines():
    """
    Real-world setup (v2_5dc): only DCs 0/1/2 have turbines; DCs 3/4 are
    brown-only. The provider's get_predicted_wind_w_per_dc() returns a
    list of length 3, not 5. The env-side padding (in
    HierarchicalMultiDCEnv._collect_crd_info) must align it to
    num_datacenters before handing it to forecast_cf_per_step, which is
    strict about per-DC array lengths.

    This test fixes the contract on the provider side: the accessor
    returns a list ordered by `self.dc_ids`, with whatever length
    `self.dc_ids` has — no automatic padding from the provider's side.
    """
    p = _make_stub_provider(
        dc_assignments={0: [10], 1: [20], 2: [30]},  # 3 DCs with turbines
        per_t_pred={
            10: np.array([100.0], dtype=np.float32),
            20: np.array([200.0], dtype=np.float32),
            30: np.array([300.0], dtype=np.float32),
        },
    )
    out = p.get_predicted_wind_w_per_dc(horizon=0)
    # Provider returns length 3 — env must pad to its num_datacenters.
    assert len(out) == 3
    assert out == [100_000.0, 200_000.0, 300_000.0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
