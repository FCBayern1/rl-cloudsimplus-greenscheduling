"""A-prime bins perturbation - the four Codex-mandated tests plus wiring locks
(V32B_ANNEAL_SPEC, anti watt-domain sign-off 2026-08-17)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gym_cloudsimplus.envs.hierarchical_multidc_env import perturb_future_bins

CAP = np.array([595.93, 545.33, 211.0, 0.0])


class TestAntiAPrime:
    def test_zero_and_capacity_swap(self):
        bins = np.zeros((4, 3)); bins[1, :] = CAP[1]
        out = perturb_future_bins(bins, "anti", CAP)
        assert np.allclose(out[0], CAP[0])       # 0 -> H_d
        assert np.allclose(out[1], 0.0)          # H_d -> 0

    def test_no_green_dc_stays_zero(self):
        bins = np.full((4, 3), 100.0)            # even nonsense input on H_d=0
        out = perturb_future_bins(bins, "anti", CAP)
        assert np.allclose(out[3], 0.0)

    def test_involution_within_range(self):
        rng = np.random.default_rng(0)
        bins = rng.uniform(0, 1, (4, 5)) * CAP[:, None]
        twice = perturb_future_bins(
            perturb_future_bins(bins, "anti", CAP), "anti", CAP)
        assert np.allclose(twice, bins, atol=1e-9)

    def test_missing_capacity_fails_fast(self):
        with pytest.raises(RuntimeError, match="capacity vector"):
            perturb_future_bins(np.zeros((4, 3)), "anti", None)

    def test_uniform_ceiling_bug_locked_out(self):
        # the rejected plain-A design: a uniform H=3000 would fabricate
        # 3000 W of future green on a turbine-less DC; A-prime returns 0.
        out = perturb_future_bins(np.zeros((4, 3)), "anti", CAP)
        assert out[3].max() == 0.0 and out[0].max() < 600


class TestShuffleAndWiring:
    def test_shuffle_reverses_dc_axis(self):
        bins = np.arange(12.0).reshape(4, 3)
        out = perturb_future_bins(bins, "shuffle", CAP)
        assert np.allclose(out, bins[::-1])

    def test_none_is_identity(self):
        bins = np.arange(12.0).reshape(4, 3)
        assert perturb_future_bins(bins, "none", CAP) is bins

    def test_both_source_branches_wired(self):
        # source-level lock: godeye AND timecap returns must route through
        # _v32_maybe_perturb_bins - a silent clean-bins bypass regresses the
        # derived-feature leak this fix closes.
        src = (Path(__file__).resolve().parents[1]
               / "gym_cloudsimplus/envs/hierarchical_multidc_env.py").read_text()
        assert src.count("return self._v32_maybe_perturb_bins(") == 2
