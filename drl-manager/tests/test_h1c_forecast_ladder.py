"""H1c ladder units: the pre-registered feature gate and arm configs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h1c_forecast_ladder import arm_config, feature_gate_flags


def _obs(best_now, best_fut, mi=None, ttd=None, present=None, backlog=0.0):
    n = len(best_now)
    return {"batch_cloudlet_best_now_carbon": np.asarray(best_now, float),
            "batch_cloudlet_best_future_carbon": np.asarray(best_fut, float),
            "batch_cloudlet_mi": np.asarray(mi if mi is not None else [1e6] * n, float),
            "batch_cloudlet_time_to_deadline": np.asarray(
                ttd if ttd is not None else [0.8] * n, float),   # x3600 = 2880s
            "batch_cloudlet_deadline_present": np.asarray(
                present if present is not None else [1.0] * n, float),
            "global_deferred_count": np.asarray([backlog / 2000.0])}


class TestFeatureGate:
    def test_relative_gain_threshold_both_sides(self):
        g = _obs([0.30, 0.30], [0.26, 0.28])   # 13.3% vs 6.7% saving
        f = feature_gate_flags(g, 2, 3600.0, t=0)
        assert f.tolist() == [True, False]

    def test_persistence_tuple_never_defers(self):
        # blind fill: best_future == best_now -> rel gain 0 -> gate closed
        g = _obs([0.30, 0.10], [0.30, 0.10])
        assert not feature_gate_flags(g, 2, 3600.0, t=0).any()

    def test_decision_horizon_binds_not_drain(self):
        # near the 7200 decision boundary the gate must close even though the
        # 10000 drain window would still leave time
        g = _obs([0.30], [0.10], ttd=[2.0])    # deadline slack huge
        assert feature_gate_flags(g, 1, 3600.0, t=1000).any()
        assert not feature_gate_flags(g, 1, 3600.0, t=7150).any()

    def test_backlog_cap_closes_gate(self):
        g = _obs([0.30], [0.10], backlog=250)
        assert not feature_gate_flags(g, 1, 3600.0, t=0).any()

    def test_padding_and_no_deadline_route(self):
        g = _obs([0.30, 0.30], [0.10, 0.10], mi=[0.0, 1e6], present=[1.0, 0.0])
        assert not feature_gate_flags(g, 2, 3600.0, t=0).any()


class TestArmConfig:
    ORACLE = {"green_oracle_mode": "godeye", "timecap": {"device": "cuda"}}
    NOFC = {"forecast_mode": "none"}
    CAP = [595.9, 0.0]

    def test_clean_forces_timecap_on_cpu(self):
        c = arm_config("clean", self.ORACLE, self.NOFC, self.CAP)
        assert c["green_oracle_mode"] == "timecap"
        assert c["timecap"]["device"] == "cpu"

    def test_anti_carries_capacity_vector(self):
        c = arm_config("anti", self.ORACLE, self.NOFC, self.CAP)
        assert c["v32_perturb_capacity_w"] == self.CAP

    def test_persistence_uses_blind_config(self):
        c = arm_config("persistence", self.ORACLE, self.NOFC, self.CAP)
        assert c.get("forecast_mode") == "none"

    def test_oracle_and_immediate_unmodified(self):
        for a in ("oracle", "immediate", "shuffle"):
            c = arm_config(a, self.ORACLE, self.NOFC, self.CAP)
            assert c["green_oracle_mode"] == "godeye"
            assert "v32_perturb_capacity_w" not in c or a == "anti"
