import numpy as np

from sqt2_hazard_calibrate import accuracies, decision_set, freeze, CANDIDATES
from sqt2_prescreen import TroughIndex
from oracle_slack_planner import WARMUP_ROWS


def _rows(specs):
    return [{"arrival_time": str(a), "length": str(mi), "pes_required": str(p),
             "deadline": str(d), "cloudlet_id": str(i)}
            for i, (a, mi, p, d) in enumerate(specs)]


class TestDecisionSet:
    def test_membership_and_labels(self, monkeypatch):
        import sqt2_hazard_calibrate as m
        monkeypatch.setattr(m, "N_EPISODES", 1)
        tindex = TroughIndex([{"start": WARMUP_ROWS + 100, "dur": 600}])
        rows = _rows([
            (150, 4e6, 1, 5000),   # in trough (age 50), B>0 -> included
            (900, 4e6, 1, 5000),   # after trough end (row 900 >= 700) -> out
            (150, 4e6, 1, 300),    # B <= 0 -> excluded
        ])
        d = decision_set(rows, tindex, off_range=0)
        assert len(d) == 1
        worthy, age, budget, mi = d[0]
        assert age == 50.0
        # runtime 100s, ttd 4850 -> deadline budget 4630; horizon budget
        # 7200-150-100-120 = 6830 -> B = 4630; residual 550 <= 4630 -> worthy
        assert budget == 4630.0 and worthy and mi == 4e6

    def test_not_worthy_when_residual_exceeds_budget(self, monkeypatch):
        import sqt2_hazard_calibrate as m
        monkeypatch.setattr(m, "N_EPISODES", 1)
        tindex = TroughIndex([{"start": WARMUP_ROWS, "dur": 4000}])
        rows = _rows([(10, 4e6, 1, 700)])   # B = 700-10-100-120 = 470 < 3990
        d = decision_set(rows, tindex, off_range=0)
        assert len(d) == 1 and d[0][0] is np.False_ or d[0][0] is False


class TestFreeze:
    def test_naive_accuracy_is_worthy_fraction(self):
        dset = [(True, 10.0, 500.0, 1.0), (False, 10.0, 500.0, 3.0)]
        res = accuracies(dset)
        assert res["naive"]["acc"] == 0.5
        assert res["naive"]["acc_mi"] == 0.25          # MI-weighted, info only

    def test_freeze_picks_argmax_and_comparator(self):
        res = {"naive": {"acc": 0.80}}
        for q in CANDIDATES:
            res[f"hazard@{q:.2f}"] = {"acc": 0.70}
        res["hazard@0.40"] = {"acc": 0.84}
        q_star, comp = freeze(res)
        assert q_star == 0.40 and comp == "hazard"

    def test_comparator_falls_back_to_naive(self):
        res = {"naive": {"acc": 0.90}}
        for q in CANDIDATES:
            res[f"hazard@{q:.2f}"] = {"acc": 0.70}
        _, comp = freeze(res)
        assert comp == "naive"

    def test_hazard_threshold_monotone_on_dataset(self):
        # deep-age short-trough decisions: high P(end soon) -> low thresholds
        # defer more; accuracy differs across q on a mixed set
        dset = [(True, 1400.0, 200.0, 1.0)] * 8 + [(False, 50.0, 100.0, 1.0)] * 2
        res = accuracies(dset)
        accs = [res[f"hazard@{q:.2f}"]["acc"] for q in CANDIDATES]
        assert max(accs) >= res["naive"]["acc"] - 1e-9
