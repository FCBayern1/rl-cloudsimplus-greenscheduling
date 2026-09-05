import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_selectivity import lift_and_auc  # noqa: E402


def test_perfectly_selective_policy_passes():
    p = np.array([0.9, 0.8, 0.85, 0.1, 0.2, 0.15])
    y = np.array([1, 1, 1, 0, 0, 0])
    r = lift_and_auc(p, y)
    assert r["auc"] == 1.0 and abs(r["lift"] - 0.7) < 1e-9 and r["pass"]


def test_indiscriminate_policy_fails_even_if_it_defers_a_lot():
    p = np.array([0.95] * 8)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    r = lift_and_auc(p, y)
    assert r["lift"] == 0.0 and abs(r["auc"] - 0.5) < 1e-9 and not r["pass"]


def test_balanced_auc_ignores_class_imbalance():
    # 2 positives above 98 negatives: AUC 1.0 regardless of imbalance
    p = np.concatenate([[0.9, 0.8], np.linspace(0.0, 0.5, 98)])
    y = np.concatenate([[1, 1], np.zeros(98, int)])
    r = lift_and_auc(p, y)
    assert r["auc"] == 1.0 and r["n_pos"] == 2 and r["n_neg"] == 98


def test_ties_get_half_credit_and_empty_class_is_reported():
    r = lift_and_auc([0.5, 0.5], [1, 0])
    assert abs(r["auc"] - 0.5) < 1e-9
    r2 = lift_and_auc([0.5, 0.6], [1, 1])
    assert r2["pass"] is False and r2["reason"] == "one class empty"


def test_pairing_takes_first_legal_defer_and_legal_route_per_job_and_excludes_forced_routes():
    from timing_selectivity import pair_corpus
    rows = [
        # job 7: deferred at step 1 (legal), deferred at step 2 (legal), routed at step 3 (legal) -> paired
        {"step": 1, "slot": 0, "cloudlet_id": 7, "is_defer": 1, "defer_allowed": 1.0},
        {"step": 2, "slot": 0, "cloudlet_id": 7, "is_defer": 1, "defer_allowed": 1.0},
        {"step": 3, "slot": 0, "cloudlet_id": 7, "is_defer": 0, "defer_allowed": 1.0},
        # job 8: deferred while legal, then routed only when the mask forbade DEFER -> route excluded
        {"step": 1, "slot": 1, "cloudlet_id": 8, "is_defer": 1, "defer_allowed": 1.0},
        {"step": 5, "slot": 1, "cloudlet_id": 8, "is_defer": 0, "defer_allowed": 0.0},
        # job 9: routed immediately, never deferred -> no pair
        {"step": 1, "slot": 2, "cloudlet_id": 9, "is_defer": 0, "defer_allowed": 1.0},
        # padding / unknown id ignored
        {"step": 1, "slot": 3, "cloudlet_id": -1, "is_defer": 1, "defer_allowed": 1.0},
    ]
    pc = pair_corpus(rows)
    assert pc["n_jobs"] == 3 and pc["n_paired"] == 1
    assert pc["pairs"][7] == {"defer": (1, 0), "route": (3, 0)}
    assert pc["n_route_forced_excluded"] == 1 and pc["n_never_deferred"] == 1
