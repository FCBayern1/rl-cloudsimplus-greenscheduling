"""The matched-pair reading applies the frozen criterion and hides nothing (Addendum D)."""

import csv
import importlib.util
import os
import sys

import pytest

G1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "g1", "compressed_timecap_s2"))


def _judge():
    spec = importlib.util.spec_from_file_location(
        "matched_pair_judge", os.path.join(G1, "matched_pair_judge.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _write(d, arm, tier, carbons, comp=1.0, ontime=1.0, forced=0.0):
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    for i, c in enumerate(carbons):
        if c is None:
            continue
        p = os.path.join(d, "eval", f"{arm}_{tier}_k{i}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["total_carbon_kg", "completion_rate",
                                              "completion_rate_mi", "ontime_mi_share",
                                              "deadline_forced_count", "global_decision_us_mean",
                                              "global_decision_us_p95", "global_decision_us_p99"])
            w.writeheader()
            w.writerow({"total_carbon_kg": c, "completion_rate": comp, "completion_rate_mi": comp,
                        "ontime_mi_share": ontime, "deadline_forced_count": forced,
                        "global_decision_us_mean": 100.0, "global_decision_us_p95": 150.0,
                        "global_decision_us_p99": 200.0})


def _full(d, j, v_carbon, e_carbon, clean_v=1.0, clean_e=1.0, **kw):
    for s in j.SEEDS:
        _write(d, f"V_s{s}", "shrink75", [v_carbon] * 6)
        _write(d, f"E_s{s}", "shrink75", [e_carbon] * 6, **kw)
        _write(d, f"V_s{s}", "godeye", [clean_v] * 6)
        _write(d, f"E_s{s}", "godeye", [clean_e] * 6)
    _write(d, "cover_argmax", "shrink75", [0.9] * 6)
    _write(d, "cover_argmax", "godeye", [0.8] * 6)


def test_benefit_needs_all_three_seeds_and_three_percent(tmp_path):
    j = _judge()
    _full(str(tmp_path), j, 1.0, 0.95)                       # 5 % better, clean equal
    r = j.judge(str(tmp_path))
    assert r["primary"]["verdict"] == "BENEFIT_ESTABLISHED"
    assert r["primary"]["all_three_negative"] is True
    assert abs(r["primary"]["rel_mean"] + 0.05) < 1e-9


def test_small_consistent_gain_is_only_suggestive(tmp_path):
    j = _judge()
    _full(str(tmp_path), j, 1.0, 0.99)                       # 1 % better on every seed
    assert j.judge(str(tmp_path))["primary"]["verdict"] == "SUGGESTIVE_NOT_ESTABLISHED"


def test_clean_regression_fails_regardless_of_the_contaminated_result(tmp_path):
    j = _judge()
    _full(str(tmp_path), j, 1.0, 0.80, clean_v=1.0, clean_e=1.10)   # 20 % better, clean 10 % worse
    assert j.judge(str(tmp_path))["primary"]["verdict"] == "FAIL_CLEAN_GUARD"


def test_mixed_signs_are_not_a_benefit_even_if_the_mean_is_negative(tmp_path):
    j = _judge()
    d = str(tmp_path)
    seeds = list(j.SEEDS)
    for s, e in zip(seeds, (0.80, 0.85, 1.10)):              # two win, one loses
        _write(d, f"V_s{s}", "shrink75", [1.0] * 6); _write(d, f"E_s{s}", "shrink75", [e] * 6)
        _write(d, f"V_s{s}", "godeye", [1.0] * 6); _write(d, f"E_s{s}", "godeye", [1.0] * 6)
    r = j.judge(d)
    assert r["primary"]["verdict"] != "BENEFIT_ESTABLISHED"
    assert r["paired"]["shrink75"]["all_negative"] is False
    assert r["paired"]["shrink75"]["rel_min"] < 0 < r["paired"]["shrink75"]["rel_max"]


def test_lower_carbon_with_lower_completion_is_flagged(tmp_path):
    j = _judge()
    _full(str(tmp_path), j, 1.0, 0.90, comp=0.7)
    r = j.judge(str(tmp_path))
    assert any("lower completion" in f for f in r["flags"])


def test_missing_cells_are_counted_not_dropped(tmp_path):
    j = _judge()
    d = str(tmp_path)
    _full(d, j, 1.0, 0.95)
    os.remove(os.path.join(d, "eval", f"E_s{j.SEEDS[0]}_shrink75_k3.csv"))
    r = j.judge(d)
    assert r["arms"][f"E_s{j.SEEDS[0]}"]["shrink75"]["n_missing"] == 1
    assert any("missing cells" in f for f in r["flags"])


def test_incomplete_seed_set_cannot_be_read_as_a_verdict(tmp_path):
    j = _judge()
    d = str(tmp_path)
    s = j.SEEDS[0]
    _write(d, f"V_s{s}", "shrink75", [1.0] * 6); _write(d, f"E_s{s}", "shrink75", [0.5] * 6)
    _write(d, f"V_s{s}", "godeye", [1.0] * 6); _write(d, f"E_s{s}", "godeye", [1.0] * 6)
    assert j.judge(d)["primary"]["verdict"] == "INCOMPLETE"


def test_per_window_differences_are_reported_so_one_window_cannot_hide(tmp_path):
    j = _judge()
    d = str(tmp_path)
    for s in j.SEEDS:                                        # gain sits in one window only
        _write(d, f"V_s{s}", "shrink75", [1.0] * 6)
        _write(d, f"E_s{s}", "shrink75", [1.0, 1.0, 0.4, 1.0, 1.0, 1.0])
        _write(d, f"V_s{s}", "godeye", [1.0] * 6); _write(d, f"E_s{s}", "godeye", [1.0] * 6)
    r = j.judge(d)
    pw = r["paired"]["shrink75"]["per_seed"][str(j.SEEDS[0])]["per_window_rel"]
    assert sum(1 for x in pw if x and abs(x) > 1e-9) == 1     # visible in the record
    assert r["primary"]["all_three_negative"] is True         # pooled looks like a win ...
    assert abs(pw[2] + 0.6) < 1e-9                            # ... but the record shows why


def _write_decisions(d, arm, tier, rows, window=0):
    """rows: list of (slot, cloudlet_id, site, kappa)"""
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    p = os.path.join(d, "eval", f"{arm}_{tier}_k{window}_decisions.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "step", "slot", "cloudlet_id", "site", "kappa"])
        w.writeheader()
        for slot, cid, site, kap in rows:
            w.writerow({"episode": 0, "step": 10 + slot, "slot": slot, "cloudlet_id": cid,
                        "site": site, "kappa": kap})


def test_action_divergence_ignores_padding_and_counts_real_changes(tmp_path):
    j = _judge()
    d = str(tmp_path)
    _full(d, j, 1.0, 0.95)
    s = j.SEEDS[0]
    # a padding slot (cloudlet_id -1) carrying action 0 must not be counted as a decision
    _write_decisions(d, f"V_s{s}", "shrink75", [(0, 11, 0, 5), (1, 12, 1, 70), (2, -1, 0, 0)])
    _write_decisions(d, f"E_s{s}", "shrink75", [(0, 11, 0, 5), (1, 12, 2, 60), (2, -1, 0, 0)])
    _write_decisions(d, "cover_argmax", "shrink75", [(0, 11, 0, 70), (1, 12, 1, 70), (2, -1, 0, 0)])
    r = j.judge(d)
    ad = r["action_divergence"]["shrink75"]
    assert ad["per_arm"][f"V_s{s}"]["n_decisions"] == 2          # the padding slot is excluded
    assert ad["per_arm"][f"V_s{s}"]["sites"] == {0: 1, 1: 1}
    pair = ad["pairs"][f"V_s{s}|E_s{s}"]
    assert pair["n_shared"] == 2
    assert abs(pair["identical_frac"] - 0.5) < 1e-9             # one of two choices differs
    assert abs(pair["same_site_frac"] - 0.5) < 1e-9
    assert abs(pair["mean_abs_dkappa"] - 5.0) < 1e-9            # (0 + 10) / 2
    # and the comparison against the initial rule is available for question 2
    assert f"V_s{s}|cover_argmax" in pair or any("cover_argmax" in k for k in ad["pairs"])


def test_action_divergence_reports_a_policy_that_never_moves(tmp_path):
    j = _judge()
    d = str(tmp_path)
    _full(d, j, 1.0, 1.0)
    s = j.SEEDS[0]
    same = [(0, 11, 0, 5), (1, 12, 0, 5)]
    _write_decisions(d, f"V_s{s}", "shrink75", same)
    _write_decisions(d, f"E_s{s}", "shrink75", same)
    ad = j.judge(d)["action_divergence"]["shrink75"]
    assert ad["pairs"][f"V_s{s}|E_s{s}"]["identical_frac"] == 1.0
    assert ad["per_arm"][f"E_s{s}"]["sites"] == {0: 2}          # collapsed onto one site, visible
