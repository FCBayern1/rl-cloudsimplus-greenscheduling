"""Round 0 must be a different set from the axis grid, and every gate must be arithmetic.

Both sets happen to number 8,640. That is a coincidence of the axis sizes, and mistaking
one for the other would silently screen the wrong thing, so the two are separated here.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402
import round0 as r0  # noqa: E402


@pytest.fixture(scope="module")
def keys():
    return r0.round0_physical_keys()


def test_there_are_exactly_the_expected_physical_units(keys):
    assert len(keys) == 8640
    uniq = {(k["pes_per_job"], k["concurrency"], k["turbines_per_site"],
             k["installed_divisor"], k["horizon"], k["triplet_index"],
             k["season_index"]) for k in keys}
    assert len(uniq) == 8640, "physical keys are not unique"


def test_physical_keys_carry_no_workload_axis(keys):
    forbidden = {"n_jobs", "wait_cap", "budget_fraction", "seed", "runtime_set"}
    for k in keys[:50]:
        assert not (forbidden & set(k)), f"a workload axis leaked in: {forbidden & set(k)}"


def test_physical_keys_span_six_triplets_and_six_seasons(keys):
    assert len({k["triplet_index"] for k in keys}) == r0.N_TRIPLETS
    assert len({k["season_index"] for k in keys}) == r0.N_SEASONS
    assert len({(k["triplet_index"], k["season_index"]) for k in keys}) == 36


def test_confirmation_turbines_are_never_touched(keys):
    conf = set(r0.confirmation_pool())
    used = {t for k in keys for site in k["triplet"] for t in site}
    assert not (used & conf), f"confirmation turbines appeared: {sorted(used & conf)}"
    assert used <= set(r0.discovery_pool())


def test_the_gate_applies_no_rho_cut(keys):
    """rho is recorded for every unit and never used to reject one."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "round0.py")).read()
    gate = src[src.index("def passes_physical_gate"):src.index("def neighbourhood")]
    assert "rho" not in gate, "the physical gate refers to rho"
    m = r0.physical_metrics(keys[0])
    assert "rho_residual" in m


def test_round0_never_calls_the_solver():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "round0.py")).read()
    for banned in ("cp_model", "exact_oracle", "solve("):
        assert banned not in src, f"round 0 references {banned}"


def test_metrics_are_reproducible(keys):
    a = r0.physical_metrics(keys[100])
    b = r0.physical_metrics(keys[100])
    ha = hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
    assert ha == hb


def test_simultaneous_poor_is_the_no_site_can_cover_one_job_fraction(keys):
    k = keys[500]
    gres, _ = r0.residual_green(k)
    p_job = k["pes_per_job"] * ig.DYN_W_PER_PE
    expect = float((gres.max(axis=0) < p_job).mean())
    assert r0.physical_metrics(k)["simultaneous_poor_fraction"] == pytest.approx(expect)


def test_best_dc_change_fraction_matches_its_definition(keys):
    k = keys[700]
    gres, _ = r0.residual_green(k)
    p = k["pes_per_job"] * ig.DYN_W_PER_PE
    cb = np.asarray(ig.BROWN_FACTORS); cg = np.asarray(ig.GREEN_FACTORS)
    marg = (cb.reshape(-1, 1) * np.maximum(p - gres, 0.0)
            + cg.reshape(-1, 1) * np.minimum(p, gres))
    best = np.argmin(marg, axis=0)
    expect = 1.0 - np.bincount(best, minlength=3).max() / len(best)
    assert r0.physical_metrics(k)["best_dc_change_fraction"] == pytest.approx(expect)


def test_a_site_that_always_wins_is_rejected():
    m = {"corr_degenerate": False, "pairwise_corr": [0.8, 0.8, 0.8],
         "simultaneous_poor_fraction": 0.5, "best_dc_change_fraction": 0.05}
    ok, why = r0.passes_physical_gate(m)
    assert not ok and "fixed" in why


def test_negative_correlation_is_rejected_not_absolutised():
    m = {"corr_degenerate": False, "pairwise_corr": [-0.85, 0.8, 0.8],
         "simultaneous_poor_fraction": 0.5, "best_dc_change_fraction": 0.5}
    ok, why = r0.passes_physical_gate(m)
    assert not ok and "correlation" in why


def test_degenerate_poor_fractions_are_rejected():
    for f in (0.0, 1.0):
        m = {"corr_degenerate": False, "pairwise_corr": [0.8] * 3,
             "simultaneous_poor_fraction": f, "best_dc_change_fraction": 0.5}
        ok, why = r0.passes_physical_gate(m)
        assert not ok and "degenerate" in why


def test_anchor_sha_is_canonical_and_ignores_incidental_fields(keys):
    k = dict(keys[0])
    h1 = r0.anchor_sha(k)
    k["triplet_index"] = 99          # not part of the payload
    k["season_index"] = 99
    assert r0.anchor_sha(k) == h1
    k2 = dict(keys[0]); k2["installed_divisor"] = 24000
    assert r0.anchor_sha(k2) != h1


def test_neighbourhoods_are_three_consecutive_divisors():
    order = list(ig.INSTALLED_DIVISOR)
    for d in order:
        nb = r0.neighbourhood(d)
        assert len(nb) == 3 and nb == sorted(nb)
        idx = [order.index(x) for x in nb]
        assert idx == list(range(idx[0], idx[0] + 3))
        assert d in nb


def test_anchor_selection_takes_the_smallest_hashes_per_layer(keys):
    subset = [k for k in keys if k["triplet_index"] < 2 and k["season_index"] < 2]
    chosen, empty = r0.select_anchors(subset)
    layers = {(k["triplet_index"], k["season_index"], k["turbines_per_site"])
              for k in subset}
    assert len(chosen) == r0.ANCHORS_PER_LAYER * len(layers)
    for lid in layers:
        mine = sorted([k for k in subset
                       if (k["triplet_index"], k["season_index"],
                           k["turbines_per_site"]) == lid], key=r0.anchor_sha)
        got = [k for k in chosen
               if (k["triplet_index"], k["season_index"],
                   k["turbines_per_site"]) == lid]
        assert ([r0.anchor_sha(k) for k in got]
                == [r0.anchor_sha(k) for k in mine[:r0.ANCHORS_PER_LAYER]])


def test_the_layer_design_is_seventy_two_not_thirty_six():
    """Both turbine counts get their own six triplets, so the layers double."""
    assert len(r0.expected_layers()) == 72
    assert r0.ANCHORS_PER_LAYER * 72 * 3 * len(ig.BUDGET_FRACTION) == 1728


def test_empty_layers_are_reported_not_skipped():
    chosen, empty = r0.select_anchors([])
    assert chosen == [] and len(empty) == 72, "a layer with no survivor went unreported"


def test_the_cli_writes_non_empty_artifacts(tmp_path):
    """The module has to run, not merely import: an entry point that does nothing is a
    silent no-op that looks like a completed screen.

    From a dirty tree the provenance gate refuses on purpose, so both outcomes are
    asserted rather than only the convenient one.
    """
    import subprocess
    out = tmp_path / "out"
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "round0.py")],
                       capture_output=True, text=True, cwd=str(tmp_path),
                       env={**os.environ, "TB13_ROUND0_OUT": str(out)})
    if r.returncode != 0:
        assert "dirty tree" in r.stderr, r.stderr[-2000:]
        return
    assert len(r.stdout) > 0, "the entry point produced no output"
    for name in ("round0_all.jsonl", "round0_anchors.json", "round0_summary.json",
                 "round0_manifest.json"):
        p = os.path.join(str(out), name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, f"{name} missing or empty"
    summary = json.load(open(os.path.join(str(out), "round0_summary.json")))
    assert summary["total_units"] == 8640
    assert summary["layers_expected"] == 72
    assert summary["seed0_solve_cap"] == 1728
