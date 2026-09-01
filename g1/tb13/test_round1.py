"""Round 1 must freeze the blind before it can see any EVPI, and must refuse bad inputs.

The denominator of every EVPI in this screen is the frozen blind. If the arm could be
chosen after the oracle ran, or per instance, the ratio would be a function of the result
rather than a measurement of it.
"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402
import round0 as r0  # noqa: E402
import round1 as r1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R0 = os.path.join(HERE, "round0_out")


def test_the_expected_instance_count_is_the_registered_product():
    assert r1.EXPECTED_INSTANCES == 36 * 3 * 3 * 4 == 1296


def test_instances_are_exactly_the_registered_cross_product():
    inst = r1.build_instances(R0)
    assert len(inst) == 1296
    assert len({(json.dumps({k: v for k, v in a.items() if k != "runtime_set"},
                            sort_keys=True, default=str)) for a in inst}) == 1296
    assert {a["n_jobs"] for a in inst} == set(ig.N_JOBS)
    assert {a["wait_cap"] for a in inst} == set(ig.WAIT_CAP_ROWS)
    assert {a["budget_fraction"] for a in inst} == set(ig.BUDGET_FRACTION)


def test_confirmation_turbines_are_never_touched():
    inst = r1.build_instances(R0)
    conf = set(r0.confirmation_pool())
    used = {t for a in inst for site in a["triplet"] for t in site}
    assert not (used & conf)
    assert used <= set(r0.discovery_pool())


def test_a_tampered_anchor_file_is_refused(tmp_path, monkeypatch):
    import shutil
    d = tmp_path / "r0"
    shutil.copytree(R0, d)
    p = d / "round0_anchors.json"
    p.write_text(p.read_text() + "\n")
    # The dirty-tree gate fires first in production; here the input check is under test.
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "")
    with pytest.raises(RuntimeError, match="does not match its manifest hash"):
        r1.preflight(str(d))


def test_phase_a_does_not_import_or_call_the_solver():
    """Freezing the blind must not be able to see an oracle result."""
    src = open(os.path.join(HERE, "round1.py")).read()
    a = src[src.index("def phase_a"):src.index("def _oracle_one")]
    for banned in ("solve(", "_oracle_one", "evpi", "EVPI"):
        assert banned not in a, f"phase A refers to {banned}"


def test_phase_b_uses_only_the_frozen_arm():
    src = open(os.path.join(HERE, "round1.py")).read()
    b = src[src.index("def phase_b"):src.index("def _quantiles")]
    assert "blind_class_diagnostic" not in b
    assert "min(" not in b.split("gates = {")[0].split("blind_c = ")[1], \
        "phase B re-selects a blind instead of using the frozen one"


def _all_fail(ax):
    """Module level so the process pool can pickle it."""
    return {"carbon": {n: None for n in r1.cbl.BLINDS},
            "valid": {n: False for n in r1.cbl.BLINDS},
            "rho_residual": 0.0, "pes_share": 0.5}


def test_no_valid_blind_stops_rather_than_dropping_cells(tmp_path, monkeypatch):
    """If an arm cannot honour the contract everywhere, the run stops."""
    monkeypatch.setattr(r1, "_blinds_one", _all_fail)
    frozen, art = r1.phase_a([{}], str(tmp_path), {"commit": "test"})
    assert frozen is None and art["status"] == "STOP_NO_VALID_BLIND"


def test_preflight_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output",
                        lambda *a, **k: " M g1/tb13/round1.py\n")
    with pytest.raises(RuntimeError, match="dirty tree"):
        r1.preflight(R0)


def test_the_poststop_diagnostic_never_reads_carbon():
    """The diagnostic must not be able to smuggle in a value-of-information number."""
    src = open(os.path.join(HERE, "poststop_feasibility.py")).read()
    for banned in ("carbon_of", "_costs_all", "evpi", "EVPI", "blind_carbon"):
        assert banned not in src, f"the diagnostic references {banned}"
    assert "feasibility_only" in src


def test_the_feasibility_model_has_no_objective():
    """A constant objective is what makes this a contract audit, not an optimisation."""
    src = open(os.path.join(HERE, "poststop_feasibility.py")).read()
    body = src[src.index("def feasibility_only"):src.index("def _one")]
    assert "Minimize" not in body and "Maximize" not in body
    for term in ("green", "brown", "cb[", "cg["):
        assert term not in body, f"the feasibility model refers to {term}"
