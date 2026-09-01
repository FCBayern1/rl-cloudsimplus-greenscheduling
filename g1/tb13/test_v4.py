"""Tests for the v4 physical map, workloads and Round 0-v4."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants_v4 as c4                        # noqa: E402
import instance_gen as ig                        # noqa: E402
import round0 as r0                              # noqa: E402
import round0_v3 as r3                           # noqa: E402
import round0_v4 as r4                           # noqa: E402
import schedule_feasibility as sf                # noqa: E402
import workload_v3 as w3                         # noqa: E402
import workload_v4 as w4                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ── the registered physical map ─────────────────────────────────────────────

def test_power_curve_closes_on_the_registered_three_points():
    assert c4.power_w(0) == pytest.approx(51.4, abs=1e-9)
    assert c4.power_w(32) == pytest.approx(132.7, abs=1e-9)
    assert c4.power_w(64) == pytest.approx(214.0, abs=1e-9)


def test_dynamic_per_pe_is_derived_not_typed():
    assert c4.DYN_W_PER_PE == pytest.approx((214.0 - 51.4) / 64 * 1.0, abs=1e-12)
    assert c4.CLOUDLET_UTILISATION == 1.0


def test_the_vm_fleet_exposes_the_whole_host():
    assert c4.VMS_PER_SITE * c4.PES_PER_VM == c4.HOST_PES == c4.CAP_PES_PER_SITE == 64


def test_job_sizes_and_the_fluid_control():
    assert c4.PES_PER_JOB == (8, 16, 32)
    assert c4.FLUID_CONTROL_PES == (8,)
    assert 8 / 64 == 0.125 and 16 / 64 == 0.25 and 32 / 64 == 0.5
    assert c4.MIN_PES_SHARE == 0.25


def test_v4_matches_the_java_sentinel_artifact():
    art = json.load(open(os.path.join(HERE, "sentinel_v4_out",
                                      "power_sentinel_layer2.json")))
    assert art["verdict"] == "PASS"
    assert art["vm_pes_total"] == c4.CAP_PES_PER_SITE
    assert art["cloudlet_cpu_utilization"] == c4.CLOUDLET_UTILISATION
    assert art["idle_host_power_down"] is False
    for phase in art["phases"]:
        assert phase["mean_power_w"] == pytest.approx(
            c4.power_w(phase["busy_pes"]), abs=1e-9)


# ── workloads ───────────────────────────────────────────────────────────────

def test_axis_rules_carry_over_unchanged():
    assert w4.compatible_axes() == w3.compatible_axes()
    assert len(w4.compatible_axes()) == 89
    assert len(w4.PES_PER_JOB) * 89 == 267


def test_acceptance_uses_the_64_pe_site():
    key = w4.workload_key(0, 144, 32, 3, 12, 24)
    acc = w4.accepted(key)
    assert acc is not None and acc["retry"] < w4.MAX_RETRIES
    wl = acc["workload"]
    b = w4.budget_for(wl, w4.STRICTEST_BUDGET_FRACTION)
    assert sf.capacity_ok(wl, b, cap=64) == "FEASIBLE"
    assert sf.reservation_edf(wl, b, cap=64)[0] is not None


def test_a_32_pe_load_needs_the_64_pe_site():
    """The same load that fits a 64-PE site cannot fit the v3 16-PE one."""
    key = w4.workload_key(0, 144, 32, 3, 12, 24)
    wl = w4.accepted(key)["workload"]
    b = w4.budget_for(wl, 0.10)
    assert sf.reservation_edf(wl, b, cap=16)[0] is None


def test_capacity_default_is_unchanged_for_v3():
    key = w3.workload_key(0, 144, 4, 3, 12, 24)
    wl = w3.draw(key, 0)
    b = w3.budget_for(wl, 0.10)
    assert sf.reservation_edf(wl, b) == sf.reservation_edf(wl, b, cap=ig.CAP_PES_PER_SITE)


def test_workload_content_is_independent_of_the_power_map():
    """Loads depend on the axes only, so v3 and v4 agree wherever the axes agree."""
    a = w3.draw(w3.workload_key(0, 96, 8, 2, 10, 12), 0)
    b = w4.draw(w4.workload_key(0, 96, 8, 2, 10, 12), 0)
    assert w3.content_hash(a) == w4.content_hash(b)


# ── round 0-v4 ──────────────────────────────────────────────────────────────

def test_physical_space_is_12960_and_unique():
    keys = r4.physical_keys()
    assert len(keys) == 12960
    assert len({r4.key_sha(k) for k in keys}) == 12960


def test_v4_keys_differ_from_v3_keys_under_the_same_axes():
    """The digest carries the power map, so a v4 cell can never be mistaken for a v3 one."""
    assert c4.grid_hash_v4() != r3.grid_hash_v3()


def test_windows_and_horizons_are_v3s():
    assert r4.base_offsets() == r3.base_offsets()
    assert r4.HORIZONS == r3.HORIZONS


def test_confirmation_turbines_are_untouched():
    conf = set(r0.confirmation_pool())
    used = set().union(*(r4._turbines(k) for k in r4.physical_keys()))
    assert not conf & used


def test_metrics_use_the_v4_power_and_capacity():
    k = dict(r4.physical_keys()[0])
    k["pes_per_job"] = 32
    m = r4.physical_metrics(k)
    assert m["pes_share"] == 0.5
    gres, _g = r4.residual_green(k)
    assert m["rho_residual"] == pytest.approx(
        k["concurrency"] * 32 * c4.DYN_W_PER_PE / max(gres.mean(), 1e-9), rel=1e-12)


def test_physical_gate_is_the_registered_one():
    assert r4.r0.CORR_BAND == (0.70, 0.95)
    assert r4.r0.BEST_DC_CHANGE_MIN == 0.10


def test_block_shape_and_cohort_rules_are_v3s():
    a = r4.physical_keys()[0]
    b = r4.build_block(a, (10, 12))
    assert len(b["cells"]) == 12 == r4.CELLS_PER_BLOCK
    assert len(b["divisors"]) == 3
    assert b["budget_fractions"] == list(c4.BUDGET_FRACTION)
    assert r4.MAX_BLOCKS == 144


def test_round0_refuses_without_a_passing_power_gate(tmp_path, monkeypatch):
    gate = json.load(open(os.path.join(HERE, "sentinel_v4_out",
                                       "power_gate_freeze.json")))
    assert gate["verdict"] == "PASS"
    bad = tmp_path / "sentinel_v4_out"
    bad.mkdir()
    gate["verdict"] = "STOP"
    (bad / "power_gate_freeze.json").write_text(json.dumps(gate))
    monkeypatch.setattr(r4, "HERE", str(tmp_path))
    with pytest.raises(RuntimeError, match="power gate"):
        r4._provenance(os.path.abspath(os.path.join(HERE, "..", "..")))


# ── zero-emissions preflight and round 1-v4 ─────────────────────────────────

import causal_blinds as cbl                      # noqa: E402
import round1_v4 as r1v4                         # noqa: E402
import zero_emission_v4 as z4                    # noqa: E402
from exact_oracle import validate_assignment     # noqa: E402


def _cohort_dir():
    return os.path.join(HERE, "round0_v4_out")


def _cells(k=None):
    cohort, _i = z4.load_cohort(_cohort_dir())
    cells = z4.cohort_cells(cohort)
    return cells if k is None else cells[:k]


def test_cohort_v4_digest_and_manifest_agree():
    _c, integrity = z4.load_cohort(_cohort_dir())
    assert integrity["cohort_sha_matches"] and integrity["manifest_sha_matches"]


def test_cohort_v4_is_1728_unique_cells():
    cells = _cells()
    assert len(cells) == 1728
    assert len({c["cell_id"] for c in cells}) == 1728


def test_preflight_v4_never_re_enumerates():
    src = open(os.path.join(HERE, "zero_emission_v4.py")).read()
    for banned in ("physical_keys(", "select_cohort(", "build_block("):
        assert banned not in src, banned


def test_preflight_v4_guarded_sources_are_blind():
    for name in z4.GUARDED_SOURCES:
        assert z4._scan_source(name) == [], name


def test_scenario_v4_carries_the_registered_site():
    sc, prov = r1v4.build_scenario(_cells(1)[0])
    assert sc.cap.tolist() == [64, 64, 64]
    assert sc.dyn == c4.DYN_W_PER_PE
    assert sc.static.tolist() == [51.4, 51.4, 51.4]
    assert prov["pes_share"] in (0.125, 0.25, 0.5)


def test_scenario_v4_power_matches_the_registered_curve():
    """A site running 32 of its 64 PEs draws the sentinel's midpoint."""
    sc, _p = r1v4.build_scenario(_cells(1)[0])
    assert sc.static[0] + 32 * sc.dyn == pytest.approx(132.7, abs=1e-9)
    assert sc.static[0] + 64 * sc.dyn == pytest.approx(214.0, abs=1e-9)


def test_every_arm_reports_a_valid_schedule_or_none():
    for cell in _cells(2):
        row = r1v4._blinds_one(cell)
        assert set(row["carbon"]) == set(cbl.BLINDS)
        for name, c in row["carbon"].items():
            assert (c is None) == (not row["valid"][name])


def test_reservation_arm_is_contract_safe_on_the_64_pe_site():
    for cell in _cells(3):
        sc, prov = r1v4.build_scenario(cell)
        c, a = cbl.BLINDS["reservation_edf_blind"](sc, prov["clim_residual_green"])
        assert c is not None
        ok, why = validate_assignment(sc, a, budget=sc.B)
        assert ok, why


def test_phase_b_v4_refuses_without_a_freeze_artifact(tmp_path):
    with pytest.raises(RuntimeError, match="Phase A"):
        r1v4.main(phase="b", out_dir=str(tmp_path))


def test_reservation_arm_reads_capacity_from_the_scenario():
    """A 64-PE scenario must not be scheduled against the module's 16-PE default."""
    cell = next(c for c in _cells() if c["physical"]["pes_per_job"] == 32)
    sc, prov = r1v4.build_scenario(cell)
    c, a = cbl.BLINDS["reservation_edf_blind"](sc, prov["clim_residual_green"])
    assert c is not None, "a 32-PE job fits a 64-PE site"
    for _i, (d, s) in a.items():
        assert 0 <= d < 3
    ok, why = validate_assignment(sc, a, budget=sc.B)
    assert ok, why
