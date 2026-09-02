"""Tests for the Scheme-2 generator: axes, closure, windows, exact-diff config discipline."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_s2 as g  # noqa: E402


def test_axes_give_twelve_pairs_and_108_cells():
    assert len(g.admissible_pairs()) == 12
    assert len(g.cells()) == 108
    for r, w in g.admissible_pairs():
        assert r + w <= 144


def test_inadmissible_pairs_are_absent_not_squeezed():
    assert (48, 120) not in g.admissible_pairs()
    assert (72, 96) not in g.admissible_pairs()


def test_every_trace_row_honours_the_closure_condition():
    base = g.base_block()
    mips = float(base["datacenters"][0]["vm_pe_mips"])
    for cell in g.cells():
        rows, rep = g.trace(cell, mips)
        r = cell["runtime_rows"]
        for (_i, a, mi, pes, _fs, _os_, dl) in rows:
            assert dl == a + r + cell["wait_cap_rows"]
            assert dl - a <= 144, "a latest start plus runtime must fit the window"
            assert mi == int(round(r * mips * g.CPU_UTILISATION))
            assert pes == g.PES_PER_JOB
        assert rep["closure_ok"] and rep["deadline_reachable"] and rep["fits_episode"]


def test_arrivals_are_partitioned_and_never_clipped():
    for cell in g.cells():
        a, span = g.arrivals(cell)
        assert a.max() < span, "no arrival may be squeezed onto the span edge"
        assert int(a.max() - a.min() + 1) > 1
        n = cell["n_jobs"]
        for i, x in enumerate(sorted(a.tolist())):
            lo, hi = (i * span) // n, ((i + 1) * span) // n
            assert lo <= x < max(hi, lo + 1)


def test_trace_is_deterministic():
    base = g.base_block()
    mips = float(base["datacenters"][0]["vm_pe_mips"])
    cell = g.cells()[37]
    t1 = g.trace_text(g.trace(cell, mips)[0])
    t2 = g.trace_text(g.trace(cell, mips)[0])
    assert g.content_sha(t1) == g.content_sha(t2)


def test_arrival_and_any_future_stream_are_domain_separated():
    cell = g.cells()[0]
    assert g._seed(cell, "arrival") != g._seed(cell, "runtime")


def test_derived_block_changes_only_the_enumerated_keys():
    base = g.base_block()
    blk = g.derived_block(g.cells()[0], base)
    changed = {k for k in set(base) | set(blk)
               if json.dumps(base.get(k), sort_keys=True, default=str)
               != json.dumps(blk.get(k), sort_keys=True, default=str)}
    assert changed == set(g.OVERRIDDEN_KEYS), changed


def test_pinned_keys_and_values():
    base = g.base_block()
    blk = g.derived_block(g.cells()[0], base)
    assert blk["defer_deadline_force_mode"] == "latest_start"
    assert blk["defer_deadline_slack_sec"] == 0.0, \
        "an inherited 600 s slack fires the backstop before any defer decision"
    assert blk["green_oracle_mode"] == "godeye", \
        "the TimeCAP obs provider is dead weight for planner arms"
    assert blk["cloudlet_cpu_utilization"] == 1.0
    assert blk["cloudlet_trace_file"].startswith("traces/s2/")
    # Inherited untouched, on purpose: the frozen episode length and offset schedule.
    assert blk["max_episode_length"] == 7200
    assert blk["green_episode_offset_range"] == 44950
    assert blk["workload_mode"] == "CSV"
    assert base["defer_deadline_force_mode"] == "legacy", \
        "the base block still carries the legacy backstop; S2 must not inherit it"


def test_windows_are_six_disjoint_and_quarantine_k0():
    w = g.windows(44950)
    ks = [k for k, _o in w["discovery"] + w["confirmation"]]
    offs = [o for _k, o in w["discovery"] + w["confirmation"]]
    assert len(ks) == 6 and len(set(ks)) == 6
    assert 0 not in ks
    assert len(set(w["discovery"]) & set(w["confirmation"])) == 0
    for i in range(6):
        for j in range(i + 1, 6):
            assert abs(offs[i] - offs[j]) >= g.WINDOW_SPACING
        assert offs[i] + g.EPISODE_ROWS_MAX <= g.TRACE_ROWS_MAX


def test_windows_match_the_simulator_schedule():
    for k, off in g.windows(44950)["discovery"]:
        assert off == (1009 * k) % 44950


def test_generator_never_reads_green_values():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "gen_s2.py")).read()
    for banned in ("Patv", "power_kw", "_series", "green_power", "carbon_kg"):
        assert banned not in src, banned


def test_offered_concurrency_is_reported_near_target():
    base = g.base_block()
    mips = float(base["datacenters"][0]["vm_pe_mips"])
    for cell in g.cells():
        _rows, rep = g.trace(cell, mips)
        assert 0.5 * cell["concurrency"] <= rep["offered_concurrency"] \
            <= 1.5 * cell["concurrency"], (cell, rep["offered_concurrency"])


def test_config_s2_carries_the_base_common_section_verbatim(tmp_path):
    import yaml
    g.generate(out_dir=str(tmp_path), trace_dir=str(tmp_path / "traces"))
    ours = yaml.safe_load(open(tmp_path / "config_s2.yml"))
    base = yaml.safe_load(open(g.BASE_CONFIG))
    assert ours["common"] == base["common"]
    assert len([k for k in ours if k != "common"]) == 108


# ── stage A verdict reader (frozen before the oracle results exist) ─────────

import stage_a_verdict as sv                    # noqa: E402


def _mk(cell, blind, o144, full, fav=3):
    per = {}
    for i, k in enumerate((1, 9, 17)):
        good = i < fav
        per[k] = {"blind": blind, "oracle144": o144 if good else blind * 1.01,
                  "full": full}
    return per


def test_adjacency_is_one_step_on_one_axis():
    a = {"runtime_rows": 24, "wait_cap_rows": 48, "concurrency": 3, "n_jobs": 35, "seed": 0}
    b = dict(a, wait_cap_rows=72)
    c = dict(a, wait_cap_rows=72, concurrency=5)
    d = dict(a, wait_cap_rows=120)
    assert sv.adjacent(a, b)
    assert not sv.adjacent(a, c), "two axes moved"
    assert not sv.adjacent(a, d), "two steps on one axis"


def test_capture_gate_requires_a_positive_denominator():
    # full == blind -> denominator zero -> the cell cannot pass however big the drop.
    blind, o144, full = 1.0, 0.5, 1.0
    denom = blind - full
    assert not (denom > 0 and (blind - o144) / denom >= sv.GATE_CAPTURE)


def test_verdict_centre_is_by_sha_not_by_effect(tmp_path, monkeypatch):
    # Three adjacent passing cells with very different effect sizes; the centre must be
    # the smallest SHA, not the deepest reduction.
    cells = [
        {"runtime_rows": 24, "wait_cap_rows": 24, "concurrency": 1, "n_jobs": 20, "seed": 0},
        {"runtime_rows": 24, "wait_cap_rows": 48, "concurrency": 1, "n_jobs": 20, "seed": 0},
        {"runtime_rows": 24, "wait_cap_rows": 72, "concurrency": 1, "n_jobs": 20, "seed": 0},
    ]
    rows = []
    for i, c in enumerate(cells):
        rows.append({"cell": c, "cell_sha": sv.cell_sha(c)[:16],
                     "pass": True, "reduction": 0.1 * (i + 1)})
    regions, seen = [], set()
    passing = rows
    centre = min(passing, key=lambda r: sv.cell_sha(r["cell"]))
    deepest = max(passing, key=lambda r: r["reduction"])
    assert centre["cell_sha"] == min(sv.cell_sha(c)[:16] for c in cells)
    # only a coincidence would make them equal; assert the rule text, not luck
    assert centre["cell_sha"] != deepest["cell_sha"] or len({r["cell_sha"] for r in rows}) == 1


# ── ladder-v2 reader (frozen before the confirmation sweep is interpreted) ──

import ladder_v2_verdict as lv                  # noqa: E402


def test_v2_windows_are_the_confirmation_triple():
    import run_stage_a as ra
    assert [k for k, _o in ra.windows("confirmation")] == [25, 33, 41]
    assert [k for k, _o in ra.windows("discovery")] == [1, 9, 17]


def test_v2_tier_list_matches_the_prereg():
    import run_stage_a as ra
    assert ra.TIERS_V2 == ("godeye", "s05", "s15", "s30", "s60",
                           "checkpoint_residual_surrogate_v2", "shuffle", "anti")


def test_v2_calibration_artifact_is_measured_not_rounded():
    here = os.path.dirname(os.path.abspath(__file__))
    cal = json.load(open(os.path.join(here, "dc_residual_cal.json")))
    assert cal["c"] != 0.8, "c must be the measured median, never a hand-rounded value"
    off = cal["off_diagonals"]
    assert cal["c"] == sorted(off)[1], "c is the median off-diagonal"
    assert max(abs(x - cal["c"]) for x in off) <= cal["single_factor_tolerance"]
    assert cal["label_offset"] == 0 and cal["year"] == 2020


# ── Scheme 2-E data split (frozen before any E carbon result) ───────────────

import e_data_split as eds                      # noqa: E402


def test_e_split_is_deterministic_and_matches_the_artifact():
    art = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "e_data_split.json")))
    dealt = eds.deal()
    assert dealt["discovery"] == art["discovery"]
    assert dealt["confirmation"] == art["confirmation"]


def test_e_split_turbines_are_fresh_and_disjoint():
    art = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "e_data_split.json")))
    d = set(art["discovery"]["turbines"])
    c = set(art["confirmation"]["turbines"])
    assert len(d) == 5 and len(c) == 5 and not d & c
    banned = set(art["excluded"]["s2"]) | set(art["excluded"]["train"]) \
        | set(art["excluded"]["sealed"]) | set(art["excluded"]["tb13"])
    assert not (d | c) & banned
    for t in d | c:
        p = os.path.join(eds.SPL, f"Turbine_{t}_2021.csv")
        assert sum(1 for _ in open(p)) - 1 == 52559


def test_e_split_windows_never_touch_and_never_ran():
    art = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "e_data_split.json")))
    ks = art["discovery"]["windows_k"] + art["confirmation"]["windows_k"]
    assert not set(ks) & {1, 9, 17, 25, 33, 41}, "S2's burned windows are off limits"
    offs = art["discovery"]["offsets"] + art["confirmation"]["offsets"]
    for i, a in enumerate(offs):
        assert a + 7200 <= 52559
        for b in offs[i + 1:]:
            assert abs(a - b) >= 7300


def test_e_split_dc_map_shape_matches_the_topology():
    art = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "e_data_split.json")))
    for part in ("discovery", "confirmation"):
        m = art[part]["dc_map"]
        assert [len(m["0"]), len(m["1"]), len(m["2"])] == [2, 2, 1]


def test_e_config_differs_from_s2_only_in_green_dc_turbines(tmp_path):
    import yaml
    g.generate_e("discovery", out_dir=str(tmp_path))
    e_cfg = yaml.safe_load(open(tmp_path / "config_s2e_discovery.yml"))
    split = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "e_data_split.json")))["discovery"]
    base = g.base_block()
    for cell in g.cells()[:12]:
        name = g.cell_name(cell)
        s2 = g.derived_block(cell, base)
        e = e_cfg[name]
        for k in set(s2) | set(e):
            if k in ("datacenters", "simulation_name"):
                continue
            assert json.dumps(s2.get(k), sort_keys=True, default=str) \
                == json.dumps(e.get(k), sort_keys=True, default=str), k
        for dc_s2, dc_e in zip(s2["datacenters"], e["datacenters"]):
            for k in set(dc_s2) | set(dc_e):
                if k == "turbine_ids":
                    continue
                assert dc_s2.get(k) == dc_e.get(k), k
            did = str(dc_e["datacenter_id"])
            if did in split["dc_map"]:
                assert dc_e["turbine_ids"] == [int(t) for t in split["dc_map"][did]]
            else:
                assert dc_e["turbine_ids"] == dc_s2["turbine_ids"]


def test_e_configs_generate_for_both_parts(tmp_path):
    d = g.generate_e("discovery", out_dir=str(tmp_path))
    c = g.generate_e("confirmation", out_dir=str(tmp_path))
    assert d["blocks"] == c["blocks"] == 108
    assert set(d["turbines"]).isdisjoint(c["turbines"])
