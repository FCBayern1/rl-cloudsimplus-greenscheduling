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
