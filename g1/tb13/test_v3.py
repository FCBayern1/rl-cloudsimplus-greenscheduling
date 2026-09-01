"""Tests for the v3 workload, the axis and window gates, and the block cohort."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig                       # noqa: E402
import preflight_v3 as p3                       # noqa: E402
import round0 as r0                             # noqa: E402
import round0_v3 as r3                          # noqa: E402
import workload_v3 as w3                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# ── workload ────────────────────────────────────────────────────────────────

def test_axis_count_is_eighty_nine():
    assert len(w3.compatible_axes()) == 89


def test_no_clipping_anywhere_in_source():
    src = open(os.path.join(HERE, "workload_v3.py")).read()
    for banned in ("clip", "min(target_span", "np.minimum(arr"):
        assert banned not in src, banned


def test_workload_is_blind_to_wind_and_carbon():
    src = open(os.path.join(HERE, "workload_v3.py")).read()
    for banned in ("green", "cb[", "cg[", "climatology", "carbon", "_series"):
        assert banned not in src, banned


def test_runtime_multiset_is_exactly_half_and_half():
    for (h, n, c, wc) in w3.compatible_axes():
        w = w3.draw(w3.workload_key(0, h, 4, c, n, wc), 0)
        r = sorted(w["runtime"].tolist())
        assert r == [6] * (n // 2) + [12] * (n // 2)
        assert int(w["runtime"].sum()) == 9 * n


def test_three_registered_assertions_hold_on_every_axis():
    for (h, n, c, wc) in w3.compatible_axes():
        for pes in w3.PES_PER_JOB:
            key = w3.workload_key(0, h, pes, c, n, wc)
            w = w3.draw(key, 0)
            checks, ok = w3.assertions(w, key)
            assert ok, (h, n, c, wc, pes, checks)
            assert w["service_span"] == math.ceil(9 * n / c)


def test_arrivals_are_partitioned_one_per_interval():
    key = w3.workload_key(0, 144, 4, 3, 12, 24)
    w = w3.draw(key, 0)
    n, S = 12, w["service_span"]
    for i, a in enumerate(sorted(w["arrival"].tolist())):
        assert (i * S) // n <= a < max(((i + 1) * S) // n, (i * S) // n + 1)


def test_streams_are_domain_separated():
    """Runtime and arrival draws come from different seeds for the same key and retry."""
    key = w3.workload_key(0, 144, 4, 3, 12, 24)
    assert w3.domain_seed(key, "arrival", 0) != w3.domain_seed(key, "runtime", 0)
    assert w3.domain_seed(key, "arrival", 0) != w3.domain_seed(key, "arrival", 1)


def test_draw_is_a_pure_function_of_the_key():
    key = w3.workload_key(0, 96, 8, 2, 10, 12)
    a, b = w3.draw(key, 3), w3.draw(key, 3)
    assert w3.content_hash(a) == w3.content_hash(b)
    assert np.array_equal(a["arrival"], b["arrival"])


def test_incompatible_axes_are_excluded_not_squeezed():
    # 12 jobs at concurrency 1 need 108 rows of service plus 12 of runtime and the cap.
    assert not w3.compatible(72, 12, 1, 6)
    assert (72, 12, 1, 6) not in w3.compatible_axes()


def test_odd_job_counts_are_refused():
    with pytest.raises(ValueError):
        w3.draw(w3.workload_key(0, 144, 4, 3, 9, 12), 0)


# ── gates ───────────────────────────────────────────────────────────────────

def test_axis_gate_passes_with_267_keys():
    g = p3.axis_gate()
    assert g["pass"] and g["compatible_axes"] == 89
    assert g["workload_keys"] == 267 == g["workload_key_cap"]


def test_window_gate_checks_every_discovery_file():
    g = p3.window_gate()
    assert g["pass"]
    assert g["turbines_checked"] == 24
    assert g["row_counts_unique"] == [52559]
    assert g["windows_payload_sha"] == p3.REGISTERED_WINDOWS_PAYLOAD_SHA
    assert g["base_offsets"] == [4307, 13067, 21827, 30587, 39347, 48107]


def test_windows_are_disjoint_and_in_bounds():
    wins = sorted(json.load(open(r3.WINDOWS))["windows"], key=lambda x: x["foot_start"])
    for a, b in zip(wins, wins[1:]):
        assert a["foot_end"] <= b["foot_start"]
    assert wins[-1]["foot_end"] <= 52559
    for w in wins:
        assert w["foot_end"] - w["foot_start"] == 144


# ── round 0-v3 ──────────────────────────────────────────────────────────────

def test_physical_space_is_12960_and_confirmation_free():
    keys = r3.physical_keys()
    assert len(keys) == 12960
    assert len({r3.key_sha(k) for k in keys}) == 12960
    confirmation = set(r0.confirmation_pool())
    assert not confirmation.intersection(set().union(*(r3._turbines(k) for k in keys)))


def test_physical_keys_use_only_v3_horizons_and_frozen_offsets():
    keys = r3.physical_keys()
    assert {k["horizon"] for k in keys} == set(r3.HORIZONS)
    assert {k["season_offset"] for k in keys} == set(r3.base_offsets())


def test_v3_reuses_the_registered_physical_gate():
    assert r3.r0.CORR_BAND == (0.70, 0.95)
    assert r3.r0.BEST_DC_CHANGE_MIN == 0.10


# ── cohort ──────────────────────────────────────────────────────────────────

def _fake_anchors(n_layers):
    pool = r0.discovery_pool()
    offs = r3.base_offsets()
    out = []
    for tps in ig.TURBINES_PER_SITE:
        triples = ig.turbine_triples(pool, tps, r3.N_TRIPLETS)
        for ti, triple in enumerate(triples):
            for si, off in enumerate(offs):
                if len(out) // r3.ANCHORS_PER_LAYER >= n_layers:
                    break
                for j, div in enumerate((3000, 6000)):
                    out.append({"pes_per_job": 8, "concurrency": 3,
                                "turbines_per_site": tps, "installed_divisor": div,
                                "horizon": 144, "triplet_index": ti, "season_index": si,
                                "triplet": triple, "season_offset": off, "year": 2021})
    return out


def test_block_has_exactly_twelve_cells_and_never_splits_an_axis():
    a = _fake_anchors(1)[0]
    b = r3.build_block(a, (10, 12))
    assert len(b["cells"]) == 12
    assert len(b["divisors"]) == 3 and b["divisors"] == r0.neighbourhood(3000)
    assert b["budget_fractions"] == list(ig.BUDGET_FRACTION)
    assert {c["budget_fraction"] for c in b["cells"]} == set(ig.BUDGET_FRACTION)


def test_cohort_is_capped_deduplicated_and_layer_spread():
    blocks, skipped, candidates = r3.select_cohort(_fake_anchors(72))
    assert len(blocks) == r3.MAX_BLOCKS
    cells = [c["cell_id"] for b in blocks for c in b["cells"]]
    assert len(cells) == 1728 and len(set(cells)) == 1728
    layers = {tuple(b["layer"]) for b in blocks}
    assert len(layers) == 72, "round-robin must reach every layer before repeating"
    assert candidates >= len(blocks)


def test_cohort_selection_is_reproducible():
    a = _fake_anchors(72)
    first = [b["block_sha"] for b in r3.select_cohort(a)[0]]
    second = [b["block_sha"] for b in r3.select_cohort(list(a))[0]]
    assert first == second


def test_every_candidate_is_either_chosen_or_recorded_as_a_collision():
    """Below the cap, nothing is dropped silently: chosen + skipped accounts for all.

    Two anchors in one layer can carry overlapping divisor neighbourhoods, and the second
    block cannot be trimmed to fit, so it is skipped whole and written to the skip list.
    """
    blocks, skipped, candidates = r3.select_cohort(_fake_anchors(2)[:2])
    assert len(blocks) + len(skipped) == candidates <= r3.MAX_BLOCKS + len(skipped)
    assert len(blocks) >= 1
    cells = [c["cell_id"] for b in blocks for c in b["cells"]]
    assert len(set(cells)) == len(cells)
    assert all(s["reason"] == "cell already in cohort" for s in skipped)


def test_cohort_never_reads_green_or_carbon():
    src = open(os.path.join(HERE, "round0_v3.py")).read()
    body = src.split("def select_cohort")[1].split("def _provenance")[0]
    for banned in ("green", "carbon", "_series", "residual", "evpi", "EVPI"):
        assert banned not in body, banned


# ── acceptance path (used later by the zero-emissions preflight) ─────────────

def test_accepted_returns_a_capacity_feasible_reservation_edf_load():
    """The strictest budget still admits a load, and the accepted one is deterministic."""
    key = w3.workload_key(0, 144, 4, 3, 8, 24)
    a = w3.accepted(key)
    assert a is not None
    assert a["retry"] < w3.MAX_RETRIES
    assert w3.assertions(a["workload"], key)[1]
    b = w3.accepted(w3.workload_key(0, 144, 4, 3, 8, 24))
    assert a["content_hash"] == b["content_hash"]
