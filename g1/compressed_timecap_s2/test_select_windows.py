"""The six scheduler windows.

What these tests are actually defending: a window chosen for its weather, a window that
runs off the end of the 2021 series, a window the simulator cannot address, and a
CONFIRMATION set drawn from the same weather as DISCOVERY. Each of those would let a carbon
result mean something other than what it claims.
"""
import json
import math
import os
import re

import pytest

import constants as C
import select_windows as S

WINDOWS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows.json")


@pytest.fixture(scope="module")
def win():
    return json.load(open(WINDOWS_PATH))


class TestBlindness:
    def test_selector_never_parses_a_power_value(self):
        """A green-informed window choice would let the weather pick the exam."""
        src = open(S.__file__).read()
        body = src.split('"""', 2)[-1]          # exclude the module docstring
        for forbidden in ("power_kw", "DictReader", "np.percentile", "float(r["):
            assert forbidden not in body, f"selector reads power data: {forbidden}"

    def test_row_count_helper_ignores_fields(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("timestamp,power_kw\n2021-01-01,3.5\n2021-01-02,9.9\n")
        assert S.row_count(str(p)) == 2


class TestArtifactMatchesTheRule:
    def test_regenerating_gives_the_same_selection(self, win):
        """A hand-edited windows.json is a void run; regeneration must reproduce it."""
        blk, tz, turbines = S.load_base()
        used = sorted({t for ts in turbines.values() for t in ts})
        assert win["turbines_used"] == used
        assert win["tz_rows"] == {str(k): v for k, v in sorted(tz.items())}
        fp = S.footprint_rows(max(tz.values()))
        assert win["footprint"] == fp
        rng = win["n_rows_eval_year"] - fp["footprint_rows"]
        assert win["green_episode_offset_range"] == rng
        k_max = (rng - 1) // C.STRIDE
        assert win["k_max_no_wrap"] == k_max
        for i in range(C.N_WINDOWS):
            centre = (2 * i + 1) * rng // (2 * C.N_WINDOWS)
            expected_k = min(range(1, k_max + 1),
                             key=lambda kk: (abs(C.STRIDE * kk - centre), kk))
            w = win["windows"][f"w{i}"]
            assert w["block_centre"] == centre
            assert w["k"] == expected_k
            assert w["offset"] == C.STRIDE * expected_k

    def test_selection_hash_matches_content(self, win):
        import hashlib
        body = {k: v for k, v in win.items() if k != "selection_hash"}
        assert win["selection_hash"] == hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()).hexdigest()


class TestGeometry:
    def test_six_windows_three_and_three(self, win):
        assert len(win["windows"]) == C.N_WINDOWS
        assert len(win["discovery"]) == 3 and len(win["confirmation"]) == 3
        assert not set(win["discovery"]) & set(win["confirmation"])

    def test_discovery_and_confirmation_interleave(self, win):
        """Blocks 0/2/4 against 1/3/5: neither set can occupy its own part of the year."""
        pos = {nm: win["windows"][nm]["position"] for nm in win["windows"]}
        assert sorted(pos[nm] for nm in win["discovery"]) == list(C.DISCOVERY_POSITIONS)
        assert sorted(pos[nm] for nm in win["confirmation"]) == list(C.CONFIRMATION_POSITIONS)

    def test_read_intervals_are_pairwise_disjoint(self, win):
        iv = sorted(w["read_rows"] for w in win["windows"].values())
        for a, b in zip(iv, iv[1:]):
            assert a[1] < b[0], f"windows overlap: {a} {b}"

    def test_no_window_runs_off_the_end_of_the_year(self, win):
        n = win["n_rows_eval_year"]
        for nm, w in win["windows"].items():
            assert 0 <= w["read_rows"][0] and w["read_rows"][1] < n, nm

    def test_footprint_fits_in_a_block(self, win):
        assert win["footprint"]["footprint_rows"] <= win["block_rows"]

    def test_every_offset_is_reachable_by_reset_skip(self, win):
        """evaluate.py addresses a window by k; an unreachable offset cannot be run."""
        rng = win["green_episode_offset_range"]
        for nm, w in win["windows"].items():
            assert (C.STRIDE * w["k"]) % rng == w["offset"], nm

    def test_reset_skip_stays_runnable(self, win):
        """--reset-skip k performs k real env.reset() calls before the measured episode,
        so a modular-inverse k of ~46000 would be an unrunnable experiment, not a window."""
        k_max = win["k_max_no_wrap"]
        for nm, w in win["windows"].items():
            assert 1 <= w["k"] <= k_max, nm
            assert C.STRIDE * w["k"] < win["green_episode_offset_range"], nm

    def test_windows_stay_near_their_block_centres(self, win):
        """The no-wrap constraint may move a window off centre, but not out of its block."""
        for nm, w in win["windows"].items():
            assert abs(w["centre_error_rows"]) < C.STRIDE, nm
            assert w["block"][0] <= w["offset"] <= w["block"][1], nm

    def test_windows_are_spread_over_the_whole_eval_year(self, win):
        """The first draft accepted k = 1, 5, 9 ... and put all six in the first half of
        the year, one footprint apart, so CONFIRMATION would have been the same weather."""
        offs = sorted(w["offset"] for w in win["windows"].values())
        F = win["footprint"]["footprint_rows"]
        assert min(b - a for a, b in zip(offs, offs[1:])) >= 2 * F
        assert offs[-1] - offs[0] >= 0.8 * win["green_episode_offset_range"]


class TestFootprintCoversEveryCell:
    def test_longest_cell_fits(self, win):
        fp = win["footprint"]
        need = (fp["clock0_rows"] + C.max_episode_steps() + fp["warmup_rows"]
                + fp["max_tz_rows"])
        assert need <= fp["footprint_rows"]
        assert fp["footprint_rows"] - need == C.GUARD_ROWS

    def test_every_cell_is_shorter_than_the_bound(self, win):
        import workload as W
        for cell in C.cells():
            assert W.episode_steps(W.draw(cell)) <= C.max_episode_steps()

    def test_bound_uses_the_analytic_arrival_span(self):
        """Window selection must not depend on a workload draw."""
        cell = max(C.cells(), key=C.episode_steps_bound)
        assert C.arrival_span_bound(cell) == math.ceil(
            (cell["n_jobs"] - 1) * cell["runtime_rows"] / cell["concurrency"])


class TestYearIsolation:
    def test_eval_year_is_2021_and_train_year_is_2020(self, win):
        assert win["year_scheduler_eval"] == 2021
        assert win["year_timecap_train"] == 2020
        assert win["year_scheduler_eval"] != win["year_timecap_train"]

    def test_forbidden_year_is_two_rows_of_nothing(self, win):
        counts = {k: v for k, v in win["turbine_row_counts"].items()
                  if k.endswith(f"_{C.YEAR_FORBIDDEN}")}
        assert counts, "2022 inventory missing"
        assert all(v <= 2 for v in counts.values()), counts

    def test_every_used_turbine_has_both_years(self, win):
        for t in win["turbines_used"]:
            for y in (C.YEAR_TIMECAP_TRAIN, C.YEAR_SCHEDULER_EVAL):
                assert win["turbine_row_counts"][f"Turbine_{t}_{y}"] > 0

    def test_eval_offsets_are_row_indices_into_the_eval_year_only(self, win):
        """Nothing in windows.json may address a 2020 row."""
        assert re.search(r"2020", json.dumps(win["windows"])) is None
