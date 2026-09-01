"""The generated workloads.

The failure this file exists to prevent is the one the TB12 window probe walked into: a
scenario where the question "does this job fit in the forecast" has the same answer for
every job at every epoch, so the forecast cannot change a decision and the arm can only
lose. Here the closure condition is enforced per job, and the arrivals are solved back out
of the target concurrency instead of being clipped onto epoch zero.
"""
import hashlib
import os

import numpy as np
import pytest

import constants as C
import workload as W

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "..", "cloudsimplus-gateway/src/main/resources/traces")


@pytest.fixture(scope="module")
def drawn():
    return {C.cell_key(c): W.draw(c) for c in C.cells()}


class TestGrid:
    def test_grid_is_the_registered_one(self):
        assert C.RUNTIME_ROWS == (24, 48, 72)
        assert C.WAIT_CAP_ROWS == (24, 48, 72, 96, 120)
        assert C.CONCURRENCY == (1, 3, 5)
        assert C.N_JOBS == (20, 35, 50)
        assert len(C.cells()) == 108

    def test_only_admissible_pairs_are_generated(self):
        for r, w in C.admissible_pairs():
            assert r + w <= C.CLOSURE_ROWS
        for r in C.RUNTIME_ROWS:
            for w in C.WAIT_CAP_ROWS:
                if r + w > C.CLOSURE_ROWS:
                    assert (r, w) not in C.admissible_pairs()

    def test_cell_keys_are_unique(self):
        keys = [C.cell_key(c) for c in C.cells()]
        assert len(set(keys)) == len(keys)


class TestInvariants:
    def test_every_cell_passes_its_assertions(self, drawn):
        bad = {}
        for k, wl in drawn.items():
            checks, ok = W.assertions(wl)
            if not ok:
                bad[k] = [n for n, v in checks.items() if not v]
        assert not bad, bad

    def test_closure_holds_per_job_not_just_on_average(self, drawn):
        """(s - a) + r <= 144 for the latest legal start of every single job."""
        for k, wl in drawn.items():
            w = wl["cell"]["wait_cap_rows"]
            assert int((w + wl["runtime"]).max()) <= C.CLOSURE_ROWS, k

    def test_latest_start_is_exactly_wait_cap_after_arrival(self, drawn):
        for k, wl in drawn.items():
            w = wl["cell"]["wait_cap_rows"]
            assert np.array_equal(wl["deadline"] - wl["runtime"], wl["arrival"] + w), k

    def test_arrivals_are_never_clipped_onto_epoch_zero(self, drawn):
        for k, wl in drawn.items():
            a = wl["arrival"]
            assert int(a[0]) == 0
            assert int(a[-1]) > 0, k
            assert len(np.unique(a)) > 1, k

    def test_offered_concurrency_tracks_the_target(self, drawn):
        """n jobs over (n-1) spacings sit n/(n-1) above target; nothing else may drift."""
        for k, wl in drawn.items():
            rep = W.report(wl)
            n, c = rep["n_jobs"], rep["target_concurrency"]
            assert rep["offered_concurrency"] == pytest.approx(c * n / (n - 1), rel=0.05), k

    def test_runtime_never_exceeds_the_registered_bound(self, drawn):
        for k, wl in drawn.items():
            assert int(wl["runtime"].max()) <= wl["cell"]["runtime_rows"], k

    def test_mi_obeys_the_cloudsim_runtime_identity(self, drawn):
        unit = int(C.VM_PE_MIPS * C.CPU_UTIL)
        for k, wl in drawn.items():
            assert np.array_equal(wl["mi"], wl["runtime"] * wl["pes"] * unit), k

    def test_no_job_can_be_split(self, drawn):
        """split_large_cloudlets is inherited true; a split would void the runtime model."""
        for k, wl in drawn.items():
            assert int(wl["pes"].max()) <= 8, k

    def test_every_job_fits_under_the_observation_bound(self, drawn):
        for k, wl in drawn.items():
            assert int(wl["mi"].max()) <= C.obs_cloudlet_mi_high(), k


class TestDeterminism:
    def test_two_draws_are_identical(self):
        for cell in C.cells()[:12]:
            a, b = W.draw(cell), W.draw(cell)
            assert W.to_csv(a) == W.to_csv(b)

    def test_streams_are_domain_separated(self):
        cell = C.cells()[0]
        assert W.stream_seed(cell, "runtime") != W.stream_seed(cell, "pes")
        other = C.cells()[1]
        assert W.stream_seed(cell, "runtime") != W.stream_seed(other, "runtime")

    def test_content_hashes_are_distinct_across_cells(self, drawn):
        hashes = {k: W.content_sha256(wl) for k, wl in drawn.items()}
        assert len(set(hashes.values())) == len(hashes)

    def test_hash_certifies_the_emitted_bytes(self, drawn):
        wl = drawn["r24w24c1n20"]
        assert W.content_sha256(wl) == hashlib.sha256(W.to_csv(wl).encode()).hexdigest()

    def test_csv_header_matches_the_reader_contract(self, drawn):
        first = W.to_csv(drawn["r24w24c1n20"]).splitlines()[0]
        assert first == ",".join(W.COLUMNS)


class TestEpisodeLength:
    def test_episode_covers_the_last_possible_finish_plus_the_drain(self, drawn):
        for k, wl in drawn.items():
            w = wl["cell"]["wait_cap_rows"]
            last = int((wl["arrival"] + w + wl["runtime"]).max())
            assert W.episode_steps(wl) == last + C.DRAIN_STEPS, k

    def test_realised_episode_never_exceeds_the_footprint_bound(self, drawn):
        for k, wl in drawn.items():
            assert W.episode_steps(wl) <= C.episode_steps_bound(wl["cell"]), k


class TestTracesOnDisk:
    """A trace that is missing or stale is silently a different experiment."""

    def test_every_trace_is_present_and_byte_identical(self, drawn):
        missing, stale = [], []
        for k, wl in drawn.items():
            p = os.path.join(TRACES_DIR, W.trace_name(wl["cell"]))
            if not os.path.isfile(p):
                missing.append(k)
            elif open(p).read() != W.to_csv(wl):
                stale.append(k)
        assert not missing, f"run generate_configs.py --write: {missing[:5]}"
        assert not stale, f"traces do not match the generator: {stale[:5]}"
