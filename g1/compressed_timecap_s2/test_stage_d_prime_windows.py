import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_d_prime_windows as sw  # noqa: E402
import window_preflight as wp  # noqa: E402


def test_selection_is_deterministic_hash_ordered_and_non_overlapping():
    read = [(1000, 3000), (20000, 25000)]
    a = sw.select(read, eval_len=600)
    b = sw.select(read, eval_len=600)
    assert a == b and a["status"] == "OK" and len(a["windows"]) == 6
    ivs = [(o - wp.PRE, o + 600) for o in a["windows"]]
    for i, x in enumerate(ivs):
        assert all(not wp.overlaps(x, r) for r in read)
        assert all(not wp.overlaps(x, y) for y in ivs[:i])
    # the first chosen offset is the hash-minimal legal offset among the free ones
    cands = sw.candidates(read, 600)
    first = min(cands, key=lambda o: hashlib.sha256(f"{sw.TAG}:{o}".encode()).hexdigest())
    assert first in a["windows"]


def test_stop_when_fewer_than_six_fit():
    read = [(0, wp.ROWS_IN_FILE - 1500)]          # only ~1500 rows free
    r = sw.select(read, eval_len=600)
    assert r["status"] == "STOP_WINDOW_SPLIT" and len(r["windows"]) < 6


def test_tag_is_the_ruled_string():
    assert sw.TAG == "stage-d-prime-judgement-v1"
