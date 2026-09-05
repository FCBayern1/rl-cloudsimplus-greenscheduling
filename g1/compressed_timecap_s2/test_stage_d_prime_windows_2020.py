import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_d_prime_windows_2020 as s20  # noqa: E402
import window_preflight as wp  # noqa: E402


def test_six_windows_fit_on_the_2020_length_with_no_read_intervals():
    r = s20.select_2020([], eval_len=2922)
    assert r["status"] == "OK" and len(r["windows"]) == 6 and r["rows"] == 32225
    ivs = [(o - wp.PRE, o + 2922) for o in r["windows"]]
    for i, x in enumerate(ivs):
        assert x[0] >= 0 and x[1] <= 32225
        assert all(not wp.overlaps(x, y) for y in ivs[:i])


def test_hash_tag_carries_the_year_and_order_is_deterministic():
    assert s20.TAG == "stage-d-prime-judgement-v1:2020"
    a = s20.select_2020([(5000, 9000)], 2922); b = s20.select_2020([(5000, 9000)], 2922)
    assert a == b
    cands = [o for o in range(wp.PRE, 32225 - 2922 + 1, wp.PRE)]
    legal = [o for o in cands if not wp.overlaps((o - wp.PRE, o + 2922), (5000, 9000))]
    first = min(legal, key=lambda o: hashlib.sha256(f"{s20.TAG}:{o}".encode()).hexdigest())
    assert first in a["windows"]


def test_stop_when_read_intervals_leave_too_little_room():
    r = s20.select_2020([(0, 32225 - 4000)], 2922)
    assert r["status"] == "STOP_WINDOW_SPLIT" and len(r["windows"]) == 1
