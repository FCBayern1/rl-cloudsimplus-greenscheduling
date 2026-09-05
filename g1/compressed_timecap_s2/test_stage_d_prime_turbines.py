import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_d_prime_turbines import STRUCTURE, TAG, choose  # noqa: E402


def test_choice_is_hash_ordered_deterministic_and_mapped_onto_the_hz_layout():
    cands = list(range(2, 60))
    a, b = choose(cands), choose(cands)
    assert a == b and a["status"] == "OK" and len(a["turbines"]) == 5
    order = sorted(cands, key=lambda i: hashlib.sha256(f"{TAG}:{i}".encode()).hexdigest())
    assert a["turbines"] == order[:5]
    assert a["dc_turbines"] == {0: order[:2], 1: order[2:4], 2: order[4:5]}
    assert sum(n for _d, n in STRUCTURE) == 5


def test_stop_when_fewer_than_five_candidates():
    assert choose([2, 4, 5])["status"] == "STOP_NO_CANDIDATES"


def test_used_ids_never_enter_when_excluded_upstream():
    # the rule itself does not know usage; eligibility is the inventory's job, so a
    # candidate list without the HZ ids can never return them
    r = choose([i for i in range(2, 146) if i not in (123, 10, 51, 53, 112)])
    assert not set(r["turbines"]) & {123, 10, 51, 53, 112}
