import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_v1 import draw_windows  # noqa: E402
from scene_v2 import KEPT_POOL_K, candidates_13_24, first_passing  # noqa: E402


def test_candidates_are_the_same_hash_sequence_continued():
    c = candidates_13_24(n_rows=52559)
    assert len(c) == 12
    pool = draw_windows(52559, 12, "scene-interface-v1:2021:")["windows"]
    assert not set(c) & set(pool)
    for o in c:                                       # no overlap with the pool's footprints either
        assert all(o + 2922 <= p or p + 2922 <= o for p in pool)


def test_first_passing_stops_at_the_earliest_pass_and_respects_the_contract():
    ref = 0.01897
    res = [{"offset": 1, "C_B": 0.0010, "C_ST": 0.0005, "contract_ok": True},    # 50 % rel but abs 0.0005 < 9.5e-4
           {"offset": 2, "C_B": 0.0040, "C_ST": 0.0020, "contract_ok": False},   # would pass, contract broken
           {"offset": 3, "C_B": 0.0040, "C_ST": 0.0025, "contract_ok": True},    # 37 % rel, abs 0.0015 -> pass
           {"offset": 4, "C_B": 0.0050, "C_ST": 0.0010, "contract_ok": True}]
    assert first_passing(res, ref) == 2
    assert first_passing(res[:2], ref) is None
    assert KEPT_POOL_K == [3, 5, 6, 8, 9]
