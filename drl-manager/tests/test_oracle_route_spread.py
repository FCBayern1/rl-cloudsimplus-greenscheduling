"""Tests for the spread routing mode of oracle_fdefer_gate.

argmax routing piles the whole batch onto the greenest DC (84-89% of load on
one site in the five-scenario probe); spread allocates the batch across
green-capable DCs proportionally to current green power, deterministically.
"""
import importlib.util, sys
from collections import Counter
from pathlib import Path

_S = importlib.util.spec_from_file_location(
    "oracle_fdefer_gate",
    Path(__file__).resolve().parents[1] / "oracle_fdefer_gate.py")
og = importlib.util.module_from_spec(_S); sys.modules["oracle_fdefer_gate"] = og
_S.loader.exec_module(og)


def test_argmax_unchanged():
    ga = og._route_batch([5.0, 100.0, 1.0], {0, 1}, 3, 10, "argmax")
    assert ga == [1] * 10


def test_spread_proportional():
    # green 300 vs 100 -> 75/25 split of a 100-batch
    ga = og._route_batch([300.0, 100.0, 0.0], {0, 1}, 3, 100, "spread")
    c = Counter(ga)
    assert len(ga) == 100 and c[0] == 75 and c[1] == 25 and 2 not in c


def test_spread_zero_green_equal_split():
    ga = og._route_batch([0.0, 0.0, 0.0], {0, 1}, 3, 10, "spread")
    c = Counter(ga)
    assert c[0] == 5 and c[1] == 5


def test_spread_never_routes_to_brown_dc():
    ga = og._route_batch([50.0, 10.0, 999.0], {0, 1}, 3, 40, "spread")
    assert 2 not in set(ga) and len(ga) == 40


def test_spread_batch_size_preserved_with_remainder():
    ga = og._route_batch([1.0, 1.0, 1.0], {0, 1, 2}, 3, 10, "spread")
    assert len(ga) == 10 and set(ga) == {0, 1, 2}


def test_spread_deterministic():
    a = og._route_batch([7.0, 3.0], {0, 1}, 2, 33, "spread")
    b = og._route_batch([7.0, 3.0], {0, 1}, 2, 33, "spread")
    assert a == b


def test_spread_is_interleaved_not_blocked():
    # weights 300/100/... -> a short prefix must NOT be all one DC (the bug:
    # blocked list put ~all DC0 in the consumed prefix when batch underfills).
    ga = og._route_batch([300.0, 100.0, 0.0], {0, 1}, 3, 100, "spread")
    prefix = ga[:8]
    assert set(prefix) == {0, 1}, f"prefix not interleaved: {prefix}"
    # and DC0 (higher green) should lead but not monopolise the prefix
    assert prefix.count(0) >= prefix.count(1) >= 1


def test_spread_prefix_roughly_proportional():
    # any reasonable prefix should reflect the 3:1 green ratio, not 100% DC0
    ga = og._route_batch([300.0, 100.0], {0, 1}, 2, 40, "spread")
    p = ga[:20]
    assert 12 <= p.count(0) <= 18 and 2 <= p.count(1) <= 8
