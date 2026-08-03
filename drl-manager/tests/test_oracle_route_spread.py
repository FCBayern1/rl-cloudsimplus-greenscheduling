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
