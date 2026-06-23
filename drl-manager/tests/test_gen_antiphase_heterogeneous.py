"""Tests for the anti-phase workload generator's heterogeneity + invariants.

The old trace made every cloudlet identical (length/pes uniq=1), which caused the
deterministic global policy to route a whole 128-cloudlet batch onto a single DC. The
generator now draws sizes from a Gaussian. These tests lock in:
  1. cloudlet sizes are heterogeneous (many distinct lengths/pes);
  2. mean length is preserved (so total load / util is unchanged vs the uniform trace);
  3. the anti-phase arrival structure is preserved (most arrivals land in brown steps);
  4. deadlines remain arrival + slack.
"""
import numpy as np
import pytest

from gen_antiphase_workload import build_cloudlets


def _fake_green_ts(T=2000, nd=5, seed=1):
    # Two real green DCs with an anti-phase-ish sinusoid; rest zero.
    t = np.arange(T)
    g0 = np.clip(np.sin(t / 120.0) * 600 + 600, 0, None)
    g1 = np.clip(np.cos(t / 130.0) * 500 + 500, 0, None)
    G = np.zeros((T, nd))
    G[:, 0] = g0
    G[:, 1] = g1
    return G


def test_sizes_are_heterogeneous():
    G = _fake_green_ts()
    rows, st = build_cloudlets(G, n=5000, mi=400000, pes=2, seed=7)
    assert st["len_uniq"] > 1000, "lengths should be heterogeneous, not a single constant"
    assert st["pes_uniq"] >= 3, "pes should take several distinct values"
    lengths = [r[2] for r in rows]
    assert min(lengths) >= 1
    # spread is real
    assert np.std(lengths) > 0.2 * 400000


def test_mean_length_preserved_load_invariant():
    G = _fake_green_ts()
    mi = 400000
    rows, st = build_cloudlets(G, n=8000, mi=mi, pes=2, seed=7)
    # realized mean length within 0.5% of target → total work preserved
    assert abs(st["mean_len"] - mi) / mi < 0.005


def test_antiphase_structure_preserved():
    G = _fake_green_ts()
    rows, st = build_cloudlets(G, n=8000, mi=400000, pes=2, brown_frac=0.9, seed=7)
    # ~90% of arrivals should land in brown (low-green) steps
    assert st["in_brown"] > 0.8


def test_deadline_is_arrival_plus_slack():
    G = _fake_green_ts()
    rows, _ = build_cloudlets(G, n=1000, mi=400000, pes=2, deadline_steps=3600, seed=7)
    for (_i, a, _len, _p, _fs, _os, ddl) in rows:
        assert ddl == a + 3600


def test_uniform_when_std_zero():
    """std_frac=0 collapses back to (near-)uniform — a strict generalization."""
    G = _fake_green_ts()
    rows, st = build_cloudlets(G, n=2000, mi=400000, pes=2, mi_std_frac=0.0, seed=7)
    assert st["len_uniq"] == 1
