"""Regression test for _to_scalar in local_schedulers.

Original PSO/GA/PackingAware schedulers used `float(local_obs.get(key, 0) or 0.0)`,
which raised `TypeError: only 0-dimensional arrays can be converted to Python
scalars` when the env handed back a numpy array of size > 1 (real-world case
that broke the full comparison run on 2026-05-07). The helper must accept
None / scalars / 0-d arrays / >0-d arrays without raising.
"""

import numpy as np

from src.baselines.local_schedulers import _to_scalar


def test_none_returns_default():
    assert _to_scalar(None) == 0.0
    assert _to_scalar(None, default=-1.5) == -1.5


def test_python_int():
    assert _to_scalar(7) == 7.0


def test_python_float():
    assert _to_scalar(3.14) == 3.14


def test_zero_d_array():
    assert _to_scalar(np.array(5)) == 5.0


def test_one_d_array_single_element():
    assert _to_scalar(np.array([5])) == 5.0


def test_one_d_array_multiple_elements_takes_first():
    # If the env hands back e.g. np.array([5, 10, 20]) we must not raise;
    # taking flat[0] is the documented behaviour.
    assert _to_scalar(np.array([5, 10, 20])) == 5.0


def test_two_d_array():
    assert _to_scalar(np.array([[5, 10], [20, 30]])) == 5.0


def test_empty_array_returns_default():
    assert _to_scalar(np.array([])) == 0.0
    assert _to_scalar(np.array([]), default=99.0) == 99.0


def test_original_failure_case():
    """Reproduce the exact call shape that crashed PSO_local on 2026-05-07."""
    obs = {"next_cloudlet_pes": np.array([5, 10])}  # the shape that broke prod
    # Plain float() would raise; _to_scalar must succeed:
    assert _to_scalar(obs.get("next_cloudlet_pes")) == 5.0
