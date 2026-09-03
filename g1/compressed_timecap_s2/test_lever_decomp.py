import numpy as np

from lever_decomp import decompose


def test_ordering_and_bounds_on_random_green():
    rng = np.random.default_rng(0)
    G = rng.uniform(0, 260, size=(3, 600))
    r = decompose(G, arrivals=[5, 40, 90, 150, 300], runtime=8, wait_cap=30, p_job=132.7)
    assert r["brown_oracle"] <= r["brown_myopic"] + 1e-12
    assert r["brown_oracle"] <= r["brown_now"] + 1e-12
    assert 0.0 <= r["wasted_wait_rate"] <= 1.0
    assert abs(r["lever_forecast_only"] - (r["brown_myopic"] - r["brown_oracle"])) < 1e-12


def test_always_green_means_no_lever_and_no_waste():
    G = np.full((2, 300), 500.0)
    r = decompose(G, arrivals=[0, 50, 100], runtime=10, wait_cap=20)
    assert r["brown_now"] == 0.0 and r["lever_forecast_only"] == 0.0
    assert r["wasted_wait_rate"] == 0.0 and r["full_green_now_rate"] == 1.0


def test_green_arrives_later_myopic_matches_oracle_when_waiting_is_free():
    # famine for rows 0..49, feast afterwards: waiting (lead-0 only) captures everything
    G = np.zeros((1, 300))
    G[0, 50:] = 1000.0
    r = decompose(G, arrivals=[0, 10], runtime=5, wait_cap=60)
    assert r["brown_now"] == 1.0
    assert r["brown_myopic"] == 0.0 and r["brown_oracle"] == 0.0
    assert r["lever_forecast_only"] == 0.0


def test_famine_that_never_ends_is_a_wasted_wait():
    G = np.zeros((1, 300))
    r = decompose(G, arrivals=[0], runtime=5, wait_cap=20)
    assert r["wasted_wait_rate"] == 1.0
    assert r["brown_myopic"] == 1.0 and r["brown_oracle"] == 1.0
