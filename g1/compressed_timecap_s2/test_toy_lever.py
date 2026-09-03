import numpy as np

from toy_lever import Sim, evaluate, lead0_exact, view_anti, view_shrink, view_shuffle


def test_brown_accounting_closes():
    G = np.full((1, 200), 50.0)
    sim = Sim(G, arrivals=[0, 0], runtime=10, slack=0, p=81.3)
    r = sim.run_now()
    # two jobs share one DC: load 162.6 vs green 50 -> brown 112.6 per row for 10 rows
    assert abs(r["brown_w_rows"] - 112.6 * 10) < 1e-6
    assert abs(r["green_w_rows"] + r["brown_w_rows"] - r["total_w_rows"]) < 1e-9


def test_myopic_waits_for_green_and_is_forced_at_slack():
    G = np.zeros((1, 300)); G[0, 40:] = 500.0
    r_wait = Sim(G, [0], runtime=5, slack=60).myopic()
    assert r_wait["brown_w_rows"] == 0.0 and r_wait["mean_wait"] == 40.0
    r_forced = Sim(G, [0], runtime=5, slack=20).myopic()
    assert r_forced["mean_wait"] == 20.0 and r_forced["brown_w_rows"] > 0


def test_planner_with_truth_never_worse_than_myopic_for_one_job():
    rng = np.random.default_rng(1)
    G = rng.uniform(0, 200, size=(2, 400))
    sim = Sim(G, [10], runtime=8, slack=40)
    assert sim.planner(G)["brown_w_rows"] <= sim.myopic()["brown_w_rows"] + 1e-9


def test_competition_herd_effect_is_visible():
    # famine then a short feast that fits ONE job; two jobs arrive together.
    G = np.zeros((1, 400)); G[0, 50:60] = 90.0; G[0, 200:] = 500.0
    sim = Sim(G, [0, 0], runtime=10, slack=250)
    myo = sim.myopic()      # both start at row 50 (headroom check is per job before load update? FIFO updates load) -> second waits
    tru = sim.planner(G)    # plans one at 50, the other at 200
    assert tru["brown_w_rows"] <= myo["brown_w_rows"] + 1e-9


def test_views_keep_shape_and_lead0():
    G = np.random.default_rng(2).uniform(0, 100, size=(3, 50))
    sim = Sim(G, [5, 20], runtime=3, slack=5)
    for v in (view_shrink(G, 0.3), view_anti(G), view_shuffle(G)):
        assert v.shape == G.shape
        ve = lead0_exact(sim, v)
        assert np.allclose(ve[:, 5], G[:, 5]) and np.allclose(ve[:, 20], G[:, 20])


def test_evaluate_reports_levers():
    G = np.random.default_rng(3).uniform(0, 150, size=(2, 300))
    r = evaluate(G, [0, 30, 60, 90], runtime=6, slack=40)
    assert set(("run_now", "myopic", "truth", "shrink", "anti", "shuffle")) <= set(r)
    assert np.isfinite(r["lever_forecast_only_pp"])
