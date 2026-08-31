import numpy as np
import pytest

from tb13_nonfluid_screen import (
    Job, calibrate_scale, load_wind, schedule_blind, schedule_clairvoyant, score,
    run, split_capacity, split_jobs,
)


def test_indivisible_job_cannot_pool_subthreshold_green_across_dcs():
    job = Job(0, 0, 1, 10.0, 0)
    green = np.full((2, 2), 6.0)
    cap = np.array([10.0, 10.0])
    placement = schedule_clairvoyant([job], green, cap)
    rec = score([job], placement, green)
    assert rec["green"] == pytest.approx(6.0)
    assert rec["brown"] == pytest.approx(4.0)


def test_clairvoyant_skips_decoy_but_causal_blind_takes_it():
    job = Job(0, 4, 2, 10.0, 0)
    green = np.zeros((1, 8))
    green[0, 0] = 10.0               # one-row decoy
    green[0, 3:5] = 10.0             # full future window
    cap = np.array([10.0])
    rb = schedule_blind([job], green, cap, threshold=1.0)
    rc = schedule_clairvoyant([job], green, cap)
    assert rb == [(0, 0)]
    assert rc == [(0, 3)]
    assert score([job], rc, green)["carbon"] < score([job], rb, green)["carbon"]


def test_blind_early_decision_does_not_read_changed_future():
    job = Job(0, 5, 2, 10.0, 0)
    cap = np.array([10.0])
    a = np.zeros((1, 10)); a[0, 0] = 10.0
    b = a.copy(); b[0, 4:8] = 100.0
    assert schedule_blind([job], a, cap, 1.0) == schedule_blind([job], b, cap, 1.0)


def test_blind_processes_pending_jobs_on_one_shared_timeline():
    jobs = [Job(0, 4, 1, 5.0, 0), Job(0, 4, 1, 5.0, 1)]
    cap = np.array([10.0])
    green = np.zeros((1, 8)); green[0, 2] = 10.0
    # Both jobs are pending when row 2 arrives and both fit there.  A
    # job-by-job future scan can reserve row 2 for one job and mishandle the
    # other; the event loop releases both from the same causal state.
    assert schedule_blind(jobs, green, cap, 1.0) == [(0, 2), (0, 2)]


def test_capacity_is_fixed_in_aggregate():
    for k in (1, 2, 3, 5):
        cap = split_capacity(5, k, 10.0)
        assert cap.sum() == pytest.approx(50.0)
        assert np.all(cap >= 10.0)


def test_integrated_kappa_calibration():
    raw = np.ones((2, 20))
    jobs = [Job(0, 0, 2, 10.0, 0)]
    # Monkey-sized episodes are represented by repeating the same offset; the
    # function's algebra must still hit the requested integrated ratio.
    scale = calibrate_scale(raw[:, :1].repeat(288, axis=1), [0], jobs, 0.8)
    green = raw[:, :1].repeat(288, axis=1).sum() * scale
    work = sum(j.power_w * j.duration for j in jobs)
    assert green / work == pytest.approx(0.8)


def test_multidc_requires_at_least_one_job_capacity_per_dc():
    with pytest.raises(ValueError, match="n_jobs must be >="):
        run((8, 9, 29), n_jobs=2)


def test_turbines_can_be_aggregated_inside_one_dc():
    grouped = load_wind(((100, 101),), 2020)
    separate = load_wind((100, 101), 2020)
    assert grouped.shape[0] == 1
    assert np.array_equal(grouped[0], separate.sum(axis=0))


def test_fluid_split_preserves_total_work():
    jobs = [Job(1, 5, 4, 12.0, 7)]
    parts = split_jobs(jobs, 4)
    assert len(parts) == 16
    assert sum(x.power_w * x.duration for x in parts) == pytest.approx(48.0)
    assert {(x.arrival, x.latest, x.duration) for x in parts} == {(1, 5, 1)}


def test_oracle_threshold_stress_cannot_have_more_blind_carbon():
    rec = run(((100, 101),), kappa=1.0)
    # A per-episode post-hoc choice contains the frozen q* as one candidate, so
    # it can only shrink the apparent clairvoyant advantage.
    assert (rec["oracle_threshold_stress"]["pooled_effect"]
            >= rec["pooled_effect"] - 1e-12)
