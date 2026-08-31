"""Reservation-ledger contract for the curve-informed planner (Codex, 2026-08-30).

The first build of this arm accumulated one capacity grid and replanned every slot
every step. Four things went wrong at once and the run still produced a plausible
looking number, so the ledger is pinned here rather than in a report.

    padding was planned      empty slots were forced to pes=1, mi=1 and booked capacity
    reservations doubled     a deferred job came back and booked its future a second time
    plans were never kept    only occupancy was remembered, never "job i at s on dc d"
    deadlines never bound    time_to_deadline never reached the arm and defaulted to 1e9

The batch shows only the first 128 queued jobs, so a deferred job can vanish for many
steps. Clearing the plan each step would lose its reservation, which is why the test
below walks a job out of the batch and back in.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("yaml")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def planner_cls():
    os.environ.setdefault("EVAL_CONFIG_PATH", os.path.join(REPO, "config_C.yml"))
    os.environ.setdefault("ORACLE_EXPERIMENT", "experiment_g1eval_matchedvan")
    from src.baselines.global_schedulers import CurveInformedPlannerGlobalScheduler
    return CurveInformedPlannerGlobalScheduler


@pytest.fixture
def planner(planner_cls):
    return planner_cls(5, 8)


def batch(n, ids, mi=None, pes=None, ttd=None, present=None, clock=0.0):
    """A planner channel with n slots, -1 marking padding."""
    real = len(ids)
    ids = list(ids) + [-1] * (n - real)

    def field(values, default, pad, dtype):
        vals = list(values) if values is not None else [default] * real
        assert len(vals) == real, "one value per real job"
        return np.array(vals + [pad] * (n - real), dtype=dtype)

    return {
        "planner": {
            "batch_cloudlet_ids": np.array(ids, dtype=np.int64),
            "batch_cloudlet_mi": field(mi, 40000, 0, np.int64),
            "batch_cloudlet_pes": field(pes, 2, 0, np.int64),
            "batch_cloudlet_time_to_deadline": field(ttd, 3600.0, 0.0, np.float64),
            "batch_cloudlet_deadline_present": field(present, 1, 0, np.int64),
            "batch_cloudlet_is_deferred": np.zeros(n, dtype=np.int64),
            "batch_cloudlet_wait_age": np.zeros(n, dtype=np.float64),
            "current_clock": clock,
        }
    }


def test_missing_planner_channel_raises_instead_of_filling(planner):
    """The 1e9 deadline default is exactly how the first build lost its constraint."""
    with pytest.raises(RuntimeError, match="planner channel"):
        planner.schedule({"batch_cloudlet_mi": [40000] * 8})


def test_padding_slots_are_never_planned(planner):
    """Eight empty slots must leave the occupancy grid untouched."""
    planner.schedule(batch(8, []))
    assert planner.occ.sum() == 0.0
    assert planner.n_plan == 0
    assert not planner.reservations and not planner.active


def test_capacity_is_real_vm_pes_not_hosts_times_64(planner):
    assert list(planner.cap) == [600.0, 480.0, 296.0, 240.0, 184.0]


def test_one_job_books_capacity_exactly_once(planner):
    planner.schedule(batch(8, [7]))
    assert planner.n_plan == 1
    booked = planner.occ.sum()
    assert booked > 0.0
    entry = dict(planner.reservations, **planner.active)[7]
    d, s, e, p = entry
    assert booked == pytest.approx(p * (e - s))


def test_a_job_that_leaves_the_batch_keeps_its_reservation(planner):
    """The batch is a 128-slot window on a longer queue, not the queue itself."""
    planner.schedule(batch(8, [7], ttd=[3600.0]))
    if 7 not in planner.reservations:
        pytest.skip("job dispatched immediately in this window, nothing to hold")
    held = planner.reservations[7]
    booked = planner.occ.sum()

    for _ in range(5):                       # job is out of the visible batch
        planner.schedule(batch(8, [101, 102]))

    assert 7 in planner.reservations, "reservation lost while the job was out of view"
    assert planner.reservations[7] == held, "reservation drifted while out of view"

    before = planner.n_plan
    planner.schedule(batch(8, [7]))          # job comes back
    assert planner.reservations.get(7, planner.active.get(7))[:2] == held[:2]
    assert planner.n_plan == before, "returning job was replanned and booked twice"
    other = planner.occ.sum() - booked
    assert other >= 0.0


def test_repeated_presentation_does_not_double_book(planner):
    planner.schedule(batch(8, [7]))
    first = planner.occ.sum()
    for _ in range(4):
        planner.schedule(batch(8, [7]))
    assert planner.occ.sum() == pytest.approx(first)


def test_active_occupancy_is_released_when_the_run_window_ends(planner):
    """A short job routed now must free its PEs once it has run."""
    planner.schedule(batch(8, [7], mi=[40000], pes=[2], ttd=[0.0], present=[1]))
    assert 7 in planner.active, "a job with no slack must be dispatched now"
    d, s, e, p = planner.active[7]
    for _ in range(e + 2):
        planner.schedule(batch(8, []))
    assert 7 not in planner.active
    assert planner.occ.sum() == pytest.approx(0.0)


def test_planned_plus_active_never_exceeds_capacity(planner_cls):
    planner = planner_cls(5, 64)
    ids = list(range(1, 40))
    for step in range(12):
        planner.schedule(batch(64, ids, pes=[8] * len(ids), ttd=[3600.0] * len(ids)))
    for d in range(5):
        assert planner.occ[d].max() <= planner.cap[d] + 1e-9, f"DC{d} oversubscribed"


def test_deadline_binds_the_latest_start(planner):
    """A job whose deadline is one runtime away cannot be deferred."""
    planner.schedule(batch(8, [7], mi=[40000], pes=[2], ttd=[1.0], present=[1]))
    assert 7 in planner.active, "tight deadline was deferred anyway"
    assert planner.n_defer == 0


def test_reset_clears_clock_and_all_three_ledgers(planner):
    planner.schedule(batch(8, [7, 8, 9]))
    assert planner.t == 1
    assert planner.occ.sum() > 0.0
    planner.reset()
    assert planner.t == 0
    assert planner.occ.sum() == 0.0
    assert planner.reservations == {} and planner.active == {}
    assert planner.n_plan == 0 and planner.n_defer == 0


# ── The arm family: one planner, one capacity ledger, one carbon model ──────────
# Codex 2026-08-30: a blind arm that cannot wait makes the gap to a waiting arm read as
# the value of the forecast when most of it is the value of waiting. The family below
# differs only in the green trace it may plan against, and in whether waiting is allowed.

def test_family_shares_planner_capacity_and_carbon(planner_cls):
    from src.baselines.global_schedulers import (
        ClimatologyPlannerGlobalScheduler,
        NoWaitPlannerGlobalScheduler,
        PersistencePlannerGlobalScheduler,
    )
    arms = [planner_cls, PersistencePlannerGlobalScheduler,
            ClimatologyPlannerGlobalScheduler, NoWaitPlannerGlobalScheduler]
    built = [a(5, 8) for a in arms]
    ref = built[0]
    for arm in built[1:]:
        assert list(arm.cap) == list(ref.cap), "capacity model differs across the family"
        assert list(arm.cb) == list(ref.cb), "brown carbon model differs across the family"
        assert list(arm.cg) == list(ref.cg), "green carbon model differs across the family"
        assert arm.mips == ref.mips and arm.dyn_per_pe == ref.dyn_per_pe


def test_only_the_green_view_differs(planner_cls):
    from src.baselines.global_schedulers import PersistencePlannerGlobalScheduler
    curve = planner_cls(5, 8)
    pers = PersistencePlannerGlobalScheduler(5, 8)
    pers.green_now = np.array([123.0, 0.0, 0.0, 0.0, 0.0])
    pers.t = 10
    view = pers._green_view(0)
    assert np.all(view[10:] == 123.0), "persistence must hold the current level flat"
    assert not np.allclose(view[10:], curve._green_view(0)[10:]), "views must differ"


def test_climatology_is_calibrated_before_the_window(planner_cls, monkeypatch):
    from src.baselines.global_schedulers import ClimatologyPlannerGlobalScheduler
    # A climatology needs history to calibrate on, so the window has to start somewhere
    # other than row zero. Every real cell sets this; at offset zero there is nothing
    # before the window and the level is correctly left at zero.
    monkeypatch.setenv("ORACLE_OFFSET_ROWS", "19171")
    clim = ClimatologyPlannerGlobalScheduler(5, 8)
    # G is laid onto the planning grid at the first decision, once the simulator's clock
    # is known, so the raw per-row series is what exists before any step.
    windy = [d for d in range(5) if clim.G_rows[d].size > 1 and clim.G_rows[d].any()]
    assert windy, "expected at least one site with a turbine"
    for d in windy:
        assert clim.clim[d] > 0.0, f"DC{d} climatology never calibrated"
    for d in range(5):
        if clim.G_rows[d].size <= 1:
            assert clim.clim[d] == 0.0, "a turbine-free site must stay at zero green"


def test_nowait_arm_never_defers_a_job(planner_cls):
    from src.baselines.global_schedulers import NoWaitPlannerGlobalScheduler
    arm = NoWaitPlannerGlobalScheduler(5, 8)
    arm.schedule(batch(8, [1, 2, 3], ttd=[3600.0] * 3))
    assert arm.n_defer == 0, "the no-wait arm deferred, so it is not the temporal control"
    assert not arm.reservations
    assert len(arm.active) == 3


# ── Two-timepoint contract ─────────────────────────────────────────────────────
# Codex 2026-08-30: running both arms to a fixed step charges the fast one idle carbon
# it would never have burned, and stopping at natural completion lets the slow one park
# its tail outside the ledger. Waiting closes for everyone at the registered boundary and
# the remainder drains to what each arm already committed to.

def test_no_new_wait_is_opened_after_the_decision_boundary(planner_cls, monkeypatch):
    monkeypatch.setenv("PLANNER_DECISION_HORIZON", "3")
    arm = planner_cls(5, 8)
    for _ in range(3):
        arm.schedule(batch(8, []))
    before = arm.n_defer
    arm.schedule(batch(8, [11, 12], ttd=[3600.0, 3600.0]))
    assert arm.n_defer == before, "a new wait was opened past the boundary"
    assert 11 in arm.active and 12 in arm.active


def test_a_reservation_is_not_pulled_forward_at_the_boundary(planner_cls, monkeypatch):
    """A frozen reservation is a commitment, not a decision still open.

    The first build pulled every open reservation forward the instant the boundary
    arrived. On a real cell that dispatched 4542 jobs in one step into sites holding 480,
    384 and 296 PEs, destroying a plan that had been feasible and driving the occupancy
    ledger away from the simulator. Executing an existing commitment is not deciding.
    """
    monkeypatch.setenv("PLANNER_DECISION_HORIZON", "4")
    arm = planner_cls(5, 8)
    arm.schedule(batch(8, [11], ttd=[3600.0]))
    if 11 not in arm.reservations:
        pytest.skip("job dispatched immediately in this window, nothing to hold")
    held = arm.reservations[11]
    for _ in range(6):
        arm.schedule(batch(8, [11], ttd=[3600.0]))
    assert arm.n_drain_pulled == 0, "the boundary pulled a reservation forward"
    if 11 in arm.reservations:
        assert arm.reservations[11] == held, "the boundary moved a frozen commitment"
    else:
        assert arm.active[11][:2] == held[:2], "the boundary changed site or start"


def test_the_boundary_does_not_dispatch_the_whole_backlog_at_once(planner_cls, monkeypatch):
    """Past the boundary the backlog leaves through a rate-limited drain, not in one go."""
    monkeypatch.setenv("PLANNER_DECISION_HORIZON", "1")
    arm = planner_cls(5, 64)
    ids = list(range(1, 50))
    obs = batch(64, ids, pes=[8] * len(ids), ttd=[3600.0] * len(ids))
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    arm.schedule(obs)                      # step 0, still planning
    arm.schedule(obs)                      # step 1, boundary reached
    dispatched_pes = sum(p for (_d, _s, _e, p) in arm.active.values())
    assert dispatched_pes <= float(np.sum(arm.cap)), \
        "the drain dispatched more PEs than the whole fleet has"
    assert arm.n_drain_waited > 0 or arm.n_drain_dispatched < len(ids), \
        "nothing was held back, so the drain is not rate limited"


def test_the_drain_never_exceeds_the_reported_free_pes_per_dc(planner_cls, monkeypatch):
    monkeypatch.setenv("PLANNER_DECISION_HORIZON", "1")
    arm = planner_cls(5, 64)
    ids = list(range(1, 60))
    free = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs = batch(64, ids, pes=[8] * len(ids), ttd=[3600.0] * len(ids))
    obs["dc_available_pes"] = free
    arm.schedule(obs)
    arm.schedule(obs)
    per_dc = np.zeros(5)
    for (d, _s, _e, p) in arm.active.values():
        per_dc[d] += p
    for d in range(5):
        assert per_dc[d] <= free[d] + 1e-9, \
            f"DC{d} drained {per_dc[d]} PEs against {free[d]} reported free"


def test_the_ledger_uses_the_measured_start_lag(planner_cls):
    """A route issued at step t is executing by the state observed at t+1.

    Measured by g1/check_route_visibility_lag.py on this gateway: three routing events at
    different steps and batch sizes, utilisation moving one step after the route every
    time, and dc_available_pes reporting that occupancy seven observations later still.
    """
    arm = planner_cls(5, 8)
    assert arm.START_LAG == 1
    assert arm.AVAIL_REPORT_LAG == 7
    obs = batch(8, [11], mi=[40000], pes=[2], ttd=[0.0], present=[1])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    arm.schedule(obs)
    assert 11 in arm.active, "a job with no slack must go now"
    d, start, end, p = arm.active[11]
    assert start == 1, f"the ledger starts the job at {start}, not at t+1"
    assert arm.occ[d, 0] == 0.0, "the ledger claims the job runs in the step it was routed"
    assert arm.occ[d, 1] == p


def test_the_boundary_is_off_by_default(planner_cls):
    arm = planner_cls(5, 8)
    assert arm.decision_horizon == 0
    assert arm.draining is False


# ── reactive_wait: a causal stopping rule, not a planner ───────────────────────
# Codex 2026-08-30: persistence flattens the future to the current level, which prices
# every candidate start alike and makes it take the earliest one. reactive_wait is a
# different rule and has historically been the strongest blind, so it is its own arm.

def test_reactive_wait_books_no_future_capacity():
    from src.baselines.global_schedulers import ReactiveWaitPlannerGlobalScheduler
    arm = ReactiveWaitPlannerGlobalScheduler(5, 8)
    for _ in range(6):
        arm.schedule(batch(8, [11, 12], ttd=[3600.0, 3600.0]))
    assert arm.reservations == {}, "a reactive arm must never hold a reservation"
    for d in range(5):
        held = arm.occ[d, arm.t:]
        assert held.sum() == pytest.approx(sum(
            p * max(0, e - arm.t) for (dc, s, e, p) in arm.active.values() if dc == d)), \
            "future occupancy must come only from jobs already running"


def test_reactive_wait_waits_while_there_is_no_green():
    from src.baselines.global_schedulers import ReactiveWaitPlannerGlobalScheduler
    arm = ReactiveWaitPlannerGlobalScheduler(5, 8)
    obs = batch(8, [11], ttd=[3600.0])
    obs["dc_current_green_power_w"] = np.zeros(5)
    arm.schedule(obs)
    assert 11 not in arm.active, "routed a job with no green on any meter"
    assert arm.n_defer == 1


def test_reactive_wait_goes_as_soon_as_the_meter_carries_the_job():
    from src.baselines.global_schedulers import ReactiveWaitPlannerGlobalScheduler
    arm = ReactiveWaitPlannerGlobalScheduler(5, 8)
    dark = batch(8, [11], ttd=[3600.0])
    dark["dc_current_green_power_w"] = np.zeros(5)
    arm.schedule(dark)
    assert 11 not in arm.active

    bright = batch(8, [11], ttd=[3599.0])
    bright["dc_current_green_power_w"] = np.array([5000.0, 0.0, 0.0, 0.0, 0.0])
    arm.schedule(bright)
    assert 11 in arm.active, "green arrived and the job still did not start"
    assert arm.active[11][0] == 0


def test_reactive_wait_routes_unconditionally_at_the_margin():
    from src.baselines.global_schedulers import ReactiveWaitPlannerGlobalScheduler
    arm = ReactiveWaitPlannerGlobalScheduler(5, 8)
    obs = batch(8, [11], mi=[40000], pes=[2], ttd=[1.0], present=[1])
    obs["dc_current_green_power_w"] = np.zeros(5)
    arm.schedule(obs)
    assert 11 in arm.active, "the latest-start margin did not force a route"
    assert arm.n_fallback == 1


def test_the_family_is_not_wrapped_by_the_threshold_defer_rule(planner_cls):
    """DeferringGlobalScheduler only adds defers. Wrapping a planner would rewrite a
    dispatch into a wait while the reservation stayed put, stranding the booking."""
    from src.baselines.global_schedulers import (
        ClimatologyPlannerGlobalScheduler, NoWaitPlannerGlobalScheduler,
        PersistencePlannerGlobalScheduler, ReactiveWaitPlannerGlobalScheduler)
    for cls in (planner_cls, PersistencePlannerGlobalScheduler,
                ClimatologyPlannerGlobalScheduler, ReactiveWaitPlannerGlobalScheduler,
                NoWaitPlannerGlobalScheduler):
        assert getattr(cls, "HANDLES_DEFER", False), f"{cls.__name__} would be wrapped"


def test_capacity_is_taken_from_the_simulator_not_from_config_arithmetic(planner):
    """Three of five sites configure more VM PEs than their hosts can carry.

    hosts*64 overstated every site and the VM total still overstated DC0, DC1 and DC4,
    so the planner reserved room that does not exist. The simulator reports the truth at
    the first step, when nothing is running yet.
    """
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(obs)
    assert list(planner.cap) == [480.0, 384.0, 296.0, 240.0, 144.0]
    assert list(planner.cap_config) == [600.0, 480.0, 296.0, 240.0, 184.0]
    assert planner.drift_abs_max == pytest.approx(0.0), "calibration left a phantom gap"


def test_capacity_is_calibrated_once_not_every_step(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(obs)
    busy = batch(8, [])
    busy["dc_available_pes"] = np.array([100.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(busy)
    assert planner.cap[0] == 480.0, "capacity shrank when a site got busy"
    assert planner.drift_abs_max > 0.0, "the sentinel must see the load it cannot explain"


# ── Capacity read is locked (Codex 2026-08-30) ─────────────────────────────────

def test_capacity_read_is_refused_against_running_load(planner):
    """A read taken while work is running understates the site and would go unnoticed."""
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([100.0, 384.0, 296.0, 240.0, 144.0])
    obs["dc_utilizations"] = np.array([0.4, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="idle initialisation point"):
        planner.schedule(obs)


def test_capacity_that_disagrees_with_the_registered_vector_stops_the_run(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([999.0, 384.0, 296.0, 240.0, 144.0])
    with pytest.raises(RuntimeError, match="registered vector"):
        planner.schedule(obs)


def test_capacity_is_reread_and_refrozen_after_reset(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(obs)
    assert planner._cap_calibrated
    planner.reset()
    assert not planner._cap_calibrated
    assert list(planner.cap) == list(planner.cap_config), "reset kept a stale capacity"
    planner.schedule(obs)
    assert list(planner.cap) == [480.0, 384.0, 296.0, 240.0, 144.0]


# ── Runtime model, calibrated against the simulator's own finish events ────────
# 2026-08-30: a cloudlet runs at cloudlet_cpu_utilization of a VM PE. Java defaults that
# key to 0.5 and this experiment never sets it, so a job occupies its site for twice the
# nominal length/mips. Measured over 12 finish events: elapsed/(length/mips) = 2.0166,
# sd 0.0156, no dependence on PES; forcing the key to 0.25 moved it to 4.0201.

def test_runtime_uses_the_effective_rate_not_the_nominal_one(planner):
    assert planner.cpu_util == 0.5, "this experiment leaves the key unset, so Java's 0.5 applies"
    assert planner.mips == pytest.approx(20000.0), "effective rate must be mips * cpu_util"


def test_a_job_is_held_for_the_measured_duration(planner):
    """A 40000 MI job nominally runs one step and really occupies the site for two."""
    obs = batch(8, [11], mi=[40000], pes=[2], ttd=[0.0], present=[1])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(obs)
    d, start, end, p = planner.active[11]
    assert end - start == 2, f"held for {end - start} steps, expected 2"


def test_the_rate_follows_the_configured_utilisation(planner_cls, monkeypatch, tmp_path):
    """Not a hard-coded 2x: the factor comes from the same key Java reads."""
    import yaml
    cfg = yaml.safe_load(open(os.environ["EVAL_CONFIG_PATH"]))
    blk = cfg[os.environ["ORACLE_EXPERIMENT"]]
    blk["cloudlet_cpu_utilization"] = 0.25
    path = tmp_path / "quarter.yml"
    path.write_text(yaml.safe_dump({os.environ["ORACLE_EXPERIMENT"]: blk}))
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(path))
    # The wind directory is derived from the config's own parent, so a config written to
    # a temporary directory has to be told where the traces still live.
    monkeypatch.setenv("ORACLE_WIND_DIR", os.path.join(
        REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"))
    arm = planner_cls(5, 8)
    assert arm.cpu_util == 0.25
    assert arm.mips == pytest.approx(10000.0)


# ── Per-id closure against the simulator's execution events ───────────────────
# Codex 2026-08-30: the old contract compared planner_occ with cap - dc_available_pes.
# That field is a VM allocation counter which never recovers once a cloudlet finishes, so
# it could neither budget the drain nor audit the ledger. The contract is now per id.

def trace(started=(), running=(), running_pes=None):
    """Execution events in the CSV encoding the gateway emits."""
    d = {}
    if started:
        d["exec_started_csv"] = ";".join(
            f"{i}:{dc}:{pes}:{t}" for i, dc, pes, t in started)
    if running:
        d["exec_running_csv"] = ";".join(f"{i}:{dc}:{pes}" for i, dc, pes in running)
    if running_pes is not None:
        d["dc_running_pes_csv"] = ",".join(str(int(v)) for v in running_pes)
    return d


def test_a_start_the_planner_never_ordered_is_counted(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs.update(trace(started=[(999, 0, 2, 1.0)]))
    planner.schedule(obs)
    assert planner.n_unplanned_start == 1, "a backstop start went unnoticed"


def test_a_start_on_the_wrong_site_is_counted(planner):
    obs = batch(8, [11], mi=[40000], pes=[2], ttd=[0.0], present=[1])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(obs)
    committed = planner.active[11][0]
    wrong = (committed + 1) % 5
    nxt = batch(8, [])
    nxt.update(trace(started=[(11, wrong, 2, 2.0)]))
    planner.schedule(nxt)
    assert planner.n_wrong_dc == 1, "the job started elsewhere and the ledger agreed anyway"
    assert planner.n_unplanned_start == 0


def test_a_start_on_the_committed_site_closes_cleanly(planner):
    obs = batch(8, [11], mi=[40000], pes=[2], ttd=[0.0], present=[1])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    planner.schedule(obs)
    committed = planner.active[11][0]
    nxt = batch(8, [])
    nxt.update(trace(started=[(11, committed, 2, 2.0)]))
    planner.schedule(nxt)
    assert planner.n_wrong_dc == 0 and planner.n_unplanned_start == 0
    assert planner.metrics()["planner_n_dispatched_never_started"] == 0


def test_execution_beyond_capacity_is_recorded(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs.update(trace(running_pes=[500, 0, 0, 0, 0]))
    planner.schedule(obs)
    assert planner.running_pes_over_cap == pytest.approx(20.0)


def test_the_drain_budgets_on_real_execution_not_on_allocation(planner_cls, monkeypatch):
    """DC0 fully busy in reality must take nothing, even though allocation says it is free."""
    monkeypatch.setenv("PLANNER_DECISION_HORIZON", "1")
    arm = planner_cls(5, 64)
    ids = list(range(1, 30))
    obs = batch(64, ids, pes=[8] * len(ids), ttd=[3600.0] * len(ids))
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    arm.schedule(obs)
    nxt = batch(64, ids, pes=[8] * len(ids), ttd=[3600.0] * len(ids))
    nxt.update(trace(running_pes=[480, 384, 296, 240, 144]))
    arm.schedule(nxt)
    drained_now = [d for (d, s, _e, _p) in arm.active.values() if s >= arm.t - 1]
    assert 0 not in drained_now or arm.n_drain_waited > 0, \
        "the drain ignored that every site was already fully executing"


def test_a_utilisation_mismatch_between_python_and_java_stops_the_run(planner):
    """A silent fallback to Java's 0.5 default is how every runtime here came to be 2x."""
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["cloudlet_cpu_utilization_effective"] = "1.0"
    with pytest.raises(RuntimeError, match="utilization mismatch"):
        planner.schedule(obs)


def test_a_matching_utilisation_passes_and_is_checked_once(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["cloudlet_cpu_utilization_effective"] = "0.5"
    planner.schedule(obs)
    assert planner._cpu_util_checked
    planner.schedule(batch(8, []))



# ── Latest start is derived, not searched (Codex 2026-08-31) ──────────────────
# The active backstop is the legacy fixed lead: it fires when now + 600 s >= deadline,
# with no reference to runtime. A job must beat both that and its own completion, so
# latest = D - max(r + eps, S + eps). A fixed margin grid of {2..64} steps could never
# reach 600 and was abandoned rather than widened.

def test_the_slack_and_mode_come_from_the_same_defaults_java_uses(planner):
    assert planner.backstop_slack == 600.0
    assert planner.backstop_mode == "legacy"
    assert planner.eps == 2


def test_a_short_job_is_bound_by_the_fixed_six_hundred_second_lead(planner):
    """Runtime 2 steps, deadline 5000 away: the 600 s lead binds, not the runtime."""
    planner._ttd = np.array([5000.0] + [0.0] * 7)
    present = np.array([1] + [0] * 7)
    latest = planner._latest_start(0, 2, present)
    assert latest == 5000 - 602, f"got {latest}, expected D - (S + eps)"


def test_a_long_job_is_bound_by_its_own_runtime(planner):
    """Runtime 900 steps exceeds the 600 s lead, so completion binds instead."""
    planner._ttd = np.array([5000.0] + [0.0] * 7)
    present = np.array([1] + [0] * 7)
    latest = planner._latest_start(0, 900, present)
    assert latest == 5000 - 902, f"got {latest}, expected D - (r + eps)"


def test_no_job_pays_an_extra_runtime_just_to_dodge_the_backstop(planner):
    """The lead is a max, not a sum: a 900 step job loses 902, not 1502."""
    planner._ttd = np.array([5000.0] + [0.0] * 7)
    present = np.array([1] + [0] * 7)
    assert planner._latest_start(0, 900, present) > 5000 - 900 - 600


def test_the_crossover_is_at_the_slack_itself(planner):
    planner._ttd = np.array([5000.0] + [0.0] * 7)
    present = np.array([1] + [0] * 7)
    assert planner._latest_start(0, 599, present) == 5000 - 602
    assert planner._latest_start(0, 601, present) == 5000 - 603


def test_a_job_with_no_deadline_is_bound_only_by_the_trace(planner):
    planner._ttd = np.array([0.0] * 8)
    present = np.zeros(8, dtype=int)
    assert planner._latest_start(0, 10, present) == planner.T - 11


def test_the_backstop_lead_is_converted_from_seconds_to_steps(planner):
    """600 s is 600 steps only while the timestep is one second."""
    assert planner.timestep_sec == 1.0
    assert planner.backstop_slack_steps == 600


def test_a_coarse_timestep_does_not_inflate_the_lead(planner_cls, monkeypatch, tmp_path):
    """TB12 runs at 600 s per step, where 600 s of slack is a single step."""
    import yaml
    cfg = yaml.safe_load(open(os.environ["EVAL_CONFIG_PATH"]))
    blk = cfg[os.environ["ORACLE_EXPERIMENT"]]
    blk["simulation_timestep"] = 600.0
    path = tmp_path / "coarse.yml"
    path.write_text(yaml.safe_dump({os.environ["ORACLE_EXPERIMENT"]: blk}))
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(path))
    monkeypatch.setenv("ORACLE_WIND_DIR", os.path.join(
        REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"))
    arm = planner_cls(5, 8)
    assert arm.backstop_slack_steps == 1, f"got {arm.backstop_slack_steps} steps"


def test_the_always_defer_arm_never_routes():
    from src.baselines.global_schedulers import AlwaysDeferGlobalScheduler
    arm = AlwaysDeferGlobalScheduler(5, 16)
    actions = arm.schedule({})
    assert actions == [5] * 16, "the diagnostic arm routed something"


def test_a_backstop_slack_mismatch_stops_the_run(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["cloudlet_cpu_utilization_effective"] = "0.5"
    obs["defer_deadline_slack_sec_effective"] = "300.0"
    with pytest.raises(RuntimeError, match="defer_deadline_slack_sec mismatch"):
        planner.schedule(obs)


def test_a_backstop_mode_mismatch_stops_the_run(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["cloudlet_cpu_utilization_effective"] = "0.5"
    obs["defer_deadline_slack_sec_effective"] = "600.0"
    obs["defer_deadline_force_mode_effective"] = "latest_start"
    with pytest.raises(RuntimeError, match="defer_deadline_force_mode mismatch"):
        planner.schedule(obs)


def test_all_three_matching_values_pass(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["cloudlet_cpu_utilization_effective"] = "0.5"
    obs["defer_deadline_slack_sec_effective"] = "600.0"
    obs["defer_deadline_force_mode_effective"] = "legacy"
    planner.schedule(obs)
    assert planner._cpu_util_checked


# ── The horizon gate (Codex 2026-08-31) ──────────────────────────────────────
# oracle144 separates "the predictor is not accurate enough" from "the horizon is too
# short to matter". A perfect forecast truncated to 144 steps, then the frozen causal
# tail every arm shares. If even this cannot beat the blind, no predictor can.

def test_the_horizon_limited_view_is_true_then_tail():
    from src.baselines.global_schedulers import HorizonLimitedOraclePlannerGlobalScheduler
    arm = HorizonLimitedOraclePlannerGlobalScheduler(5, 8)
    arm.t = 100
    view = arm._green_view(0)
    assert np.array_equal(view[100:244], arm.G[0, 100:244]), "the near field is not the truth"
    assert np.all(view[244:] == arm.clim[0]), "the far field is not the shared tail"


def test_the_tail_model_is_shared_and_selectable(monkeypatch):
    from src.baselines.global_schedulers import HorizonLimitedOraclePlannerGlobalScheduler
    arm = HorizonLimitedOraclePlannerGlobalScheduler(5, 8)
    assert arm.tail_model == "climatology"
    monkeypatch.setenv("PLANNER_TAIL_MODEL", "persistence")
    arm2 = HorizonLimitedOraclePlannerGlobalScheduler(5, 8)
    arm2.green_now = np.array([7.0, 0, 0, 0, 0])
    arm2.t = 10
    assert arm2._green_view(0)[10 + 144] == 7.0


def test_the_horizon_length_is_configurable(monkeypatch):
    from src.baselines.global_schedulers import HorizonLimitedOraclePlannerGlobalScheduler
    monkeypatch.setenv("PLANNER_HORIZON_STEPS", "12")
    arm = HorizonLimitedOraclePlannerGlobalScheduler(5, 8)
    arm.t = 0
    view = arm._green_view(0)
    assert np.array_equal(view[:12], arm.G[0, :12])
    assert np.all(view[12:] == arm.clim[0])


def test_the_full_oracle_is_unaffected_by_the_horizon_knob(planner):
    assert planner.info_source == "curve"
    planner.t = 100
    assert np.array_equal(planner._green_view(0), planner.G[0]), "full oracle got truncated"


# ── The weather clock is shared with Java, not guessed (Codex 2026-08-31) ─────
# Java resolves the wind row from the absolute simulation clock. The planner counts steps
# from zero, and at the first decision the clock already stands at the CloudSim start-up
# cost. Under a one-second row that is whole rows of weather, which is what the hard-coded
# `warmup = 13` was patching over; under a 600 second row it is none.

def test_no_hard_coded_warmup_rows_remain(planner):
    import inspect
    from src.baselines import global_schedulers as gs
    src = inspect.getsource(gs.CurveInformedPlannerGlobalScheduler)
    assert "warmup = 13" not in src.replace("`warmup = 13`", ""), \
        "the measured-not-configured constant is back"
    assert planner.weather_warmup_rows == 0


def test_the_grid_is_laid_from_the_observed_clock(planner):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["planner"]["current_clock"] = 13.0
    planner.schedule(obs)
    assert planner._grid_built and planner._clock0 == 13.0
    # One second per row here, so start-up moved the weather by whole rows. That is the
    # defect of this time base; it is recorded rather than hidden.
    assert planner._startup_row_shift == 13
    d = next(i for i in range(5) if planner.G_rows[i].size > 1)
    base = planner.row_base[d]
    # One second per row here, so a clock of 13 means the first step reads row base + 13.
    assert planner.G[d, 0] == planner.G_rows[d][base + 13]


def test_a_six_hundred_second_row_absorbs_the_start_up_cost(planner_cls, monkeypatch, tmp_path):
    import yaml
    cfg = yaml.safe_load(open(os.environ["EVAL_CONFIG_PATH"]))
    blk = cfg[os.environ["ORACLE_EXPERIMENT"]]
    for dc in blk["datacenters"]:
        dc["time_scaling_mode"] = "REAL_TIME"
    path = tmp_path / "phys.yml"
    path.write_text(yaml.safe_dump({os.environ["ORACLE_EXPERIMENT"]: blk}))
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(path))
    monkeypatch.setenv("ORACLE_WIND_DIR", os.path.join(
        REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"))
    arm = planner_cls(5, 8)
    assert arm.row_seconds == 600.0
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["planner"]["current_clock"] = 13.0
    arm.schedule(obs)
    d = next(i for i in range(5) if arm.G_rows[i].size > 1)
    base = arm.row_base[d]
    assert arm.G[d, 0] == arm.G_rows[d][base], "start-up advanced the weather by a row"


def test_rows_change_on_absolute_boundaries_not_from_the_decision(planner_cls, monkeypatch, tmp_path):
    """At clock 350 the current row must end at 600, not 350 + 600."""
    import yaml
    cfg = yaml.safe_load(open(os.environ["EVAL_CONFIG_PATH"]))
    blk = cfg[os.environ["ORACLE_EXPERIMENT"]]
    for dc in blk["datacenters"]:
        dc["time_scaling_mode"] = "REAL_TIME"
    path = tmp_path / "phys2.yml"
    path.write_text(yaml.safe_dump({os.environ["ORACLE_EXPERIMENT"]: blk}))
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(path))
    monkeypatch.setenv("ORACLE_WIND_DIR", os.path.join(
        REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"))
    arm = planner_cls(5, 8)
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["planner"]["current_clock"] = 350.0
    arm.schedule(obs)
    d = next(i for i in range(5) if arm.G_rows[i].size > 1)
    base = arm.row_base[d]
    assert arm.G[d, 249] == arm.G_rows[d][base], "row ended before clock 600"
    assert arm.G[d, 250] == arm.G_rows[d][base + 1], "row did not change at clock 600"
    assert arm.G[d, 849] == arm.G_rows[d][base + 1]
    assert arm.G[d, 850] == arm.G_rows[d][base + 2], "row did not change at clock 1200"


# ── Rows are enumerated, not assumed (Codex 2026-08-31) ──────────────────────
# A 7200 step episode does not touch 7200/600 = 12 wind rows when the clock starts part
# way into a row, and the terminal drain does not touch 20. Both are computed from the
# shared mapping, and the start-up phase is checked rather than trusted.

def _phys_arm(planner_cls, monkeypatch, tmp_path, name="ph.yml"):
    import yaml
    cfg = yaml.safe_load(open(os.environ["EVAL_CONFIG_PATH"]))
    blk = cfg[os.environ["ORACLE_EXPERIMENT"]]
    for dc in blk["datacenters"]:
        dc["time_scaling_mode"] = "REAL_TIME"
    path = tmp_path / name
    path.write_text(yaml.safe_dump({os.environ["ORACLE_EXPERIMENT"]: blk}))
    monkeypatch.setenv("EVAL_CONFIG_PATH", str(path))
    monkeypatch.setenv("ORACLE_WIND_DIR", os.path.join(
        REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified"))
    monkeypatch.setenv("ORACLE_OFFSET_ROWS", "19171")
    return planner_cls(5, 8)


def _step_once(arm, clock):
    obs = batch(8, [])
    obs["dc_available_pes"] = np.array([480.0, 384.0, 296.0, 240.0, 144.0])
    obs["planner"]["current_clock"] = clock
    arm.schedule(obs)
    return arm


def test_the_touched_rows_are_counted_not_assumed(planner_cls, monkeypatch, tmp_path):
    arm = _step_once(_phys_arm(planner_cls, monkeypatch, tmp_path), 13.0)
    m = arm.metrics()
    assert m["planner_rows_7200"], "no row span was recorded"
    d = next(i for i in range(5) if arm.G_rows[i].size > 1)
    first, last, count = arm._row_span[(d, 7200)]
    # clock 13 leaves 587 s of the first row, so the episode reaches into row 12 and
    # touches thirteen rows, not the twelve a naive 7200/600 would give.
    assert count == 13, f"touched {count} rows over 7200 steps"
    assert last - first == 12


def test_the_terminal_drain_span_is_also_counted(planner_cls, monkeypatch, tmp_path):
    arm = _step_once(_phys_arm(planner_cls, monkeypatch, tmp_path, "ph2.yml"), 13.0)
    d = next(i for i in range(5) if arm.G_rows[i].size > 1)
    first, last, count = arm._row_span[(d, 12000)]
    assert count == 21, f"touched {count} rows over 12000 steps"


def test_a_start_up_longer_than_a_row_is_refused(planner_cls, monkeypatch, tmp_path):
    arm = _phys_arm(planner_cls, monkeypatch, tmp_path, "ph3.yml")
    with pytest.raises(RuntimeError, match="registered offset no longer names"):
        _step_once(arm, 601.0)


def test_two_arms_on_the_same_window_share_a_row_signature(planner_cls, monkeypatch, tmp_path):
    from src.baselines.global_schedulers import ClimatologyPlannerGlobalScheduler
    a = _step_once(_phys_arm(planner_cls, monkeypatch, tmp_path, "ph4.yml"), 13.0)
    monkeypatch.setenv("ORACLE_OFFSET_ROWS", "19171")
    b = _step_once(ClimatologyPlannerGlobalScheduler(5, 8), 13.0)
    assert a._rows_signature == b._rows_signature != ""
    assert a._clock0 == b._clock0


def test_the_signature_separates_windows_that_weight_rows_differently(
        planner_cls, monkeypatch, tmp_path):
    """clock 13 and clock 14 visit the same rows but spend different seconds in the first."""
    a = _step_once(_phys_arm(planner_cls, monkeypatch, tmp_path, "sg1.yml"), 13.0)
    b = _step_once(_phys_arm(planner_cls, monkeypatch, tmp_path, "sg2.yml"), 14.0)
    d = next(i for i in range(5) if a.G_rows[i].size > 1)
    assert a._row_span[(d, 12000)][:2] == b._row_span[(d, 12000)][:2], \
        "expected the same first and last row for this comparison to be meaningful"
    assert a._rows_signature != b._rows_signature, \
        "the signature ignored how long each row was actually served"


def test_the_segment_table_accounts_for_every_second(planner_cls, monkeypatch, tmp_path):
    arm = _step_once(_phys_arm(planner_cls, monkeypatch, tmp_path, "sg3.yml"), 13.0)
    d = next(i for i in range(5) if arm.G_rows[i].size > 1)
    assert sum(sec for _row, sec in arm._segments[d]) == 12000
    assert arm._segments[d][0][1] == 587, "the first row should carry 600 - 13 seconds"
