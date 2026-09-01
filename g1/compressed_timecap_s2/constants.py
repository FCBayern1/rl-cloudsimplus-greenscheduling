"""Frozen constants for the COMPRESSED short-horizon TimeCAP positive control (scheme 2).

Registered by reports/COMPRESSED_TIMECAP_S2_PREREG.md. Nothing in this module may be
changed once the prereg is committed; an amendment must be appended to the prereg with
its own commit and the affected artifacts must be regenerated from scratch.

Identity of this line, restated so it cannot drift:

    accelerated-weather synthetic mechanism positive control

One wind row is one synthetic control epoch. It is not ten minutes, and nothing here may
be described in hours, days, or "24 hour forecast".
"""
from __future__ import annotations

import itertools

# --- data isolation -------------------------------------------------------------------
YEAR_TIMECAP_TRAIN = 2020      # TimeCAP train/validation only; never a scheduler window
YEAR_SCHEDULER_EVAL = 2021     # DISCOVERY + CONFIRMATION only; never TimeCAP training
YEAR_FORBIDDEN = 2022          # two rows of zeros per turbine; excluded everywhere

# --- frozen base ----------------------------------------------------------------------
BASE_CONFIG_REL = "g1/config_C_2020.yml"
BASE_BLOCK = "experiment_g1eval_matchedvan"
BLOCK_PREFIX = "experiment_cts2"
TRACE_PREFIX = "cts2"

# --- weather clock (COMPRESSED) -------------------------------------------------------
# GreenEnergyProvider serves row r at simulation second r under COMPRESSED
# (TimeScalingMode.getTypicalInterval -> 1.0), and the row actually read by DC d at
# simulation clock t is
#
#     row(d, t) = episode_offset + tz_rows[d] + simulation_warmup_rows + floor(t / 1.0)
#
# simulation_warmup_rows is unset in the base block, so it is 0 by the Java default.
ROW_SECONDS = 1.0
SIM_TIMESTEP = 1.0
WARMUP_ROWS = 0
# CloudSim start-up cost: the clock already stands here at the first observation, so the
# first row read is offset + tz + CLOCK0_ROWS rather than offset + tz. Measured on this
# gateway for the C-regime and reused by g1/select_phys_windows.py. It is a small additive
# term and the guard band below absorbs a re-measurement of +/- a few rows, but Stage A
# must still re-verify it before the first carbon run (see prereg section "clock zero").
CLOCK0_SEC = 13.0

# Deterministic episode-offset schedule implemented in
# MultiDatacenterSimulationCore.episodeOffsetFor: offset = (1009 * k) mod range, where k
# is the reset counter driven by evaluate.py --reset-skip.
STRIDE = 1009

# --- TimeCAP task shape (unchanged from the existing checkpoint's task) ----------------
SEQ_LEN = 96
PRED_LEN = 144
# Scheme 2's closure condition: a candidate placement must start AND finish inside the
# forecast, i.e. (s_i - a_i) + r_i <= CLOSURE_ROWS. "Wait <= 144" is not enough.
CLOSURE_ROWS = PRED_LEN

# --- frozen Stage A grid --------------------------------------------------------------
RUNTIME_ROWS = (24, 48, 72)
WAIT_CAP_ROWS = (24, 48, 72, 96, 120)
CONCURRENCY = (1, 3, 5)
N_JOBS = (20, 35, 50)

# --- frozen workload generation parameters --------------------------------------------
BASE_SEED = 20260901
# Runtime is drawn from {ceil(f*r) .. r} so the registered r is a hard upper bound and the
# closure condition holds for every job by construction.
RUNTIME_FLOOR_FRACTION = 0.75
PES_CHOICES = (2, 4)
FILE_SIZE = 300
OUTPUT_SIZE = 150
# Execution physics: runtime_rows = MI / (PES * VM_PE_MIPS * CPU_UTIL) with 1 row = 1 s.
VM_PE_MIPS = 40000.0
CPU_UTIL = 1.0
# Backstop: latest_start fires iff now + MI/(PES*MIPS*util) + slack >= deadline. With
# deadline = arrival + wait_cap + runtime this puts the forced start at
# arrival + wait_cap - slack, i.e. exactly at the registered latest legal start.
DEFER_SLACK_ROWS = 1.0
DEFER_URGENCY_WINDOW_ROWS = 144.0

# --- episode / footprint --------------------------------------------------------------
# Steps appended after the last job can finish, so completion events flush before the
# episode is cut. Carbon keeps accruing during the drain, identically for every arm.
DRAIN_STEPS = 120
# Slack added on top of the analytic footprint so a re-measured CLOCK0_SEC, an off-by-one
# in the row floor, or an extra terminal event cannot push a window past the year end.
GUARD_ROWS = 64

# --- window selection -----------------------------------------------------------------
N_WINDOWS = 6
DISCOVERY_POSITIONS = (0, 2, 4)      # of the six, sorted ascending by offset
CONFIRMATION_POSITIONS = (1, 3, 5)

# --- observation bound ----------------------------------------------------------------
# obs_cloudlet_mi_high must exceed the largest MI in the whole grid or the observation is
# silently clipped. Frozen grid-wide (not per cell) so cells stay comparable.
OBS_MI_MARGIN = 1.25


def admissible_pairs():
    """(runtime, wait_cap) pairs that satisfy the closure condition."""
    return tuple((r, w) for r in RUNTIME_ROWS for w in WAIT_CAP_ROWS
                 if r + w <= CLOSURE_ROWS)


def cells():
    """Every Stage A cell of the frozen grid, in canonical order."""
    return tuple({"runtime_rows": r, "wait_cap_rows": w, "concurrency": c, "n_jobs": n}
                 for (r, w), c, n in itertools.product(
                     admissible_pairs(), CONCURRENCY, N_JOBS))


def cell_key(cell):
    """Canonical short key. Used for the trace name, the block name and the RNG domain."""
    return (f"r{cell['runtime_rows']}w{cell['wait_cap_rows']}"
            f"c{cell['concurrency']}n{cell['n_jobs']}")


def arrival_span_bound(cell):
    """Upper bound on the arrival span, independent of the RNG draw.

    Arrivals are spaced by mean(runtime)/concurrency and mean(runtime) <= runtime_rows,
    so (n-1)*r/c bounds the span for every seed. The window footprint uses this bound so
    window selection never depends on a workload draw.
    """
    import math
    return int(math.ceil((cell["n_jobs"] - 1) * cell["runtime_rows"] / cell["concurrency"]))


def episode_steps_bound(cell):
    """Steps the episode must run for this cell, using the arrival-span bound."""
    return (arrival_span_bound(cell) + cell["wait_cap_rows"]
            + cell["runtime_rows"] + DRAIN_STEPS)


def max_episode_steps():
    return max(episode_steps_bound(c) for c in cells())


def max_job_mi():
    return int(round(max(RUNTIME_ROWS) * max(PES_CHOICES) * VM_PE_MIPS * CPU_UTIL))


def obs_cloudlet_mi_high():
    import math
    return int(math.ceil(max_job_mi() * OBS_MI_MARGIN))
