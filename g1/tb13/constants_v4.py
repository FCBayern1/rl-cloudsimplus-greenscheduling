"""TB13-v4 physical map. One registered configuration, not a scan.

Every number here follows from the SPECpower record for the ASUSTeK RS500A-E10-PS4
(64 cores, idle 51.4 W, peak 214 W) and from the decision to schedule the whole host:

    per site              1 x RS500A
    host capacity         64 PE
    VMs                   2 x 32 PE
    schedulable capacity  64 PE
    job sizes             8 (fluid control), 16, 32 PE
    cloudlet utilisation  1.0, a scenario assumption for compute-bound batch work
    dynamic per PE        (214 - 51.4) / 64 = 2.540625 W

The module refuses to import if the three registered points do not close, so a later edit
to one constant cannot quietly break the curve the power sentinel measured.
"""
from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402

N_DC = ig.N_DC
HOSTS_PER_SITE = 1
HOST_PES = 64
VMS_PER_SITE = 2
PES_PER_VM = 32
CAP_PES_PER_SITE = VMS_PER_SITE * PES_PER_VM          # 64
HOST_IDLE_W = 51.4
HOST_PEAK_W = 214.0
CLOUDLET_UTILISATION = 1.0
DYN_W_PER_PE = (HOST_PEAK_W - HOST_IDLE_W) / HOST_PES * CLOUDLET_UTILISATION
STATIC_W_PER_SITE = HOST_IDLE_W * HOSTS_PER_SITE

PES_PER_JOB = (8, 16, 32)
FLUID_CONTROL_PES = (8,)
MIN_PES_SHARE = ig.MIN_PES_SHARE                      # 0.25, unchanged

# Carried over from v3 without change.
EPOCH_SECONDS = ig.EPOCH_SECONDS
BROWN_FACTORS = ig.BROWN_FACTORS
GREEN_FACTORS = ig.GREEN_FACTORS
INSTALLED_DIVISOR = ig.INSTALLED_DIVISOR
TURBINES_PER_SITE = ig.TURBINES_PER_SITE
BUDGET_FRACTION = ig.BUDGET_FRACTION

REGISTERED_CURVE = {0: 51.4, 32: 132.7, 64: 214.0}


def power_w(busy_pes):
    return HOST_IDLE_W + busy_pes * DYN_W_PER_PE


def _closes():
    return all(abs(power_w(b) - w) < 1e-9 for b, w in REGISTERED_CURVE.items())


if not _closes():
    raise AssertionError(
        f"the v4 power map does not close on {REGISTERED_CURVE}: "
        f"{{b: power_w(b) for b in REGISTERED_CURVE}}")

if CAP_PES_PER_SITE != HOST_PES:
    raise AssertionError("the VM fleet must expose the whole host")

if not all(p <= CAP_PES_PER_SITE for p in PES_PER_JOB):
    raise AssertionError("a job may not be larger than a site")


def grid_hash_v4():
    import hashlib
    payload = repr((HOSTS_PER_SITE, HOST_PES, VMS_PER_SITE, PES_PER_VM, CAP_PES_PER_SITE,
                    HOST_IDLE_W, HOST_PEAK_W, CLOUDLET_UTILISATION, DYN_W_PER_PE,
                    PES_PER_JOB, FLUID_CONTROL_PES, MIN_PES_SHARE, EPOCH_SECONDS,
                    BROWN_FACTORS, GREEN_FACTORS, INSTALLED_DIVISOR, TURBINES_PER_SITE,
                    BUDGET_FRACTION))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
