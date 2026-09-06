"""Fixed-schedule replay arm (ERROR_LADDER_PLANNER_PREREG §2.2): (site, start) -> (site, κ) at the
job's first sighting, late starts and masked pairs counted, never silently clipped."""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.baselines.global_schedulers import ScheduleReplayGlobalScheduler  # noqa: E402

N = 3
GRID = list(range(73))
K = len(GRID)


def _arm(tmp_path, plan):
    p = tmp_path / "sched.json"
    p.write_text(json.dumps({"schedule": {str(k): list(v) for k, v in plan.items()}, "grid": GRID}))
    os.environ["SCHEDULE_JSON"] = str(p)
    try:
        return ScheduleReplayGlobalScheduler(num_datacenters=N, batch_size=3)
    finally:
        os.environ.pop("SCHEDULE_JSON", None)


def _obs(ids, mask=None):
    return {"planner": {"batch_cloudlet_ids": ids},
            **({} if mask is None else {"batch_cloudlet_offset_allowed": mask})}


def test_start_maps_to_kappa_from_the_current_step(tmp_path):
    arm = _arm(tmp_path, {7: (1, 10), 8: (2, 1)})
    out = arm.schedule(_obs([7, 8, -1], mask=np.ones((3, N * K))))     # t = 0
    assert out[0] == 1 * K + GRID.index(9)                                # start 10 = 0 + 9 + lag
    assert out[1] == 2 * K + GRID.index(0)                                # start 1 = now + lag
    assert out[2] == 0
    assert arm.n_late == 0 and arm.n_masked == 0


def test_late_start_and_masked_pair_are_counted_not_clipped_silently(tmp_path):
    arm = _arm(tmp_path, {7: (1, 3), 8: (0, 40), 9: (0, 5)})
    for _ in range(5):                                                    # advance to t = 5
        arm.schedule(_obs([-1, -1, -1]))
    mask = np.ones((3, N * K)); mask[1, 0 * K + GRID.index(34)] = 0.0
    out = arm.schedule(_obs([7, 8, 9], mask=mask))                        # t = 5
    assert out[0] == 1 * K + 0                                            # start 3 already past
    assert out[1] == 0 * K + 0 and arm.n_masked == 1                      # (0, 34) refused by the mask
    assert out[2] == 0 * K + 0                                            # start 5 = 5 - 5 - 1 < 0 -> late too
    assert arm.n_late == 2                                                # jobs 7 and 9


def test_executor_on_another_grid_is_refused_not_misindexed(tmp_path):
    # ladder_v3 k0: the simulator ran the dyadic 12-value grid while the plan used 0..72, so
    # site*K + index pointed at the wrong (site, κ) and 26 holds were "masked"; refuse loudly.
    import pytest
    arm = _arm(tmp_path, {7: (1, 10)})
    with pytest.raises(RuntimeError, match="OFFSET_GRID_DENSE"):
        arm.schedule(_obs([7, -1, -1], mask=np.ones((3, N * 12))))


def test_unplanned_job_is_counted(tmp_path):
    arm = _arm(tmp_path, {1: (0, 2)})
    arm.schedule(_obs([99, -1, -1]))
    assert arm.n_unplanned == 1
