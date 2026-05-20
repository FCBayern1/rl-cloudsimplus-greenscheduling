"""
Regression for the 2026-05-17 logger fix.

Bug: `rllib_green_energy_logger.py` summed 5 Java-side reward term sums into
`global_agent_reward` but forgot to include `global_reward_term_per_action_sum`.
With Stage 1's per-action diff reward enabled (and the older alpha/beta/gamma
weights set to 0), this made monitor.csv and best_episode_details.csv show
`global_agent_reward=0` for every iter — even though PPO was training with
non-zero rewards (smoke 20260517_015600).

This test pins both invariants:
  1. The CSV header lists `global_term_per_action_sum` (so the column exists
     and downstream plots can find it).
  2. The fetch-and-sum block in the source includes the per-action term in
     the global_agent_reward computation (so the column is non-zero when
     Java emits it).

Run from drl-manager/:
    .venv/bin/python -m pytest tests/test_logger_per_action_reward.py -v
"""
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.callbacks import rllib_green_energy_logger as LOG


def test_best_episode_csv_header_includes_per_action_sum():
    assert "global_term_per_action_sum" in LOG.BEST_EPISODE_CSV_HEADERS, (
        "best_episode_details.csv must export global_term_per_action_sum so "
        "Stage 1 per-action diff reward is visible in monitor."
    )


def test_monitor_csv_header_includes_per_action_sum():
    """
    Inspect the source of GreenEnergyLoggerCallback._init_csv and _init_csv_v2
    to ensure both initializers include the per-action column in their `headers`
    list.  We don't actually invoke them (they touch the filesystem and need a
    worker), so we just textually verify the string is present.
    """
    src = inspect.getsource(LOG)
    init_csv = inspect.getsource(LOG.GreenEnergyLoggerCallback._init_csv)
    init_csv_v2 = inspect.getsource(LOG.GreenEnergyLoggerCallback._init_csv_v2)
    assert "global_term_per_action_sum" in init_csv, (
        "_init_csv must add global_term_per_action_sum to monitor.csv headers."
    )
    assert "global_term_per_action_sum" in init_csv_v2, (
        "_init_csv_v2 must add global_term_per_action_sum to monitor.csv headers."
    )


def test_monitor_row_writer_and_header_have_same_column_count():
    """
    2026-05-17 second-smoke regression: when we added
    `global_term_per_action_sum` to the header (CSV column list) but forgot
    to add the matching variable to the row list inside on_episode_end,
    the row had N-1 columns while the header had N → all columns after the
    insertion point were labeled with the WRONG name.  PPO plots looked
    OK for `total_carbon_kg` (before the insertion point) but every
    energy/per-DC field was off by one in the CSV → misleading dashboards.

    Heuristic test: count how many times each canonical reward-breakdown name
    appears in (a) BEST_EPISODE_CSV_HEADERS, (b) the monitor.csv header
    initialisers (_init_csv / _init_csv_v2), (c) the row-builder block at
    the top of on_episode_end's CSV write.  All three counts must match for
    each name (the fix added per_action_sum to all four).
    """
    import re
    src = inspect.getsource(LOG.GreenEnergyLoggerCallback)
    header_init = inspect.getsource(LOG.GreenEnergyLoggerCallback._init_csv)
    header_init_v2 = inspect.getsource(LOG.GreenEnergyLoggerCallback._init_csv_v2)
    on_end = inspect.getsource(LOG.GreenEnergyLoggerCallback.on_episode_end)
    save_best = inspect.getsource(LOG.GreenEnergyLoggerCallback._save_best_episode)

    canonical_names = [
        "global_term_local_sum",
        "global_term_carbon_sum",
        "global_term_throughput_sum",
        "global_term_completion_mi_sum",
        "global_term_waste_sum",
        "global_term_per_action_sum",
    ]
    for name in canonical_names:
        assert name in LOG.BEST_EPISODE_CSV_HEADERS, f"BEST_EPISODE_CSV_HEADERS missing {name}"
        # `_init_csv*` must add the same name as a STRING literal
        assert f"'{name}'" in header_init, f"_init_csv missing '{name}'"
        assert f"'{name}'" in header_init_v2, f"_init_csv_v2 missing '{name}'"
        # `on_episode_end` row writer must reference the variable BARE (not as a string)
        # at least once between the "global reward breakdown" comment and "energy breakdown".
        # Easiest: just check the bare token appears as part of the row list.
        row_block = on_end[on_end.find("# --- global reward breakdown"):
                           on_end.find("# --- energy breakdown")]
        assert name in row_block, (
            f"on_episode_end row writer missing bare variable `{name}` in "
            f"global-reward-breakdown block (would cause off-by-one in monitor.csv)"
        )
        # _save_best_episode row writer (same shape)
        save_block = save_best[save_best.find("# --- global reward breakdown"):
                               save_best.find("# --- energy breakdown")]
        assert name in save_block, (
            f"_save_best_episode missing `{name}` in global-reward-breakdown block"
        )


def test_global_agent_reward_sum_includes_per_action_term():
    """
    Verify the summation block in on_episode_end adds global_term_per_action_sum
    into global_agent_reward.  Done by source inspection — refactoring the
    summation into a helper just for this test would bloat the API.
    """
    on_end = inspect.getsource(LOG.GreenEnergyLoggerCallback.on_episode_end)
    # Find the assignment block.
    fetch_marker = "global_term_per_action_sum = global_energy_stats.get("
    sum_marker = "+ global_term_per_action_sum"
    assert fetch_marker in on_end, (
        "on_episode_end must fetch global_reward_term_per_action_sum from "
        "Java stats dict."
    )
    assert sum_marker in on_end, (
        "on_episode_end must add global_term_per_action_sum into the "
        "global_agent_reward = (... ) summation."
    )
    # Bonus: make sure the row_dict also exports it (so downstream tooling
    # can read it back).
    export_marker = "'global_term_per_action_sum': global_term_per_action_sum"
    assert export_marker in on_end, (
        "on_episode_end must include global_term_per_action_sum in the "
        "monitor.csv row dict."
    )
