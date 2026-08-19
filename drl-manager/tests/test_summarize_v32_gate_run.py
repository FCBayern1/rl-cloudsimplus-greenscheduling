import csv
import json
import sys
from pathlib import Path

DRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL_ROOT))

from summarize_v32_gate_run import (
    load_rollouts,
    summarize_completion,
    summarize_rollouts,
    summarize_training,
)


def test_gate_run_summary_combines_rollout_learner_and_completion(tmp_path):
    counts = [[0] * 6 for _ in range(6)]
    defers = [[0] * 6 for _ in range(6)]
    counts[0][3] = 100; defers[0][3] = 10
    counts[3][3] = 100; defers[3][3] = 70
    payload = {
        "real_slot_count": counts,
        "defer_count": defers,
        "waited_resolution_count": 8,
        "wait_predicted_positive_count": 8,
        "wait_carbon_improved_count": 6,
        "wait_forecast_half_realized_count": 4,
        "forced_route_count": 2,
        "deferred_job_count": 10,
        "pending_deferral_count": 0,
        "job_key_collision_count": 0,
        "mean_wait_duration_sec": 300.0,
        "mean_predicted_gain_at_first_defer": 0.4,
        "mean_observed_gain_at_resolution": 0.3,
    }
    (tmp_path / "v32_rollout_worker1.jsonl").write_text(json.dumps(payload) + "\n")
    rollout = summarize_rollouts(load_rollouts(tmp_path))
    assert rollout["rollout_temporal_delta"] == 0.6
    assert rollout["wait_carbon_improvement_rate"] == 0.75
    assert rollout["backstop_resolution_ratio"] == 0.2
    assert rollout["backstop_dominant"] is False

    headers = [
        "global_v32_td_abs_defer", "global_v32_td_defer_count",
        "global_v32_td_abs_route", "global_v32_td_route_count",
        "global_v32_adv_defer", "global_v32_adv_defer_count",
        "global_v32_adv_route", "global_v32_adv_route_count",
        "global_v32_adv_defer_wait_0_60",
        "global_v32_adv_defer_wait_0_60_count",
    ]
    with (tmp_path / "training_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow({
            "global_v32_td_abs_defer": 3, "global_v32_td_defer_count": 4,
            "global_v32_td_abs_route": 1, "global_v32_td_route_count": 8,
            "global_v32_adv_defer": 0.2, "global_v32_adv_defer_count": 4,
            "global_v32_adv_route": -0.1, "global_v32_adv_route_count": 8,
            "global_v32_adv_defer_wait_0_60": 0.4,
            "global_v32_adv_defer_wait_0_60_count": 2,
        })
    learner = summarize_training(tmp_path)
    assert learner["defer_route_td_residual_ratio"] == 3.0
    assert learner["advantage_by_wait_sec"]["0-60"] == 0.4

    with (tmp_path / "monitor_worker1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["completion_rate_mi"])
        writer.writeheader(); writer.writerow({"completion_rate_mi": 0.999})
    assert summarize_completion(tmp_path)["completion_collapse"] is False
