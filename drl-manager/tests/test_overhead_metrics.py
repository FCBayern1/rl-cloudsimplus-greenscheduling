"""Tests for the efficiency-overhead metrics added to the evaluator:
peak memory capture and the decision-latency summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines.evaluate import _capture_memory_mb, _summarize_decision_latency


def test_capture_memory_keys_and_positivity():
    m = _capture_memory_mb()
    assert set(m) == {"peak_cpu_rss_mb", "peak_gpu_mem_mb"}
    assert m["peak_cpu_rss_mb"] > 0.0          # the running process has some RSS
    assert m["peak_gpu_mem_mb"] >= 0.0         # zero on CPU-only, never negative


def test_decision_latency_percentiles_ordered():
    # 1..1000 microseconds worth of samples (given in ns).
    samples_ns = [i * 1000 for i in range(1, 1001)]
    s = _summarize_decision_latency(samples_ns, "global_decision")
    assert s["global_decision_count"] == 1000
    # mean/percentiles are in microseconds and monotone p50 <= p95 <= p99.
    assert s["global_decision_us_p50"] <= s["global_decision_us_p95"] <= s["global_decision_us_p99"]
    assert abs(s["global_decision_us_mean"] - 500.5) < 1.0


def test_decision_latency_empty_is_zero_not_nan():
    s = _summarize_decision_latency([], "local_decision")
    assert s["local_decision_count"] == 0
    assert s["local_decision_us_mean"] == 0.0
    assert s["local_decision_us_p99"] == 0.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
