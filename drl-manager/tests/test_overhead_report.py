"""Tests for the scheduling-overhead aggregation used by the efficiency table."""

import csv
from pathlib import Path

import pytest

from src.baselines.overhead_report import (
    OVERHEAD_FIELDS,
    aggregate_seeds,
    build_report,
    format_latex,
    format_markdown,
    main,
    parse_arm_spec,
    read_overhead_csv,
)


def _write_csv(path: Path, rows, extra_cols=None):
    cols = list(OVERHEAD_FIELDS) + list(extra_cols or [])
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _row(latency=100.0, wall=10.0, rss=500.0, gpu=0.0):
    return {
        "global_decision_us_mean": latency,
        "global_decision_us_p95": latency * 1.5,
        "global_decision_us_p99": latency * 2.0,
        "local_decision_us_mean": latency / 4,
        "local_decision_us_p95": latency / 3,
        "local_decision_us_p99": latency / 2,
        "episode_wall_s": wall,
        "peak_cpu_rss_mb": rss,
        "peak_gpu_mem_mb": gpu,
    }


def test_read_single_episode(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [_row(latency=120.0, wall=8.0)])
    got = read_overhead_csv(p)
    assert got["global_decision_us_mean"] == pytest.approx(120.0)
    assert got["episode_wall_s"] == pytest.approx(8.0)


def test_multiple_episodes_are_averaged(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [_row(latency=100.0, wall=10.0), _row(latency=200.0, wall=20.0)])
    got = read_overhead_csv(p)
    assert got["global_decision_us_mean"] == pytest.approx(150.0)
    assert got["episode_wall_s"] == pytest.approx(15.0)


def test_csv_without_overhead_columns_returns_none(tmp_path):
    p = tmp_path / "old.csv"
    with open(p, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["total_carbon_kg"])
        writer.writeheader()
        writer.writerow({"total_carbon_kg": "0.18"})
    assert read_overhead_csv(p) is None


def test_missing_file_returns_none(tmp_path):
    assert read_overhead_csv(tmp_path / "nope.csv") is None


def test_blank_and_question_mark_values_are_skipped(tmp_path):
    p = tmp_path / "partial.csv"
    row = _row(latency=100.0)
    row["peak_gpu_mem_mb"] = "?"
    row["episode_wall_s"] = ""
    _write_csv(p, [row])
    got = read_overhead_csv(p)
    assert "peak_gpu_mem_mb" not in got
    assert "episode_wall_s" not in got
    assert got["global_decision_us_mean"] == pytest.approx(100.0)


def test_seeds_reduced_by_median():
    rows = [_row(latency=100.0), _row(latency=300.0), _row(latency=200.0)]
    agg = aggregate_seeds(rows)
    assert agg["global_decision_us_mean"] == pytest.approx(200.0)
    assert agg["n_seeds"] == 3


def test_ratio_columns_against_baseline(tmp_path):
    base = tmp_path / "rr_s1.csv"
    arm = tmp_path / "rl_s1.csv"
    _write_csv(base, [_row(latency=50.0, wall=10.0)])
    _write_csv(arm, [_row(latency=200.0, wall=25.0)])
    report = build_report({"RR": [str(base)], "RL": [str(arm)]}, baseline="RR")
    rl = next(e for e in report if e["arm"] == "RL")
    rr = next(e for e in report if e["arm"] == "RR")
    assert rl["latency_vs_baseline"] == pytest.approx(4.0)
    assert rl["wall_vs_baseline"] == pytest.approx(2.5)
    assert rr["latency_vs_baseline"] == pytest.approx(1.0)


def test_no_baseline_means_no_ratio_columns(tmp_path):
    p = tmp_path / "a_s1.csv"
    _write_csv(p, [_row()])
    report = build_report({"A": [str(p)]}, baseline=None)
    assert "latency_vs_baseline" not in report[0]


def test_unknown_baseline_label_is_tolerated(tmp_path):
    p = tmp_path / "a_s1.csv"
    _write_csv(p, [_row()])
    report = build_report({"A": [str(p)]}, baseline="does-not-exist")
    assert len(report) == 1
    assert "latency_vs_baseline" not in report[0]


def test_arms_without_usable_csv_are_dropped(tmp_path):
    good = tmp_path / "good.csv"
    _write_csv(good, [_row()])
    report = build_report({"Good": [str(good)], "Empty": []})
    assert [e["arm"] for e in report] == ["Good"]


def test_parse_arm_spec_expands_glob(tmp_path):
    for i in (1, 2):
        _write_csv(tmp_path / f"kn_s{i}.csv", [_row()])
    label, paths = parse_arm_spec(f"EU-CRD={tmp_path}/kn_s*.csv")
    assert label == "EU-CRD"
    assert len(paths) == 2


def test_parse_arm_spec_rejects_missing_equals():
    with pytest.raises(Exception):
        parse_arm_spec("no-equals-sign")


def test_markdown_and_latex_render_all_arms(tmp_path):
    base = tmp_path / "rr.csv"
    arm = tmp_path / "rl.csv"
    _write_csv(base, [_row(latency=50.0)])
    _write_csv(arm, [_row(latency=200.0)])
    report = build_report({"RR": [str(base)], "EU_CRD": [str(arm)]}, baseline="RR")
    md = format_markdown(report)
    assert "RR" in md and "EU_CRD" in md and "4.00x" in md
    tex = format_latex(report)
    assert r"\begin{tabular}" in tex and r"\bottomrule" in tex
    # LaTeX must escape underscores in arm labels.
    assert r"EU\_CRD" in tex


def test_cli_writes_output_file(tmp_path):
    csv_path = tmp_path / "van_s1.csv"
    _write_csv(csv_path, [_row()])
    out = tmp_path / "table.tex"
    rc = main([
        "--arm", f"Vanilla={tmp_path}/van_s*.csv",
        "--format", "latex",
        "--output", str(out),
    ])
    assert rc == 0
    assert r"\begin{tabular}" in out.read_text()


def test_cli_returns_error_when_nothing_usable(tmp_path):
    rc = main(["--arm", f"Empty={tmp_path}/missing_*.csv"])
    assert rc == 1
