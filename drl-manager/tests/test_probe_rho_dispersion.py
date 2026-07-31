"""Tests for the rho-dispersion probe's result-extraction helpers."""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "probe_rho_dispersion",
    Path(__file__).resolve().parents[1] / "scripts" / "probe_rho_dispersion.py",
)
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["probe_rho_dispersion"] = probe
_SPEC.loader.exec_module(probe)


def _result(**crd):
    stats = {f"crd/{k}": v for k, v in crd.items()}
    return {"learners": {"global_policy": stats}}


def test_extract_from_new_api_layout():
    got = probe.extract_crd_metrics(_result(reweight_w_std=0.31, c_t_mean=0.6))
    assert got["crd/reweight_w_std"] == 0.31


def test_extract_from_legacy_layout():
    r = {"info": {"learner": {"global_policy": {"crd/reweight_w_std": 0.2}}}}
    assert probe.extract_crd_metrics(r)["crd/reweight_w_std"] == 0.2


def test_extract_missing_module_returns_empty():
    assert probe.extract_crd_metrics({"learners": {}}) == {}


def test_extract_ignores_non_numeric_and_non_crd():
    r = {"learners": {"global_policy": {"crd/x": "nan-str", "policy_loss": 1.0,
                                        "crd/rho_routing_std": 0.05}}}
    got = probe.extract_crd_metrics(r)
    assert got == {"crd/rho_routing_std": 0.05}


def test_format_report_orders_headline_first():
    line = probe.format_report({"crd/rho_routing_std": 0.05,
                                "crd/reweight_w_std": 0.31})
    assert line.startswith("RHO_DISPERSION reweight_w_std=0.3100")


def test_format_report_empty():
    assert probe.format_report({}) == "RHO_DISPERSION <empty>"
