"""The energy-carbon decomposition must reproduce the ledger, not approximate it.

Codex 2026-08-30: a carbon difference between two arms cannot be attributed until the
identity C = sum_d (E_green,d*c_green,d + E_brown,d*c_brown,d) is shown to close against
what the simulator reports. These tests pin the arithmetic on a trace whose answer is
known by hand, so a later change to the meter shows up here rather than in a verdict.
"""
import csv
import os
import subprocess
import sys

import pytest

yaml = pytest.importorskip("yaml")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "g1", "energy_carbon_identity.py")


@pytest.fixture
def rig(tmp_path):
    """Two sites, one hour, hand-computable totals.

    DC0 draws 100 W against 60 W of wind, so 60 W green and 40 W brown, and spills
    nothing. DC1 draws 100 W against 160 W of wind, so it is fully green and spills 60 W.
    Over 3600 one-second steps that is exactly 60/40 Wh and 100/0 Wh, with 60 Wh spilled.
    """
    cfg = {
        "exp": {
            "simulation_timestep": 1.0,
            "datacenters": [
                {"datacenter_id": 0, "brown_carbon_factor": 0.5, "green_carbon_factor": 0.01},
                {"datacenter_id": 1, "brown_carbon_factor": 0.2, "green_carbon_factor": 0.01},
            ],
        }
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    trace = tmp_path / "tr.csv"
    cols = ["step", "green_w_dc0", "green_w_dc1", "power_w_dc0", "power_w_dc1"]
    with open(trace, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in range(3600):
            w.writerow([t, 60.0, 160.0, 100.0, 100.0])

    e_green = [60.0, 100.0]
    e_brown = [40.0, 0.0]
    carbon = sum(g * 0.01 + b * f for g, b, f in zip(e_green, e_brown, [0.5, 0.2])) / 1000.0
    totals = tmp_path / "tot.csv"
    with open(totals, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "total_carbon_kg", "green_used_wh", "brown_used_wh", "total_energy_wh",
            "green_waste_wh", "received_dc_0", "received_dc_1"])
        w.writeheader()
        w.writerow({
            "total_carbon_kg": carbon,
            "green_used_wh": sum(e_green),
            "brown_used_wh": sum(e_brown),
            "total_energy_wh": sum(e_green) + sum(e_brown),
            "green_waste_wh": 60.0,
            "received_dc_0": 10, "received_dc_1": 20,
        })
    return cfg_path, trace, totals, carbon


def run(cfg, trace, totals):
    return subprocess.run(
        [sys.executable, SCRIPT, str(trace), str(totals),
         "--config", str(cfg), "--experiment", "exp"],
        capture_output=True, text=True)


def test_identity_closes_on_a_hand_computed_trace(rig):
    cfg, trace, totals, carbon = rig
    out = run(cfg, trace, totals)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("MATCH") >= 4, out.stdout
    assert "MISMATCH" not in out.stdout, out.stdout
    assert f"{carbon:.6f}" in out.stdout


def test_green_is_capped_by_demand_and_the_rest_is_spill(rig):
    """DC1 generates more than it draws, so it must show no brown and 60 Wh spilled."""
    cfg, trace, totals, _ = rig
    out = run(cfg, trace, totals)
    line = [l for l in out.stdout.splitlines() if l.strip().startswith("1 ")][0]
    green, brown, spill = [float(x) for x in line.split()[2:5]]
    assert green == pytest.approx(100.0)
    assert brown == pytest.approx(0.0)
    assert spill == pytest.approx(60.0)


def test_a_broken_ledger_is_reported_not_absorbed(rig):
    """Halving the reported carbon must fail the audit rather than pass quietly."""
    cfg, trace, totals, carbon = rig
    rows = list(csv.DictReader(open(totals)))
    rows[0]["total_carbon_kg"] = carbon / 2.0
    with open(totals, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = run(cfg, trace, totals)
    assert out.returncode == 1
    assert "MISMATCH" in out.stdout
