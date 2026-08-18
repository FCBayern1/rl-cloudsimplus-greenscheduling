"""Official preflight entry test (Codex step: exercise MAIN, not the helper)."""
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]


def run_preflight(*args):
    return subprocess.run(
        [str(D / ".venv/bin/python"), str(D / "preflight_scenario.py"), *args],
        capture_output=True, text=True, cwd=str(D))


class TestOfficialSqt2Preflight:
    def test_sqt2_pair_full_cert_green(self):
        r = run_preflight("experiment_sqt2_oracle", "experiment_sqt2_noforecast",
                          "--v32-cert", "--p02-cert")
        assert "AUDIT PASSED" in r.stdout, r.stdout[-2000:]
        # the v2 protocol must actually run in the official entry
        for line in ("sqt2v2: decision exposure (MAIN)",
                     "sqt2v2: P(worthy|tight) in band",
                     "sqt2v2: distinct trough coverage",
                     "sqt2: latest-start backstop on"):
            assert f"[PASS] {line}" in r.stdout, line
        assert "[ N/A ] job longer than a peak" in r.stdout
        assert "slack_p50" not in r.stdout      # mixed-p50 check deleted

    def test_legacy_pair_keeps_geometry_checks(self):
        r = run_preflight("experiment_v3_2_oracle", "experiment_v3_2_noforecast",
                          "--v32-cert")
        assert "job longer than a peak" in r.stdout
        assert "N/A ] job longer" not in r.stdout
        assert "sqt2v2" not in r.stdout
