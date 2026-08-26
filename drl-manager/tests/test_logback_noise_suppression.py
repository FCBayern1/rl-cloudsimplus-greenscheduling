"""The gateway's logback config must keep the per-step simulator chatter off disk.

A 7200-step multi-DC episode used to emit 2.2M log lines / ~250 MB per gateway log,
96.6% of it from three loggers, written twice (CONSOLE + FILE) synchronously on the
simulation thread. Four scenarios filled ~1.8 GB in one evening.

These are config assertions, not behaviour assertions: verifying that logback
actually applies the levels needs a JVM, which is done once by hand (see
``docs``/commit message) rather than on every pytest run. What is checked here is
that the config keeps saying what we think it says, and — the failure mode that
already bit us once with the turbine CSVs — that the copy on the JVM's runtime
classpath has not drifted from the source.

Run from repo root:
    cd drl-manager && python -m pytest tests/test_logback_noise_suppression.py -v
"""
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

GATEWAY = REPO_ROOT.parent / "cloudsimplus-gateway"
SRC_LOGBACK = GATEWAY / "src" / "main" / "resources" / "logback.xml"
BUILD_LOGBACK = GATEWAY / "build" / "resources" / "main" / "logback.xml"

# logger -> the level at which its flood is suppressed. The flood is INFO for
# DatacenterBroker and WARN for the two placement loggers, so the configured
# level must sit strictly above it.
NOISY = {
    "DatacenterBroker": "WARN",
    "VmAllocationPolicy": "ERROR",
    "VmAllocationPolicyCustom": "ERROR",
}

pytestmark = pytest.mark.skipif(
    not SRC_LOGBACK.exists(), reason="cloudsimplus-gateway/logback.xml not present"
)


def loggers(path):
    root = ElementTree.parse(path).getroot()
    return {el.get("name"): el.get("level") for el in root.findall("logger")}


@pytest.mark.parametrize("name, level", sorted(NOISY.items()))
def test_noisy_logger_is_capped(name, level):
    configured = loggers(SRC_LOGBACK).get(name)
    assert configured is not None, f"{name} has no <logger> entry — the flood is back"
    assert configured.endswith(f":-{level}}}") or configured == level, (
        f"{name} is at {configured!r}, expected default {level!r}"
    )


def test_root_stays_at_info():
    """Only the three chatty loggers are capped; everything else keeps logging."""
    root = ElementTree.parse(SRC_LOGBACK).getroot().find("root")
    assert root.get("level") == "INFO"


def test_caps_are_overridable_for_debugging():
    """Each cap must be a ${prop:-DEFAULT} so a run can restore the firehose."""
    for name, level in loggers(SRC_LOGBACK).items():
        if name in NOISY:
            assert level.startswith("${") and ":-" in level, (
                f"{name} is hardcoded to {level!r}; make it a system property so "
                f"the full log can be restored without editing the config"
            )


def test_rolling_policy_has_a_total_size_cap():
    """maxHistory only prunes within one run's dated dir — it never bounds disk."""
    root = ElementTree.parse(SRC_LOGBACK).getroot()
    policy = root.find(".//rollingPolicy")
    assert policy is not None
    cap = policy.find("totalSizeCap")
    assert cap is not None and cap.text.strip(), "no totalSizeCap: disk is unbounded"
    assert policy.find("maxHistory") is not None, "totalSizeCap needs maxHistory to apply"


def test_no_logger_silences_the_turbine_fallback_warning():
    """GreenEnergyProvider's 'Could not find CSV for turbine' is how a silently
    zero-green datacenter is detected — it must never be suppressed."""
    assert "GreenEnergyProvider" not in loggers(SRC_LOGBACK)


@pytest.mark.skipif(not BUILD_LOGBACK.exists(), reason="gateway not built")
def test_runtime_classpath_copy_matches_source():
    """The JVM runs with build/resources/main on the classpath, not the jar, so a
    stale copy there means the running config is not the one in git."""
    assert BUILD_LOGBACK.read_text() == SRC_LOGBACK.read_text(), (
        "build/resources/main/logback.xml has drifted from src/main/resources — "
        "run ./gradlew processResources (or cp) before trusting a run's logging"
    )
