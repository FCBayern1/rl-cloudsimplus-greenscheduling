"""Green-DC derivation replaces the hardcoded GREEN_DCS = [0, 1, 2] in the oracles.

The bug this locks down: with the 8-DC sweep topology the green sites are
0, 1, 2 and *5* (DC_Nordic2), and 6/7 are extra brown sites. A hardcoded
[0, 1, 2] silently drops DC5 from routing and from the "is it windy now" sum,
so the oracle would grade a topology it wasn't actually exercising.

Run from repo root:
    cd drl-manager && python -m pytest tests/test_green_dcs.py -v
"""
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.green_dcs import (  # noqa: E402
    describe_green_dcs,
    green_capable_dcs,
    green_dcs_from_env,
)

CONFIG_C = REPO_ROOT.parent / "config_C.yml"


def dc(name, turbines, enabled=True):
    return {"name": name, "turbine_ids": turbines, "green_energy_enabled": enabled}


# --- the regression the change exists for -----------------------------------

def test_eight_dc_topology_includes_the_non_contiguous_green_site():
    """DC5 owns turbines; a hardcoded [0, 1, 2] would have missed it."""
    dcs = [
        dc("DC_Nordic", [7012, 7036]),
        dc("DC_Germany", [7095, 7091]),
        dc("DC_US_East", [7096]),
        dc("DC_US_West", [], enabled=False),
        dc("DC_APAC", [], enabled=False),
        dc("DC_Nordic2", [7101, 7103]),
        dc("DC_EU_South", [], enabled=False),
        dc("DC_SEA", [], enabled=False),
    ]
    assert green_capable_dcs(dcs) == [0, 1, 2, 5]
    assert green_capable_dcs(dcs) != [0, 1, 2]


def test_five_dc_topology_still_yields_the_old_constant():
    """The 5-DC scenarios must be unchanged by the fix — same list as before."""
    dcs = [
        dc("DC_Nordic", [7012, 7036]),
        dc("DC_Germany", [7095, 7091]),
        dc("DC_US_East", [7096]),
        dc("DC_US_West", [], enabled=False),
        dc("DC_APAC", [], enabled=False),
    ]
    assert green_capable_dcs(dcs) == [0, 1, 2]


@pytest.mark.skipif(not CONFIG_C.exists(), reason="config_C.yml not present")
@pytest.mark.parametrize(
    "experiment, expected",
    [
        ("experiment_sweep_dc8", [0, 1, 2, 5]),
        ("experiment_sweep_rwv3l", [0, 1, 2]),
        ("experiment_sweep_rwv3m", [0, 1, 2]),
        ("experiment_sweep_scarce", [0, 1, 2]),
        ("experiment_sweep_offset", [0, 1, 2]),
    ],
)
def test_against_the_real_sweep_configs(experiment, expected):
    """Derivation matches the actual scenarios the sweep runs."""
    cfg = yaml.safe_load(CONFIG_C.read_text())
    if experiment not in cfg:
        pytest.skip(f"{experiment} not in config_C.yml")
    assert green_capable_dcs(cfg[experiment].get("datacenters", [])) == expected


@pytest.mark.skipif(not CONFIG_C.exists(), reason="config_C.yml not present")
def test_dc8_green_dcs_match_the_green_timeseries_columns():
    """The derived green DCs must be exactly the non-zero columns of the 8-DC
    green series — otherwise the oracle's forecast and its routing disagree."""
    np = pytest.importorskip("numpy")
    ts_path = REPO_ROOT / "data" / "green_stretch_8dc.npy"
    if not ts_path.exists():
        pytest.skip("green_stretch_8dc.npy not present")
    cfg = yaml.safe_load(CONFIG_C.read_text())
    if "experiment_sweep_dc8" not in cfg:
        pytest.skip("experiment_sweep_dc8 not in config_C.yml")

    ts = np.load(ts_path)
    nonzero_cols = sorted(int(c) for c in np.flatnonzero(ts.max(axis=0) > 0))
    derived = green_capable_dcs(cfg["experiment_sweep_dc8"].get("datacenters", []))
    assert derived == nonzero_cols, (
        f"config says green DCs are {derived} but the green series has power in "
        f"columns {nonzero_cols}"
    )


# --- predicate details -------------------------------------------------------

def test_turbines_present_but_green_energy_disabled_is_not_green():
    dcs = [dc("A", [1, 2], enabled=False), dc("B", [3])]
    assert green_capable_dcs(dcs) == [1]


def test_green_energy_enabled_defaults_to_true_when_absent():
    dcs = [{"name": "A", "turbine_ids": [1]}, {"name": "B", "turbine_ids": []}]
    assert green_capable_dcs(dcs) == [0]


def test_empty_turbine_list_and_missing_key_are_both_brown():
    dcs = [dc("A", []), {"name": "B"}, dc("C", [7])]
    assert green_capable_dcs(dcs) == [2]


def test_num_dc_truncates_indices_beyond_the_running_env():
    """A config may list more DCs than the env instantiated."""
    dcs = [dc("A", [1]), dc("B", []), dc("C", [2]), dc("D", [3])]
    assert green_capable_dcs(dcs) == [0, 2, 3]
    assert green_capable_dcs(dcs, num_dc=3) == [0, 2]


def test_non_mapping_entries_are_ignored_not_crashed_on():
    dcs = [dc("A", [1]), None, "garbage", dc("B", [2])]
    assert green_capable_dcs(dcs) == [0, 3]


# --- the fallback that keeps callers from dividing by zero -------------------

def test_all_brown_topology_falls_back_to_every_dc():
    """roundrobin_routing does `% len(green_dcs)`; an empty list would raise."""
    dcs = [dc("A", []), dc("B", []), dc("C", [])]
    assert green_capable_dcs(dcs) == [0, 1, 2]


def test_missing_datacenters_key_falls_back_to_num_dc():
    assert green_capable_dcs(None, num_dc=4) == [0, 1, 2, 3]
    assert green_capable_dcs([], num_dc=2) == [0, 1]


def test_fallback_is_logged_as_a_warning(caplog):
    with caplog.at_level("WARNING"):
        green_capable_dcs([dc("A", [])], num_dc=1)
    assert "No green-capable DC" in caplog.text


def test_result_is_never_empty_so_modulo_is_always_safe():
    for dcs, n in ([], 3), ([dc("A", [])], 1), (None, 1):
        got = green_capable_dcs(dcs, num_dc=n)
        assert got, "empty green list would break `% len(green_dcs)` in the oracles"
        assert 0 % len(got) == 0


# --- env adapter -------------------------------------------------------------

class FakeEnv:
    def __init__(self, dc_configs, num_datacenters):
        self.dc_configs = dc_configs
        self.num_datacenters = num_datacenters


def test_green_dcs_from_env_reads_the_envs_cached_configs():
    env = FakeEnv([dc("A", [1]), dc("B", []), dc("C", [2])], 3)
    assert green_dcs_from_env(env) == [0, 2]


def test_green_dcs_from_env_on_a_single_dc_env_without_configs():
    env = FakeEnv(None, 1)
    assert green_dcs_from_env(env) == [0]


# --- banner ------------------------------------------------------------------

def test_describe_green_dcs_names_the_sites():
    dcs = [dc("DC_Nordic", [1]), dc("DC_Brown", []), dc("DC_Nordic2", [2])]
    assert describe_green_dcs([0, 2], dcs) == "dc0(DC_Nordic) dc2(DC_Nordic2)"


def test_describe_green_dcs_without_config_or_names():
    assert describe_green_dcs([0, 3]) == "dc0 dc3"
    assert describe_green_dcs([]) == "(none)"
