"""Static sanity checks for the compare_algorithms ALGORITHMS dict.

Catches typos in checkpoint paths, missing experiment fields, and accidental
flag drift before the comparison run is launched (the real run is multi-hour,
expensive to discover misconfig late).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "drl-manager" / "scripts" / "rl" / "compare_algorithms.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "drl-manager"))
    spec = importlib.util.spec_from_file_location("compare_algorithms", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_algorithms_contains_eight_targets():
    mod = _load_module()
    expected = {
        "Round-Robin", "PSO", "GA",
        "PPO_Simple", "PPO_MLP", "PPO_ResMLP", "PPO_gMLP", "PPO_GTrXL",
    }
    assert set(mod.ALGORITHMS.keys()) == expected


def test_each_algo_pins_an_experiment():
    mod = _load_module()
    for name, cfg in mod.ALGORITHMS.items():
        assert "experiment" in cfg, f"{name} missing 'experiment' field"


@pytest.mark.parametrize("name,expected_shared_local,expected_new_api", [
    ("PPO_Simple", False, True),
    ("PPO_MLP",    True,  False),
    ("PPO_ResMLP", True,  True),
    ("PPO_gMLP",   True,  True),
    ("PPO_GTrXL",  True,  True),
])
def test_ppo_variant_flags(name, expected_shared_local, expected_new_api):
    """Flag values come from inspecting each checkpoint's policies/ subdir.
    Per-DC policies => shared_local=False; shared_local_policy => True."""
    mod = _load_module()
    cfg = mod.ALGORITHMS[name]
    assert cfg["shared_local"] is expected_shared_local
    assert cfg["new_api"] is expected_new_api


def test_rllib_checkpoint_paths_exist():
    """Resolve each checkpoint path (relative to drl-manager/) and verify the
    rllib_checkpoint.json file is present — the canonical sentinel for a
    valid Ray checkpoint."""
    mod = _load_module()
    drl_root = REPO_ROOT / "drl-manager"
    for name, cfg in mod.ALGORITHMS.items():
        if cfg["type"] != "rllib":
            continue
        ckpt = (drl_root / cfg["checkpoint"]).resolve()
        sentinel = ckpt / "rllib_checkpoint.json"
        assert sentinel.exists(), f"{name}: checkpoint sentinel missing at {sentinel}"


if __name__ == "__main__":
    test_algorithms_contains_eight_targets()
    test_each_algo_pins_an_experiment()
    for n, sl, na in [("PPO_Simple", False, True), ("PPO_MLP", True, False),
                      ("PPO_ResMLP", True, True), ("PPO_gMLP", True, True),
                      ("PPO_GTrXL", True, True)]:
        test_ppo_variant_flags(n, sl, na)
    test_rllib_checkpoint_paths_exist()
    print("OK")
