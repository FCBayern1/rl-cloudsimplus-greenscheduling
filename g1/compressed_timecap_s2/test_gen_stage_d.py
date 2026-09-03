import json
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("yaml")
import gen_stage_d as sd  # noqa: E402


@pytest.fixture(scope="module")
def built():
    if not os.path.exists(sd.WINDOWS):
        pytest.skip("window preflight artifact missing")
    with tempfile.TemporaryDirectory() as d:
        blocks, man = sd.build(50_000, out_dir=d, trace_dir=d)
        cfg = yaml.safe_load(open(os.path.join(d, "config_stage_d.yml")))
        yield blocks, man, cfg


def test_four_lines_with_the_registered_identities(built):
    blocks, man, cfg = built
    names = sorted(blocks)
    assert names == sorted(f"sd_{l}_s2_r48_w72_c3_n35" for l in ("NV", "V", "NE", "E"))
    for n, b in blocks.items():
        line = n.split("_")[1]
        assert b["crd"]["enabled"] is (line in ("NE", "E"))
        assert b["forecast_mode"] == ("none" if line in ("NV", "NE") else "full")
        assert b["green_oracle_mode"] == "perturbed_godeye" and b["perturb_tier"] == "godeye"
        assert b["training"]["total_timesteps"] == 50_000


def test_diff_against_the_hz_block_is_whitelisted(built):
    blocks, _, _ = built
    hz = yaml.safe_load(open(sd.HZ_CONFIG))[sd.HZ_CELL]
    for n, b in blocks.items():
        extra = set(sd.diff_keys(hz, b)) - sd.WHITELIST
        assert not extra, (n, extra)
        # physics untouched
        assert b["datacenters"] == hz["datacenters"]
        assert b["max_cloudlet_pes"] == 32 and b["split_large_cloudlets"] is False
        assert b["compressed_power_divisor"] == 3000.0


def test_lines_differ_only_in_identity_hollow_and_crd(built):
    blocks, _, _ = built
    ref = blocks["sd_V_s2_r48_w72_c3_n35"]
    for n, b in blocks.items():
        extra = set(sd.diff_keys(ref, b)) - sd.BETWEEN_LINES
        assert not extra, (n, extra)
        assert {k: v for k, v in b["crd"].items() if k != "enabled"} == \
               {k: v for k, v in ref["crd"].items() if k != "enabled"}


def test_physical_variant_changes_only_the_three_reward_keys():
    if not os.path.exists(sd.WINDOWS):
        pytest.skip("window preflight artifact missing")
    with tempfile.TemporaryDirectory() as d:
        legacy, _ = sd.build(50_000, out_dir=d, trace_dir=d)
        phys, man = sd.build(50_000, out_dir=d, trace_dir=d, reward_variant="physical")
        assert man["config"] == "config_stage_d_physical.yml"
        assert os.path.exists(os.path.join(d, "stage_d_manifest_physical.json"))
    for n in legacy:
        extra = set(sd.diff_keys(legacy[n], phys[n])) - sd.REWARD_KEYS
        assert not extra, (n, extra)
        assert phys[n]["defer_base_cost"] == 0.0 and phys[n]["defer_urgency_weight"] == 0.0
        assert phys[n]["per_action_carbon_weight"] == 0.0
        assert phys[n]["global_reward_beta"] == legacy[n]["global_reward_beta"] == 1.0


def test_eval_blocks_cover_cells_tiers_and_hollow_with_whitelisted_diff():
    with tempfile.TemporaryDirectory() as d:
        blocks, man = sd.build_eval(out_dir=d)
    assert man["blocks"] == len(blocks) == 6 * (4 + 1)
    hz = yaml.safe_load(open(sd.HZ_CONFIG))
    for n, b in blocks.items():
        cell = "_".join(n.split("_")[1:6])
        extra = set(sd.diff_keys(hz[cell], b)) - sd.EVAL_WHITELIST
        assert not extra, (n, extra)
        assert "green_episode_offset_allowlist" not in b
        assert b["defer_base_cost"] == 0.0 and b["per_action_carbon_weight"] == 0.0
        if n.endswith("_hollow"):
            assert b["forecast_mode"] == "none" and b["perturb_tier"] == "godeye"
        else:
            assert b["forecast_mode"] == "full" and b["perturb_tier"] in sd.EVAL_TIERS


def test_training_checkpoints_every_iteration(built):
    blocks, _, _ = built
    for b in blocks.values():
        assert b["training"]["checkpoint_freq_timesteps"] == 8000


def test_manifest_records_hashes_and_windows(built):
    _, man, _ = built
    assert len(man["crd_subtree_sha256"]) == 64 and len(man["config_sha256"]) == 64
    assert man["train_trace"]["file"].endswith("s2_r48_w72_c3_n35_pes32.csv")
    assert len(man["train_windows"]) >= 4 and len(man["eval_windows"]) == 6
    assert json.dumps(man)  # serialisable
