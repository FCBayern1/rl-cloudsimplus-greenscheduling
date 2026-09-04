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
        if b["perturb_tier"] == "calibrated_shrink_v1":
            assert b["perturb_error_params"] == sd.AUDIT_JSON and os.path.exists(b["perturb_error_params"])
        else:
            assert "perturb_error_params" not in b


def test_judgement_eval_blocks_carry_the_six_unread_offsets_only_there():
    if not os.path.exists(sd.WINDOWS):
        pytest.skip("window preflight artifact missing")
    with tempfile.TemporaryDirectory() as d:
        blocks, man = sd.build_eval(out_dir=d, windows="judgement")
        cert, _ = sd.build_eval(out_dir=d, windows="certified")
    assert man["config"] == "config_stage_d_eval_judgement.yml" and man["blocks"] == 30
    offs = [int(x) for x in man["allowlist"].split(";")]
    assert offs == sd.judgement_offsets() and len(offs) == 6
    for n, b in blocks.items():
        assert b["green_episode_offset_allowlist"] == man["allowlist"]
        assert "green_episode_offset_allowlist" not in cert[n]
        assert set(sd.diff_keys(cert[n], b)) == {"green_episode_offset_allowlist"}


def test_longrun_config_has_400k_and_40k_checkpoints():
    if not os.path.exists(sd.WINDOWS):
        pytest.skip("window preflight artifact missing")
    with tempfile.TemporaryDirectory() as d:
        blocks, man = sd.build(400_000, out_dir=d, trace_dir=d, reward_variant="physical",
                               checkpoint_freq=40_000, out_name="config_stage_d_longrun.yml")
    assert man["config"] == "config_stage_d_longrun.yml" and man["checkpoint_freq_timesteps"] == 40_000
    for b in blocks.values():
        assert b["training"]["total_timesteps"] == 400_000
        assert b["training"]["checkpoint_freq_timesteps"] == 40_000
        assert b["defer_base_cost"] == 0.0


def test_training_checkpoints_every_iteration_kept_and_init_saved(built):
    blocks, _, _ = built
    for b in blocks.values():
        assert b["training"]["checkpoint_freq_timesteps"] == 8000
        assert b["training"]["checkpoint_num_to_keep"] == 0
        assert b["training"]["save_init_checkpoint"] is True


def test_ledger_aligned_is_an_alias_of_the_physical_variant():
    assert sd.REWARD_VARIANTS["ledger_aligned"] == sd.REWARD_VARIANTS["physical"]


def test_manifest_records_hashes_and_windows(built):
    _, man, _ = built
    assert len(man["crd_subtree_sha256"]) == 64 and len(man["config_sha256"]) == 64
    assert man["train_trace"]["file"].endswith("s2_r48_w72_c3_n35_pes32.csv")
    assert len(man["train_windows"]) >= 4 and len(man["eval_windows"]) == 6
    assert json.dumps(man)  # serialisable


def test_cca_lines_are_vanilla_backbone_with_the_hindsight_baseline_on():
    if not os.path.exists(sd.WINDOWS):
        pytest.skip("window preflight artifact missing")
    with tempfile.TemporaryDirectory() as d:
        four, _ = sd.build(400_000, out_dir=d, trace_dir=d, reward_variant="physical",
                           checkpoint_freq=40_000, out_name="config_stage_d_longrun.yml")
        cca, man = sd.build(400_000, out_dir=d, trace_dir=d, reward_variant="physical",
                            checkpoint_freq=40_000, out_name="config_stage_d_cca.yml",
                            lines=sd.CCA_LINES)
    assert man["config"] == "config_stage_d_cca.yml" and set(cca) == {
        "sd_NC_s2_r48_w72_c3_n35", "sd_C_s2_r48_w72_c3_n35"}
    ref = four["sd_V_s2_r48_w72_c3_n35"]
    for n, b in cca.items():
        assert b["crd"]["enabled"] is False           # vanilla backbone, no ensemble
        assert b["crd"]["cca"] == sd.CCA_CFG
        assert b["forecast_mode"] == ("none" if "_NC_" in n else "full")
        assert b["training"]["total_timesteps"] == 400_000
        extra = set(sd.diff_keys(ref, b)) - sd.BETWEEN_LINES
        assert not extra, (n, extra)                   # identical to V apart from identity/hollow/crd
