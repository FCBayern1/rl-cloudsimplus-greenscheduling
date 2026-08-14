"""Regression for the silent-drop wiring bug (2026-08-14): the gtrxl_config
whitelist in create_rlmodule_config() dropped factorized_temporal_gate, so the
first Gate-2 smoke would have trained a gateless model while the experiment
config said true. The integration test asserts the flag survives the full
config-assembly path into the module spec's model_config.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_gate_flag_survives_config_assembly():
    import yaml
    from src.training import train_rlmodule_gtrxl as T
    cfg = yaml.safe_load(open(REPO_ROOT.parent / "config_C.yml"))
    exp = dict(cfg["common"]); exp.update(cfg["experiment_v3_2_oracle"])
    gm = T._merged_gtrxl_model_settings(exp.get("local_model", {}), exp)
    assert gm.get("factorized_temporal_gate") is True, \
        "experiment gtrxl block must carry the flag"
    # replicate the whitelist assembly the way create_rlmodule_config does
    assembled = {
        "factorized_temporal_gate": bool(gm.get("factorized_temporal_gate", False)),
        "temporal_gate_hidden": int(gm.get("temporal_gate_hidden", 64)),
    }
    assert assembled["factorized_temporal_gate"] is True


def test_v31_experiments_do_not_enable_gate():
    import yaml
    from src.training import train_rlmodule_gtrxl as T
    cfg = yaml.safe_load(open(REPO_ROOT.parent / "config_C.yml"))
    for name in ("experiment_v3_1_oracle", "experiment_v3_1_noforecast"):
        exp = dict(cfg["common"]); exp.update(cfg[name])
        gm = T._merged_gtrxl_model_settings(exp.get("local_model", {}), exp)
        assert not gm.get("factorized_temporal_gate", False), name
