"""Gate-0 dry-run: CLI-resolved seed reaches Tune's result config."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.train_rlmodule_gtrxl import create_rlmodule_config, load_config

CONFIG_YML = REPO_ROOT.parent / "config.yml"
EXPERIMENT = "experiment_multi_5dc_carbon_v2"


def test_cli_seed_is_serialized_in_tuner_param_space_without_training():
    exp = load_config(str(CONFIG_YML))[EXPERIMENT]
    cli_seed = 271828
    config = create_rlmodule_config(
        exp,
        exp.get("global_model", {}),
        exp.get("local_model", {}),
        exp.get("training", {}),
        seed=cli_seed,
    )

    # train_rlmodule_gtrxl passes this exact dict to tune.Tuner(param_space=...).
    # Tune persists param_space under result.json["config"], so this is the
    # requested no-training/mocked equivalent of result.json.config.seed.
    assert config.seed == cli_seed
    assert config.to_dict()["seed"] == cli_seed


def test_omitted_seed_preserves_legacy_unspecified_behavior():
    exp = load_config(str(CONFIG_YML))[EXPERIMENT]
    config = create_rlmodule_config(
        exp,
        exp.get("global_model", {}),
        exp.get("local_model", {}),
        exp.get("training", {}),
    )
    assert config.to_dict()["seed"] is None
