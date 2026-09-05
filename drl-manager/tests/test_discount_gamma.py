"""P0' discounted-return metric must use the GLOBAL policy's discount (Addendum A2 of
reports/OPTION_ACTION_DESIGN.md): the hierarchical configs keep it under global_model,
while the top-level gamma is the local / legacy value."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.baselines.evaluate import discount_gamma  # noqa: E402


def test_global_model_gamma_wins_over_the_top_level_key():
    cfg = {"gamma": 0.99, "global_model": {"gamma": 0.999, "gae_lambda": 0.98},
           "local_model": {"gamma": 0.99}}
    assert discount_gamma(cfg) == 0.999


def test_flat_config_falls_back_to_the_top_level_key():
    assert discount_gamma({"gamma": 0.97}) == 0.97


def test_missing_everything_uses_the_default():
    assert discount_gamma({}) == 0.99
    assert discount_gamma(None) == 0.99
    assert discount_gamma({"global_model": {"learning_rate": 3e-4}}, default=0.5) == 0.5
