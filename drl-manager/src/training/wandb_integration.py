"""
Weights & Biases integration for the GTrXL multi-DC trainer.

Two public entry points:

* :func:`build_wandb_callbacks` — returns a list of Ray Tune
  ``LoggerCallback`` instances to attach to ``RunConfig``.  Empty list
  when wandb is disabled or the ``wandb`` package is missing.

* :func:`upload_run_artifacts` — called once after ``tuner.fit()``
  finishes.  Opens the same wandb run (by trial id, via
  ``resume="allow"``) and attaches files under the trial directory as a
  single artifact.  Safe no-op when wandb is disabled.

Configuration is read from ``env_config['wandb']`` (see
``config.yml::experiment_multi_5dc_carbon_v2.wandb``).  CLI overrides
in the entrypoint mutate this dict before training starts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_WANDB_CFG: Dict[str, Any] = {
    "enabled": False,
    "project": "rl-cloudsimplus-greenscheduling",
    "entity": "",
    "group": "",
    "job_type": "train",
    "tags": [],
    "notes": "",
    "mode": "online",
    "log_artifacts": True,
    "api_key_file": "",
}


def get_wandb_config(env_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user ``wandb:`` block on top of defaults."""
    user = (env_config or {}).get("wandb") or {}
    merged = dict(_DEFAULT_WANDB_CFG)
    if isinstance(user, dict):
        merged.update({k: v for k, v in user.items() if v is not None})
    return merged


def _has_wandb_package() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def build_wandb_callbacks(
    env_config: Dict[str, Any],
    *,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> List[Any]:
    """Return a list with a single ``WandbLoggerCallback`` when enabled.

    Empty list when:
      * ``env_config['wandb']['enabled']`` is false,
      * wandb python package isn't installed (logs a warning),
      * mode is ``"disabled"``.
    """
    cfg = get_wandb_config(env_config)
    if not cfg.get("enabled", False):
        return []
    if str(cfg.get("mode", "online")).lower() == "disabled":
        logger.info("wandb mode=disabled — skipping callback")
        return []
    if not _has_wandb_package():
        logger.warning(
            "wandb.enabled=true but the 'wandb' package is not installed in "
            "the current python env. Install with `pip install wandb` or set "
            "wandb.enabled=false. Continuing without wandb."
        )
        return []

    from ray.air.integrations.wandb import WandbLoggerCallback

    init_kwargs: Dict[str, Any] = {
        "job_type": cfg.get("job_type") or "train",
        "tags": list(cfg.get("tags") or []) + [experiment_name],
        "notes": cfg.get("notes") or "",
        "mode": cfg.get("mode") or "online",
    }
    if cfg.get("entity"):
        init_kwargs["entity"] = cfg["entity"]
    if run_name:
        init_kwargs["name"] = run_name

    callback_kwargs: Dict[str, Any] = {
        "project": cfg["project"],
        "log_config": True,           # so wandb.config holds the full PPO config
        "upload_checkpoints": False,  # checkpoints can be huge; opt-in only
    }
    if cfg.get("group"):
        callback_kwargs["group"] = cfg["group"]
    if cfg.get("api_key_file"):
        callback_kwargs["api_key_file"] = os.path.expanduser(cfg["api_key_file"])

    logger.info(
        "wandb enabled — project=%s entity=%s tags=%s",
        callback_kwargs["project"],
        cfg.get("entity") or "(default)",
        init_kwargs["tags"],
    )
    return [WandbLoggerCallback(**callback_kwargs, **init_kwargs)]


def _find_trial_id(output_dir: Path) -> Optional[str]:
    """
    Ray Tune writes each trial as a sub-directory under
    ``<storage_path>/multidc_gtrxl_training/``.  The directory name
    ends with the trial id (the same id WandbLoggerCallback used as
    wandb's run id), e.g. ``PPO_multidc_env_abc12_00000``.
    """
    trial_root = output_dir / "multidc_gtrxl_training"
    if not trial_root.is_dir():
        return None
    candidates = [p for p in trial_root.iterdir() if p.is_dir() and p.name.startswith("PPO_")]
    if not candidates:
        return None
    # If multiple trials existed, pick the most recently modified one
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    name = candidates[0].name
    # Trial id is the last underscore-separated chunk in the directory name.
    return name.split("_")[-2] if "_" in name else None


def upload_run_artifacts(
    env_config: Dict[str, Any],
    output_dir: str,
    *,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> None:
    """Attach training output files as a wandb artifact.

    Resumes the same run that ``WandbLoggerCallback`` opened for the
    trial (via ``resume="allow"`` + matching id), so the artifact
    appears under the per-iteration metrics rather than as a new run.
    Silent no-op when wandb is disabled or missing.
    """
    cfg = get_wandb_config(env_config)
    if not cfg.get("enabled", False) or not cfg.get("log_artifacts", True):
        return
    if str(cfg.get("mode", "online")).lower() == "disabled":
        return
    if not _has_wandb_package():
        logger.warning("log_artifacts requested but wandb is not installed — skipping")
        return

    out = Path(output_dir)
    candidates = [
        out / "experiment_config.yml",
        out / "monitor.csv",
        out / "best_episode_details.csv",
        out / "best_episode.json",
        out / "carbon_signal_debug.csv",
    ]
    plots_dir = out / "plots"

    files_to_upload = [p for p in candidates if p.exists()]
    if plots_dir.is_dir():
        files_to_upload.extend(sorted(plots_dir.glob("*.png")))

    if not files_to_upload:
        logger.info("No artifact files found under %s — nothing to upload to wandb", output_dir)
        return

    trial_id = _find_trial_id(out)
    if not trial_id:
        logger.warning(
            "Could not locate Ray Tune trial directory under %s — uploading "
            "as a fresh wandb run instead of resuming the trial run.",
            out,
        )

    import wandb

    init_kwargs: Dict[str, Any] = {
        "project": cfg["project"],
        "job_type": "artifact-upload",
        "tags": list(cfg.get("tags") or []) + [experiment_name, "artifacts"],
        "mode": cfg.get("mode") or "online",
    }
    if cfg.get("entity"):
        init_kwargs["entity"] = cfg["entity"]
    if trial_id:
        init_kwargs["id"] = trial_id
        init_kwargs["resume"] = "allow"
    if run_name:
        init_kwargs["name"] = run_name

    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        logger.warning("wandb.init failed during artifact upload: %s", e)
        return

    try:
        artifact = wandb.Artifact(
            name=f"{experiment_name}-results",
            type="training-results",
            description=f"Training artifacts for {experiment_name} (trial {trial_id})",
        )
        for f in files_to_upload:
            artifact.add_file(str(f), name=f.relative_to(out).as_posix())
        run.log_artifact(artifact)
        logger.info("Uploaded %d files to wandb artifact '%s'", len(files_to_upload), artifact.name)
    finally:
        wandb.finish()
