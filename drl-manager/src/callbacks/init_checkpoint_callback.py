"""Save the untrained algorithm before its first SGD step (Stage D health gate).

RLlib's periodic checkpoints start after the first training iteration, so
`checkpoint_000000` is already one PPO update old. The health gate needs the true
initialisation as its reference, so this callback writes `<output_dir>/checkpoint_init`
from `on_algorithm_init`, which RLlib fires once the algorithm (env runners, learner
group, RLModule) is fully built and before `train()` is ever called.
"""
from __future__ import annotations

import logging
import os

from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


class InitCheckpointCallback(DefaultCallbacks):
    def __init__(self, output_dir: str, name: str = "checkpoint_init"):
        super().__init__()
        self._path = os.path.join(output_dir, name)

    def on_algorithm_init(self, *, algorithm=None, metrics_logger=None, **kwargs):
        if algorithm is None:
            return
        os.makedirs(self._path, exist_ok=True)
        saved = algorithm.save_to_path(self._path)
        with open(os.path.join(self._path, "INIT_MARKER"), "w") as f:
            f.write("saved from on_algorithm_init before the first training iteration\n")
        logger.info("Init checkpoint saved before first SGD: %s", saved)
