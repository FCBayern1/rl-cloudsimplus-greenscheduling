"""
Optional data augmentation utilities.

Some dataset loaders in this repo import `run_augmentation_single` to optionally
augment training windows when `args.augmentation_ratio > 0`.

In this codebase, augmentation is not required for core functionality, so this
module provides a safe no-op default implementation to keep training runnable.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def run_augmentation_single(
    data_x: np.ndarray,
    data_y: np.ndarray,
    args: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    No-op augmentation by default.

    Returns:
        data_x: unchanged
        data_y: unchanged
        augmentation_tags: zeros array (len(data_x),) indicating no augmentation
    """
    n = int(getattr(data_x, "shape", [0])[0] or 0)
    augmentation_tags = np.zeros((n,), dtype=np.int32)
    return data_x, data_y, augmentation_tags





