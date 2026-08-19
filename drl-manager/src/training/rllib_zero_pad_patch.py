"""
Monkeypatch for an RLlib 2.40 bug: ``split_and_zero_pad`` does not descend into
nested-``Dict`` observations on stateful (recurrent) RLModules.

Background
----------
On the new RLModule/Learner API stack, ``AddStatesFromEpisodesToBatch`` chunks
each episode's per-column data into ``max_seq_len``-long pieces via
``ray.rllib.utils.postprocessing.zero_padding.split_and_zero_pad``. The upstream
implementation only splits along the time axis when an item is a
``BatchedNdArray``. When the item is a *struct* (e.g. a ``Dict`` observation
whose leaves are ``BatchedNdArray`` — exactly what our hierarchical multi-DC env
produces), it falls through to the "single item" branch and treats an entire
variable-length episode as one timestep. The un-chunked, variable-length
episodes then fail to stack downstream in ``BatchIndividualItems`` with::

    ValueError: all input arrays must have the same shape

This bites the GTrXL (recurrent) experiments: the local-agent observation is a
nested ``Dict`` and the ``shared_local_policy`` batches several episodes of
differing length. The OLD API stack used a different
(``pad_batch_to_sequences_of_same_size``) path that handled ``Dict`` obs
leaf-by-leaf, so it never triggered this — which is why old-stack PPO runs fine.

Fix
---
Replace ``split_and_zero_pad`` with a version that, when an item is a struct
whose leaves are ``BatchedNdArray``, splits/zero-pads each leaf along axis 0 in
lockstep (treating the struct exactly like a ``BatchedNdArray`` of structs).
All other inputs behave identically to upstream. The upstream ``zero_element``
logic already descends into structs, so padding the final partial chunk works
unchanged.

Two ways to apply it
--------------------
:func:`apply_patch` rebinds the function *in this process only*. That is not
enough on its own: the crash happens inside the **Learner actor**, a separate
Python process that imports Ray fresh from disk. So the primary mechanism is
:func:`ensure_patched`, which repairs the installed
``ray/rllib/utils/postprocessing/zero_padding.py`` on disk (the same fix
``isambard/setup_env.sh`` does with ``cp``) so every Ray worker picks it up, and
*also* monkeypatches the current process (whose ``zero_padding`` module object is
already imported).

The disk repair is gated on a functional probe, so it is a no-op on a Ray that
already handles structs (upstream fixed this in 2.47.0, PR #52818) — it will
never downgrade a newer Ray.

Call :func:`ensure_patched` once at import of the GTrXL trainer. Idempotent.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import tree  # dm_tree

from ray.rllib.utils.postprocessing import zero_padding as _zp
from ray.rllib.utils.spaces.space_utils import batch as _batch, BatchedNdArray

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_dict_obs_split_patch_applied"


def _leaves_are_batched(item) -> bool:
    """True for a struct (not itself a BatchedNdArray) whose every leaf is one."""
    if isinstance(item, BatchedNdArray):
        return False
    leaves = tree.flatten(item)
    return len(leaves) > 0 and all(isinstance(leaf, BatchedNdArray) for leaf in leaves)


def split_and_zero_pad(item_list, max_seq_len):
    """Drop-in replacement for RLlib's split_and_zero_pad with struct support.

    Mirrors the upstream implementation exactly, adding one branch: a struct
    whose leaves are ``BatchedNdArray`` is split along axis 0 per leaf, so a
    ``Dict`` observation is chunked into ``max_seq_len``-long sequences just like
    a plain ``BatchedNdArray`` would be.
    """
    zero_element = tree.map_structure(
        lambda s: np.zeros_like([s[0]] if isinstance(s, BatchedNdArray) else s),
        item_list[0],
    )

    ret = []
    current_time_row = []
    current_t = 0

    item_list = deque(item_list)
    while len(item_list) > 0:
        item = item_list.popleft()
        # `item` is a batched np.array: split along axis 0 if necessary.
        if isinstance(item, BatchedNdArray):
            t = max_seq_len - current_t
            current_time_row.append(item[:t])
            if len(item) <= t:
                current_t += len(item)
            else:
                current_t += t
                item_list.appendleft(item[t:])
        # `item` is a struct (e.g. Dict obs) of BatchedNdArrays: split every leaf
        # along axis 0 in lockstep. (Upstream lacks this and never chunks here.)
        elif _leaves_are_batched(item):
            item_len = len(tree.flatten(item)[0])
            t = max_seq_len - current_t
            current_time_row.append(tree.map_structure(lambda s: s[:t], item))
            if item_len <= t:
                current_t += item_len
            else:
                current_t += t
                item_list.appendleft(tree.map_structure(lambda s: s[t:], item))
        # `item` is a single item (no batch axis): append and continue.
        else:
            current_time_row.append(item)
            current_t += 1

        if current_t == max_seq_len:
            ret.append(
                _batch(current_time_row, individual_items_already_have_batch_dim="auto")
            )
            current_time_row = []
            current_t = 0

    # Unfinished row: pad to max_seq_len and append.
    if current_t > 0 and current_t < max_seq_len:
        current_time_row.extend([zero_element] * (max_seq_len - current_t))
        ret.append(
            _batch(current_time_row, individual_items_already_have_batch_dim="auto")
        )

    return ret


def apply_patch() -> bool:
    """Install the patched split_and_zero_pad everywhere it is bound. Idempotent.

    Returns True if the patch was applied now, False if it was already applied.
    """
    if getattr(_zp, _PATCH_FLAG, False):
        return False

    _zp.split_and_zero_pad = split_and_zero_pad
    setattr(_zp, _PATCH_FLAG, True)

    # `add_states_from_episodes_to_batch` does `from ...zero_padding import
    # split_and_zero_pad`, binding a module-local name that we must rebind too —
    # that is the call site (line ~261) that chunks the obs column.
    try:
        from ray.rllib.connectors.common import (
            add_states_from_episodes_to_batch as _asb,
        )

        _asb.split_and_zero_pad = split_and_zero_pad
    except Exception:  # pragma: no cover - defensive; module name may move
        pass

    return True


# --------------------------------------------------------------------------- #
# Self-check / self-heal of the *installed* Ray (covers Ray worker processes)
# --------------------------------------------------------------------------- #

# Vendored, already-patched copy of ray 2.40.0's zero_padding.py. This is the
# same file isambard/setup_env.sh copies over site-packages on the cluster.
VENDORED_PATCH = (
    Path(__file__).resolve().parents[3] / "isambard" / "patches" / "ray_zero_padding_dictobs.py"
)


def _probe(split_fn) -> bool:
    """True if `split_fn` chunks a Dict-obs item along the time axis.

    Feeds one 3-timestep struct with ``BatchedNdArray`` leaves and asks for
    ``max_seq_len=2``. A working implementation returns 2 rows of length 2 (the
    second zero-padded); the buggy one treats the whole struct as a single
    timestep and returns 1 row.
    """
    item = {"observation": BatchedNdArray(np.arange(3, dtype=np.float32)[:, None])}
    try:
        out = split_fn([item], 2)
        return len(out) == 2 and all(len(o["observation"]) == 2 for o in out)
    except Exception:
        return False


def installed_ray_is_patched() -> bool:
    """True if the Ray *installed on disk* already handles Dict observations.

    Loads ``zero_padding.py`` from disk as a standalone module and probes that,
    rather than the (possibly already monkeypatched) imported one — the file is
    what a freshly spawned Ray worker/Learner process will import.
    """
    path = _installed_zero_padding_path()
    try:
        spec = importlib.util.spec_from_file_location("_ray_zero_padding_ondisk", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return _probe(mod.split_and_zero_pad)
    except Exception:  # pragma: no cover - unreadable / broken file
        logger.exception("Could not probe installed zero_padding.py at %s", path)
        return False


def _installed_zero_padding_path() -> Path:
    import ray

    return Path(ray.__file__).resolve().parent / "rllib/utils/postprocessing/zero_padding.py"


def repair_installed_ray(source: Path | None = None) -> bool:
    """Overwrite the installed zero_padding.py with the vendored patched copy.

    Only ever called when the installed Ray fails :func:`installed_ray_is_patched`,
    so a newer (already-fixed) Ray is never downgraded. Keeps a one-time
    ``.orig`` backup and rolls back if the copy does not actually fix the probe.

    Returns True if the installed Ray is patched afterwards.
    """
    source = Path(source) if source is not None else VENDORED_PATCH
    if not source.is_file():
        logger.error("Cannot repair Ray: vendored patch not found at %s", source)
        return False

    target = _installed_zero_padding_path()
    if not os.access(target.parent, os.W_OK):
        logger.error("Cannot repair Ray: %s is not writable", target.parent)
        return False

    backup = target.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(target, backup)

    # Atomic replace: several Ray processes may import this module at once, and a
    # half-written zero_padding.py would break every one of them.
    staged = target.with_suffix(f".py.{os.getpid()}.tmp")
    shutil.copy2(source, staged)
    os.replace(staged, target)

    if installed_ray_is_patched():
        logger.warning(
            "Installed Ray had the unpatched split_and_zero_pad (Dict-obs bug); "
            "repaired %s from %s (original saved as %s)",
            target, source, backup.name,
        )
        return True

    # Vendored copy did not fix it (e.g. wrong Ray version): roll back.
    shutil.copy2(backup, target)
    logger.error(
        "Repair of %s from %s did not fix the Dict-obs probe; rolled back", target, source
    )
    return False


def ensure_patched(strict: bool = True) -> bool:
    """Guarantee Dict-obs-safe zero padding in this process *and* in Ray workers.

    1. Probe the Ray installed on disk (the file a spawned worker will import).
    2. If broken, rewrite it from the vendored patched copy.
    3. Patch this process too (its ``zero_padding`` module is already imported).

    With ``strict=True`` (default) an unrepairable installation raises instead of
    letting training run for minutes and then die inside the Learner actor with
    ``ValueError: all input arrays must have the same shape``.
    """
    on_disk_ok = installed_ray_is_patched()
    if not on_disk_ok:
        on_disk_ok = repair_installed_ray()

    if not on_disk_ok:
        msg = (
            "Installed Ray's split_and_zero_pad does not handle Dict observations. "
            "Ray worker/Learner processes will crash with 'all input arrays must have "
            "the same shape'. Fix with:\n"
            f"  cp {VENDORED_PATCH} {_installed_zero_padding_path()}\n"
            "(or upgrade to ray >= 2.47.0, which contains the upstream fix)."
        )
        if strict:
            raise RuntimeError(msg)
        logger.error(msg)

    # This process already imported zero_padding, so the file fix alone is not
    # enough here — rebind in memory as well.
    if not _probe(_zp.split_and_zero_pad):
        apply_patch()

    return on_disk_ok
