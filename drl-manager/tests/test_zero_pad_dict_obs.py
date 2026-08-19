"""
Regression test for the RLlib 2.40 stateful-connector bug where
``split_and_zero_pad`` does not descend into nested-``Dict`` observations.

Without the fix, a Dict-obs episode is treated as a single timestep (never
chunked to ``max_seq_len``), so episodes of different length stay at their raw
length and later fail to ``np.stack`` in ``BatchIndividualItems`` with
``ValueError: all input arrays must have the same shape``. This is what crashed
the GTrXL (recurrent) ``shared_local_policy``.

These tests exercise the patched ``split_and_zero_pad`` directly (no Java / no
RLlib training loop needed) and assert:
  1. a nested Dict-obs episode is chunked into ceil(T / max_seq_len) rows,
  2. each leaf row has length exactly max_seq_len (tail zero-padded),
  3. concatenating the chunks reproduces the original T timesteps,
  4. chunks from episodes of *different* length stack without error
     (the exact regression that crashed training),
  5. plain BatchedNdArray / scalar inputs still behave like upstream,
  6. apply_patch() rebinds the call site used by the learner connector.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ray.rllib.utils.spaces.space_utils import BatchedNdArray  # noqa: E402
from src.training.rllib_zero_pad_patch import split_and_zero_pad  # noqa: E402

VM = 175       # vm_loads feature width
MASK = 176     # action_mask width
MAX_SEQ = 128


def _episode_obs(T: int):
    """One episode's obs item: nested Dict with BatchedNdArray leaves, len T.

    Leaf values encode their global timestep index so we can verify ordering /
    padding after chunking.
    """
    idx = np.arange(T, dtype=np.float32)
    return {
        "observation": {
            "vm_loads": BatchedNdArray(np.tile(idx[:, None], (1, VM))),
        },
        "action_mask": BatchedNdArray(np.tile(idx[:, None], (1, MASK))),
    }


def test_dict_obs_episode_is_chunked_and_padded():
    T = 378
    chunks = split_and_zero_pad([_episode_obs(T)], MAX_SEQ)

    assert len(chunks) == math.ceil(T / MAX_SEQ) == 3
    for ch in chunks:
        assert ch["observation"]["vm_loads"].shape == (MAX_SEQ, VM)
        assert ch["action_mask"].shape == (MAX_SEQ, MASK)

    # Concatenated chunks reproduce the original T timesteps (col 0 == index).
    recon = np.concatenate([ch["action_mask"][:, 0] for ch in chunks])
    assert recon.shape == (3 * MAX_SEQ,)
    np.testing.assert_array_equal(recon[:T], np.arange(T, dtype=np.float32))
    # The tail (3*128 - 378 = 6 rows) is zero-padded.
    assert np.all(recon[T:] == 0.0)


def test_chunks_from_different_length_episodes_stack():
    # The exact downstream regression: two episodes of differing length.
    chunks_a = split_and_zero_pad([_episode_obs(378)], MAX_SEQ)  # 3 chunks
    chunks_b = split_and_zero_pad([_episode_obs(278)], MAX_SEQ)  # 3 chunks
    all_chunks = chunks_a + chunks_b

    # Stacking every leaf across all chunks must not raise and must be uniform.
    masks = np.stack([c["action_mask"] for c in all_chunks], axis=0)
    vms = np.stack([c["observation"]["vm_loads"] for c in all_chunks], axis=0)
    assert masks.shape == (len(all_chunks), MAX_SEQ, MASK)
    assert vms.shape == (len(all_chunks), MAX_SEQ, VM)


def test_exact_multiple_has_no_padding():
    T = 256  # exactly 2 * MAX_SEQ
    chunks = split_and_zero_pad([_episode_obs(T)], MAX_SEQ)
    assert len(chunks) == 2
    recon = np.concatenate([ch["action_mask"][:, 0] for ch in chunks])
    np.testing.assert_array_equal(recon, np.arange(T, dtype=np.float32))


def test_plain_batched_ndarray_unchanged():
    # Upstream behaviour for BatchedNdArray inputs must be preserved.
    out = split_and_zero_pad(
        [BatchedNdArray([0, 1]), BatchedNdArray([2, 3, 4, 5]), BatchedNdArray([6, 7, 8])],
        5,
    )
    np.testing.assert_array_equal(np.asarray(out[0]), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(out[1]), [5, 6, 7, 8, 0])


def test_scalar_items_unchanged():
    out = split_and_zero_pad([0, 1, 2, 3, 4, 5, 6, 7], 5)
    np.testing.assert_array_equal(np.asarray(out[0]), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(out[1]), [5, 6, 7, 0, 0])


def test_apply_patch_rebinds_call_site():
    from src.training.rllib_zero_pad_patch import apply_patch, split_and_zero_pad as patched
    from ray.rllib.utils.postprocessing import zero_padding as zp

    apply_patch()  # idempotent
    from ray.rllib.connectors.common import add_states_from_episodes_to_batch as asb

    assert zp.split_and_zero_pad is patched
    assert asb.split_and_zero_pad is patched


# --------------------------------------------------------------------------- #
# Self-check / self-heal of the installed Ray (what Ray worker processes import)
# --------------------------------------------------------------------------- #

def _stock_split_and_zero_pad(item_list, max_seq_len):
    """Stand-in for ray 2.40's buggy version: no struct branch."""
    out, row, t = [], [], 0
    for item in item_list:
        row.append(item)
        t += 1
        if t == max_seq_len:
            out.append(row)
            row, t = [], 0
    if row:
        out.append(row)
    return out


def test_probe_distinguishes_broken_from_fixed():
    from src.training.rllib_zero_pad_patch import _probe

    assert _probe(split_and_zero_pad) is True
    assert _probe(_stock_split_and_zero_pad) is False
    assert _probe(lambda *a, **k: (_ for _ in ()).throw(ValueError("boom"))) is False


def test_installed_ray_handles_dict_obs():
    """Regression guard: a fresh `pip install ray` silently drops the patch.

    This is exactly the failure mode that crashed the GTrXL run -- the driver was
    fine, but the Learner actor imported the stock zero_padding.py from disk.
    """
    from src.training.rllib_zero_pad_patch import ensure_patched, installed_ray_is_patched

    ensure_patched()
    assert installed_ray_is_patched(), (
        "installed ray still cannot chunk Dict observations; Ray workers will crash"
    )


def test_ensure_patched_is_idempotent():
    from src.training.rllib_zero_pad_patch import ensure_patched

    assert ensure_patched() is True
    assert ensure_patched() is True


def test_repair_returns_false_when_source_missing(tmp_path):
    from src.training.rllib_zero_pad_patch import repair_installed_ray

    assert repair_installed_ray(tmp_path / "does_not_exist.py") is False


def test_repair_writes_target_and_keeps_backup(tmp_path, monkeypatch):
    import src.training.rllib_zero_pad_patch as zp_patch

    target = tmp_path / "zero_padding.py"
    target.write_text("# stock, unpatched\n")
    monkeypatch.setattr(zp_patch, "_installed_zero_padding_path", lambda: target)
    monkeypatch.setattr(zp_patch, "installed_ray_is_patched", lambda: True)

    assert zp_patch.repair_installed_ray() is True
    assert "_is_batched_struct" in target.read_text()
    assert target.with_suffix(".py.orig").read_text() == "# stock, unpatched\n"


def test_repair_rolls_back_when_still_broken(tmp_path, monkeypatch):
    import src.training.rllib_zero_pad_patch as zp_patch

    target = tmp_path / "zero_padding.py"
    target.write_text("# stock, unpatched\n")
    monkeypatch.setattr(zp_patch, "_installed_zero_padding_path", lambda: target)
    monkeypatch.setattr(zp_patch, "installed_ray_is_patched", lambda: False)

    assert zp_patch.repair_installed_ray() is False
    assert target.read_text() == "# stock, unpatched\n"  # rolled back, venv intact


def test_vendored_patch_exists():
    from src.training.rllib_zero_pad_patch import VENDORED_PATCH

    assert VENDORED_PATCH.is_file(), f"vendored ray patch missing: {VENDORED_PATCH}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
