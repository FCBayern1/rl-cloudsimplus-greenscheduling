"""The frozen spatial base must resolve to the SAME bytes on every machine.

It was trained on the 5080 under logs/ (gitignored), so the 3060 could not
find it and the formal PPO-base verdict was blocked. A hash-verified copy now
lives at frozen_ckpt/ and the resolver prefers the training path when present,
falling back to the tracked copy - so both boxes load one shared base rather
than two similar ones.
"""
import hashlib
import pathlib

import pytest

from sqt2_prescreen import resolve_blind_ck, _BLIND_CK_LOCAL, _BLIND_CK_TRACKED

REPO = pathlib.Path(__file__).resolve().parents[1]
POLICY = "learner_group/learner/rl_module/global_policy/module_state.pt"


class TestResolver:
    def test_resolves_to_a_real_policy(self):
        p = resolve_blind_ck(REPO)
        assert (p / POLICY).exists()

    def test_prefers_the_training_path_when_present(self):
        local = (REPO / _BLIND_CK_LOCAL).resolve()
        if (local / POLICY).exists():
            assert resolve_blind_ck(REPO) == local

    def test_tracked_copy_exists_for_other_machines(self):
        tracked = (REPO / _BLIND_CK_TRACKED).resolve()
        assert (tracked / POLICY).exists(), "the tracked fallback must be committed"

    def test_missing_base_raises_not_silently_passes(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_blind_ck(tmp_path)


class TestManifest:
    def test_tracked_copy_matches_its_sha256_manifest(self):
        tracked = (REPO / _BLIND_CK_TRACKED).resolve()
        man = tracked.parent / f"{tracked.name}.sha256"
        assert man.exists(), "hash manifest missing"
        n = 0
        for line in man.read_text().splitlines():
            want, rel = line.split(None, 1)
            f = tracked / rel.strip().lstrip("./")
            assert f.exists(), f"manifest lists a missing file: {rel}"
            got = hashlib.sha256(f.read_bytes()).hexdigest()
            assert got == want, f"checkpoint byte mismatch: {rel}"
            n += 1
        assert n >= 60, f"manifest looks truncated ({n} files)"

    def test_two_paths_agree_when_both_present(self):
        local = (REPO / _BLIND_CK_LOCAL).resolve()
        tracked = (REPO / _BLIND_CK_TRACKED).resolve()
        if not (local / POLICY).exists():
            pytest.skip("training path absent on this machine")
        a = hashlib.sha256((local / POLICY).read_bytes()).hexdigest()
        b = hashlib.sha256((tracked / POLICY).read_bytes()).hexdigest()
        assert a == b, "the tracked copy diverged from the trained checkpoint"
