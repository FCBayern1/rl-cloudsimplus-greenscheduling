import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_d_longrun as lr  # noqa: E402


def test_main_jobs_are_360_final_plus_144_init():
    cks = {L: {"init": f"/ck/{L}/init", "final": f"/ck/{L}/final"} for L in lr.LINES}
    jobs = lr.build_jobs(20260904, cks, range(6))
    assert lr.assert_main_counts(jobs) == (360, 144)
    assert len(jobs) == 504
    assert sum(1 for j in jobs if j["line"] in ("V", "E") and j["tag"] == "final") == 2 * 6 * 4 * 6
    assert all(j["tier"] == lr.CLEAN[j["line"]] for j in jobs if j["tag"] == "init")


def test_wrong_window_count_is_refused():
    cks = {L: {"init": "i", "final": "f"} for L in lr.LINES}
    with pytest.raises(AssertionError):
        lr.assert_main_counts(lr.build_jobs(1, cks, range(3)))


def test_init_hash_is_over_state_files_only_and_equal_for_equal_weights():
    with tempfile.TemporaryDirectory() as d:
        for name, w in (("a", b"weights-1"), ("b", b"weights-1"), ("c", b"weights-2")):
            root = os.path.join(d, name, "learner_group", "learner", "rl_module", "global_policy")
            os.makedirs(root)
            open(os.path.join(root, "module_state.pt"), "wb").write(w)
            open(os.path.join(root, "metadata.json"), "w").write(name)   # differs, must not count
        ha, hb, hc = (lr.init_hash(os.path.join(d, n)) for n in ("a", "b", "c"))
        assert ha == hb and ha != hc and len(ha) == 64
        assert lr.init_hash(os.path.join(d, "nothing")) is None


def test_env_pins_hash_seed_and_ray_tmpdir():
    e = lr.env_for()
    assert e["PYTHONHASHSEED"] == "0" and e["RAY_TMPDIR"] == lr.RAY_TMPDIR
    assert e["TMPDIR"] == lr.TMPDIR and lr.TMPDIR.startswith("/home/")
    ee = lr.env_for(lr.EVAL_ENV)
    assert ee["OMP_NUM_THREADS"] == "1"


def test_disk_gate_refuses_when_below_threshold():
    free = lr.disk_free_gb("/")
    assert lr.disk_gate(min_gb=0, path="/") == free
    with pytest.raises(SystemExit):
        lr.disk_gate(min_gb=free + 1e6, path="/")
    with pytest.raises(SystemExit):
        lr.disk_gate(min_gb=0, path="/", tmp_min_gb=free + 1e6, tmp_path="/")


def test_seeds_and_budget_are_the_frozen_ones():
    assert lr.SEEDS == (20260904, 20260905, 20260906, 20260907, 20260908)
    assert lr.STEPS == 400_000 and lr.PAIRS == (("NV", "V"), ("NE", "E"))
