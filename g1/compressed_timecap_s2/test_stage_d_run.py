import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_d_run as sr  # noqa: E402


def test_job_list_is_exactly_180_final_plus_72_init():
    cks = {L: {"init": f"/ck/{L}/init", "final": f"/ck/{L}/final"} for L in sr.LINES}
    jobs = sr.build_jobs(cks)
    assert len(jobs) == 252
    final = [j for j in jobs if j["tag"] == "final"]
    init = [j for j in jobs if j["tag"] == "init"]
    assert len(final) == 180 and len(init) == 72
    # V and E: 6 cells x 4 tiers x 3 windows; NV and NE: 6 x 1 x 3
    assert sum(1 for j in final if j["line"] in ("V", "E")) == 2 * 6 * 4 * 3
    assert sum(1 for j in final if j["line"] in ("NV", "NE")) == 2 * 6 * 1 * 3
    assert all(j["tier"] == sr.CLEAN[j["line"]] for j in init)
    assert all(j["ck"].endswith("init") for j in init) and all(j["ck"].endswith("final") for j in final)


def test_job_list_refuses_a_wrong_shape(monkeypatch):
    monkeypatch.setattr(sr, "KS", (26, 34))
    cks = {L: {"init": "i", "final": "f"} for L in sr.LINES}
    with pytest.raises(AssertionError):
        sr.build_jobs(cks)


def test_frozen_sources_exist():
    for p in sr.FROZEN_SOURCES:
        assert os.path.exists(os.path.join(sr.REPO, p)), p
