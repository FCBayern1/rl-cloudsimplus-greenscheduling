"""Python mirror of the Java episode-offset rule (schedule vs Stage D allowlist)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_cloudsimplus.envs.hierarchical_multidc_env import episode_offset_rows  # noqa: E402


def test_schedule_without_allowlist_matches_java():
    assert episode_offset_rows(0, 44950) == 0
    assert episode_offset_rows(2, 44950) == 2018
    assert episode_offset_rows(45, 44950) == (1009 * 45) % 44950
    assert episode_offset_rows(7, 0) == 0


def test_allowlist_is_cycled_and_wins():
    allow = [13016, 21088, 29160]
    assert [episode_offset_rows(i, 44950, allow) for i in range(4)] == [13016, 21088, 29160, 13016]
    assert episode_offset_rows(1, 0, allow) == 21088


def test_allowlist_string_forms():
    assert episode_offset_rows(1, 44950, "13016;21088") == 21088
    assert episode_offset_rows(1, 44950, "13016, 21088") == 21088
    assert episode_offset_rows(1, 44950, "") == 1009
    assert episode_offset_rows(1, 44950, None) == 1009
