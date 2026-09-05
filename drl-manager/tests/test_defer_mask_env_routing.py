"""Env-side enforcement of the deadline-safe DEFER mask for arms that cannot be masked at
the logit level (heuristics, an adversarial always-defer policy)."""
from gym_cloudsimplus.envs.hierarchical_multidc_env import route_disallowed_defers

NUM_DCS = 3          # DEFER index == 3


def test_disallowed_defer_goes_to_the_greenest_dc_with_room():
    acts, n = route_disallowed_defers(
        actions=[3, 3, 1], defer_allowed=[0.0, 1.0, 0.0], pes=[4, 4, 4],
        dc_green_ratio=[0.2, 0.9, 0.5], dc_available_pes=[8, 2, 8], num_dcs=NUM_DCS)
    # slot 0: DEFER not allowed -> greenest DC with >= 4 PEs is DC2 (DC1 has only 2)
    # slot 1: DEFER allowed -> untouched; slot 2: already routed -> untouched
    assert acts == [2, 3, 1] and n == 1


def test_falls_back_to_the_greenest_dc_when_nothing_has_room():
    acts, n = route_disallowed_defers([3], [0.0], [64], [0.1, 0.7, 0.3], [8, 8, 8], NUM_DCS)
    assert acts == [1] and n == 1


def test_mask_off_or_missing_arrays_change_nothing():
    acts, n = route_disallowed_defers([3, 0], [1.0, 1.0], None, None, None, NUM_DCS)
    assert acts == [3, 0] and n == 0


def test_padding_slots_are_never_rerouted_nor_counted():
    # slot 2 is padding (pes 0) and carries a DEFER with defer_allowed 0: must stay untouched
    acts, n = route_disallowed_defers([3, 3, 3], [0.0, 1.0, 0.0], [4, 4, 0], [0.2, 0.9, 0.5], [8, 8, 8], NUM_DCS)
    assert acts == [1, 3, 3] and n == 1
