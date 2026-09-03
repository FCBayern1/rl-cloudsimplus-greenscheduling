"""The step-2 pilot's two arms: derived, not hand-written, and provably minimal.

The pilot answers "fooled or blind-escape". That answer is only attributable if the two
arms differ in the forecast tier and nothing else, so the diff is pinned in both
directions, and the boundary conditions the work order set (Vanilla only, DISCOVERY
turbines only) are asserted mechanically rather than promised.
"""
from __future__ import annotations

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_rl_step2_pilot as g  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return g.base_block()


@pytest.fixture(scope="module")
def blocks():
    return g.build(100_000)


class TestExactDiff:
    def test_each_arm_differs_from_the_frozen_cell_only_as_declared(self, base, blocks):
        for n, b in blocks.items():
            extra = g.diff_keys(base, b) - g.WHITELIST
            assert not extra, f"{n}: undeclared drift {sorted(extra)}"

    def test_the_whitelist_has_no_unused_entries(self, base, blocks):
        touched = set()
        for b in blocks.values():
            touched |= g.diff_keys(base, b)
        assert not (g.WHITELIST - touched), sorted(g.WHITELIST - touched)

    def test_the_two_arms_differ_only_in_the_tier_and_names(self, blocks):
        a, b = (blocks[f"rlp2_{g.BASE_CELL}_{k}"] for k in ("godeye", "shrink50"))
        extra = g.diff_keys(a, b) - g.ARM_WHITELIST
        assert not extra, f"arms differ beyond the tier: {sorted(extra)}"

    def test_the_two_arms_really_do_differ_in_the_tier(self, blocks):
        a, b = (blocks[f"rlp2_{g.BASE_CELL}_{k}"] for k in ("godeye", "shrink50"))
        assert a["perturb_tier"] == "godeye" and b["perturb_tier"] == "shrink50"

    def test_the_base_block_is_not_mutated(self, base):
        before = yaml.safe_dump(base, sort_keys=True)
        g.derive(base, "shrink50", 50_000)
        assert yaml.safe_dump(base, sort_keys=True) == before


class TestBoundaries:
    def test_both_arms_are_vanilla(self, blocks):
        for n, b in blocks.items():
            assert (b.get("crd") or {}).get("enabled", False) is False, n

    def test_confirmation_turbines_are_untouched(self, blocks):
        for n, b in blocks.items():
            t = {int(x) for dc in b["datacenters"] for x in (dc.get("turbine_ids") or [])}
            assert not (t & g.CONFIRMATION_TURBINES), f"{n} reaches {sorted(t & g.CONFIRMATION_TURBINES)}"

    def test_the_discovery_turbines_are_the_ones_carried_over(self, base, blocks):
        want = [dc.get("turbine_ids") for dc in base["datacenters"]]
        for n, b in blocks.items():
            assert [dc.get("turbine_ids") for dc in b["datacenters"]] == want, n

    def test_the_pilot_does_not_publish(self, blocks):
        for n, b in blocks.items():
            assert b["wandb"]["enabled"] is False and b["wandb"]["mode"] == "disabled", n

    def test_observation_forecast_comes_from_the_ladder(self, blocks):
        for n, b in blocks.items():
            assert b["green_oracle_mode"] == "perturbed_godeye", n

    def test_the_budget_is_a_pilot_budget(self, base):
        assert base["training"]["total_timesteps"] == 600_000
        for steps in (50_000, 100_000):
            b = g.derive(base, "godeye", steps)
            assert b["training"]["total_timesteps"] == steps
            assert b["training"]["train_batch_size"] == base["training"]["train_batch_size"]

    def test_a_crd_enabled_base_is_refused(self, monkeypatch, base):
        import copy as _c
        bad = _c.deepcopy(base)
        bad["crd"] = dict(bad["crd"], enabled=True)
        monkeypatch.setattr(g, "base_block", lambda path=None: bad)
        with pytest.raises(AssertionError, match="crd enabled"):
            g.build(50_000)

    def test_tiers_are_supported_by_the_provider(self):
        sys.path.insert(0, os.path.join(g.REPO, "drl-manager"))
        from src.prediction.perturbed_godeye_provider import SUPPORTED_TIERS
        for t in g.ARMS.values():
            assert t in SUPPORTED_TIERS
