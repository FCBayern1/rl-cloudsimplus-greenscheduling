"""The derived experiment blocks.

Two directions are pinned, and both matter. A key outside the whitelist that drifts makes
the cells incomparable. A key inside the whitelist that does not carry its registered value
is worse: latest_start silently reverting to the legacy fixed-lead backstop would force
every job 600 rows early, so the 144-row exam would be decided before the scheduler ever
saw it, and the run would look perfectly healthy while measuring nothing.
"""
import copy
import json
import os

import numpy as np
import pytest
import yaml

import constants as C
import generate_configs as G
import workload as W

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WIND = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified")
TRACES = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/traces")
STAGE_A_YML = os.path.join(HERE, "config_cts2_stage_a.yml")


@pytest.fixture(scope="module")
def base():
    return G.load_base()


@pytest.fixture(scope="module")
def windows():
    return G.load_windows()


@pytest.fixture(scope="module")
def blocks(base, windows):
    return G.build_stage_a(base, windows)[0]


@pytest.fixture(scope="module")
def sample(base, windows):
    return G.derive_stage_a(base, C.cells()[0], windows)


class TestExactDiff:
    def test_top_level_diff_is_inside_the_whitelist(self, base, blocks):
        for name, b in blocks.items():
            extra = G.diff_keys(base, b) - G.STAGE_A_WHITELIST
            assert not extra, f"{name}: undeclared drift {sorted(extra)}"

    def test_every_whitelisted_key_actually_changes_or_is_pinned(self, base, sample):
        """The whitelist is a contract, not a permission slip: nothing in it is unused."""
        touched = G.diff_keys(base, sample)
        unused = G.STAGE_A_WHITELIST - touched
        assert not unused, f"whitelist lists keys the derivation never sets: {sorted(unused)}"

    def test_datacenter_diff_is_only_the_two_green_keys(self, base, blocks):
        for name, b in blocks.items():
            for i, (o, n) in enumerate(zip(base["datacenters"], b["datacenters"])):
                extra = {k for k in set(o) | set(n) if o.get(k) != n.get(k)}
                extra -= G.STAGE_A_DC_WHITELIST
                assert not extra, f"{name} dc{i}: {sorted(extra)}"

    def test_topology_is_untouched(self, base, blocks):
        """Five DCs, three with turbines and two without: the C-regime map, unchanged."""
        want = [(d["datacenter_id"], tuple(d.get("turbine_ids") or []),
                 d["brown_carbon_factor"], d.get("time_zone_offset_rows", 0))
                for d in base["datacenters"]]
        for name, b in blocks.items():
            got = [(d["datacenter_id"], tuple(d.get("turbine_ids") or []),
                    d["brown_carbon_factor"], d.get("time_zone_offset_rows", 0))
                   for d in b["datacenters"]]
            assert got == want, name
        assert sum(1 for d in base["datacenters"] if d.get("turbine_ids")) == 3

    def test_base_block_is_never_mutated(self, windows):
        fresh = G.load_base()
        before = yaml.safe_dump(fresh, sort_keys=True)
        G.derive_stage_a(fresh, C.cells()[0], windows)
        assert yaml.safe_dump(fresh, sort_keys=True) == before

    def test_missing_base_block_is_loud(self, tmp_path):
        p = tmp_path / "empty.yml"
        p.write_text("{}\n")
        with pytest.raises(KeyError):
            G.load_base(str(p))


class TestRegisteredValues:
    def test_backstop_is_runtime_aware_not_legacy(self, blocks):
        for name, b in blocks.items():
            assert b["defer_deadline_force_mode"] == "latest_start", name
            assert float(b["defer_deadline_slack_sec"]) == C.DEFER_SLACK_ROWS, name

    def test_backstop_fires_at_the_registered_latest_start(self, blocks):
        """now + MI/(PES*MIPS*util) + slack >= deadline, with deadline = a + w + r,
        puts the forced start one row before the latest legal start and not 600 early."""
        for cell in C.cells()[:12]:
            wl = W.draw(cell)
            b = blocks[G.block_name(cell)]
            slack = float(b["defer_deadline_slack_sec"])
            runtime_est = wl["mi"] / (wl["pes"] * float(b["datacenters"][0]["vm_pe_mips"])
                                      * float(b["cloudlet_cpu_utilization"]))
            forced_at = wl["deadline"] - runtime_est - slack
            latest = wl["arrival"] + cell["wait_cap_rows"]
            assert np.allclose(latest - forced_at, slack), G.block_name(cell)
            # The legacy fixed-lead rule would have fired 600 rows early, i.e. before the
            # job even arrives in every cell of this grid.
            assert np.all(wl["deadline"] - 600.0 < wl["arrival"])

    def test_execution_physics_is_full_utilization(self, blocks):
        """0.5 doubles every runtime and voids the closure condition."""
        for name, b in blocks.items():
            assert float(b["cloudlet_cpu_utilization"]) == 1.0, name

    def test_row_semantics_are_compressed_and_step(self, blocks):
        for name, b in blocks.items():
            assert b["green_interpolation_mode"] == "STEP", name
            for dc in b["datacenters"]:
                assert dc["time_scaling_mode"] == "COMPRESSED", name
                assert dc["green_interpolation_mode"] == "STEP", name
                assert float(dc["green_power_scale"]) == 1.0, name
            assert float(b["compressed_power_divisor"]) == 1500.0, name

    def test_scheduler_year_is_the_eval_year_not_the_training_year(self, blocks):
        for name, b in blocks.items():
            assert int(b["wind_csv_year"]) == C.YEAR_SCHEDULER_EVAL, name
            assert int(b["wind_csv_year"]) != C.YEAR_TIMECAP_TRAIN, name

    def test_offset_range_comes_from_windows_json(self, blocks, windows):
        for name, b in blocks.items():
            assert b["green_episode_offset_range"] == windows["green_episode_offset_range"], name

    def test_observation_bound_clears_the_largest_job(self, blocks):
        for name, b in blocks.items():
            assert b["obs_cloudlet_mi_high"] >= C.max_job_mi(), name

    def test_stage_a_cannot_reach_a_checkpoint(self, blocks):
        for name, b in blocks.items():
            assert "timecap" not in b, name
            assert b["forecast_mode"] == "none", name
            assert b["green_oracle_mode"] != "timecap", name

    def test_episode_length_covers_the_last_possible_finish(self, blocks):
        for cell in C.cells():
            wl = W.draw(cell)
            b = blocks[G.block_name(cell)]
            last = int((wl["arrival"] + cell["wait_cap_rows"] + wl["runtime"]).max())
            assert b["max_episode_length"] == last + C.DRAIN_STEPS

    def test_no_cell_reads_past_its_window(self, blocks, windows):
        fp = windows["footprint"]
        for name, b in blocks.items():
            need = fp["clock0_rows"] + b["max_episode_length"] + fp["max_tz_rows"]
            assert need <= fp["footprint_rows"], name


class TestStageCArms:
    def test_arms_differ_only_as_intended(self, sample):
        arms = {a: G.derive_stage_c(sample, a) for a in G.STAGE_C_ARMS}
        for a1 in arms:
            for a2 in arms:
                if a1 < a2:
                    extra = G.diff_keys(arms[a1], arms[a2]) - G.STAGE_C_ARM_WHITELIST
                    assert not extra, f"{a1}/{a2}: {sorted(extra)}"

    def test_no_forecast_arm_is_the_stage_a_scenario_renamed(self, sample):
        nf = G.derive_stage_c(sample, "noforecast")
        assert G.diff_keys(sample, nf) == {"experiment_name", "simulation_name"}

    def test_negative_controls_carry_the_perturbation_switch(self, sample):
        for arm, want in (("clean", "none"), ("shuffle", "shuffle"), ("anti", "anti")):
            b = G.derive_stage_c(sample, arm)
            assert b["timecap"]["forecast_perturbation"] == want
            assert b["green_oracle_mode"] == "timecap"

    def test_unknown_arm_is_loud(self, sample):
        with pytest.raises(ValueError):
            G.derive_stage_c(sample, "godeye")

    def test_stage_a_block_is_never_mutated_by_stage_c(self, sample):
        before = json.dumps(sample, sort_keys=True)
        for a in G.STAGE_C_ARMS:
            G.derive_stage_c(sample, a)
        assert json.dumps(sample, sort_keys=True) == before


class TestArtifactsOnDisk:
    def test_every_referenced_trace_exists(self, blocks):
        missing = [n for n, b in blocks.items()
                   if not os.path.isfile(os.path.join(TRACES,
                                                      os.path.basename(b["cloudlet_trace_file"])))]
        assert not missing, f"run generate_configs.py --write: {missing[:5]}"

    def test_every_turbine_csv_for_the_eval_year_exists(self, blocks):
        """A missing turbine CSV does not crash: it silently becomes zero green power."""
        missing = []
        for name, b in blocks.items():
            y = int(b["wind_csv_year"])
            for dc in b["datacenters"]:
                for t in dc.get("turbine_ids") or []:
                    if not os.path.isfile(os.path.join(WIND, f"Turbine_{t}_{y}.csv")):
                        missing.append((name, t, y))
        assert not missing, missing[:5]

    def test_emitted_config_matches_a_fresh_build(self, blocks):
        assert os.path.isfile(STAGE_A_YML), "run generate_configs.py --write"
        on_disk = yaml.safe_load(open(STAGE_A_YML))
        assert set(on_disk) == set(blocks)
        for name in blocks:
            assert on_disk[name] == blocks[name], f"{name} was hand edited"

    def test_manifest_matches_the_generator(self, blocks):
        mpath = os.path.join(HERE, "workloads.json")
        assert os.path.isfile(mpath), "run generate_configs.py --write"
        man = json.load(open(mpath))
        assert man["n_cells"] == len(blocks)
        assert man["stage_a_config_sha256"] == G.sha256_file(STAGE_A_YML)
        got = {r["key"]: r["content_sha256"] for r in man["workloads"]}
        want = {C.cell_key(c): W.content_sha256(W.draw(c)) for c in C.cells()}
        assert got == want
