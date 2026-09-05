import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_scene_v1 import scene_block  # noqa: E402


def _src():
    return {"experiment_name": "old", "simulation_name": "OLD", "global_action_mode": "offset_v1",
            "green_episode_offset_allowlist": "1;2;3", "green_episode_offset_range": 44950,
            "datacenters": [
                {"datacenter_id": 1, "turbine_ids": [51, 53], "vm_pe_mips": 40000},
                {"datacenter_id": 0, "turbine_ids": [123, 10], "vm_pe_mips": 40000},
                {"datacenter_id": 2, "turbine_ids": [112]},
                {"datacenter_id": 3, "turbine_ids": []},
                {"datacenter_id": 4, "turbine_ids": []}]}


def test_turbines_and_allowlist_replaced_everything_else_kept():
    src = _src()
    b = scene_block(src, {"0": [133, 78], "1": [22, 81], "2": [94]}, [24398, 10829], "sv1_x")
    ids = {d["datacenter_id"]: d["turbine_ids"] for d in b["datacenters"]}
    assert ids == {0: [133, 78], 1: [22, 81], 2: [94], 3: [], 4: []}
    assert b["green_episode_offset_allowlist"] == "24398;10829"
    assert b["global_action_mode"] == "offset_v1" and b["green_episode_offset_range"] == 44950
    assert b["datacenters"][1]["vm_pe_mips"] == 40000
    assert b["experiment_name"] == "sv1_x" and b["simulation_name"] == "SCENE_V1_sv1_x"
    assert src["datacenters"][0]["turbine_ids"] == [51, 53]          # source untouched
