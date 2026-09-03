import os
import tempfile

import yaml

from gen_s2 import PILOT_CELLS, generate_h


def _gen(zero_floor):
    with tempfile.TemporaryDirectory() as d:
        r = generate_h(1, PILOT_CELLS, out_dir=d, trace_dir=d, zero_floor=zero_floor)
        cfg = yaml.safe_load(open(os.path.join(d, r["config"])))
    return r, cfg


def test_zero_floor_swaps_every_host_to_the_dyn_twin_and_keeps_the_fleet():
    r, cfg = _gen(True)
    assert r["config"] == "config_s2hz_m1.yml"
    for name in PILOT_CELLS:
        blk = cfg[name]
        assert blk["simulation_name"].startswith("S2HZ_m1_")
        assert blk["split_large_cloudlets"] is False and blk["max_cloudlet_pes"] == 32
        for dc in blk["datacenters"]:
            assert "host_count_spec_asus_rs500a" not in dc
            assert "host_count_spec_asus_rs700a" not in dc
            host_pes = (dc.get("host_count_spec_asus_rs500a_dyn", 0) * 64
                        + dc.get("host_count_spec_asus_rs700a_dyn", 0) * 128)
            assert host_pes > 0
            assert dc["initial_s_vm_count"] == host_pes // 32 and dc["small_vm_pes"] == 32


def test_default_generate_h_is_unchanged_by_the_new_flag():
    r, cfg = _gen(False)
    assert r["config"] == "config_s2h_m1.yml" and r["zero_floor"] is False
    dc0 = cfg[PILOT_CELLS[0]]["datacenters"][0]
    assert "host_count_spec_asus_rs500a" in dc0 and "host_count_spec_asus_rs500a_dyn" not in dc0
