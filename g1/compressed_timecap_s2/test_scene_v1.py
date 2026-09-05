import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene_v1 import (  # noqa: E402
    ABS_FRAC, P_DYN_PE_W, c_brown_ref_kg, draw_windows, dynamic_energy_wh, energy_weighted,
    headroom_ok, mean_brown_factor)


def test_unit_lock_one_wh_times_half_kg_per_kwh():
    assert c_brown_ref_kg(1.0, 0.5) == 0.0005                      # B1: kg per kWh, /1000


def test_dynamic_energy_is_pes_times_draw_times_runtime():
    # one job: 32 PEs, 1920000 MI at 40000 MIPS, u = 1 -> 48 s; energy = 32 * P * 48/3600 Wh
    e = dynamic_energy_wh([32], [1920000.0], 40000.0, 1.0)
    assert abs(e - 32 * P_DYN_PE_W * 48.0 / 3600.0) < 1e-9
    # utilisation stretches the runtime and scales the draw: u = 0.5 -> runtime 96 s, draw halved
    e2 = dynamic_energy_wh([32], [1920000.0], 40000.0, 0.5)
    assert abs(e2 - 32 * P_DYN_PE_W * 0.5 * 96.0 / 3600.0) < 1e-9
    assert dynamic_energy_wh([], [], 40000.0, 1.0) == 0.0


def test_mean_brown_factor_and_headroom_gate():
    assert mean_brown_factor([{"brown_carbon_factor": 0.5}, {"brown_carbon_factor": 0.7}, {}]) == 0.6
    ref = 1.0
    assert headroom_ok(1.0, 0.8, ref) is True                       # 20 % rel, 0.2 >= 0.05
    assert headroom_ok(1.0, 0.9, ref) is False                      # 10 % rel
    assert headroom_ok(0.1, 0.05, ref) is True                      # 50 % rel, gap 0.05 == 0.05 * ref (>=)
    assert headroom_ok(0.1, 0.06, ref) is False                     # 40 % rel but gap 0.04 < 0.05 * ref
    assert headroom_ok(0.0, 0.0, ref) is False
    assert ABS_FRAC == 0.05


def test_windows_are_hash_ordered_greedy_and_non_overlapping():
    r = draw_windows(20000, 3, "t:", footprint=2922)
    assert r["status"] == "OK" and len(r["windows"]) == 3
    ws = sorted(r["windows"])
    assert all(ws[i] + 2922 <= ws[i + 1] for i in range(len(ws) - 1))
    order = sorted(range(0, 20000 - 2922 + 1), key=lambda o: hashlib.sha256(f"t:{o}".encode()).hexdigest())
    assert r["windows"][0] == order[0]                              # first accepted = first in hash order
    assert draw_windows(6000, 3, "t:", footprint=2922)["status"] == "STOP_WINDOW_SPLIT"   # room for two only
    assert draw_windows(6000, 3, "t:", footprint=2922) == draw_windows(6000, 3, "t:", footprint=2922)


def test_energy_weighted_coverage_prefers_big_jobs():
    # a big job fully covered and a tiny job uncovered -> weighted coverage near 1
    v = energy_weighted([1.0, 0.0], pes=[32, 1], mi=[1920000.0, 40000.0], vm_pe_mips=40000.0, cpu_util=1.0)
    assert v > 0.99
    assert energy_weighted([], [], [], 40000.0, 1.0) == 0.0
