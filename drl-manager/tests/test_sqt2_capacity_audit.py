import pytest

from sqt2_capacity_audit import (PROFILES, VM_MIPS, audit, dc_profile,
                                 executable_mi_per_on_second, w_per_pe_eff)
from oracle_slack_planner import WARMUP_ROWS


class TestFormula:
    def test_w_per_pe_eff_rs500a(self):
        # 51.36/64 + (214-51.36)/64 = 214/64 = 3.34375 exactly
        assert w_per_pe_eff("rs500a") == pytest.approx(214.0 / 64)

    def test_w_per_pe_eff_is_idle_share_plus_dynamic(self):
        pes, peak, idle_pct = PROFILES["rs700a"]
        idle = peak * idle_pct / 100
        assert w_per_pe_eff("rs700a") == pytest.approx(
            idle / pes + (peak - idle) / pes)

    def test_dc_profile_reads_config_keys(self):
        assert dc_profile({"datacenter_id": 0,
                           "host_count_spec_asus_rs500a": 10}) == ("rs500a", 10)
        with pytest.raises(ValueError):
            dc_profile({"datacenter_id": 9})

    def test_power_cap_binds_before_pe_cap(self):
        # DC0 H_d=595.93W / 3.34375 = 178.2 PEs << 640 available
        dcs = [{"datacenter_id": 0, "host_count_spec_asus_rs500a": 10}]
        mi = executable_mi_per_on_second(dcs)
        assert mi == pytest.approx((595.93 / (214.0 / 64)) * VM_MIPS, rel=1e-3)

    def test_brown_dcs_excluded(self):
        dcs = [{"datacenter_id": 3, "host_count_spec_asus_rs700a": 4}]
        assert executable_mi_per_on_second(dcs) == 0.0


class TestAudit:
    def _art(self, rows=20000, troughs=()):
        return {"rows": rows, "troughs": [{"start": s, "dur": d}
                                          for s, d in troughs]}

    def test_all_on_window_passes_when_capacity_ample(self):
        art = self._art(rows=WARMUP_ROWS + 200000)
        trace = [{"length": "1000000"}] * 10          # tiny load
        dcs = [{"datacenter_id": 0, "host_count_spec_asus_rs500a": 10}]
        res = audit(art, trace, dcs, off_range=0)
        assert res["pass"] and res["worst_window_ratio"] > 1.2

    def test_fails_when_load_exceeds_green_window(self):
        art = self._art(rows=WARMUP_ROWS + 200000)
        trace = [{"length": "1e15"}]                  # absurd load
        dcs = [{"datacenter_id": 0, "host_count_spec_asus_rs500a": 10}]
        res = audit(art, trace, dcs, off_range=0)
        assert not res["pass"]

    def test_trough_rows_do_not_count(self):
        rows = WARMUP_ROWS + 8000
        art_on = self._art(rows=rows)
        art_off = self._art(rows=rows,
                            troughs=((WARMUP_ROWS, 3600),))
        trace = [{"length": "1000000"}]
        dcs = [{"datacenter_id": 0, "host_count_spec_asus_rs500a": 10}]
        a = audit(art_on, trace, dcs, off_range=0)
        b = audit(art_off, trace, dcs, off_range=0)
        w_a = a["windows"][0]["on_seconds"]
        w_b = b["windows"][0]["on_seconds"]
        assert w_a == 7200 and w_b == 7200 - 3600
