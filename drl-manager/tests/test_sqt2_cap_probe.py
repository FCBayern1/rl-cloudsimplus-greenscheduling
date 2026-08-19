import pytest

from sqt2_cap_probe import ARMS, PE_PER_VM, scale_fleet


def cfg():
    """Baseline SQT2 fleet: 600/480/296/240/184/296/240/184 = 2520 PE."""
    tri = [(100, 50, 25), (80, 40, 20), (50, 25, 12), (40, 20, 10),
           (30, 15, 8), (50, 25, 12), (40, 20, 10), (30, 15, 8)]
    return {"datacenters": [{"initial_s_vm_count": s, "initial_m_vm_count": m,
                             "initial_l_vm_count": l} for s, m, l in tri]}


class TestScaleFleet:
    def test_identity_scale_reproduces_baseline(self):
        c = cfg()
        assert scale_fleet(c, 1.0) == 2520
        assert c["datacenters"][0] == {"initial_s_vm_count": 100,
                                       "initial_m_vm_count": 50,
                                       "initial_l_vm_count": 25}

    def test_half_scale_halves_fleet(self):
        assert scale_fleet(cfg(), 0.5) == pytest.approx(1260, abs=8)

    def test_heterogeneity_preserved(self):
        c = cfg()
        scale_fleet(c, 0.35)
        pes = [2 * d["initial_s_vm_count"] + 4 * d["initial_m_vm_count"]
               + 8 * d["initial_l_vm_count"] for d in c["datacenters"]]
        assert pes[0] > pes[1] > pes[2] > pes[3] > pes[4]   # order intact
        assert len(set(pes)) > 3                            # not homogenised

    def test_never_zeroes_a_tier(self):
        c = cfg()
        scale_fleet(c, 0.01)
        for d in c["datacenters"]:
            assert min(d.values()) >= 1

    def test_pe_weights_match_config_semantics(self):
        assert PE_PER_VM == {"s": 2, "m": 4, "l": 8}

    def test_preregistered_arms(self):
        assert ARMS == ("nowait", "naive", "clairvoyant")
