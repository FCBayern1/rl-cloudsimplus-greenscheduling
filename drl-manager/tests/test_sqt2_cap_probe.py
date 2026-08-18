from sqt2_cap_probe import VM_GRID, ARMS, squeeze


class TestCapProbe:
    def test_grid_totals_match_labels(self):
        for cap, (s, m, l) in VM_GRID.items():
            per_dc = s * 2 + m * 4 + l * 8
            assert per_dc * 8 == cap

    def test_squeeze_overrides_every_dc(self):
        cfg = {"datacenters": [{"initial_s_vm_count": 100,
                                "initial_m_vm_count": 50,
                                "initial_l_vm_count": 25} for _ in range(8)]}
        squeeze(cfg, 224)
        for dc in cfg["datacenters"]:
            assert (dc["initial_s_vm_count"], dc["initial_m_vm_count"],
                    dc["initial_l_vm_count"]) == (6, 2, 1)

    def test_preregistered_arms(self):
        assert ARMS == ("nowait", "naive", "clairvoyant")
