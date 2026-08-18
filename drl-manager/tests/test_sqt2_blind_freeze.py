from sqt2_blind_freeze import ARMS, CANDIDATE_QS, arm_spec, freeze_by_carbon


def rec(arm, carbon, c7200=0.999, term=1.0, ep=0):
    return {"arm": arm, "episode_index": ep, "total_carbon_kg": carbon,
            "carbon_at_7200": carbon * 0.98, "completion_at_7200": c7200,
            "completion_rate_mi": term}


class TestArmSpec:
    def test_hazard_labels_carry_q(self):
        assert arm_spec("hazard@0.25") == ("hazard", 0.25)
        assert arm_spec("naive") == ("naive", 0.5)
        assert arm_spec("nowait") == ("nowait", 0.5)

    def test_preregistered_arm_set(self):
        assert ARMS == ("nowait", "naive", "hazard@0.25", "hazard@0.40",
                        "hazard@0.50", "hazard@0.60")
        assert CANDIDATE_QS == (0.25, 0.40, 0.50, 0.60)


class TestFreezeByCarbon:
    def test_winner_is_min_median_carbon_among_eligible(self):
        records = ([rec("nowait", 0.10, ep=i) for i in range(3)]
                   + [rec("naive", 0.09, ep=i) for i in range(3)]
                   + [rec("hazard@0.50", 0.08, ep=i) for i in range(3)])
        stats, winner = freeze_by_carbon(records)
        assert winner == "hazard@0.50"
        assert stats["hazard@0.50"]["median_terminal_carbon"] == 0.08

    def test_nowait_is_control_not_candidate(self):
        records = ([rec("nowait", 0.01, ep=i) for i in range(3)]
                   + [rec("naive", 0.09, ep=i) for i in range(3)])
        _, winner = freeze_by_carbon(records)
        assert winner == "naive"

    def test_dual_sla_filter_drops_7200_violator(self):
        records = ([rec("naive", 0.05, c7200=0.99, ep=0)]     # cheap but late
                   + [rec("naive", 0.05, ep=1)]
                   + [rec("hazard@0.50", 0.09, ep=i) for i in range(2)])
        stats, winner = freeze_by_carbon(records)
        assert winner == "hazard@0.50"
        assert not stats["naive"]["dual_sla_all_anchors"]

    def test_dual_sla_filter_drops_terminal_violator(self):
        records = ([rec("naive", 0.05, term=0.99, ep=0)]
                   + [rec("hazard@0.50", 0.09, ep=0)])
        _, winner = freeze_by_carbon(records)
        assert winner == "hazard@0.50"

    def test_empty_filter_returns_none_never_softens(self):
        records = [rec("naive", 0.05, c7200=0.98, ep=0),
                   rec("hazard@0.50", 0.06, c7200=0.97, ep=0)]
        stats, winner = freeze_by_carbon(records)
        assert winner is None
        assert stats["naive"]["min_completion_at_7200"] == 0.98

    def test_tie_break_by_label_accuracy(self):
        records = ([rec("hazard@0.40", 0.08, ep=i) for i in range(2)]
                   + [rec("hazard@0.50", 0.08, ep=i) for i in range(2)])
        acc = {"hazard@0.40": {"acc": 0.77}, "hazard@0.50": {"acc": 0.78}}
        _, winner = freeze_by_carbon(records, accuracies=acc)
        assert winner == "hazard@0.50"
