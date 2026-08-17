"""P0-1 hierarchical deterministic decode - Codex-mandated test list
(V32B_ANNEAL_SPEC R1 item 1)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.rlmodule_gtrxl_models import hierarchical_deterministic_logits


def _slot_logits(p_hold, route_probs):
    """Build one slot's 9-way normalized log-probs from (p, q)."""
    q = np.asarray(route_probs, dtype=np.float64)
    q = q / q.sum()
    probs = np.concatenate([(1 - p_hold) * q, [p_hold]])
    return np.log(np.maximum(probs, 1e-12))


def _decode(slots):
    logits = torch.tensor(np.concatenate([_slot_logits(*s) for s in slots]),
                          dtype=torch.float32).unsqueeze(0)
    out = hierarchical_deterministic_logits(logits, len(slots), 9)
    return out.reshape(len(slots), 9).argmax(-1).tolist()


class TestCodexMandatedCases:
    def test_diffuse_routing_bug_case(self):
        # p_hold=0.2, 8 uniform DCs: flat argmax defers (0.2 > 0.8/8);
        # hierarchical MUST route.
        slots = [(0.2, [1] * 8)]
        flat = torch.tensor(_slot_logits(*slots[0])).argmax().item()
        assert flat == 8                      # documents the flat-argmax bug
        assert _decode(slots)[0] != 8         # the fix routes

    def test_threshold_both_sides(self):
        assert _decode([(0.49, [1] * 8)])[0] != 8
        assert _decode([(0.51, [1] * 8)])[0] == 8

    def test_tie_goes_route(self):
        assert _decode([(0.5, [1] * 8)])[0] != 8

    def test_route_choice_preserved(self):
        # when routing, the chosen DC must be the route-argmax (DC 3 here)
        assert _decode([(0.3, [1, 1, 1, 9, 1, 1, 1, 1])])[0] == 3

    def test_per_slot_independence_and_padding_isolation(self):
        # slot 0 defers, slot 1 routes; transforming one never leaks into the
        # other (padding slots are just slots the env ignores - same property)
        acts = _decode([(0.9, [1] * 8), (0.1, [1, 5, 1, 1, 1, 1, 1, 1])])
        assert acts == [8, 1]

    def test_high_confidence_defer_kept(self):
        assert _decode([(0.95, [8, 1, 1, 1, 1, 1, 1, 1])])[0] == 8
