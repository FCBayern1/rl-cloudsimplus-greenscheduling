import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_d_credit_cross import cross  # noqa: E402


def test_low_weight_on_negative_defer_advantages_and_guard_recovery():
    # DEFER: negative advantages get w 0.06 (erased), positive get 1.1; ROUTE all 1.0
    share = np.array([1, 1, 1, 1, 0, 0], float)
    adv = np.array([-1.0, -2.0, 0.5, 0.5, -1.0, 1.0])
    w = np.array([0.06, 0.06, 1.1, 1.1, 1.0, 1.0])
    r = cross(w, adv, share, eta=0.5)
    d = r["DEFER"]
    assert d["n"] == 4 and d["n_neg"] == 2 and abs(d["E_w_neg"] - 0.06) < 1e-12 and abs(d["E_w_pos"] - 1.1) < 1e-12
    assert d["P_low_neg"] == 1.0 and d["P_low_pos"] == 0.0
    assert abs(d["neg_mass_retained_raw"] - 0.06) < 1e-12
    assert abs(d["neg_mass_retained_guarded"] - 0.53) < 1e-12        # 1 + 0.5 (0.06 - 1)
    assert r["low_weight_falls_on_negative"] and r["guard_recovers_negative_mass"]
    assert r["ROUTE"]["neg_mass_retained_raw"] == 1.0


def test_no_effect_when_weights_are_uniform():
    share = np.array([1, 1, 0, 0], float)
    adv = np.array([-1.0, 1.0, -1.0, 1.0])
    w = np.ones(4)
    r = cross(w, adv, share)
    assert r["DEFER"]["P_low_neg"] == 0.0 and not r["low_weight_falls_on_negative"]
    assert not r["guard_recovers_negative_mass"]


def test_guard_gate_uses_actual_guarded_weights_and_bitwise_check():
    share = np.ones(200); adv = -np.ones(200); w = np.full(200, 0.5)
    r = cross(w, adv, share, eta=0.5, w_guarded=1.0 + 0.5 * (w - 1.0))
    g = r["guard_gate"]
    # R_raw 0.5 -> R_guarded 0.75: retained_ok fails (< 0.90) but recovery is exactly half the gap
    assert g["n_neg_ok"] and g["bitwise_ok"] and g["recovery_ok"] and not g["retained_ok"] and not g["pass"]
    r2 = cross(np.full(200, 0.9), adv, share, eta=0.5, w_guarded=np.full(200, 0.95))
    assert r2["guard_gate"]["pass"]
    r3 = cross(np.full(200, 0.9), adv, share, eta=0.5, w_guarded=np.full(200, 0.97))   # not the ruled formula
    assert not r3["guard_gate"]["bitwise_ok"]
    r4 = cross(np.full(50, 0.9), adv[:50], share[:50], eta=0.5)
    assert not r4["guard_gate"]["n_neg_ok"]
