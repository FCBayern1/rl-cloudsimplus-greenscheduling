"""Blind arms must be CAUSAL (2026-08-20, 5080 diagnosis).

offline_proof_tvci3's blind arms select their release time by scanning the
whole realised window, e.g.

    ow = np.flatnonzero(onset[win])          # sees the entire future
    R["onset_wait"][i] = win[ow[0]] if ow.size else win[gi[0]]

which is not a blind policy: it knows in advance whether a green onset will
arrive inside the budget, and only falls back to "release now" when it knows
none will. A blind policy must commit online with the current observation
plus the registered distributions, and eat the forced release when the
budget expires.

Consequence: the DP (the only genuinely blind arm) appeared to FAIL its
correctness check by 6%, while against the causal version of the same rule
it wins by ~50%. Every "blind" arm therefore has to be causal, or the
measured value of information is really the value of OPTIMISATION inside a
clairvoyant information set.
"""
import numpy as np
import pytest


def _causal_onset(g, onset, lo, hi):
    """Release at the first onset actually encountered, else at budget end."""
    for t in range(lo, hi + 1):
        if onset[t]:
            return t
    return hi


def _peeking_onset(g, onset, lo, hi):
    """The implementation under test: scan the window, fall back to first green."""
    w = np.arange(lo, hi + 1)
    ow = np.flatnonzero(onset[w])
    if ow.size:
        return int(w[ow[0]])
    gi = np.flatnonzero(g[w] == 1)
    return int(w[gi[0]]) if gi.size else hi


class TestPeekingIsNotBlind:
    def test_fallback_reveals_the_future(self):
        # green now (mid-window), no onset inside the budget: the peeking rule
        # releases NOW because it has verified no onset is coming; a causal
        # rule cannot know that and must wait out its budget.
        T = 100
        g = np.zeros(T, dtype=int); g[:30] = 1        # green only at the start
        onset = np.zeros(T, dtype=bool); onset[0] = True
        lo, hi = 10, 60                                # arrive mid green window
        assert _peeking_onset(g, onset, lo, hi) == 10  # "release now, it is green"
        assert _causal_onset(g, onset, lo, hi) == 60   # waits, then forced

    def test_agree_when_an_onset_exists(self):
        T = 100
        g = np.zeros(T, dtype=int); g[40:80] = 1
        onset = np.zeros(T, dtype=bool); onset[40] = True
        assert _peeking_onset(g, onset, 10, 60) == 40
        assert _causal_onset(g, onset, 10, 60) == 40

    def test_causal_never_beats_peeking(self):
        """The peeking rule is an upper bound on its own causal version."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            T = 200
            g = (rng.random(T) < 0.5).astype(int)
            onset = np.zeros(T, dtype=bool)
            onset[1:] = (g[1:] == 1) & (g[:-1] == 0)
            lo = int(rng.integers(0, 100)); hi = lo + int(rng.integers(5, 80))
            p, c = _peeking_onset(g, onset, lo, hi), _causal_onset(g, onset, lo, hi)
            # peeking releases no later than causal, and into green whenever
            # causal does (it has strictly more information)
            assert p <= c or g[p] >= g[c]
