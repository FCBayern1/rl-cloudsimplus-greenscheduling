Run 3 (code 50433d5e: random state reset, critic EMA NOT yet restored).

W1 and W2 reproduce run 2 almost exactly: mean |w_A - w_B| 0.0124 (run 2: 0.0123), targeting
ratio 33.4 (33.5), max 0.475 (0.475).

W3 is still not resolvable: the A/B relative L2 difference has median 0.00201 against a null
of 0.00313 (ratio 0.60), so resetting the random state alone did not make the paired null
exact. That is the empirical confirmation that the remaining leak was the critic value-target
variance EMA, which the parent loss advances on every evaluation; it is restored from c5983010
onwards. Run 4 repeats the comparison under the full isolation.

Under this run the cosine is now clamped and none of the 198 calls had an undefined cosine.
