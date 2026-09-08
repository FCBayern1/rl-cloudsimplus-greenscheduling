Run 3 (code 50433d5e: random state reset, critic EMA NOT yet restored).

W1 and W2 reproduce run 2 almost exactly: mean |w_A - w_B| 0.0124 (run 2: 0.0123), targeting
ratio 33.4 (33.5), max 0.475 (0.475).

W3 is still not resolvable: the A/B relative L2 difference has median 0.00201 against a null
of 0.00313 (ratio 0.60), so resetting the random state alone did not make the paired null
exact. That is all run 3 establishes. A source audit of the loss chain found one remaining
state write, the critic value-target variance EMA that the parent loss advances on every
evaluation; it is restored from c5983010 onwards. Whether that write is what produced the
non-zero null is only supported if run 4, under the full isolation, brings the null to zero.

Under this run the cosine is now clamped and none of the 198 calls had an undefined cosine.
