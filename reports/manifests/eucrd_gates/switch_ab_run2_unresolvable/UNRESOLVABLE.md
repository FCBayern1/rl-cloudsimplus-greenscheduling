W1 and W2 pass, W3 is not resolvable.

Weights: mean |w_A - w_B| 0.0123, max 0.475; on firing cells 0.209 against 0.0062 off them,
a targeting ratio of 33.5 against the registered 2. The forecast responsibility does change
the weights, and it changes them where the forecast changed a decision.

Gradient: the A/B relative L2 difference has median 0.00203 while the null (the same
advantages scored twice) has median 0.00325, ratio 0.58, and only 24 percent of calls exceed
their own null. The gradient comparison is dominated by the loss path randomness, so W3 as
measured says nothing either way. Per the ruling of 2026-09-08 this is a diagnostic pairing
problem, not evidence that the mechanism is inert, and the responsibility coefficients are
not touched.

Fix: the random state is now reset to the same point before each of the three gradient
evaluations and restored afterwards, which makes the null exactly zero (asserted in a test).
Re-run under the same frozen W1-W3.
