Invalidated by an implementation fault, not a finding.

The 198 recorded loss calls all fell inside the 450-call reweight warmup, where
_compute_responsibilities returns before computing the weights. Both variants therefore fell
back to unit weights, so w_delta was exactly 0 everywhere and the STOP verdict says nothing
about the mechanism. EUCRD_SWITCH_AB_PREREG section 1 requires the comparison to score both
variants with their weights applied regardless of the warmup counter; the implementation did
not do that. Fixed by a diagnostic-only warmup bypass (with the counter snapshotted and
restored), and the run repeated. The criteria W1-W3 were not touched.

One measurement from this run is kept: scoring the SAME advantages twice gave a relative
gradient difference of 0.003 (cosine about 0.999996), so the gradient comparison has a
non-zero noise floor. The repeat run records that null baseline per call.
