# margin = 4, low window — superseded, never compared

Superseded by source-derived backstop alignment before any valid carbon comparison.

The fixed-margin grid {2,4,8,16,32,64} could not succeed. The active backstop is the
legacy fixed-lead rule with defer_deadline_slack_sec = 600 s, which fires 600 steps
before the deadline regardless of runtime, and no margin in that grid reaches 600. These
cells are kept as evidence of the search and must not enter any comparison. The run was
stopped part way, so several cells are incomplete.

margin = 2 carried the same defect. Its curve arm showed carbon 0.071052 against the
strongest passing blind at 0.125985, a 43.6% reduction. That is an INVALID diagnostic:
the curve arm failed its own contract with 1618 forced routes. It does not enter the
paper and did not inform the replacement rule, which is derived from the source and the
configuration alone.
