# Availability limitation

Only checkpoint_000014 is locally complete for primary Isambard V seed 20260911.
The planned intermediate primary replay attempts failed at module loading, before
starting a simulator. The three final-checkpoint rows did run. These missing
checkpoints are not scientific failures and no intermediate primary curve can be
inferred from them. The complete 15-checkpoint bisect is the separately labelled
local-replication trajectory. The diagnostic launcher now validates every selected
checkpoint before launching any jobs.
