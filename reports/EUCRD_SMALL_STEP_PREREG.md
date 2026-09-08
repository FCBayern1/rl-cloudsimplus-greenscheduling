# Small-step update trial: does one iteration with the forecast responsibility move the policy in a lower-carbon direction? (frozen 2026-09-08)

Frozen by the commit that adds this file, before any run. An **independent development experiment**, registered as the ruling of 2026-09-08 (evening) asked: it does not replace the frozen verdicts (G5 STOP, switch comparison, the matched pair's registration), it uses only the development (read) windows, and the 2020 confirmation windows stay sealed. Anything chosen from its reading still needs independent verification.

## 0. Why this and not zero-training screening

EU-CRD reweights training experience. With the model fixed, changing its parameters changes nothing at deployment, so the only cheap way to see whether the mechanism moves the policy in a useful direction is to let it take a small, fixed number of updates and deploy the result. This is not zero training, and it is cheaper than six 120 000-step runs.

## 1. Arms, all from the same checkpoint

Start state for every arm: the G5-err final global module (185 tensors, restored strictly through the warm-start path). The local policy and the learner's optimiser state and running estimators are not in that checkpoint and start fresh, identically for every arm — stated, not hidden. Same seed (20260910), so the first sampled batch is the same across arms because the sampling policy is the same.

| arm | forecast responsibility | reweighting | why |
|---|---|---|---|
| `ss_on` | on: `candidate_carbon_regret` | on, guardrail strength 0.5 as configured | the mechanism as it would train |
| `ss_off` | off: `instantaneous_carbon_cf`, which reads exactly zero under shrink75 (lead 0 exact) | on | the same machinery with the forecast channel silent |
| `ss_norw` | on | **off** (`reweight_advantages: false`) | shares computed but never applied: the "reweighting off" control the ruling named |
| `ss_base` | — | — | the unchanged checkpoint, zero updates |

Everything else identical: `perturb_tier: shrink75`, one PPO iteration (8 000 steps, 5 SGD passes over 2 048-minibatches), the last checkpoint of that iteration. **`reweight_warmup_calls` is set to 0 in every trained arm**, because one iteration (~100 loss calls) sits entirely inside the configured 450-call warmup and would make `ss_on` identical to `ss_off` by construction; this is disclosed as a deliberate departure from the training configuration, valid for this trial only.

## 2. Deployment

Deterministic decode (argmax, index ties), the six development windows, tiers godeye (clean) and shrink75 (the training contamination), one episode each. Read from the evaluation CSV: absolute carbon (`total_carbon_kg`, primary), completion (`completion_rate`, `completion_rate_mi`), on-time share (`ontime_mi_share`), forced deadline count.

## 3. Reading rules, fixed before running

Pooled over the six windows, per tier. Comparisons are `ss_on` against `ss_off` (the effect of the forecast responsibility), and each trained arm against `ss_base` (the effect of one iteration at all).

- **Clearly worse**: `ss_on` carbon higher than `ss_off` by more than 5 % at shrink75, with completion not higher. Worth finding the cause before any long run.
- **Clearly better**: `ss_on` carbon lower than `ss_off` by more than 5 % at shrink75, with completion not lower. Grounds to continue investing.
- **Little change**: anything else. This does **not** show long-run ineffectiveness; one iteration may not be enough to change a deterministic action.
- The clean tier is read the same way as a guard: a shrink75 gain paid for by a clean loss is reported as such.

The trial screens direction only. No verdict on the matched pair is taken from it by the numbers alone; the reading goes to a ruling.

## 3a. Addendum A — comparison conditions (2026-09-08, ruling received while `ss_on` was training; no deployment result had been produced)

1. **Primary pair is `ss_on` against `ss_norw`**, which differ in exactly one key (`crd.responsibility.reweight_advantages`, asserted by the generator): that isolates the effect of applying the responsibility weights. `ss_on` against `ss_off` differs in the forecast source only — the Q-ensemble is trained in both — and is kept as the broader control for the forecast channel. The §3 margins and the completion guard apply unchanged to the primary pair.
2. **Attribution to one update needs matched samples and optimiser state.** Optimiser state: fresh Adam moments in every arm, identical by construction, disclosed. Samples: the first iteration's batch is sampled by the same restored policy with the same seed before any update, so it should be identical across arms; this is **checked, not assumed** — the judge compares the iteration-1 sampling statistics (episode count, env steps, episode-return mean) across the three trained arms and reports a mismatch as a caveat on the whole reading. Restoring the module (Q heads included) is necessary but not sufficient for that identity.
3. **The pre-update checkpoint is the reference.** `ss_base` is deployed identically, so each trained arm is read against it: whether one iteration moved carbon at all, not only who beat whom.
4. **Reading order**: contracts first (completion, on-time share, forced deadlines), then same-window absolute carbon per window and pooled, then the clean tier for regression. No single best window and no mean-only reading.

The trial's reading serves the machine-time decision only; it carries no conclusion on whether EU-CRD is finally effective or ineffective.

## 4. What it cannot say

Nothing about the trained-from-scratch matched pair, nothing about seeds (one seed), nothing about the mechanism's long-run effect, and nothing about the 2020 windows. A "little change" reading leaves the pair's case exactly where the switch comparison left it.
