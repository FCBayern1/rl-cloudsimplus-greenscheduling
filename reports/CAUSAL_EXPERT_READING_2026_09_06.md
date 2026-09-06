# Causal rolling expert: reading (2026-09-06)

Preregistration: reports/CAUSAL_EXPERT_PREREG.md, frozen at 6f7b5807 (a harness defect, the decision core accidentally nested inside the class, was fixed append-only at 62ef872b before any window ran). Run: `stage_a_out/causal_v1`, archive `reports/manifests/causal_v1/run1`. Verdict: **CAUSAL_READ**.

## 1. The expert

At every step it plans only the jobs the simulator has presented and it has not yet committed, on its own committed reservations and its own curve, as one proven-optimal MILP (version-2 model, envelope cuts), and emits (site, κ) on the dense grid. 35 decisions per window, every one proven optimal within seconds, no unsolved step, no fallback, contract green (completion and on-time 1.0, no forced start) on all 42 runs.

## 2. Gate A, reachability (certification twin, simulator-settled carbon, kg)

| window | causal expert (truth) | offline exact (truth) | offline flat (λ = 0) | capture of the offline headroom |
|---|---|---|---|---|
| k0 | 0.000698 | 0.000629 | 0.005616 | 0.986 |
| k1 | 0.001186 | 0.001028 | 0.005523 | 0.965 |
| k2 | 0.001347 | 0.001225 | 0.006626 | 0.978 |
| k3 | 0.000549 | 0.000489 | 0.006387 | 0.990 |
| k4 | 0.000545 | 0.000525 | 0.002100 | 0.988 |
| k5 | 0.002635 | 0.002159 | 0.007981 | 0.918 |

Pooled capture 0.968 (gate 0.80); six of six windows ≥ 0.70 (gate ≥ 5). Knowing every future job is worth 3.2 % of the headroom on this scene; the rest is reachable with arrived jobs only.

## 3. Gate B, causal error (pooled over six windows, loss vs the expert on truth)

| rung | pooled loss (kg) | share of the expert's truth carbon | windows harmed |
|---|---|---|---|
| shrink λ = 0.75 | 0.003604 | 51.8 % | 6 / 6 |
| shrink λ = 0.5 | 0.014045 | 202 % | 6 / 6 |
| shrink λ = 0.25 | 0.026801 | 385 % | 6 / 6 |
| shrink λ = 0 | 0.030277 | 435 % | 6 / 6 |
| shuffle | 0.024346 | 350 % | 6 / 6 |
| anti | 0.030555 | 439 % | 6 / 6 |

Gate B (λ = 0.75 harms on ≥ 5/6 windows and ≥ 5 % pooled): PASS. The causal profile tracks the offline one (41.5 / 211 / 444 / 465 / 419 / 496 %) rung by rung.

## 4. What this establishes

The certified ladder's value is reachable online, and forecast error hurts an expert that has only the information an RL policy has, at every rung, on every window, with the mildest rung costing half again the truth carbon. Both bridges the ruling asked for are crossed. Nothing is claimed about a learned policy; F1–F3 (frozen f81328ca) run next with this expert's decisions as labels; the 2020 windows stay sealed.
