# G1 Freeze Manifest

Frozen 2026-08-23. Everything a G1 run is allowed to depend on is pinned here.
Any run whose artefacts do not match this manifest is not a G1 run.

## Source

| item | value |
|---|---|
| repo commit | `4921ebd2992b1ffed5d4b7b193c7004100eaac18` |
| working tree | clean at freeze time (`git status --porcelain` empty) |
| Python entrypoint | `drl-manager/entrypoint_rlmodule_gtrxl.py` |
| config file | `config_C.yml`, sha256 `11f52e052711fa9569a491294129e0b14a510604abb1286f11d22ec5519e67f5` |

## Frozen gateway

The build directory is not the authority. A rebuild by any worktree, on this
machine or another, must not reach a G1 run, so the installDist was copied out
and made read-only. Every training and evaluation command sets `GATEWAY_LIBS`
to this path explicitly.

| item | value |
|---|---|
| `GATEWAY_LIBS` | `/home/joshua/frozen/g1_gateway/lib` (mode `dr-xr-xr-x`, files `r--r--r--`) |
| main jar sha256 | `aba6f0edf473871406e96e9a4f1f2375b5976f3f9be67c4ee3d5fb665962498e` |
| all 16 jars | `/home/joshua/frozen/g1_gateway/SHA256SUMS` |
| built from | gateway HEAD, measured against `61043cf` at −0.28% carbon, 100% completion |

The 3060 may keep changing Java in its own worktree. Later changes do not enter
this round by construction, because nothing in G1 reads the build directory.

## Arms

| arm | config key | sha256 of resolved block |
|---|---|---|
| EU-CRD | `experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b` | `b1cbbb353d2440731b7594b2e87e15c94ea01f2bbd743eebba173c0e3df21013` |
| matched Vanilla | `experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan` | `af12ee169d64e3d7085f73f3e237236714aa613b08389364f086f11a431951e9` |

The two blocks differ in exactly `crd.enabled`, `experiment_name`,
`simulation_name` (`drl-manager/verify_matched_vanilla.py`, 147/147 keys,
differing=3, PASS).

Evaluation uses separate blocks, because training is deliberately open-book on
one window (`green_episode_offset_range` unset) while evaluation must reach the
three registered ones.

| arm | eval config key | sha256 |
|---|---|---|
| EU-CRD | `experiment_g1eval_knSV3b` | `e52944c70563b544cecd6ea9ab73fbe1128d2cef10faec03ea050da4ecda2134` |
| matched Vanilla | `experiment_g1eval_matchedvan` | `7d6eebc0b05dacd5c94d3f40728e3e3ac40f28dc413da546f0d822080afc04ab` |

Each is generated as an exact copy of its training block plus
`green_episode_offset_range`, and `tests/test_g1_eval_blocks.py` asserts the
difference set is exactly that one key, that the matched-arm property survives
the copy, and that no G1 runner mentions the old blocks.

The old evaluation block for the Vanilla arm, `experiment_p0cprobe_van`, was
**not** a copy of the matched Vanilla. It was a stale pre-v5 block differing in
twelve keys, eight of them objective-level: `carbon_penalty_mode`,
`per_action_carbon_weight`, `per_action_completion_weight`, `global_reward_beta`,
`global_completion_rate_mi_coef`, `carbon_normalization_fixed_max`, and two model
hyperparameters. That is the same family of parameters that invalidated the
original Vanilla comparator, so the error P0-B was built to catch was sitting
unguarded in the evaluation path. It is retired. Every P0-C feasibility number
carrying the `van` label came from it and is not a matched-Vanilla measurement.
`experiment_p0cprobe_knSV3b` was clean (differing=1) but is retired with it.

## Fixed artefacts

| artefact | sha256 |
|---|---|
| workload `traces/probe_C_2xjob_dl6500.csv` | `641041f2ab221e6a354f20a613e549ad04af38c6fca6cc945e1e2525bdb32c67` |
| TimeCAP `ckpt_best.pth` | `fa86c59df99d4fa0228ba07e018bdd399017e5e1f673edc316032a5871a9fb59` |
| `Turbine_12_2021.csv` | `e6512599834c352a…` |
| `Turbine_36_2021.csv` | `14452bd95440401e…` |
| `Turbine_95_2021.csv` | `32ac9baf98d1ac88…` |
| `Turbine_91_2021.csv` | `6279066901ab3053…` |
| `Turbine_96_2021.csv` | `ff5df78a51c09f1d…` |

TimeCAP runs on `device: cpu`. This is not a performance choice, it is required.

## Seed table

Twelve pairs, seeds 101–112. Odd seeds run Vanilla first, even seeds run EU-CRD
first, so arm order is not confounded with seed. Both members of a pair run on
the same machine, the same jar and the same commit. Pairs may sit on different
machines: the seed-level paired log-ratio estimator differences the machine out.

| stage | seeds |
|---|---|
| futility gate | 101, 102, 103, 104 |
| expansion, only if the gate passes | 105–112 |

The gate is a stop rule, not a peek. Four pairs complete, then the decision is
to stop or to run the fixed remaining eight. No effect-driven resizing.

`smoke_matchedvan_s101` (2026-08-23, 7h11m, checkpoint_000010) is a smoke run
on the pre-freeze build. It is not a G1 run and does not count as seed 101.

## Checkpoint rule

Every arm trains 600k environment steps. The reported checkpoint is the final
one at 600k, always. No checkpoint is selected on any result, validation or
otherwise. Intermediate checkpoints are kept for diagnostics only.

## Evaluation constants

| item | value |
|---|---|
| decode | deterministic argmax (sampling appendix is separate and secondary) |
| eval RNG seed | 20260823 |
| conditions | clean, blend, shuffle |
| windows | the three registered green-scarcity windows |
| episodes | one per (seed, arm, condition, window) = 216 |
| estimator | pool the three windows, then seed-level `d_i = log(C_EU/C_Van)`, report `100(exp(d̄)−1)` |
| primary carbon endpoint | Shuffle |
| SLA co-primary | `C_min,i ≥ 99.5%`, exact McNemar |

One episode per cell, not three. Under argmax with a fixed trace and a fixed
window, repeating an episode copies the trajectory, and the evaluator's
`--episodes 3` does not repeat a window at all: it advances to k+1 and k+2.

## Registered windows

Stratified by green scarcity under per-turbine read semantics. Turbine
timezones are 0 (T12, T36), 18 (T95, T91), 54 (T96). tz=108 belongs to
DC\_APAC, which carries no turbines, and must not appear in any span
computation.

| stratum | k | offset | mean total | percentile | overlap with training |
|---|---|---|---|---|---|
| low | 19 | 19171 | 1682.26 kW | p20.60 | 0 |
| mid | 56 | 11554 | 1834.17 kW | p49.06 | 0 |
| high | 34 | 34306 | 2021.10 kW | p81.84 | 0 |
| *(training)* | 0 | 0 | 1419.15 kW | p10.15 | — |

Percentiles are quoted against the 44950 offsets the schedule can produce.
Frozen in `drl-manager/calib/p0c_green_windows.json`, guarded by 14 tests that
recompute scarcity from the CSVs.

## Environment

| item | value |
|---|---|
| Python | 3.12.3 (`drl-manager/.venv`) |
| Ray | 2.40.0 |
| PyTorch | 2.11.0+cu130, CUDA 13.0 |
| Java | OpenJDK 21.0.11 |
| host | 5080, 8 cores, 60 GB RAM, RTX 5080 16 GB |

Required environment for every run:

    export GATEWAY_LIBS=/home/joshua/frozen/g1_gateway/lib
    export EVAL_CONFIG_PATH=/home/joshua/rl-cloudsimplus-greenscheduling/config_C.yml
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

`EVAL_CONFIG_PATH` is not optional. Without it the evaluator silently falls back
to a single-datacentre SWF environment and still exits 0.

## Sentinels

| sentinel | status |
|---|---|
| matched-config exact diff | PASS, 147/147 keys, differing=3 |
| VM dispersion (dispatcher wiring) | PASS, VMs carrying 90% of load 65 → 155 |
| green / TimeCAP offset alignment at the three windows | **OPEN**, P0-C step 5, rerunning on the corrected eval blocks |

The third must close before any formal evaluation. Training may proceed in
parallel.

## Warmup: 13 rows, measured

Java documents `sim_step=0` as CSV row `tz + simulation_warmup_rows`, and
`simulation_warmup_rows` is absent from every relevant block, so the documented
prediction is row `tz`. The simulator in fact consumes 13 rows during startup
before it emits the first observation. Cross-correlating the observed per-DC
green series against the CSV puts the peak at lag 13 with r = 1.0000 in 9 of 9
cells, identically across all three windows and all three green datacentres.

The registered windows therefore read `[offset + 13 + tz_i, + 7200)`. Reading
the constant off the config gives 0 and silently shifts every window by 13 rows.
The artifact records the measured value and its provenance, and a test asserts
both.

## Java fast path is inactive, on every run

`hierarchical_multidc_env.py` tries `getStepAsFlatMap()` and falls back to the
legacy 200-getter parser when it is absent. The method does not exist in the
Java source at all, so every run on this testbed uses the legacy path: the A/B
dispatcher probe, the P0-C feasibility runs, the concurrency probe and the smoke
training all log the fallback. This is the status quo, not a property of the
frozen jar. It is recorded here so that implementing the fast path mid-campaign
is recognised as changing the data path, not as a speed-up.

## Concurrency

Training saturates the box: 6 env-runners, one learner and a driver on 8 cores,
so trainings run one at a time. Evaluation does not: three concurrent cells
finished in 631 s against a 610 s solo baseline, a 2.9× throughput gain for a
3.4% per-cell slowdown. Evaluation is run three-way concurrent.
