# G1 Freeze Manifest

Frozen 2026-08-23. Everything a G1 run is allowed to depend on is pinned here.
Any run whose artefacts do not match this manifest is not a G1 run.

## Source

| item | value |
|---|---|
| repo commit | `4921ebd2992b1ffed5d4b7b193c7004100eaac18` |
| working tree | clean at freeze time (`git status --porcelain` empty) |
| Python entrypoint | `drl-manager/entrypoint_rlmodule_gtrxl.py` |
| config file | `config_C.yml`, sha256 `74b5e24bf3026f9a6c0c61a96670b55928521bbbc7f614bfe3a69b3e8e204cf1` |

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
| green / TimeCAP offset alignment at the three windows | **OPEN**, P0-C step 5 |

The third must close before any formal evaluation. Training may proceed in
parallel.

## Concurrency

Training saturates the box: 6 env-runners, one learner and a driver on 8 cores,
so trainings run one at a time. Evaluation does not: three concurrent cells
finished in 631 s against a 610 s solo baseline, a 2.9× throughput gain for a
3.4% per-cell slowdown. Evaluation is run three-way concurrent.
