# Isambard-AI equivalence smoke for the Stage D long run (2026-09-04)

Job 6301901 on aip2 node nid010259, account brics.u6tx, 2 GH200 / 72 CPU / 100 GB, elapsed 26:02, state COMPLETED. Repository commit b6d9d7ef, gateway jar built on the cluster, SHA256 prefix b291f47a67833a51 (the workstation's jar differs by construction: different architecture). Environment: conda env `sd`, Python 3.12.14, OpenJDK 21, torch 2.11.0+cu126, ray 2.40.0, pyarrow 22.0.0, msgpack 1.1.2, numpy 2.4.4, gymnasium 1.0.0, pettingzoo 1.24.3, `PYTHONHASHSEED=0`, `RAY_LIMIT_CPUS=16`, TMPDIR and RAY_TMPDIR on $SCRATCHDIR.

## Checks required by the draft addendum §4

| check | result |
|---|---|
| CUDA on the compute node | `True`, NVIDIA GH200 120GB, capability (9, 0) |
| two lines trained, one GPU each, 8000 steps | both completed, `checkpoint_init` written before the first SGD step, one periodic checkpoint each |
| init weight hashes N_V = V on this hardware | `a55d8802635e` = `a55d8802635e`, MATCH |
| **same init hash as the workstation** | the RTX 5080 run of seed 20260904 also produced `a55d8802635e`; identical initial weights across x86-64 and aarch64 |
| one deployment evaluation row, judgement window | every gate field present, `MISSING none` |
| the window actually replayed | `green_episode_offset_rows = 13016`, the first registered judgement offset for `--reset-skip 0` |

Evaluation row: episode_length 237, completion_rate_mi 1.000, ontime_mi_share 1.000, deadline_forced_count 0, total_carbon_kg 0.006953, global_reward_sum −118.76, global_defer_action_rate 0.0476, ep_carbon_norm_clip_count 0.

## Throughput, measured

| platform | 8000 steps (one PPO iteration) | 400k per line | 20 lines (5 seeds × 4) |
|---|---|---|---|
| workstation, RTX 5080, 2 lines in parallel | about 8 min | about 6.7 h | about 67 h |
| Isambard, GH200, one line per GPU | 23.9 min (first iteration, includes start-up) | about 20 h | about 20 h, all 20 jobs resident |

Per line Isambard is roughly three times slower: the simulation is single-threaded JVM work, and this cluster's cores are slower than the workstation's for it, with the gateway's log and checkpoint I/O on Lustre. The gain is parallelism, not speed. Queue limits: partition `workq` MaxTime unlimited, QoS `workq_qos` MaxWall 24 h, 256 jobs per user, so twenty single-GPU jobs fit but each 400k line needs about 20 h against a 24 h wall.

## Operational findings

- A whole four-GPU node never became free (940 nodes allocated, 358 partially allocated); the same job asking for two GPUs started immediately. Submit per line, one GPU each.
- The training entry point resolves a relative `--config` against its own directory, not the working directory; cluster scripts must pass absolute paths.
- `gradle-wrapper.jar` was excluded by the repository's `*.jar` rule, so a fresh clone could not build the gateway. Now tracked.
- The clone carries the wind data (1297 files) and traces, so no manual data transfer is needed.
