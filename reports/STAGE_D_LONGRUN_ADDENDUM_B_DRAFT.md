# Stage D long run, Addendum B — DRAFT: moving the five seeds to Isambard-AI

Status: DRAFT, not in force. It applies only if the user decides to migrate; until then the long run stays on the workstation under the frozen §8. Written before any Isambard result exists.

## 1. Why

§8 R-u fixed the hardware as "the local RTX 5080, two lines at a time, seeds serial". Measured on that machine: 8 of 50 iterations in one hour, so 400k steps take about 6.5 h per pair, 13 h per seed, and 65 h (2.7 days) for five seeds. On Isambard-AI phase 2 one node carries four GH200s and 288 cores, so a seed's four lines run at once, and the five seeds run on five nodes at once. The workload is CPU-bound in the Java gateway (GPU utilisation on the workstation stays near zero), and the node has 36 cores per line against the workstation's two, so the expected wall-clock is a few hours for the whole run rather than three days.

## 2. What changes, exactly

- Hardware: Isambard-AI aip2 GH200 nodes. Every seed runs on the same GPU model, as R-u requires; the model is GH200 instead of RTX 5080. Hardware is still never tied to an arm.
- Job shape, decided by what the scheduler actually grants: on 2026-09-04 the cluster held 940 fully allocated and 358 partially allocated nodes, a whole four-GPU node never became free (Slurm moved the start time to Unknown), and the same job asking for two GPUs started immediately. The twenty runs (five seeds × four lines) are therefore submitted as **twenty single-GPU jobs** rather than five whole-node jobs. A seed's four lines then sit on whichever GH200s are free, possibly on different nodes of the same model. This is the same hardware environment in the sense R-u protects (identical accelerator, identical software stack, no arm tied to a machine), and it is the arrangement that gets all twenty runs resident at once instead of queued behind whole-node reservations.
- The init-hash equality check (N_V = V, N_E = E) then runs as a separate step over the four `checkpoint_init` directories of a seed once all four jobs have written them, before any of them is allowed past its first evaluation; a mismatch cancels that seed's four jobs.
- All five seeds are re-run from scratch on Isambard. Nothing from the workstation's partial seed 20260904 enters the analysis; it is archived, uninterpreted, as `logs/stage_d_longrun_SUPERSEDED_5080/`.
- Determinism controls unchanged and re-verified per seed: `PYTHONHASHSEED=0`, init-hash equality N_V = V and N_E = E, `checkpoint_init` before the first SGD step, keep-all checkpoints.
- Cluster-specific environment, recorded in each seed's freeze: `TMPDIR`/`RAY_TMPDIR` on `$SCRATCHDIR` (never `/run/user`, which systemd reclaims mid-job), `RAY_LIMIT_CPUS=16` (Ray otherwise starts 288 workers that stampede the NFS import), explicit `--mem=200G`, gateway launched from installDist.

## 3. What does not change

Budget (400k, 50 iterations, last checkpoint), five seeds, the six unread judgement windows, the ledger-aligned reward, the four lines, the primary corruption and negative controls, gates 1–5 with direction ≥ 4/5, the contract rules of §8, and the reader. The verdict is computed by the same `stage_d_longrun_verdict.py`.

## 4. Equivalence evidence required before the first seed

From `isambard/stage_d_smoke.sbatch` on a compute node, recorded in the migration manifest: CUDA available and the GH200 identified; two lines trained 8000 steps with one GPU each; `checkpoint_init` present and the N_V = V init hash equal on this hardware; one deployment evaluation row carrying every field the gate reads (`completion_rate_mi`, `ontime_mi_share`, `deadline_forced_count`, `total_carbon_kg`, `global_reward_sum`, `global_defer_action_rate`, `green_episode_offset_rows`, `ep_carbon_norm_clip_count`) and the registered window offset for its `--reset-skip`. Bit-identical results across hardware are not required and not claimed.

## 5. Provenance

Repository cloned from the public remote at the frozen commit; the wind data (1297 files) and traces travel in the clone. Per-seed `freeze.json` records the node, GPU model and driver, CUDA, PyTorch, Ray, `PYTHONHASHSEED`, jar SHA256, config SHA256 and the sixteen frozen source hashes, exactly as on the workstation. Gateway jar built on the cluster (aarch64), so its SHA differs from the workstation's by construction; both are recorded.
