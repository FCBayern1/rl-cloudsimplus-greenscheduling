# RL CloudSimPlus Green Scheduling — Architecture Overview

## Purpose

Train reinforcement learning (RL) agents to schedule and scale VMs for datacenter workloads simulated by CloudSim Plus, with optional green energy modeling and rich training analytics.

## Tech Stack

- Python: Gymnasium, Stable-Baselines3 (SB3), sb3-contrib, PyTorch, Matplotlib
- Java: CloudSim Plus (simulation), exposed via Py4J gateway
- Analysis: Python scripts to parse logs and generate plots/reports

## Repository Layout

- Python RL manager
  - `drl-manager/mnt/entrypoint.py` — Main launcher: loads config, seeds RNGs, sets up logging, dispatches to `train.py`/`test.py`.
  - `drl-manager/mnt/train.py` — Training orchestration: env setup, algorithm selection, policy/hyperparams, callbacks, logging, and `model.learn`.
  - `drl-manager/mnt/test.py` — Evaluation/testing (structure mirrors training).
  - `drl-manager/mnt/utils/config_loader.py` — Loads `config.yml`, merges `common` with a selected experiment block into a single dict.
  - `drl-manager/mnt/callbacks/save_on_best_training_reward_callback.py` — Saves best model and logs per-episode details; avoids early `logger.dump` to keep `train/*` columns.
  - `drl-manager/gym_cloudsimplus/envs/loadbalancing_env.py` — Gymnasium Env that proxies actions/observations to Java via Py4J; provides action masks for MaskablePPO; computes episode-level aggregates for Monitor logging; idempotent shutdown.

- Java simulation (CloudSim Plus)
  - `cloudsimplus-gateway/src/main/java/giu/edu/cspg/LoadBalancerGateway.java` — Simulation orchestrator; calculates rewards; returns `SimulationStepResult` each step.
  - `cloudsimplus-gateway/src/main/java/giu/edu/cspg/SimulationCore.java` — Core setup/run loops (hosts, datacenter, broker, VMs/cloudlets).
  - `cloudsimplus-gateway/src/main/java/giu/edu/cspg/LoadBalancingBroker.java` — Queues cloudlets, event handling, wait time accounting.
  - `cloudsimplus-gateway/src/main/java/giu/edu/cspg/ObservationState.java` — Observation payload to Python (host/vm loads, queue, etc.).
  - `cloudsimplus-gateway/src/main/java/giu/edu/cspg/SimulationStepInfo.java` — Info map fields returned to Python for logging/analysis.
  - Energy: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/energy/*`.

- Configuration & docs
  - `config.yml` — Central config (defaults in `common`, experiment overrides under `experiment_*`).
  - `README.md`, `README_CN.md`, `docs/*` — Guides and analysis notes.

- Analysis
  - `drl-manager/analyze_training_complete.py` — Comprehensive plots (reward components, PPO losses/metrics, energy, success rates) and text report.
  - `drl-manager/analyze_training.py`, `data-analysis/*` — Additional analysis.

## Execution Flow

1. Entrypoint (`drl-manager/mnt/entrypoint.py`)
   - Reads env vars: `CONFIG_FILE` (default `config.yml`) and `EXPERIMENT_ID` (default `experiment_1`).
   - Loads/merges config, sets global seeds (Python/NumPy/Torch), and configures file + console logging under `logs/<type>/<experiment_name>/`.
   - Dynamically imports and runs mode function (`train`, `test`).

2. Training (`drl-manager/mnt/train.py`)
   - Creates Gym env with `env_id` (default `LoadBalancingScaling-v0`) and passes merged `params`.
   - Wraps with `Monitor` to record rewards/length and whitelisted `info` keys into `progress.csv`/`monitor.csv`.
   - Vectorizes with `DummyVecEnv` (or `SubprocVecEnv` for A2C).
   - Algorithm selection:
     - MaskablePPO (sb3-contrib) — supports action masks.
     - PPO/A2C/DQN/SAC/TD3 (SB3).
     - RecurrentPPO (sb3-contrib) — LSTM support (no masks).
   - Policy selection:
     - If `policy` set in config, use it.
     - Else infer by observation space: Dict → `MultiInputPolicy`, Box → `MlpPolicy`.
   - Hyperparameters: common + algorithm-specific (e.g., `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`).
   - Callbacks:
     - Best model saver based on Monitor mean reward.
     - Optional `EvalCallback`-based early stopping via `early_stop` config.
   - Runs `model.learn(total_timesteps=...)`, saves `final_model` and replay buffer where supported.

3. Testing (`drl-manager/mnt/test.py`)
   - Loads model from `train_model_dir`, runs evaluation episodes, logs results.

## Gym Environment (`loadbalancing_env.py`)

- Connects to Java via Py4J (`JavaGateway`). On reset/step, calls gateway methods.
- Observation (Gym Dict):
  - `vm_loads`: float array (size = num VMs)
  - `vm_available_pes`: int array (size = num VMs)
  - `waiting_cloudlets`: shape (1,)
  - `next_cloudlet_pes`: shape (1,)
- Action: `Discrete(N+1)` — 0 → no-assign (-1), 1..N → VM id 0..N-1.
- Action masking (MaskablePPO): `action_masks()` returns a boolean mask limiting invalid actions.
- Episode aggregates: accumulates per-step values and emits means on the terminal step (e.g., `episode_reward_*_mean`, `episode_avg_power_w`).
- Close: Idempotent; silences Py4J shutdown races and closes client/gateway safely.

## Java Gateway & Reward (`LoadBalancerGateway.java`)

- `step(int targetVmId)` (or variant with more action fields) advances simulation and returns:
  - `SimulationStepResult(observation, reward, terminated, truncated, info)`.
- Reward components (negative penalties):
  - Wait time: `-reward_wait_time_coef * log1p(avg_finished_wait_time)`.
  - Unutilization/balance: variance + deviation from a target utilization, scaled by `reward_unutilization_coef`.
  - Queue penalty: waiting/arrived ratio, scaled.
  - Invalid action penalty: scaled by `reward_invalid_action_coef`.
  - Energy: currently logged (power/energy/green ratio) but not part of reward total.
- `SimulationStepInfo.toMap()` exposes many metrics to Python Monitor (`progress.csv`).

## Configuration (`config.yml`)

- `common` — Base defaults for simulation, rewards, green energy, and RL hyperparameters.
- `experiment_*` — Per-experiment overrides (workload file, algorithm, timesteps, seeds, etc.).
- `early_stop` — Optional evaluation-based early stopping:
  - `type: no_improvement` with `patience`/`min_evals`, or `type: reward_threshold`.

### Key RL Parameters

- `algorithm`: `MaskablePPO`, `PPO`, `A2C`, `RecurrentPPO`, etc.
- `policy`: `MultiInputPolicy`/`MlpPolicy` or a custom class name if you implement one.
- `timesteps`: total env steps for training (`model.learn(total_timesteps=...)`).
- `n_steps`: rollout length (on-policy batch size before update).
- `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`.
- `policy_kwargs`: network structure (e.g., `net_arch`, `activation_fn`); for LSTM with `RecurrentPPO`: `lstm_hidden_size`, `n_lstm_layers`, `shared_lstm`.

## Algorithms & Policies

- MaskablePPO (sb3-contrib)
  - Supports action masks; use `MultiInputPolicy` for Dict observations.
  - Customize network via `policy_kwargs` (e.g., `net_arch: {pi: [...], vf: [...]}`, `activation_fn`).
  - LSTM + mask is not provided out-of-the-box — requires custom maskable recurrent policy.

- RecurrentPPO (sb3-contrib)
  - Supports LSTM policies (`MultiInputLstmPolicy` for Dict obs); no masking.
  - `policy_kwargs`: `lstm_hidden_size`, `n_lstm_layers`, `shared_lstm`, plus `net_arch` heads.

- SB3 PPO/A2C
  - Non-masked counterparts; `MultiInputPolicy` for Dict, `MlpPolicy` for Box.

## Logging & Artifacts

- Output path: `logs/<experiment_type_dir>/<experiment_name>/`
  - `progress.csv`: Monitor + SB3 logger (time, rollout, train/*, and selected `info` keys).
  - `monitor.csv`: Per-episode rows (SB3 Monitor format); includes episode aggregates exposed by env at the last step.
  - `best_model.zip` / `final_model` (and replay buffer if supported).
  - TensorBoard logs when enabled.

## Common Tasks

- Run training
  - PowerShell:
    - `cd d:\rl-cloudsimplus-greenscheduling`
    - `$env:EXPERIMENT_ID = 'experiment_3'`
    - `python drl-manager\mnt\entrypoint.py`

- Switch algorithm/policy
  - In `config.yml`, set `algorithm` and `policy` appropriately.
  - Dict observations → `MultiInputPolicy` / `MultiInputLstmPolicy`.

- Customize network (no code changes)
  - Use `policy_kwargs`:
    ```yaml
    policy_kwargs:
      net_arch:
        pi: [256, 256]
        vf: [512, 256]
      activation_fn: "torch.nn.SiLU"
    ```

- Custom Maskable policy (code changes)
  - Create `drl-manager/mnt/policies/custom_policy.py` and define `CustomMaskablePolicy` inheriting `MaskableActorCriticPolicy`.
  - In `train.py`, import and map `policy: "CustomMaskablePolicy"` from `config.yml`.

## Troubleshooting

- `train/*` columns missing in `progress.csv`
  - Ensure no early `logger.dump` before first update (fixed in custom callback).
  - Make sure `timesteps >= n_steps` so at least one optimization step occurs.

- Reward components blank in plots
  - Run training with the updated env that adds episode aggregates; re-run `analyze_training_complete.py`.
  - Energy penalty is 0 by design; to include energy in reward, modify `LoadBalancerGateway.calculateReward`.

- Py4J shutdown errors

  - Env `close()` is idempotent and silences `Py4JNetworkError` — safe to ignore; use latest code.

## Extending

- Add new info fields
  - Update Java `SimulationStepInfo.toMap()` and include keys in Monitor `info_keywords` in `train.py`.

- Add new observation features
  - Extend Java `ObservationState` and Python `_get_obs`/space definition in env.

- Early stopping
  - Configure `early_stop` in `config.yml` (`no_improvement` or `reward_threshold`).

## Notes on Recurrent + Masking

- If you need LSTM + action masking simultaneously, implement a custom maskable recurrent policy by adapting sb3-contrib’s recurrent policies to use mask-aware distributions (`MaskableDistribution`). This requires:
  - Managing `rnn_states` and `episode_starts` in the policy.
  - Returning masked action distributions in `forward`/`get_distribution`.
  - Preserving SB3’s policy interface so MaskablePPO can train with masks.

---

For deeper customization (e.g., Transformer-based features), consider adding a `CustomMaskablePolicy` template. If you want, we can scaffold this file under `drl-manager/mnt/policies/custom_policy.py` with placeholders for your network.

