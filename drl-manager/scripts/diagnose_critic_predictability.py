#!/usr/bin/env python3
"""
Diagnostic: is the global agent's discounted return PREDICTABLE from its obs?

Procedure
---------
1. Build the env (godeye mode, no GPU).
2. Run 1 episode with a RoundRobin global policy + random-valid local actions,
   logging per-step global obs (flattened) and per-step global reward.
3. Compute discounted returns (γ matches training).
4. Fit several sklearn regressors from obs features → return on a 70/30 split.
5. Report test R²:

       R² > 0.3   → returns ARE predictable from obs; the critic architecture
                    (mean-pooling) is throwing the signal away.  Fix critic.
       R² ≈ 0     → returns are genuinely unpredictable from current obs
                    (exogenous arrival noise dominates); the critic cannot
                    help in this problem.  Stop chasing it.

Notes
-----
Uses Round-Robin to drive the env: any persistent state→return signal (queue
buildup, diurnal green, in-flight cloudlet finish times) should still produce
non-trivial R² since the OBS is informative; only the RR action choices are
uninformed.  The TRAINED policy would give different returns but the
predictability ceiling depends on the obs, not the policy.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
log = logging.getLogger("critic-predictability")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2")
    ap.add_argument("--gamma", type=float, default=0.995,
                    help="discount used at training time (PPO gae_gamma)")
    ap.add_argument("--max-steps", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.baselines.evaluate import load_config, _apply_overrides
    from src.training.bc_warmstart import _build_pettingzoo_env
    from src.baselines.global_schedulers import RoundRobinGlobalScheduler

    cfg = load_config(args.experiment)
    cfg = _apply_overrides(cfg, ["green_oracle_mode=godeye"])
    cfg.pop("py4j_port", None)
    cfg["gateway_log_dir"] = "/tmp/critic_diagnostic"

    log.warning(f"Building env (experiment={args.experiment}, godeye, no GPU)…")
    env = _build_pettingzoo_env(cfg)
    num_dcs = env.num_datacenters
    batch_size = env.global_routing_batch_size
    log.warning(f"env: N_dc={num_dcs}, batch_size={batch_size}")

    rr = RoundRobinGlobalScheduler(num_dcs, batch_size)
    rng = np.random.default_rng(args.seed)
    obs, _ = env.reset(seed=args.seed)

    obs_history = []
    reward_history = []

    for step in range(args.max_steps):
        gobs = obs["global_agent"]["observation"]
        # Flatten the global obs dict into a single feature vector.
        flat = []
        for k in sorted(gobs.keys()):
            v = np.asarray(gobs[k]).flatten().astype(np.float32)
            flat.extend(v.tolist())
        obs_history.append(np.array(flat, dtype=np.float32))

        # Build hierarchical actions: RR for global, random-valid for each local.
        rr_action = rr.schedule(gobs)
        step_actions = {"global_agent": np.asarray(rr_action)}
        for i in range(num_dcs):
            mask = obs[f"local_agent_{i}"]["action_mask"]
            valid = np.where(np.asarray(mask) > 0.5)[0]
            step_actions[f"local_agent_{i}"] = int(rng.choice(valid)) if len(valid) > 0 else 0

        obs, rewards, terms, truncs, _ = env.step(step_actions)
        reward_history.append(float(rewards["global_agent"]))

        if terms.get("__all__") or truncs.get("__all__"):
            break
        if step % 1000 == 0 and step > 0:
            log.warning(f"  step {step}: avg reward so far = {np.mean(reward_history):.3f}")

    try:
        env.close()
    except Exception:
        pass

    n = len(reward_history)
    print(f"\n=== Rollout summary ===")
    print(f"  steps: {n}")
    print(f"  per-step global reward:  mean={np.mean(reward_history):.3f}  std={np.std(reward_history):.3f}")

    # Compute discounted return at each step (reverse cumulative).
    rewards_arr = np.array(reward_history, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    G = 0.0
    for i in range(n - 1, -1, -1):
        G = rewards_arr[i] + args.gamma * G
        returns[i] = G

    print(f"  discounted return (γ={args.gamma}):  mean={returns.mean():.1f}  std={returns.std():.1f}")
    print(f"  Var(returns) = {returns.var():.1f}")
    print(f"  → if V predicts mean, vf_loss ≈ Var(returns); training showed vf_loss ≈ 2500")
    print(f"  → if V predicts the FULL signal, vf_loss ≈ 0 and R² ≈ 1")

    X = np.stack(obs_history)
    print(f"  feature dim: {X.shape[1]}")

    # Skip first/last 200 steps to avoid burn-in / tail artefacts; use middle.
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    # Use middle window to avoid boundary effects.
    burn = 200
    X_use = X[burn:n - burn]
    y_use = returns[burn:n - burn]
    print(f"  using middle {len(y_use)} steps for fitting (skip {burn} burn-in + {burn} tail)")

    # 70/30 sequential split (NOT random, to avoid temporal leakage).
    split = int(0.7 * len(y_use))
    X_train, X_test = X_use[:split], X_use[split:]
    y_train, y_test = y_use[:split], y_use[split:]

    print(f"\n=== Predictability test (obs → discounted return) ===")
    print(f"  baseline: predict-mean R²    =  {r2_score(y_test, np.full_like(y_test, y_train.mean())):.4f}")
    for name, model in [
        ("LinearRegression       ", LinearRegression()),
        ("Ridge(alpha=10.0)      ", Ridge(alpha=10.0)),
        ("GradientBoosting(d=4)  ", GradientBoostingRegressor(
            max_depth=4, n_estimators=300, learning_rate=0.05, random_state=0)),
    ]:
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            print(f"  {name}  test R² = {r2:+.4f}")
        except Exception as e:
            print(f"  {name}  FAILED: {e}")

    print(f"\n=== Interpretation ===")
    print(f"  R² > 0.3   → returns ARE predictable; critic mean-pooling drops the signal.")
    print(f"                 Action: fix critic architecture (use non-pooled features).")
    print(f"  R² ∈ [0,0.3] → partial signal; critic could help but not dramatically.")
    print(f"  R² ≤ 0     → returns NOT predictable from obs; H3 (exogenous noise).")
    print(f"                 Action: stop chasing critic, accept vf_explained_var ≈ 0.")


if __name__ == "__main__":
    main()
