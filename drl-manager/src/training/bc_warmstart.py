"""
Behavioral-Cloning warm-start for the score-based global router.

The 2026-05-17 smoke (20260518_002024) showed Stage 3 + A+B finally beating
Round-Robin's c/c (2.069 vs 2.077 at iter 25) but only by a tiny margin, and
only after PPO spends ~10 iters relearning load balancing from scratch.

BC warm-start short-circuits that:
  Phase 1 (this module): run env with RR for ~5k env steps, collect
    (global_obs, RR_action) pairs, train GTrXLScoreBasedGlobalRLModule to
    imitate RR via per-slot cross-entropy.  Result: a checkpoint whose
    policy already matches RR's stationary distribution.
  Phase 2 (handled by entrypoint_rlmodule_gtrxl.py): PPO resumes from this
    checkpoint and only has to refine "toward greener DCs".

Locals are NOT BC-warmed — they already learn well in PPO from scratch
(entropy 2.7 → 1.3 over 25 iter).  Adding BC for locals would invite
distribution-shift issues without obvious payoff.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from gym_cloudsimplus.envs import (
    HierarchicalMultiDCParallelEnv,
    HierarchicalMultiDCParallelEnvSimple,
    HierarchicalMultiDCParallelEnvAblation,
)
from src.baselines.global_schedulers import RoundRobinGlobalScheduler
from src.models.rlmodule_gtrxl_models import GTrXLScoreBasedGlobalRLModule

logger = logging.getLogger(__name__)


def _build_pettingzoo_env(env_config: Dict[str, Any]):
    """
    Pick the right PettingZoo wrapper for the given env_config.

    Mirrors `train_rlmodule_gtrxl.env_creator`: strip `py4j_port` so the env
    launches its own Java gateway on a free port rather than trying to connect
    to the config-declared default (which is rarely actually running).
    """
    env_config = dict(env_config)  # shallow copy — don't mutate caller's dict
    env_config.pop("py4j_port", None)
    env_id = env_config.get("env_id", "")
    if "Ablation" in env_id:
        return HierarchicalMultiDCParallelEnvAblation(config=env_config)
    if "Simple" in env_id:
        return HierarchicalMultiDCParallelEnvSimple(config=env_config)
    return HierarchicalMultiDCParallelEnv(config=env_config)


def collect_rr_rollout(
    env_config: Dict[str, Any],
    num_steps: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run env with RoundRobinGlobalScheduler + random-valid local actions, collect
    (global_obs, RR_action) tuples until ``num_steps`` env steps have elapsed.

    Returns:
        dict with keys:
          - "observations": List[Dict[str, np.ndarray]] — one global obs per step
          - "actions":      List[List[int]]            — RR action vector per step
          - "obs_space":    gym Space for the global agent's observation
          - "action_space": gym Space for the global agent's action
    """
    env = _build_pettingzoo_env(env_config)
    num_dcs = env.num_datacenters
    batch_size = env.global_routing_batch_size
    num_local_actions = env.max_actions

    rr = RoundRobinGlobalScheduler(num_dcs, batch_size)
    rng = np.random.default_rng(seed)
    obs, _info = env.reset(seed=seed)

    observations: List[Dict[str, np.ndarray]] = []
    actions: List[List[int]] = []
    steps = 0
    episodes = 0
    while steps < num_steps:
        global_inner = obs["global_agent"]["observation"]
        rr_action = rr.schedule(global_inner)
        observations.append({k: np.asarray(v) for k, v in global_inner.items()})
        actions.append(list(rr_action))

        # Random valid local actions (just to drain queues; locals are not
        # being BC-trained — global is the only target here).
        step_actions: Dict[str, Any] = {"global_agent": np.asarray(rr_action)}
        for i in range(num_dcs):
            mask = obs[f"local_agent_{i}"]["action_mask"]
            valid_idx = np.where(np.asarray(mask) > 0.5)[0]
            if len(valid_idx) > 0:
                step_actions[f"local_agent_{i}"] = int(rng.choice(valid_idx))
            else:
                step_actions[f"local_agent_{i}"] = 0

        obs, _rewards, terms, truncs, _infos = env.step(step_actions)
        steps += 1

        if terms.get("__all__") or truncs.get("__all__"):
            rr.reset()
            obs, _info = env.reset(seed=seed + episodes + 1)
            episodes += 1

    obs_space = env.observation_space("global_agent")
    action_space = env.action_space("global_agent")
    try:
        env.close()
    except Exception:
        logger.warning("env.close() raised — ignoring", exc_info=True)

    logger.info(
        f"BC rollout: collected {len(observations)} (obs, action) tuples "
        f"across {episodes + 1} episode(s)."
    )

    return {
        "observations": observations,
        "actions": actions,
        "obs_space": obs_space,
        "action_space": action_space,
    }


def _stack_observations(
    observations: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Stack a list of per-step obs dicts into (N, ...) tensors per key.
    """
    keys = sorted(observations[0].keys())
    stacked: Dict[str, torch.Tensor] = {}
    for k in keys:
        arr = np.stack([np.asarray(o[k]) for o in observations])
        stacked[k] = torch.from_numpy(arr).to(device).float()
    return stacked


def _bc_logits_to_per_slot(
    logits: torch.Tensor, num_batch_slots: int, num_dcs: int
) -> torch.Tensor:
    """Reshape (B, T=1, A=N_b*N_d) logits to (B, N_b, N_d) per-slot logits."""
    # squeeze the singleton time dim if present
    if logits.dim() == 3 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    return logits.reshape(-1, num_batch_slots, num_dcs)


def train_bc_policy(
    obs_space: spaces.Space,
    action_space: spaces.MultiDiscrete,
    model_config: Dict[str, Any],
    rollout: Dict[str, Any],
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
) -> Tuple[GTrXLScoreBasedGlobalRLModule, Dict[str, float]]:
    """
    Train the score-based global RLModule to imitate the RR actions in
    ``rollout``.  Returns the trained module and a small stats dict for
    logging (initial loss, final loss, accuracy).
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    nvec = [int(x) for x in action_space.nvec]
    num_dcs = nvec[0]
    num_batch_slots = len(nvec)
    if not all(n == num_dcs for n in nvec):
        raise ValueError(f"BC assumes uniform MultiDiscrete, got nvec={nvec}")

    spec = RLModuleSpec(
        module_class=GTrXLScoreBasedGlobalRLModule,
        observation_space=obs_space,
        action_space=action_space,
        model_config=dict(model_config),
    )
    module = spec.build().to(device_t)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

    obs_tensors = _stack_observations(rollout["observations"], device_t)
    actions_tensor = torch.tensor(rollout["actions"], dtype=torch.long, device=device_t)
    N = actions_tensor.shape[0]

    initial_loss: Optional[float] = None
    final_loss = 0.0
    final_acc = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device_t)
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_obs = {k: v[idx] for k, v in obs_tensors.items()}
            batch_targets = actions_tensor[idx]  # (B, N_batch_slots)

            out = module._forward_train(
                {Columns.OBS: {"observation": batch_obs}}
            )
            logits = out[Columns.ACTION_DIST_INPUTS]
            logits_pb = _bc_logits_to_per_slot(logits, num_batch_slots, num_dcs)

            loss = F.cross_entropy(
                logits_pb.reshape(-1, num_dcs),
                batch_targets.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.detach())
            preds = logits_pb.argmax(dim=-1)
            running_correct += int((preds == batch_targets).sum().detach())
            running_total += int(batch_targets.numel())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        acc = running_correct / max(1, running_total)
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss
        final_acc = acc
        logger.info(
            f"BC epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} top1_acc={acc:.3f}"
        )

    stats = {
        "initial_loss": float(initial_loss or 0.0),
        "final_loss": float(final_loss),
        "final_top1_acc": float(final_acc),
        "n_samples": N,
        "epochs": epochs,
    }
    return module, stats


def run_bc_warmstart(
    env_config: Dict[str, Any],
    model_config: Dict[str, Any],
    output_path: str,
    num_steps: int = 5000,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Top-level driver: collect RR rollout, train BC, save state_dict.
    Returns training stats dict.
    """
    logger.info(
        f"BC warm-start: rolling out {num_steps} env steps under "
        f"RoundRobinGlobalScheduler (seed={seed})…"
    )
    rollout = collect_rr_rollout(env_config, num_steps=num_steps, seed=seed)
    logger.info(
        f"BC training: {epochs} epochs, batch={batch_size}, lr={learning_rate}"
    )
    module, stats = train_bc_policy(
        obs_space=rollout["obs_space"],
        action_space=rollout["action_space"],
        model_config=model_config,
        rollout=rollout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Move to CPU before saving so loading is device-agnostic (PPO learner
    # decides cuda/cpu itself; mismatch here would assert).
    state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    torch.save(state, output_path)
    logger.info(
        f"BC checkpoint saved → {output_path} "
        f"(loss {stats['initial_loss']:.4f} → {stats['final_loss']:.4f}, "
        f"top1 acc {stats['final_top1_acc']:.3f})"
    )
    return stats
