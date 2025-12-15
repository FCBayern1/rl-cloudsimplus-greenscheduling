"""
Tianshou Training Script for Hierarchical Multi-Datacenter MARL with PettingZoo.

This script trains both the Global Agent and Local Agents using Tianshou
with the PettingZoo ParallelEnv wrapper. Supports A2C, PPO and other on-policy algorithms.

Features:
- Native PettingZoo support via Tianshou
- Supports A2C, PPO algorithms (RLlib only supports PPO)
- Global Agent: Policy gradient for datacenter routing
- Local Agents: Policy gradient with action masking for VM scheduling
- TensorBoard logging
- Checkpoint management

Usage:
    python train_tianshou_multidc.py --experiment experiment_multi_dc_simple_tianshou
"""

import os
import sys
import argparse
import yaml
import logging
import warnings
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm

# Tianshou imports
from tianshou.algorithm import A2C, PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput

# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnvSimple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DictFlattenWrapper(nn.Module):
    """
    Wrapper to flatten Dict observation space into a single vector.

    Used for both global and local agents since their observations are Dict spaces.
    """

    def __init__(self, observation_space: spaces.Dict):
        super().__init__()
        self.observation_space = observation_space

        # Calculate flattened size
        self.flat_size = 0
        self._keys = []
        self._sizes = {}

        for key in sorted(observation_space.spaces.keys()):
            space = observation_space.spaces[key]
            if isinstance(space, spaces.Box):
                size = int(np.prod(space.shape))
            elif isinstance(space, spaces.Discrete):
                size = 1
            elif isinstance(space, spaces.Dict):
                # Recursive flattening for nested Dict
                size = sum(int(np.prod(s.shape)) if isinstance(s, spaces.Box) else 1
                          for s in space.spaces.values())
            else:
                size = 1
            self._keys.append(key)
            self._sizes[key] = size
            self.flat_size += size

        logger.debug(f"DictFlattenWrapper: flat_size={self.flat_size}, keys={self._keys}")

    def forward(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Flatten dict observation to tensor."""
        if isinstance(obs, Batch):
            # Handle Tianshou Batch objects
            obs = dict(obs)

        parts = []
        for key in self._keys:
            if key not in obs:
                continue
            value = obs[key]
            if isinstance(value, dict):
                # Nested dict - flatten recursively
                for subkey in sorted(value.keys()):
                    subval = value[subkey]
                    if isinstance(subval, torch.Tensor):
                        parts.append(subval.flatten(start_dim=-len(subval.shape)+1))
                    else:
                        parts.append(torch.as_tensor(subval, dtype=torch.float32).flatten())
            elif isinstance(value, torch.Tensor):
                parts.append(value.flatten(start_dim=-len(value.shape)+1) if value.dim() > 1 else value.unsqueeze(-1))
            elif isinstance(value, np.ndarray):
                parts.append(torch.from_numpy(value.astype(np.float32)).flatten())
            else:
                parts.append(torch.tensor([float(value)], dtype=torch.float32))

        if not parts:
            return torch.zeros(self.flat_size)

        result = torch.cat(parts, dim=-1)
        return result


class DiscreteActorWithMask(ModuleWithVectorOutput):
    """
    Discrete actor network with optional action masking support.

    This is used for local agents that need action masking for VM scheduling.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        hidden_sizes: List[int] = [256, 256],
        use_action_mask: bool = True,
        device: str = "cpu"
    ):
        # Flatten dict observation
        obs_flatten = DictFlattenWrapper(observation_space)
        input_dim = obs_flatten.flat_size

        # Build MLP network to calculate output_dim first
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            prev_dim = hidden_size

        # Call parent with output_dim
        super().__init__(output_dim=prev_dim)

        self.device = device
        self.use_action_mask = use_action_mask
        self.action_dim = action_space.n
        self.obs_flatten = obs_flatten

        # Build MLP network
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size

        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, self.action_dim)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: Dict[str, Any],
        state: Optional[Any] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass returning action logits."""
        # Extract actual observation (may be nested under "observation" key)
        if isinstance(obs, Batch):
            obs = dict(obs)

        actual_obs = obs.get("observation", obs)
        action_mask = obs.get("action_mask", None) if self.use_action_mask else None

        # Flatten observation
        flat_obs = self.obs_flatten(actual_obs)
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)

        # Forward through network
        features = self.net(flat_obs.to(self.device))
        logits = self.output_layer(features)

        # Apply action mask if available
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask).float().to(self.device)
            elif isinstance(action_mask, torch.Tensor):
                action_mask = action_mask.float().to(self.device)

            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            # Mask invalid actions with large negative value
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        return logits, state


class MultiDiscreteActor(ModuleWithVectorOutput):
    """
    Actor network for MultiDiscrete action space (used by global agent).

    The global agent routes cloudlets to datacenters, selecting one DC per cloudlet
    in a batch. Action space is MultiDiscrete([num_dcs] * batch_size).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        hidden_sizes: List[int] = [256, 256],
        device: str = "cpu"
    ):
        # Calculate output_dim first for parent class
        output_dim = hidden_sizes[-1] if hidden_sizes else 256
        super().__init__(output_dim=output_dim)

        self.device = device
        self.nvec = action_space.nvec  # Array of num_dcs for each position
        self.num_positions = len(self.nvec)
        self.num_dcs = self.nvec[0]  # Assume all positions have same num_dcs

        # Flatten dict observation
        self.obs_flatten = DictFlattenWrapper(observation_space)
        input_dim = self.obs_flatten.flat_size

        # Build shared MLP backbone
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size

        self.net = nn.Sequential(*layers)

        # Output heads for each position (or shared if all positions use same num_dcs)
        # Using a shared head since all positions select from same DC set
        self.output_layer = nn.Linear(prev_dim, self.num_dcs)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: Dict[str, Any],
        state: Optional[Any] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass returning action logits for each position."""
        if isinstance(obs, Batch):
            obs = dict(obs)

        actual_obs = obs.get("observation", obs)

        # Flatten observation
        flat_obs = self.obs_flatten(actual_obs)
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)

        batch_size = flat_obs.shape[0]

        # Forward through network
        features = self.net(flat_obs.to(self.device))

        # Get logits for DC selection (shared across all positions)
        base_logits = self.output_layer(features)  # (batch, num_dcs)

        # Expand to all positions: (batch, num_positions, num_dcs)
        logits = base_logits.unsqueeze(1).expand(batch_size, self.num_positions, self.num_dcs)

        return logits, state


class ValueNet(ModuleWithVectorOutput):
    """
    Value network (critic) for both global and local agents.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_sizes: List[int] = [256, 256],
        device: str = "cpu"
    ):
        # Calculate output_dim first for parent class
        output_dim = hidden_sizes[-1] if hidden_sizes else 256
        super().__init__(output_dim=output_dim)

        self.device = device

        # Flatten dict observation
        self.obs_flatten = DictFlattenWrapper(observation_space)
        input_dim = self.obs_flatten.flat_size

        # Build MLP network
        layers = []
        prev_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.ReLU())
            prev_dim = hidden_size

        self.net = nn.Sequential(*layers)
        self.value_layer = nn.Linear(prev_dim, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Forward pass returning state value."""
        if isinstance(obs, Batch):
            obs = dict(obs)

        actual_obs = obs.get("observation", obs)

        flat_obs = self.obs_flatten(actual_obs)
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)

        features = self.net(flat_obs.to(self.device))
        value = self.value_layer(features)
        return value.squeeze(-1)


def create_agent_algorithm(
    agent_name: str,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    algorithm_name: str,
    hidden_sizes: List[int],
    lr: float,
    gamma: float,
    gae_lambda: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    device: str,
    use_action_mask: bool = False
) -> A2C:
    """
    Create an A2C or PPO algorithm instance for a single agent.

    Args:
        agent_name: Name of the agent (e.g., "global_agent", "local_agent_0")
        observation_space: Agent's observation space
        action_space: Agent's action space
        algorithm_name: "A2C" or "PPO"
        hidden_sizes: Hidden layer sizes for networks
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        max_grad_norm: Max gradient norm
        device: Device to use ("cpu" or "cuda")
        use_action_mask: Whether to use action masking (for local agents)

    Returns:
        Configured algorithm instance
    """
    logger.info(f"Creating {algorithm_name} algorithm for {agent_name}")
    logger.info(f"  Observation space: {observation_space}")
    logger.info(f"  Action space: {action_space}")

    # Create actor network based on action space type
    if isinstance(action_space, spaces.MultiDiscrete):
        # Global agent with MultiDiscrete action space
        actor = MultiDiscreteActor(
            observation_space=observation_space,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            device=device
        ).to(device)

        # Distribution function for MultiDiscrete
        def dist_fn(logits):
            # logits shape: (batch, num_positions, num_dcs)
            # Return independent categorical for each position
            return torch.distributions.Independent(
                Categorical(logits=logits),
                reinterpreted_batch_ndims=1
            )
    else:
        # Local agent with Discrete action space
        actor = DiscreteActorWithMask(
            observation_space=observation_space,
            action_space=action_space,
            hidden_sizes=hidden_sizes,
            use_action_mask=use_action_mask,
            device=device
        ).to(device)

        # Distribution function for Discrete
        def dist_fn(logits):
            return Categorical(logits=logits)

    # Create critic network
    critic = ValueNet(
        observation_space=observation_space,
        hidden_sizes=hidden_sizes,
        device=device
    ).to(device)

    # Create policy
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn,
        action_space=action_space,
        observation_space=observation_space,
        action_scaling=False,
        action_bound_method=None
    )

    # Create optimizer factory
    class AdamFactory(OptimizerFactory):
        def __init__(self, lr: float):
            super().__init__()
            self.lr = lr

        def _create_optimizer_for_params(self, params) -> torch.optim.Optimizer:
            return torch.optim.Adam(params, lr=self.lr)

    optim_factory = AdamFactory(lr=lr)

    # Create algorithm (we will use its hyperparameters, actor, critic, and distribution,
    # but handle optimization manually to support our custom multi-agent update loop).
    if algorithm_name.upper() == "A2C":
        algo = A2C(
            policy=policy,
            critic=critic,
            optim=optim_factory,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            gamma=gamma
        )
    elif algorithm_name.upper() == "PPO":
        algo = PPO(
            policy=policy,
            critic=critic,
            optim=optim_factory,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            gamma=gamma,
            clip_eps=0.2,  # PPO clip parameter
            dual_clip=None,
            advantage_normalization=True,
            recompute_advantage=False
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # Attach a manual optimizer over both actor and critic parameters.
    # This avoids relying on Tianshou's internal Optimizer wrapper, which expects
    # to be driven by its own trainer/update loop.
    manual_params = list(policy.actor.parameters()) + list(critic.parameters())
    algo.manual_optim = torch.optim.Adam(manual_params, lr=lr)

    logger.info(f"  Created {algorithm_name} algorithm for {agent_name}")
    return algo


def _obs_to_tensor(obs: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    """
    Convert observation dictionary to tensors.

    Args:
        obs: Observation dictionary (may contain nested dicts)
        device: Device to place tensors on

    Returns:
        Dictionary of tensors
    """
    result = {}

    for key, value in obs.items():
        if isinstance(value, dict):
            # Nested dict (e.g., "observation" key containing the actual obs)
            result[key] = _obs_to_tensor(value, device)
        elif isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value.astype(np.float32)).to(device)
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, (int, float)):
            result[key] = torch.tensor([value], dtype=torch.float32, device=device)
        else:
            result[key] = value

    return result


def _update_agent_policy(
    algo: Any,
    buffer: Dict[str, List],
    batch_size: int,
    gamma: float,
    gae_lambda: float,
    device: str
):
    """
    Update agent's policy using collected experiences.

    This is a simplified A2C/PPO update step.

    Args:
        algo: Algorithm instance (A2C or PPO)
        buffer: Experience buffer with obs, act, rew, done, next_obs
        batch_size: Mini-batch size for updates
        gamma: Discount factor
        gae_lambda: GAE lambda
        device: Device
    """
    if len(buffer["obs"]) < batch_size:
        return

    # Convert buffer lists
    obs_list = buffer["obs"]
    act_list = buffer["act"]
    rew_list = buffer["rew"]
    done_list = buffer["done"]
    next_obs_list = buffer["next_obs"]

    n_steps = len(rew_list)
    if n_steps == 0:
        return

    # Compute state values V(s) and V(s') using the critic (no grad here;
    # critic will still be trained via value_loss below).
    with torch.no_grad():
        values = []
        next_values = []
        for obs, next_obs in zip(obs_list, next_obs_list):
            v = algo.critic(_obs_to_tensor(obs, device))
            v_next = algo.critic(_obs_to_tensor(next_obs, device))
            values.append(v.squeeze())
            next_values.append(v_next.squeeze())

        values = torch.stack(values).to(device)
        next_values = torch.stack(next_values).to(device)

    # Compute GAE advantages and returns
    returns = torch.zeros(n_steps, dtype=torch.float32, device=device)
    advantages = torch.zeros(n_steps, dtype=torch.float32, device=device)

    gae = 0.0
    for t in reversed(range(n_steps)):
        done = float(done_list[t])
        reward = float(rew_list[t])
        delta = reward + gamma * (1.0 - done) * next_values[t] - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Mini-batch updates
    indices = np.arange(len(obs_list))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        # Get batch data
        batch_obs = [obs_list[i] for i in batch_indices]
        batch_act = [act_list[i] for i in batch_indices]
        batch_ret = returns[batch_indices]
        batch_adv = advantages[batch_indices]

        # Convert observations to tensors
        batch_obs_tensors = [_obs_to_tensor(obs, device) for obs in batch_obs]

        # Determine action space type (MultiDiscrete vs Discrete)
        action_space = getattr(algo.policy, "action_space", None)
        is_multidiscrete = isinstance(action_space, spaces.MultiDiscrete)

        log_probs = []
        entropies = []
        values_pred = []

        for obs_tensor, act in zip(batch_obs_tensors, batch_act):
            logits, _ = algo.policy.actor(obs_tensor)

            if is_multidiscrete:
                # logits: (1, num_positions, num_dcs)
                logits_md = logits.squeeze(0)  # (num_positions, num_dcs)
                dist = torch.distributions.Categorical(logits=logits_md)

                act_tensor = torch.as_tensor(act, dtype=torch.long, device=device)
                lp = dist.log_prob(act_tensor).sum()  # sum over positions
                ent = dist.entropy().sum()
            else:
                # Discrete: logits shape (1, num_actions) or (num_actions,)
                logits_dis = logits.squeeze(0)
                dist = torch.distributions.Categorical(logits=logits_dis)

                act_tensor = torch.as_tensor(act, dtype=torch.long, device=device)
                lp = dist.log_prob(act_tensor)
                ent = dist.entropy().mean()

            log_probs.append(lp)
            entropies.append(ent)

            v_pred = algo.critic(obs_tensor)
            values_pred.append(v_pred.squeeze())

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values_pred = torch.stack(values_pred)

        # Compute losses (A2C-style)
        policy_loss = -(log_probs * batch_adv).mean()
        value_loss = ((values_pred - batch_ret) ** 2).mean()
        entropy_loss = -entropies.mean()

        total_loss = policy_loss + algo.vf_coef * value_loss + algo.ent_coef * entropy_loss

        # Backward and optimize using our manual optimizer attached to the algo.
        # We don't rely on Tianshou's internal Optimizer wrapper here.
        optim = getattr(algo, "manual_optim", None)
        if optim is None:
            raise RuntimeError("Algorithm is missing manual_optim; please ensure create_agent_algorithm attaches it.")

        optim.zero_grad()
        total_loss.backward()
        if algo.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(algo.policy.actor.parameters()) + list(algo.critic.parameters()),
                algo.max_grad_norm
            )
        optim.step()


def train_tianshou(
    env_config: Dict[str, Any],
    global_model_config: Dict[str, Any],
    local_model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    output_dir: str
):
    """
    Main training function using Tianshou with manual training loop.

    This implementation handles heterogeneous observation/action spaces
    by using a custom training loop instead of Tianshou's PettingZooEnv wrapper.

    Args:
        env_config: Environment configuration
        global_model_config: Global agent model configuration
        local_model_config: Local agent model configuration
        training_config: Training hyperparameters
        output_dir: Output directory for logs and checkpoints
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Tianshou Multi-Agent Training (Heterogeneous Spaces)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")

    # Get device
    device = training_config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get algorithm name
    algorithm_name = training_config.get("algorithm", "A2C").upper()
    logger.info(f"Using algorithm: {algorithm_name}")

    # Create PettingZoo environment directly (no Tianshou wrapper)
    logger.info("Creating PettingZoo environment...")
    env = HierarchicalMultiDCParallelEnvSimple(env_config)

    logger.info(f"Environment created with {len(env.possible_agents)} agents:")
    for agent in env.possible_agents:
        logger.info(f"  - {agent}")

    # Get hyperparameters
    hidden_sizes = local_model_config.get("hidden_sizes", [256, 256])
    lr = local_model_config.get("learning_rate", 3e-4)
    gamma = local_model_config.get("gamma", 0.99)
    gae_lambda = local_model_config.get("gae_lambda", 0.95)
    vf_coef = local_model_config.get("vf_coef", 0.5)
    ent_coef = local_model_config.get("ent_coef", 0.01)
    max_grad_norm = local_model_config.get("max_grad_norm", 0.5)

    # Create algorithms for each agent
    algorithms: Dict[str, Any] = {}
    for agent_name in env.possible_agents:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)

        # Use action masking for local agents
        use_mask = agent_name.startswith("local_agent")

        algo = create_agent_algorithm(
            agent_name=agent_name,
            observation_space=obs_space,
            action_space=act_space,
            algorithm_name=algorithm_name,
            hidden_sizes=hidden_sizes,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            use_action_mask=use_mask
        )
        algorithms[agent_name] = algo

    # Setup TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = output_path / "tensorboard"
    tb_log_dir.mkdir(exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Training parameters
    total_timesteps = training_config.get("total_timesteps", 100000)
    batch_size = training_config.get("batch_size", 64)
    step_per_collect = training_config.get("step_per_collect", 2048)
    update_per_step = training_config.get("update_per_step", 1)

    logger.info(f"Training configuration:")
    logger.info(f"  Total timesteps: {total_timesteps}")
    logger.info(f"  Steps per collect: {step_per_collect}")
    logger.info(f"  Batch size: {batch_size}")

    # Checkpoint directory
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Monitor file for episode metrics
    monitor_file = output_path / "monitor.csv"
    # Use the same header format as RLlib's GreenEnergyLoggerCallback so that
    # existing analysis scripts (e.g. scripts/visualize_monitor.py) work unchanged.
    with open(monitor_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        headers = [
            'episode',
            'green_waste_wh',
            'green_used_wh',
            'brown_used_wh',
            'total_energy_wh',
            'green_ratio',
            'waste_ratio',
            'total_carbon_kg',
            'carbon_intensity_kg_per_kwh',
            'episode_reward',
            'episode_length',
            'global_agent_reward',
            'local_agents_avg_reward',
            'completion_rate',
        ]

        # For compatibility with existing RLlib logs and visualization scripts,
        # we always reserve 10 DC slots, even if the environment uses fewer DCs.
        num_dcs_header = max(getattr(env, "num_datacenters", 0), 10)

        # Per-DC local rewards
        for dc_id in range(num_dcs_header):
            headers.append(f'local_reward_{dc_id}')

        # Per-DC mean completion times
        for dc_id in range(num_dcs_header):
            headers.append(f'mean_completion_time_dc_{dc_id}')

        # Per-DC finished cloudlets
        for dc_id in range(num_dcs_header):
            headers.append(f'cloudlets_finished_dc_{dc_id}')

        # Per-DC cloudlet completion rates
        for dc_id in range(num_dcs_header):
            headers.append(f'completion_rate_dc_{dc_id}')

        csv_writer.writerow(headers)

    # Training loop
    logger.info("Starting manual training loop...")
    global_step = 0
    episode_count = 0
    best_reward = float('-inf')

    # Experience buffers for each agent
    agent_buffers: Dict[str, Dict[str, List]] = {
        agent: {"obs": [], "act": [], "rew": [], "done": [], "next_obs": [], "info": []}
        for agent in env.possible_agents
    }

    pbar = tqdm(total=total_timesteps, desc="Training", unit=" steps")

    try:
        while global_step < total_timesteps:
            # Reset environment
            observations, infos = env.reset()
            episode_rewards = {agent: 0.0 for agent in env.possible_agents}
            episode_step = 0
            episode_done = False

            # Clear buffers
            for agent in env.possible_agents:
                for key in agent_buffers[agent]:
                    agent_buffers[agent][key].clear()

            while not episode_done:
                # Get actions from each agent's policy
                actions = {}
                for agent_name in env.agents:
                    obs = observations[agent_name]
                    algo = algorithms[agent_name]

                    # Convert observation to tensor
                    with torch.no_grad():
                        # Get action from policy
                        obs_tensor = _obs_to_tensor(obs, device)
                        action, _ = algo.policy.actor(obs_tensor)

                        # Sample action from distribution
                        if isinstance(action, torch.Tensor):
                            if action.dim() == 3:
                                # MultiDiscrete: logits shape (batch, num_positions, num_actions)
                                # Sample for each position independently
                                dist = torch.distributions.Categorical(logits=action)
                                sampled = dist.sample()  # (batch, num_positions)
                                # Squeeze batch dimension
                                actions[agent_name] = sampled.squeeze(0).cpu().numpy()
                            elif action.dim() == 2:
                                # Discrete with batch: (batch, num_actions)
                                dist = torch.distributions.Categorical(logits=action)
                                sampled = dist.sample()  # (batch,)
                                actions[agent_name] = sampled.squeeze().item()
                            else:
                                # Discrete without batch: (num_actions,)
                                dist = torch.distributions.Categorical(logits=action)
                                sampled = dist.sample()
                                actions[agent_name] = sampled.item()
                        else:
                            actions[agent_name] = action

                # Step environment
                next_observations, rewards, terminations, truncations, infos = env.step(actions)

                # Store experiences
                for agent_name in env.agents:
                    agent_buffers[agent_name]["obs"].append(observations[agent_name])
                    agent_buffers[agent_name]["act"].append(actions[agent_name])
                    agent_buffers[agent_name]["rew"].append(rewards[agent_name])
                    agent_buffers[agent_name]["done"].append(terminations[agent_name] or truncations[agent_name])
                    agent_buffers[agent_name]["next_obs"].append(next_observations[agent_name])
                    agent_buffers[agent_name]["info"].append(infos[agent_name])

                    episode_rewards[agent_name] += rewards[agent_name]

                # Update state
                observations = next_observations
                global_step += 1
                episode_step += 1
                pbar.update(1)

                # Check if episode is done
                episode_done = any(terminations.values()) or any(truncations.values())

                # Update policies periodically
                if global_step % step_per_collect == 0 and global_step > 0:
                    for agent_name in env.possible_agents:
                        if len(agent_buffers[agent_name]["obs"]) > batch_size:
                            _update_agent_policy(
                                algorithms[agent_name],
                                agent_buffers[agent_name],
                                batch_size,
                                gamma,
                                gae_lambda,
                                device
                            )

            # Episode finished
            episode_count += 1
            total_reward = sum(episode_rewards.values())
            global_reward = episode_rewards.get("global_agent", 0)
            local_rewards = [v for k, v in episode_rewards.items() if k.startswith("local_agent")]
            local_mean = np.mean(local_rewards) if local_rewards else 0
            num_local_agents = len(local_rewards)

            # Extract metrics from final info (use global_agent's info dict)
            info = infos.get("global_agent", {}) if isinstance(infos, dict) else {}

            # Global energy statistics from Java MultiDatacenterSimulationCore
            global_energy_stats = info.get("global_energy_stats", {})
            if isinstance(global_energy_stats, str) or global_energy_stats is None:
                global_energy_stats = {}

            green_waste = float(global_energy_stats.get('total_wasted_green_wh', 0.0))
            green_used = float(global_energy_stats.get('total_green_energy_wh', 0.0))
            brown_used = float(global_energy_stats.get('total_brown_energy_wh', 0.0))
            total_energy = green_used + brown_used
            green_ratio = float(global_energy_stats.get('green_energy_ratio', 0.0))

            available_green = green_used + green_waste
            waste_ratio = green_waste / available_green if available_green > 0 else 0.0

            total_carbon_kg = float(global_energy_stats.get('total_carbon_emission_kg', 0.0))
            carbon_intensity = float(global_energy_stats.get('carbon_intensity_kg_per_kwh', 0.0))

            total_cloudlets_created = int(global_energy_stats.get('total_created_cloudlets', 0))
            total_cloudlets_finished = int(global_energy_stats.get('total_finished_cloudlets', 0))
            completion_rate = (
                total_cloudlets_finished / total_cloudlets_created
                if total_cloudlets_created > 0
                else 0.0
            )

            # Per-DC energy metrics (mean completion time, finished cloudlets, received cloudlets)
            dc_energy_metrics = info.get("datacenter_energy_metrics", {})
            if isinstance(dc_energy_metrics, str) or dc_energy_metrics is None:
                dc_energy_metrics = {}

            per_dc_mean_completion_times: Dict[int, float] = {}
            per_dc_cloudlets_finished: Dict[int, int] = {}
            per_dc_cloudlets_received: Dict[int, int] = {}

            if isinstance(dc_energy_metrics, dict):
                for dc_id_raw, dc_metrics in dc_energy_metrics.items():
                    try:
                        dc_id = int(dc_id_raw) if isinstance(dc_id_raw, str) else int(dc_id_raw)
                    except (ValueError, TypeError):
                        continue

                    if isinstance(dc_metrics, str) or dc_metrics is None:
                        continue

                    mean_ct = float(dc_metrics.get('mean_completion_time', 0.0))
                    finished = int(dc_metrics.get('cloudlets_finished', 0))
                    received = int(dc_metrics.get('cloudlets_received', 0))

                    per_dc_mean_completion_times[dc_id] = mean_ct
                    per_dc_cloudlets_finished[dc_id] = finished
                    per_dc_cloudlets_received[dc_id] = received

            # Log to TensorBoard
            tb_writer.add_scalar("reward/total", total_reward, global_step)
            tb_writer.add_scalar("reward/global", global_reward, global_step)
            tb_writer.add_scalar("reward/local_mean", local_mean, global_step)
            tb_writer.add_scalar("episode/length", episode_step, global_step)
            tb_writer.add_scalar("metrics/green_ratio", green_ratio, global_step)
            tb_writer.add_scalar("metrics/carbon_kg", total_carbon_kg, global_step)

            # Write to monitor.csv with the same schema as RLlib's GreenEnergyLoggerCallback
            # so that scripts/visualize_monitor.py can be reused.
            num_dcs_row = max(
                num_local_agents,
                len(per_dc_mean_completion_times),
                len(per_dc_cloudlets_finished),
                len(per_dc_cloudlets_received),
                10,  # keep at least 10 for compatibility with existing logs
            )

            row = [
                episode_count,
                green_waste,
                green_used,
                brown_used,
                total_energy,
                green_ratio,
                waste_ratio,
                total_carbon_kg,
                carbon_intensity,
                total_reward,
                episode_step,
                global_reward,
                local_mean,
                completion_rate,
            ]

            # Per-DC local rewards (local_reward_0, ..., local_reward_9)
            for dc_id in range(num_dcs_row):
                row.append(float(episode_rewards.get(f"local_agent_{dc_id}", 0.0)))

            # Per-DC mean completion times (mean_completion_time_dc_0, ..., _dc_9)
            for dc_id in range(num_dcs_row):
                row.append(float(per_dc_mean_completion_times.get(dc_id, 0.0)))

            # Per-DC cloudlets finished (cloudlets_finished_dc_0, ..., _dc_9)
            for dc_id in range(num_dcs_row):
                row.append(int(per_dc_cloudlets_finished.get(dc_id, 0)))

            # Per-DC cloudlet completion rates (completion_rate_dc_0, ..., _dc_9)
            for dc_id in range(num_dcs_row):
                finished = per_dc_cloudlets_finished.get(dc_id, 0)
                received = per_dc_cloudlets_received.get(dc_id, 0)
                rate = finished / received if received and received > 0 else 0.0
                row.append(rate)

            with open(monitor_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(row)

            # Update progress bar
            pbar.set_postfix({
                "ep": episode_count,
                "reward": f"{total_reward:.1f}",
                "green": f"{green_ratio:.2f}"
            })

            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = output_path / "best_model.pt"
                torch.save({
                    'algorithms': {name: algo.policy.state_dict() for name, algo in algorithms.items()},
                    'best_reward': best_reward,
                    'episode': episode_count
                }, best_path)
                logger.info(f"New best model saved with reward {best_reward:.2f}")

            # Save checkpoint periodically
            if episode_count % 10 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode_count}.pt"
                torch.save({
                    'algorithms': {name: algo.policy.state_dict() for name, algo in algorithms.items()},
                    'episode': episode_count,
                    'global_step': global_step
                }, checkpoint_path)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        pbar.close()
        tb_writer.close()
        env.close()

    # Save final model
    final_path = output_path / "final_model.pt"
    torch.save({
        'algorithms': {name: algo.policy.state_dict() for name, algo in algorithms.items()},
        'final_step': global_step,
        'final_episode': episode_count
    }, final_path)

    logger.info("\n" + "=" * 70)
    logger.info("Training completed!")
    logger.info("=" * 70)
    logger.info(f"Total episodes: {episode_count}")
    logger.info(f"Total timesteps: {global_step}")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Results saved to: {output_dir}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Tianshou Multi-Agent Training for Multi-DC MARL"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config.yml',
        help='Path to config.yml'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment_multi_dc_simple_tianshou',
        help='Experiment ID from config.yml'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default=None,
        choices=['A2C', 'PPO'],
        help='Override algorithm (A2C or PPO)'
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=None,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script
        config_path = Path(__file__).parent.parent.parent.parent / 'config.yml'

    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    logger.info(f"Loading configuration from {config_path}")
    all_config = load_config(str(config_path))

    if args.experiment not in all_config:
        logger.error(f"Experiment '{args.experiment}' not found in config")
        logger.error(f"Available experiments: {list(all_config.keys())}")
        sys.exit(1)

    exp_config = all_config[args.experiment]

    # Extract sub-configurations
    env_config = exp_config
    global_model_config = exp_config.get('global_model', {})
    local_model_config = exp_config.get('local_model', {})
    training_config = exp_config.get('training', {})

    # Apply command-line overrides
    if args.algorithm:
        training_config['algorithm'] = args.algorithm
    if args.total_timesteps:
        training_config['total_timesteps'] = args.total_timesteps
    if args.device:
        training_config['device'] = args.device

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        algorithm_name = training_config.get('algorithm', 'A2C').upper()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"../logs/{args.experiment}_{algorithm_name}/{timestamp}"

    # Start training
    train_tianshou(
        env_config=env_config,
        global_model_config=global_model_config,
        local_model_config=local_model_config,
        training_config=training_config,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
