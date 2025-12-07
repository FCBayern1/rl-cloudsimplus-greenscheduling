"""
Baseline Scheduling Algorithms for Multi-Datacenter Cloud Task Scheduling

This module provides pluggable Global and Local schedulers that can be
combined to create different baseline algorithms for evaluation.

Global Schedulers (select Datacenter for cloudlets):
    - random: Random DC selection
    - round_robin: Round-robin DC assignment
    - min_queue: Select DC with shortest queue
    - green_aware: Prioritize DCs with highest green energy ratio
    - green_queue_balanced: Balance green energy and queue length
    - rl: Use trained RL model

Local Schedulers (select VM within Datacenter):
    - random: Random VM selection (respecting action mask)
    - first_fit: First available VM
    - best_fit: VM with closest matching resources
    - worst_fit: VM with most remaining resources (load balancing)
    - round_robin: Round-robin VM assignment
    - min_load: VM with lowest current load
    - rl: Use trained RL model

Usage:
    from src.baselines import GLOBAL_SCHEDULERS, LOCAL_SCHEDULERS

    # Create schedulers
    global_sched = GLOBAL_SCHEDULERS['random'](num_dcs=10, batch_size=10)
    local_sched = LOCAL_SCHEDULERS['first_fit'](num_vms=18)

    # Use in evaluation loop
    global_actions = global_sched.schedule(global_obs)
    local_action = local_sched.schedule(local_obs, action_mask)
"""
from .base import GlobalScheduler, LocalScheduler
from .global_schedulers import (
    GLOBAL_SCHEDULERS,
    RLlibGlobalScheduler,
    load_rllib_algorithm,
)
from .local_schedulers import (
    LOCAL_SCHEDULERS,
    RLlibLocalScheduler,
    create_rllib_schedulers,
)

__all__ = [
    'GlobalScheduler',
    'LocalScheduler',
    'GLOBAL_SCHEDULERS',
    'LOCAL_SCHEDULERS',
    # RLlib (for Multi-DC)
    'load_rllib_algorithm',
    'create_rllib_schedulers',
    'RLlibGlobalScheduler',
    'RLlibLocalScheduler',
]
