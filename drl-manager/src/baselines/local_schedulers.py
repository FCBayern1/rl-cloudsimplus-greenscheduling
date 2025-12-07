import numpy as np
from typing import Dict, Any
from .base import LocalScheduler

# Type hint for RLlib Algorithm (avoid import at module level)
AlgorithmType = Any


class RandomLocalScheduler(LocalScheduler):
    """Random Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return int(np.random.choice(valid_actions))
        return 0  # NoAssign


class FirstFitLocalScheduler(LocalScheduler):
    """First Fit Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        # Start from VM 1
        for i in range(1, len(action_mask)):
            if action_mask[i]:
                return i
        return 0  # NoAssign


class BestFitLocalScheduler(LocalScheduler):
    """Best Fit Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        vm_available_pes = local_obs.get('vm_available_pes', [])
        next_cloudlet_pes = local_obs.get('next_cloudlet_pes', 1)

        best_vm = 0
        min_waste = float('inf')

        for i in range(1, len(action_mask)):
            if action_mask[i] and i-1 < len(vm_available_pes):
                available = vm_available_pes[i-1]
                waste = available - next_cloudlet_pes
                if 0 <= waste < min_waste:
                    min_waste = waste
                    best_vm = i

        return best_vm


class WorstFitLocalScheduler(LocalScheduler):
    """Worst Fit Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        vm_available_pes = local_obs.get('vm_available_pes', [])
        next_cloudlet_pes = local_obs.get('next_cloudlet_pes', 1)

        best_vm = 0
        max_remaining = -1

        for i in range(1, len(action_mask)):
            if action_mask[i] and i-1 < len(vm_available_pes):
                available = vm_available_pes[i-1]
                remaining = available - next_cloudlet_pes
                if remaining >= 0 and remaining > max_remaining:
                    max_remaining = remaining
                    best_vm = i

        return best_vm


class RoundRobinLocalScheduler(LocalScheduler):
    """Round Robin Local Scheduler"""

    def __init__(self, num_vms: int):
        super().__init__(num_vms)
        self.current_vm = 1

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        for _ in range(self.num_vms):
            if self.current_vm < len(action_mask) and action_mask[self.current_vm]:
                selected = self.current_vm
                self.current_vm = (self.current_vm % self.num_vms) + 1
                return selected
            self.current_vm = (self.current_vm % self.num_vms) + 1
        return 0  # NoAssign

    def reset(self):
        self.current_vm = 1


class MinLoadLocalScheduler(LocalScheduler):
    """Greedy Min Load Local Scheduler"""

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        vm_loads = local_obs.get('vm_loads', [])

        best_vm = 0
        min_load = float('inf')

        for i in range(1, len(action_mask)):
            if action_mask[i] and i-1 < len(vm_loads):
                load = vm_loads[i-1]
                if load < min_load:
                    min_load = load
                    best_vm = i

        return best_vm


class RLlibLocalScheduler(LocalScheduler):
    """
    RLlib-based Local Scheduler (for Multi-DC training with Ray)

    Uses shared Algorithm instance for all DCs.
    Each DC uses its own policy (local_policy_{dc_id}).
    """

    def __init__(self, num_vms: int, algo, dc_id: int, env=None):
        """
        Args:
            num_vms: Max number of VMs across all datacenters (for base class)
            algo: Shared RLlib Algorithm instance
            dc_id: Datacenter ID (for policy selection)
            env: HierarchicalMultiDCEnv instance (used to trim obs/mask to DC-specific sizes)
        """
        super().__init__(num_vms)
        self.algo = algo
        self.dc_id = dc_id
        self.env = env
        self.policy_id = f"local_policy_{dc_id}"

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        使用 RLlib 模型选择 VM

        RLlib 通过 observation 中的 'action_mask' 键来处理动作掩码。
        这里需要将 HierarchicalMultiDCEnv 返回的、按 max_vms/max_hosts 填充的观测
        裁剪为训练时每个 DC 实际的 host/vm 数量，以匹配策略的 obs_space。
        """
        # Trim local_obs and action_mask to DC-specific sizes (mirror PettingZoo wrapper)
        if self.env is not None:
            try:
                dc_vm_count = self.env._get_dc_vm_count(self.dc_id)
                dc_host_count = self.env._get_dc_host_count(self.dc_id)

                trimmed_obs = {
                    "host_loads": local_obs["host_loads"][:dc_host_count],
                    "host_ram_usage": local_obs["host_ram_usage"][:dc_host_count],
                    "vm_loads": local_obs["vm_loads"][:dc_vm_count],
                    "vm_types": local_obs["vm_types"][:dc_vm_count],
                    "vm_available_pes": local_obs["vm_available_pes"][:dc_vm_count],
                    "waiting_cloudlets": local_obs["waiting_cloudlets"],
                    "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
                }
                trimmed_mask = action_mask[: dc_vm_count + 1].astype(np.float32)
            except Exception:
                # Fallback to original obs/mask on any error
                trimmed_obs = local_obs
                trimmed_mask = action_mask.astype(np.float32)
        else:
            trimmed_obs = local_obs
            trimmed_mask = action_mask.astype(np.float32)

        # 构造带 action_mask 的观察，需与训练时 PettingZoo env 的格式保持一致：
        # {"observation": <local_obs_dict>, "action_mask": <mask_array>}
        obs_with_mask = {
            "observation": trimmed_obs,
            "action_mask": trimmed_mask,
        }

        action = self.algo.compute_single_action(
            obs_with_mask,
            policy_id=self.policy_id,
            explore=False
        )

        return int(action)


def create_rllib_schedulers(algo, env, num_dcs: int, batch_size: int, num_vms: int):
    """
    创建 RLlib 版本的 Global 和 Local 调度器

    Args:
        algo: 已加载的 RLlib Algorithm 实例
        env: HierarchicalMultiDCEnv 实例（用于裁剪本地观测到每个 DC 的实际规模）
        num_dcs: 数据中心数量
        batch_size: Global routing batch size
        num_vms: 每个 DC 的最大 VM 数量

    Returns:
        (global_scheduler, local_schedulers_dict)
    """
    from .global_schedulers import RLlibGlobalScheduler

    global_scheduler = RLlibGlobalScheduler(num_dcs, batch_size, algo)

    local_schedulers = {
        dc_id: RLlibLocalScheduler(num_vms, algo, dc_id, env=env)
        for dc_id in range(num_dcs)
    }

    return global_scheduler, local_schedulers


# === Register Local Schedulers ===
LOCAL_SCHEDULERS = {
    'random': RandomLocalScheduler,
    'first_fit': FirstFitLocalScheduler,
    'best_fit': BestFitLocalScheduler,
    'worst_fit': WorstFitLocalScheduler,
    'round_robin': RoundRobinLocalScheduler,
    'min_load': MinLoadLocalScheduler,
    # Note: 'rllib' requires special handling via create_rllib_local_schedulers()
}
