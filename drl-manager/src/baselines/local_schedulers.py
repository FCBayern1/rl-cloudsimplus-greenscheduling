import numpy as np
from typing import Dict, Any, Optional
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

    def __init__(self, num_vms: int, algo, dc_id: int, env=None, policy_id: Optional[str] = None,
                 use_parameter_sharing: bool = False, num_datacenters: int = 10,
                 max_hosts: int = 16, max_vms: int = 224):
        """
        Args:
            num_vms: Max number of VMs across all datacenters (for base class)
            algo: Shared RLlib Algorithm instance
            dc_id: Datacenter ID (for policy selection)
            env: HierarchicalMultiDCEnv instance (used to trim obs/mask to DC-specific sizes)
            use_parameter_sharing: Whether using shared_local_policy (parameter sharing mode)
            num_datacenters: Number of datacenters (for dc_id_onehot)
            max_hosts: Max hosts per DC (for padding)
            max_vms: Max VMs per DC (for padding)
        """
        super().__init__(num_vms)
        self.algo = algo
        self.dc_id = dc_id
        self.env = env
        self.use_parameter_sharing = use_parameter_sharing
        self.num_datacenters = num_datacenters
        self.max_hosts = max_hosts
        self.max_vms = max_vms
        # If a specific policy_id is given (e.g. for parameter sharing with
        # "shared_local_policy"), use it; otherwise default to per-DC policy.
        self.policy_id = policy_id or f"local_policy_{dc_id}"

    def schedule(self, local_obs: Dict[str, Any], action_mask: np.ndarray) -> int:
        """
        使用 RLlib 模型选择 VM

        RLlib 通过 observation 中的 'action_mask' 键来处理动作掩码。

        对于参数共享模式 (shared_local_policy):
        - 使用完整的 padded 观测 (max_hosts, max_vms)
        - 添加 dc_id_onehot 和 valid_vm_mask

        对于独立策略模式:
        - 裁剪观测到每个 DC 实际的 host/vm 数量
        """
        if self.use_parameter_sharing:
            # Parameter sharing mode: use full padded observations
            # Mirror the PettingZoo wrapper's unified observation format
            dc_vm_count = self.env._get_dc_vm_count(self.dc_id) if self.env else self.max_vms

            # Build valid_vm_mask (1 for real VMs, 0 for padding)
            valid_vm_mask = np.zeros(self.max_vms, dtype=np.float32)
            valid_vm_mask[:dc_vm_count] = 1.0

            # DC ID one-hot
            dc_id_onehot = np.zeros(self.num_datacenters, dtype=np.float32)
            dc_id_onehot[self.dc_id] = 1.0

            # Use full padded observations (already padded by HierarchicalMultiDCEnv)
            unified_obs = {
                "host_loads": local_obs["host_loads"],
                "host_ram_usage": local_obs["host_ram_usage"],
                "vm_loads": local_obs["vm_loads"],
                "vm_types": local_obs["vm_types"],
                "vm_available_pes": local_obs["vm_available_pes"],
                "waiting_cloudlets": local_obs["waiting_cloudlets"],
                "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
                "dc_id_onehot": dc_id_onehot,
                "valid_vm_mask": valid_vm_mask,
            }

            # Full action mask (max_vms + 1)
            full_mask = action_mask.astype(np.float32)

            obs_with_mask = {
                "observation": unified_obs,
                "action_mask": full_mask,
            }
        else:
            # Non-parameter-sharing mode: trim to DC-specific sizes
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

            obs_with_mask = {
                "observation": trimmed_obs,
                "action_mask": trimmed_mask,
            }

        # 选择实际使用的 policy_id：
        # - 如果算法的本地 worker 中包含 shared_local_policy，且当前 policy_id 不在 policy_map 里，
        #   则说明使用了参数共享，回退到 shared_local_policy，避免访问不存在的 local_policy_*。
        policy_id = self.policy_id
        worker_mgr = getattr(self.algo, "workers", None)
        if worker_mgr is not None:
            try:
                local_worker = self.algo.workers.local_worker()
                policy_ids = set(local_worker.policy_map.keys())
                if "shared_local_policy" in policy_ids and policy_id not in policy_ids:
                    policy_id = "shared_local_policy"
            except Exception:
                # 如果无法读取 policy_map，就继续使用原始 policy_id，由 RLlib 自己决定是否报错
                pass

        action = self.algo.compute_single_action(
            obs_with_mask,
            policy_id=policy_id,
            explore=False
        )

        return int(action)


def create_rllib_schedulers(algo, env, num_dcs: int, batch_size: int, num_vms: int,
                            max_hosts: int = 16):
    """
    创建 RLlib 版本的 Global 和 Local 调度器

    Args:
        algo: 已加载的 RLlib Algorithm 实例
        env: HierarchicalMultiDCEnv 实例（用于裁剪本地观测到每个 DC 的实际规模）
        num_dcs: 数据中心数量
        batch_size: Global routing batch size
        num_vms: 每个 DC 的最大 VM 数量
        max_hosts: 每个 DC 的最大主机数量 (for parameter sharing)

    Returns:
        (global_scheduler, local_schedulers_dict)
    """
    from .global_schedulers import RLlibGlobalScheduler

    global_scheduler = RLlibGlobalScheduler(num_dcs, batch_size, algo)

    # Inspect the Algorithm's local worker policy map to detect whether
    # parameter sharing was used (i.e., presence of "shared_local_policy").
    worker = getattr(algo, "workers", None)
    policy_ids = set()
    if worker is not None:
        try:
            local_worker = algo.workers.local_worker()
            policy_ids = set(local_worker.policy_map.keys())
        except Exception:
            policy_ids = set()

    has_shared_local = "shared_local_policy" in policy_ids

    local_schedulers = {}
    for dc_id in range(num_dcs):
        if has_shared_local:
            # All DCs share the same local policy trained with parameter sharing.
            pid = "shared_local_policy"
        else:
            pid = f"local_policy_{dc_id}"
        local_schedulers[dc_id] = RLlibLocalScheduler(
            num_vms, algo, dc_id, env=env, policy_id=pid,
            use_parameter_sharing=has_shared_local,
            num_datacenters=num_dcs,
            max_hosts=max_hosts,
            max_vms=num_vms,
        )

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
