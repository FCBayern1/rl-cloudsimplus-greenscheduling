import numpy as np
from typing import Dict, List, Any
from .base import GlobalScheduler


class RandomGlobalScheduler(GlobalScheduler):
    """Random Global Scheduler"""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        return np.random.randint(0, self.num_datacenters,
                                 size=self.batch_size).tolist()


class RoundRobinGlobalScheduler(GlobalScheduler):
    """ Round Robin Global Scheduler"""

    def __init__(self, num_datacenters: int, batch_size: int):
        super().__init__(num_datacenters, batch_size)
        self.current_dc = 0

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        actions = []
        for _ in range(self.batch_size):
            actions.append(self.current_dc)
            self.current_dc = (self.current_dc + 1) % self.num_datacenters
        return actions

    def reset(self):
        self.current_dc = 0


class MinQueueGlobalScheduler(GlobalScheduler):
    """Min-length-WaitingQueue Global Scheduler"""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        queue_sizes = np.array(global_obs.get('dc_queue_sizes', [0] * self.num_datacenters))
        actions = []
        for _ in range(self.batch_size):
            # Select the DC with the smallest size waiting queue
            best_dc = int(np.argmin(queue_sizes))
            actions.append(best_dc)
            # Simulated Queue Increment (for Continuous Decision-Making)
            queue_sizes[best_dc] += 1
        return actions


class GreenAwareGlobalScheduler(GlobalScheduler):
    """GreenOpt Global Scheduler"""

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        green_ratios = np.array(global_obs.get('dc_green_ratio', [0.5] * self.num_datacenters))
        # Assign Cloudlet to the DC with highest Green Energy
        best_dc = int(np.argmax(green_ratios))
        return [best_dc] * self.batch_size


class GreenQueueBalancedGlobalScheduler(GlobalScheduler):
    """Green Energy + Queue Balancing Global Scheduling: Comprehensive consideration of green energy proportion and queue length"""

    def __init__(self, num_datacenters: int, batch_size: int, green_weight: float = 0.6):
        super().__init__(num_datacenters, batch_size)
        self.green_weight = green_weight

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        green_ratios = np.array(global_obs.get('dc_green_ratio', [0.5] * self.num_datacenters))
        queue_sizes = np.array(global_obs.get('dc_queue_sizes', [0] * self.num_datacenters))

        # Normalization
        green_norm = green_ratios / (green_ratios.max() + 1e-8)
        queue_norm = 1 - (queue_sizes / (queue_sizes.max() + 1e-8))

        # Conprehensive Scoring
        scores = self.green_weight * green_norm + (1 - self.green_weight) * queue_norm

        actions = []
        temp_queues = queue_sizes.copy()
        for _ in range(self.batch_size):
            # Dynamic Update Score
            queue_norm = 1 - (temp_queues / (temp_queues.max() + 1e-8))
            scores = self.green_weight * green_norm + (1 - self.green_weight) * queue_norm
            best_dc = int(np.argmax(scores))
            actions.append(best_dc)
            temp_queues[best_dc] += 1

        return actions


class RLlibGlobalScheduler(GlobalScheduler):
    """
    RLlib-based Global Scheduler (for Multi-DC training with Ray)

    Uses a pre-loaded RLlib Algorithm for inference.
    """

    def __init__(self, num_datacenters: int, batch_size: int, algo):
        """
        Args:
            num_datacenters: 数据中心数量
            batch_size: 每步路由的 cloudlet 数量
            algo: 已加载的 RLlib Algorithm 实例
        """
        super().__init__(num_datacenters, batch_size)
        self.algo = algo
        self.policy_id = "global_policy"

    def schedule(self, global_obs: Dict[str, Any]) -> List[int]:
        """
        使用 RLlib 模型为 batch 中的 cloudlet 选择目标 DC
        """
        # Wrap observation to match training-time PettingZoo format:
        # {"observation": <global_obs_dict>}
        wrapped_obs = {
            "observation": global_obs
        }

        action = self.algo.compute_single_action(
            wrapped_obs,
            policy_id=self.policy_id,
            explore=False
        )

        if isinstance(action, np.ndarray):
            return action.tolist()
        elif isinstance(action, (list, tuple)):
            return list(action)
        else:
            return [int(action)] * self.batch_size


def load_rllib_algorithm(checkpoint_path: str):
    """
    加载 RLlib checkpoint 并返回 Algorithm 实例

    Args:
        checkpoint_path: checkpoint 目录路径

    Returns:
        RLlib Algorithm 实例
    """
    import ray
    from ray import tune
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # 注册自定义模型
    from src.models.masked_action_model import MaskedActionModel, DictObsModel
    try:
        ModelCatalog.register_custom_model('masked_action_model', MaskedActionModel)
        ModelCatalog.register_custom_model('dict_obs_model', DictObsModel)
    except Exception:
        pass

    # 注册环境
    from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv

    def env_creator(cfg):
        env = HierarchicalMultiDCParallelEnv(cfg)
        return ParallelPettingZooEnv(env)

    tune.register_env('multidc_env', env_creator)

    # 加载 checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)
    return algo


# === Register all global schedulers ===
GLOBAL_SCHEDULERS = {
    'random': RandomGlobalScheduler,
    'round_robin': RoundRobinGlobalScheduler,
    'min_queue': MinQueueGlobalScheduler,
    'green_aware': GreenAwareGlobalScheduler,
    'green_queue_balanced': GreenQueueBalancedGlobalScheduler,
    'rllib': RLlibGlobalScheduler,  # For Multi-DC (RLlib/Ray)
}
