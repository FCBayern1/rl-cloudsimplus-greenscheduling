"""
RLlib Multi-Agent Model Loader

加载训练好的 RLlib checkpoint 用于推理。

Usage:
    # 1. 先启动 Java Gateway
    cd cloudsimplus-gateway && ./gradlew run

    # 2. 然后运行
    cd drl-manager
    python -m src.baselines.load_rllib_model --checkpoint <path> --test
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def register_rllib_components():
    """注册 RLlib 所需的自定义模型和环境"""
    import ray
    from ray import tune
    from ray.rllib.models import ModelCatalog
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # 注册自定义模型
    from src.models.masked_action_model import MaskedActionModel, DictObsModel
    try:
        ModelCatalog.register_custom_model('masked_action_model', MaskedActionModel)
        ModelCatalog.register_custom_model('dict_obs_model', DictObsModel)
    except Exception:
        pass  # Already registered

    # 注册环境
    from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv

    def env_creator(config):
        env = HierarchicalMultiDCParallelEnv(config)
        return ParallelPettingZooEnv(env)

    tune.register_env('multidc_env', env_creator)


def load_rllib_checkpoint(checkpoint_path: str):
    """
    加载 RLlib checkpoint

    Args:
        checkpoint_path: checkpoint 目录路径

    Returns:
        RLlib Algorithm 实例
    """
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # 注册组件
    register_rllib_components()

    # 加载 checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("✓ Checkpoint loaded successfully!")

    return algo


def get_policy_ids(algo) -> Dict[str, list]:
    """获取所有策略 ID"""
    policy_ids = list(algo.config.policies.keys())
    global_policies = [p for p in policy_ids if 'global' in p]
    local_policies = [p for p in policy_ids if 'local' in p]
    return {
        'global': global_policies,
        'local': sorted(local_policies, key=lambda x: int(x.split('_')[-1]))
    }


class RLlibInferenceWrapper:
    """
    RLlib 模型推理包装器

    提供简单的 API 用于获取动作
    """

    def __init__(self, checkpoint_path: str):
        self.algo = load_rllib_checkpoint(checkpoint_path)
        self.policy_ids = get_policy_ids(self.algo)
        self.num_dcs = len(self.policy_ids['local'])
        print(f"Loaded {len(self.policy_ids['global'])} global + {self.num_dcs} local policies")

    def get_global_action(self, global_obs: Dict[str, Any]) -> list:
        """
        获取全局调度动作

        Args:
            global_obs: 全局观察（Dict格式）

        Returns:
            list: 每个 cloudlet 的目标 DC 索引
        """
        action = self.algo.compute_single_action(
            global_obs,
            policy_id="global_policy",
            explore=False
        )
        if hasattr(action, 'tolist'):
            return action.tolist()
        return list(action) if isinstance(action, (list, tuple)) else [action]

    def get_local_action(self, dc_id: int, local_obs: Dict[str, Any],
                         action_mask) -> int:
        """
        获取本地调度动作

        Args:
            dc_id: 数据中心 ID
            local_obs: 本地观察（Dict格式）
            action_mask: 动作掩码

        Returns:
            int: VM 动作索引
        """
        # RLlib 使用 observations + action_mask 格式
        obs_with_mask = {
            "observations": local_obs,
            "action_mask": action_mask
        }

        action = self.algo.compute_single_action(
            obs_with_mask,
            policy_id=f"local_policy_{dc_id}",
            explore=False
        )
        return int(action)

    def close(self):
        """清理资源"""
        import ray
        if ray.is_initialized():
            ray.shutdown()


def find_latest_checkpoint(experiment_dir: str) -> Optional[str]:
    """在实验目录中找到最新的 checkpoint"""
    exp_path = Path(experiment_dir)

    # 查找 multidc_training/PPO_*/checkpoint_* 目录
    training_dir = exp_path / "multidc_training"
    if not training_dir.exists():
        return None

    # 找到 PPO 目录
    ppo_dirs = list(training_dir.glob("PPO_*"))
    if not ppo_dirs:
        return None

    ppo_dir = ppo_dirs[0]

    # 找到最新的 checkpoint
    checkpoints = sorted(ppo_dir.glob("checkpoint_*"),
                         key=lambda x: int(x.name.split('_')[-1]))
    if not checkpoints:
        return None

    return str(checkpoints[-1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load RLlib multi-agent model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (directory)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment directory (auto-find latest checkpoint)")
    parser.add_argument("--test", action="store_true",
                        help="Run a quick inference test")

    args = parser.parse_args()

    # 确定 checkpoint 路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.experiment:
        checkpoint_path = find_latest_checkpoint(args.experiment)
        if not checkpoint_path:
            print(f"No checkpoint found in {args.experiment}")
            sys.exit(1)
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        # 默认使用最新实验
        default_exp = "/home/joshua/rl-cloudsimplus-greenscheduling/logs/experiment_multi_dc_10/20251201_204434"
        checkpoint_path = find_latest_checkpoint(default_exp)
        if checkpoint_path:
            print(f"Using default: {checkpoint_path}")
        else:
            print("Please provide --checkpoint or --experiment")
            sys.exit(1)

    # 加载模型
    wrapper = RLlibInferenceWrapper(checkpoint_path)

    if args.test:
        print("\nRunning inference test...")
        import numpy as np

        # 模拟全局观察（需要与训练时的观察空间匹配）
        num_dcs = wrapper.num_dcs
        batch_size = 10

        # 创建模拟观察
        mock_global_obs = {
            "dc_current_green_power_w": np.zeros(num_dcs, dtype=np.float32),
            "dc_current_power_w": np.zeros(num_dcs, dtype=np.float32),
            "dc_green_ratio": np.random.rand(num_dcs).astype(np.float32),
            "dc_cumulative_wasted_green_wh": np.zeros(num_dcs, dtype=np.float32),
            "dc_future_short_mean": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_short_trend": np.zeros(num_dcs, dtype=np.float32),
            "dc_future_long_mean": np.random.rand(num_dcs).astype(np.float32),
            "dc_future_long_peak_timing": np.zeros(num_dcs, dtype=np.float32),
            "dc_queue_sizes": np.random.randint(0, 100, num_dcs).astype(np.int32),
            "dc_utilizations": np.random.rand(num_dcs).astype(np.float32),
            "dc_available_pes": np.random.randint(0, 100, num_dcs).astype(np.int32),
            "dc_ram_utilizations": np.random.rand(num_dcs).astype(np.float32),
            "upcoming_cloudlets_count": 50,
            "batch_cloudlet_pes": np.random.randint(1, 8, batch_size).astype(np.int32),
            "batch_cloudlet_mi": np.random.randint(1000, 10000, batch_size).astype(np.int64),
            "upcoming_pes_distribution": np.array([10, 20, 20], dtype=np.int32),
            "load_imbalance": np.array([0.1], dtype=np.float32),
            "recent_completed": 100,
        }

        print(f"Global obs keys: {list(mock_global_obs.keys())}")
        global_action = wrapper.get_global_action(mock_global_obs)
        print(f"Global action (batch routing): {global_action}")

        print("\n✓ Test completed!")

    wrapper.close()
