from gymnasium.envs.registration import register

register(
    id="LoadBalancingScaling-v0",
    entry_point="gym_cloudsimplus.envs:LoadBalancingEnv",
    # Optional: Add max_episode_steps if you want Gym to handle truncation
    # max_episode_steps=1000,
)

register(
    id="HierarchicalMultiDC-v0",
    entry_point="gym_cloudsimplus.envs:HierarchicalMultiDCEnv",
    # Multi-datacenter hierarchical MARL environment
    # max_episode_steps=2000,
)

register(
    id="HierarchicalMultiDCSimple-v0",
    entry_point="gym_cloudsimplus.envs:HierarchicalMultiDCEnvSimple",
    # Simplified multi-DC environment WITHOUT God's Eye future prediction features
    # Suitable for fair comparison and standard RL algorithms (PPO, A2C, DQN)
)

register(
    id="HierarchicalMultiDCAblation-v0",
    entry_point="gym_cloudsimplus.envs:HierarchicalMultiDCEnvAblation",
    # Ablation env with config-knob ``forecast_mode``:
    #   full | none | short_only | long_only | no_peak | raw
    # See gym_cloudsimplus/envs/hierarchical_multidc_env_ablation.py
)
