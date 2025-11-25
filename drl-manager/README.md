# DRL Manager - CloudSimPlus Reinforcement Learning

Deep Reinforcement Learning manager for cloud scheduling using CloudSimPlus and Gymnasium.

## Project Structure

```
drl-manager/
├── gym_cloudsimplus/          # Gymnasium environments
│   ├── __init__.py
│   └── envs/
│       ├── __init__.py
│       ├── loadbalancing_env.py              # Single-datacenter environment
│       ├── hierarchical_multidc_env.py       # Multi-datacenter environment
│       └── joint_training_env.py             # Joint training wrapper for MARL
│
├── src/                       # Source code for training and utilities
│   ├── __init__.py
│   ├── training/              # Training scripts
│   │   ├── __init__.py
│   │   ├── train_single_dc.py                # Single-DC training
│   │   ├── train_hierarchical_multidc.py     # Multi-DC training
│   │   └── train_hierarchical_multidc_joint.py  # Joint MARL training
│   ├── callbacks/             # Custom callbacks for training
│   │   ├── __init__.py
│   │   ├── save_on_best_reward.py            # Checkpoint on best reward
│   │   └── monitoring.py                     # Green energy monitoring
│   ├── networks/              # Custom neural networks
│   │   ├── __init__.py
│   │   └── custom_networks.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── config_loader.py                  # Configuration loader
│
├── scripts/                   # Utility scripts
│   ├── __init__.py
│   ├── generate_workload.py                  # Workload generation
│   ├── analyze_training.py                   # Training analysis
│   ├── verify_data_pipeline.py               # Data pipeline verification
│   └── monitor_success_rate.py               # Success rate monitoring
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_joint_training_integration.py    # Integration tests
│   ├── test_green_energy.py                  # Green energy tests
│   ├── cuda_test.py                          # CUDA availability test
│   └── gateway_test.py                       # Py4J gateway test
│
├── entrypoint.py              # Main entry point
├── setup.py                   # Package configuration
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## Installation

### From Source

```bash
cd drl-manager
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Using Entry Point

The main entry point dynamically loads training scripts based on the mode specified in `config.yml`:

```bash
# From drl-manager directory
python entrypoint.py --config ../config.yml --experiment experiment_1
```

### Direct Training Scripts

You can also run training scripts directly:

```bash
# Single-DC training
python -m src.training.train_single_dc

# Multi-DC hierarchical training
python -m src.training.train_hierarchical_multidc

# Joint MARL training
python -m src.training.train_hierarchical_multidc_joint \
    --config ../config.yml \
    --output_dir ../logs/joint_training \
    --total_timesteps 100000 \
    --strategy alternating
```

### Running Tests

```bash
# Run integration tests
python tests/test_joint_training_integration.py

# Run green energy tests
python tests/test_green_energy.py

# Check CUDA availability
python tests/cuda_test.py
```

### Utility Scripts

```bash
# Generate workload
python scripts/generate_workload.py --output ../traces/my_workload.csv

# Analyze training results
python scripts/analyze_training.py --log_dir ../logs/experiment_1

# Monitor success rate
python scripts/monitor_success_rate.py --experiment experiment_1
```

## Configuration

Training configuration is managed through `config.yml` in the parent directory. Key sections:

- `common`: Shared parameters for all experiments
- `experiment_X`: Specific experiment configurations
- `experiment_multi_dc_3`: Multi-datacenter MARL configuration

## Key Features

### Single-Datacenter Training
- VM scaling and load balancing
- MaskablePPO with action masking
- Real-time workload processing

### Multi-Datacenter MARL
- Hierarchical decision-making (Global + Local agents)
- Green energy optimization (50% of global reward)
- Parameter sharing across local agents
- Action masking for efficient exploration

### Green Energy Optimization
- Real-time wind power integration
- Waste penalty for unused green energy
- Carbon emission reduction
- Power consumption tracking

## Development

### Adding New Training Scripts

1. Create your script in `src/training/`
2. Add the main function with signature: `def train(params: dict)`
3. Update `entrypoint.py` mode mapping if needed

### Adding Custom Callbacks

1. Create callback in `src/callbacks/`
2. Inherit from `stable_baselines3.common.callbacks.BaseCallback`
3. Import in `src/callbacks/__init__.py`

### Adding Utility Scripts

1. Place script in `scripts/`
2. Make executable: `chmod +x scripts/your_script.py`
3. Add shebang: `#!/usr/bin/env python`

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. You're running from the `drl-manager` directory
2. The package is installed: `pip install -e .`
3. Python path includes drl-manager root

### Java Connection Issues

Ensure Java gateway is running and Py4J port matches `config.yml`:
```bash
# Check gateway
python tests/gateway_test.py
```

### CUDA Not Available

Check CUDA installation:
```bash
python tests/cuda_test.py
```

## Dependencies

Core:
- gymnasium >= 0.29.0
- py4j >= 0.10.9
- numpy >= 1.24.0
- torch >= 2.0.0
- stable-baselines3 >= 2.0.0
- sb3-contrib >= 2.0.0
- pyyaml >= 6.0

Development:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0

## License

See parent directory for license information.

## Contributing

See parent directory for contribution guidelines.
