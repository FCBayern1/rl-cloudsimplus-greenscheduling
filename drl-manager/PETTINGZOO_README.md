# PettingZoo Environment for Hierarchical Multi-DC MARL

This directory contains the PettingZoo-compatible wrapper for the Hierarchical Multi-Datacenter environment.

## 📁 New Files

### Core Environment
```
gym_cloudsimplus/envs/
└── hierarchical_multidc_pettingzoo.py  (NEW)
    - PettingZoo ParallelEnv wrapper
    - ~500 lines
    - Wraps HierarchicalMultiDCEnv without modifying it
```

### Testing
```
tests/
├── test_pettingzoo_env.py           (NEW)
│   - Comprehensive test suite
│   - ~350 lines
│   - Requires Java gateway
│
└── quick_verify_pettingzoo.py       (NEW)
    - Quick import/structure verification
    - ~90 lines
    - No Java gateway required
```

### Examples
```
examples/
└── example_pettingzoo_usage.py      (NEW)
    - Demonstrates basic usage
    - Random policy example
    - Action masking demo
    - Observation structure demo
```

### Documentation
```
ROOT/
└── PETTINGZOO_MIGRATION_SUMMARY.md  (NEW)
    - Complete migration guide
    - Architecture explanation
    - Usage instructions
```

---

## 🚀 Quick Start

### 1. Verify Installation (No Java Required)

```bash
cd drl-manager/tests
../.venv/Scripts/python.exe quick_verify_pettingzoo.py
```

Expected output:
```
[1/6] Testing PettingZoo import...          [OK]
[2/6] Testing base environment import...    [OK]
[3/6] Testing PettingZoo wrapper import...  [OK]
[4/6] Testing class structure...            [OK]
[5/6] Testing environment metadata...       [OK]
[6/6] Testing convenience import...         [OK]

[SUCCESS] All verification checks passed!
```

### 2. Run Example (Requires Java Gateway)

Terminal 1 - Start Java Gateway:
```bash
cd cloudsimplus-gateway
./gradlew run
```

Terminal 2 - Run Example:
```bash
cd drl-manager/examples
../.venv/Scripts/python.exe example_pettingzoo_usage.py
```

### 3. Run Full Tests (Requires Java Gateway)

```bash
cd drl-manager/tests
../.venv/Scripts/python.exe test_pettingzoo_env.py
```

---

## 💻 Basic Usage

```python
from gym_cloudsimplus.envs import HierarchicalMultiDCParallelEnv

# Create environment
config = {...}  # Your config dict
env = HierarchicalMultiDCParallelEnv(config)

# Reset
observations, infos = env.reset(seed=42)

# observations = {
#     "global_agent": {...},      # DC routing agent
#     "local_agent_0": {...},     # VM scheduling agent for DC 0
#     "local_agent_1": {...},     # VM scheduling agent for DC 1
#     ...
# }

# Step
actions = {
    "global_agent": np.array([0, 1, 2, 0, 1]),  # Route 5 cloudlets
    "local_agent_0": 3,  # Assign to VM 3 in DC 0
    "local_agent_1": 1,  # Assign to VM 1 in DC 1
    ...
}

observations, rewards, terminations, truncations, infos = env.step(actions)

# Action masking
mask = env.get_action_mask("local_agent_0")
valid_actions = np.where(mask)[0]

# Close
env.close()
```

---

## 🎓 Key Concepts

### Agents

The environment creates N+1 agents for N datacenters:

- **1 Global Agent** (`"global_agent"`)
  - Routes arriving cloudlets to datacenters
  - Action space: `MultiDiscrete([num_dcs] * batch_size)`
  - No action masking

- **N Local Agents** (`"local_agent_0"`, `"local_agent_1"`, ...)
  - Schedule cloudlets to VMs within each datacenter
  - Action space: `Discrete(num_vms + 1)` (0 = NoAssign, 1-N = VM indices)
  - Support action masking

### Parameter Sharing

All local agents can share the same neural network via policy mapping:

```python
# RLlib example
"policy_mapping_fn": lambda agent_id: (
    "global_policy" if agent_id == "global_agent" else "local_policy"
)
```

### Action Masking

Local agents support action masking to prevent invalid VM assignments:

```python
# Get mask for specific agent
mask = env.get_action_mask("local_agent_0")

# Get masks for all agents
all_masks = env.get_all_action_masks()

# Use with MaskablePPO (sb3-contrib)
from sb3_contrib import MaskablePPO
```

---

## 🔍 Architecture

```
┌────────────────────────────────────────┐
│  Training Framework                     │
│  (RLlib, CleanRL, SB3 + adapter)       │
└───────────────┬────────────────────────┘
                │ PettingZoo API
                ↓
┌────────────────────────────────────────┐
│  HierarchicalMultiDCParallelEnv (NEW)  │
│  - Format conversion                    │
│  - Agent management                     │
│  - Action masking                       │
└───────────────┬────────────────────────┘
                │ Wraps (unchanged)
                ↓
┌────────────────────────────────────────┐
│  HierarchicalMultiDCEnv (ORIGINAL)     │
│  - Hierarchical structure               │
│  - Py4J communication                   │
└───────────────┬────────────────────────┘
                │ Py4J
                ↓
┌────────────────────────────────────────┐
│  Java CloudSim Plus (UNCHANGED)        │
└────────────────────────────────────────┘
```

---

## 📊 Compatibility

### Works With
- ✅ RLlib (Ray) - Native support
- ✅ CleanRL - Native support
- ✅ AgileRL - Native support
- ✅ Stable-Baselines3 - Via PettingZoo-to-VecEnv adapter

### Training Frameworks

#### Option 1: RLlib (Recommended)
```python
from ray.rllib.algorithms.ppo import PPO

config = {
    "env": HierarchicalMultiDCParallelEnv,
    "multiagent": {...}
}

algo = PPO(config=config)
algo.train()
```

#### Option 2: CleanRL
```python
# Use CleanRL's PPO implementation
# With custom multi-agent training loop
```

#### Option 3: SB3 (via adapter)
```python
from supersuit import pettingzoo_env_to_vec_env_v1

vec_env = pettingzoo_env_to_vec_env_v1(env)
model = PPO("MlpPolicy", vec_env)
```

---

## 🆚 Comparison: Original vs PettingZoo

| Feature | Original (SB3) | PettingZoo |
|---------|---------------|------------|
| **API** | Custom hierarchical | Standard PettingZoo |
| **Framework** | SB3 only | RLlib, CleanRL, SB3 |
| **Agent format** | Nested dict | Flat dict |
| **Training** | Custom loop | Standard MARL |
| **Compatibility** | Limited | Wide |
| **Code changes** | None needed | None needed |
| **Files modified** | Original preserved | Original preserved |

Both versions coexist - use whichever fits your needs!

---

## 🔧 Troubleshooting

### Import Error
```
ImportError: cannot import name 'HierarchicalMultiDCParallelEnv'
```

**Solution**: Check `__init__.py` has the import:
```python
from gym_cloudsimplus.envs.hierarchical_multidc_pettingzoo import HierarchicalMultiDCParallelEnv
```

### Connection Error
```
RuntimeError: Could not connect to Java gateway
```

**Solution**: Start Java gateway first:
```bash
cd cloudsimplus-gateway
./gradlew run
```

### SuperSuit Not Available
SuperSuit is optional and not needed for core functionality. If you need it:
```bash
# Option 1: Try older version
pip install supersuit==3.7.0

# Option 2: Use conda
conda install -c conda-forge supersuit
```

---

## 📚 Further Reading

- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [ParallelEnv API](https://pettingzoo.farama.org/api/parallel/)
- [RLlib Multi-Agent Guide](https://docs.ray.io/en/latest/rllib/rllib-training.html#multi-agent-training)
- [PETTINGZOO_MIGRATION_SUMMARY.md](../PETTINGZOO_MIGRATION_SUMMARY.md)

---

## 📝 Next Steps

1. **Run full tests** with Java gateway
2. **Create training script** for RLlib/CleanRL
3. **Run first experiment** with PettingZoo environment
4. **Compare performance** with SB3 baseline

---

## ❓ FAQ

**Q: Do I need to modify Java code?**
A: No, zero Java changes required.

**Q: Can I still use the original SB3 training?**
A: Yes, both versions coexist without conflicts.

**Q: Is SuperSuit required?**
A: No, core PettingZoo functionality works without it.

**Q: Which training framework should I use?**
A: RLlib for production, CleanRL for research, SB3 for familiarity.

**Q: How do I enable parameter sharing?**
A: Use policy mapping in RLlib/CleanRL to map all local agents to one policy.

---

**Status**: ✅ Phase 1 Complete
**Last Updated**: 2025-01-12
**Contact**: See project README
