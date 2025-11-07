# Migration Guide - Project Restructuring

## Overview

The DRL Manager project has been reorganized for better clarity and maintainability. This guide explains the changes and how to update your code.

## What Changed

### Directory Structure

**Before**:
```
drl-manager/
├── gym_cloudsimplus/
├── mnt/
│   ├── train.py
│   ├── train_hierarchical_multidc.py
│   ├── train_hierarchical_multidc_joint.py
│   ├── callbacks.py                    # ⚠️ Conflict with callbacks/
│   ├── callbacks/
│   ├── custom_networks.py
│   └── utils/
├── test_joint_training_integration.py  # In root
├── test_green_energy.py                # In root
├── analyze_training_complete.py        # In root
├── generate_workload.py                # In root
└── ...
```

**After**:
```
drl-manager/
├── gym_cloudsimplus/                   # Environment library (unchanged)
├── src/                                # NEW: Source code
│   ├── training/                       # Training scripts
│   ├── callbacks/                      # Callbacks (resolved conflict)
│   ├── networks/                       # Custom networks
│   └── utils/                          # Utilities
├── scripts/                            # NEW: Utility scripts
├── tests/                              # Test suite (consolidated)
├── entrypoint.py                       # Main entry (moved from mnt/)
└── setup.py                            # Updated
```

## Import Path Changes

### 1. Training Scripts

**Old**:
```python
# Direct import from mnt
import train
from train import train
```

**New**:
```python
# Import from src.training
from src.training import train_single_dc
from src.training.train_single_dc import train
```

### 2. Callbacks

**Old**:
```python
from callbacks.save_on_best_training_reward_callback import SaveOnBestTrainingRewardCallback
from mnt.callbacks import GreenEnergyMonitorCallback
```

**New**:
```python
from src.callbacks.save_on_best_reward import SaveOnBestTrainingRewardCallback
from src.callbacks.monitoring import GreenEnergyMonitorCallback
```

### 3. Utilities

**Old**:
```python
from utils.config_loader import load_config
from mnt.utils.config_loader import load_config
```

**New**:
```python
from src.utils.config_loader import load_config
```

### 4. Custom Networks

**Old**:
```python
from mnt.custom_networks import MyNetwork
```

**New**:
```python
from src.networks.custom_networks import MyNetwork
```

## File Renames

Several files were renamed for clarity:

| Old Name | New Name | Location |
|----------|----------|----------|
| `mnt/train.py` | `train_single_dc.py` | `src/training/` |
| `mnt/callbacks/save_on_best_training_reward_callback.py` | `save_on_best_reward.py` | `src/callbacks/` |
| `mnt/callbacks.py` | `monitoring.py` | `src/callbacks/` |
| `analyze_training_complete.py` | `analyze_training.py` | `scripts/` |

## Path Resolution in Scripts

### Training Scripts

**Old** (`mnt/train.py`):
```python
import sys
import os
# No sys.path manipulation needed (in mnt/)
```

**New** (`src/training/train_single_dc.py`):
```python
import sys
import os
# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### Test Scripts

**Old** (in root):
```python
# No sys.path manipulation needed
```

**New** (`tests/`):
```python
import sys
import os
# Add drl-manager root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```

## Entrypoint Changes

The `entrypoint.py` now uses a mode mapping to dynamically load training modules:

**Old**:
```python
mode_module = mode  # e.g., "train"
module = importlib.import_module(mode_module)
```

**New**:
```python
mode_mapping = {
    "train": "src.training.train_single_dc",
    "test": "src.training.test",
    "transfer": "src.training.transfer"
}
mode_module = mode_mapping.get(mode, f"src.training.{mode}")
module = importlib.import_module(mode_module)
```

## Running Scripts

### Entrypoint (Recommended)

**Before**:
```bash
cd drl-manager
python mnt/entrypoint.py --config ../config.yml
```

**After**:
```bash
cd drl-manager
python entrypoint.py --config ../config.yml
```

### Direct Training Scripts

**Before**:
```bash
cd drl-manager/mnt
python train.py
```

**After**:
```bash
cd drl-manager
python -m src.training.train_single_dc
# or
python src/training/train_single_dc.py
```

### Utility Scripts

**Before**:
```bash
cd drl-manager
python generate_workload.py
```

**After**:
```bash
cd drl-manager
python scripts/generate_workload.py
# or
python -m scripts.generate_workload
```

### Test Scripts

**Before**:
```bash
cd drl-manager
python test_joint_training_integration.py
```

**After**:
```bash
cd drl-manager
python tests/test_joint_training_integration.py
# or
python -m tests.test_joint_training_integration
```

## Config.yml - No Changes Needed

The `config.yml` file in the parent directory **does not need to be modified**. The entrypoint handles the new path structure automatically.

## Breaking Changes

### 1. Removed `mnt/` Directory

The entire `mnt/` directory has been removed. All functionality has been reorganized into `src/`, `scripts/`, and root-level files.

### 2. Callback Naming Conflict Resolved

- **Old**: Both `mnt/callbacks.py` and `mnt/callbacks/` existed (conflict!)
- **New**: `callbacks.py` → `src/callbacks/monitoring.py`

### 3. Module Paths

All imports from `mnt.*` must be updated to `src.*`.

## Benefits of New Structure

1. **Clear Separation**: Training, callbacks, networks, and utils are clearly organized
2. **No Conflicts**: Resolved `callbacks.py` vs `callbacks/` naming conflict
3. **Better Testability**: All tests in one `tests/` directory
4. **Professional Layout**: Follows Python best practices
5. **Easier Navigation**: Logical grouping by functionality

## Checklist for Migration

If you have custom code or scripts:

- [ ] Update all imports from `mnt.*` to `src.*`
- [ ] Update callback imports to use new names
- [ ] Update script paths if hardcoded
- [ ] Reinstall package: `pip install -e .`
- [ ] Run tests to verify: `python tests/test_joint_training_integration.py`
- [ ] Check that entrypoint works: `python entrypoint.py --config ../config.yml`

## Rollback (Not Recommended)

If you need to temporarily rollback, the old `mnt/` directory structure can be restored from git history:

```bash
git show HEAD~1:drl-manager/mnt/ > /tmp/mnt_backup
```

However, it's recommended to migrate to the new structure as it resolves several issues and improves maintainability.

## Support

If you encounter import errors:

1. **Verify installation**: `pip install -e .`
2. **Check Python path**: Ensure you're running from `drl-manager/` root
3. **Clear cache**: `find . -type d -name __pycache__ -exec rm -rf {} +`
4. **Reinstall**: `pip uninstall gym_cloudsimplus_lb && pip install -e .`

For questions or issues, check the main README.md or project documentation.
