import os
import sys
import logging
import random
import shutil
import argparse
import importlib
import traceback
import yaml
from datetime import datetime

# Import from reorganized src package
try:
    from src.utils.config_loader import load_config
except ImportError:
    # Handle potential import issues if structure changes
    print("Error: Could not import ConfigLoader. Make sure src/utils/config_loader.py exists.")
    sys.exit(1)

# Configure basic logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("Entrypoint")

# --- Constants ---
DEFAULT_CONFIG_FILE = "config.yml"
DEFAULT_EXPERIMENT_ID = "experiment_1"
DEFAULT_MODE = "train"


def set_seed_globally(seed):
    """Sets random seeds for Python, NumPy, and PyTorch."""
    try:
        # Import heavy deps lazily so importing entrypoint.py doesn't fail in minimal environments.
        import numpy as np
        import torch

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info(f"Global random seeds set to: {seed}")
    except Exception as e:
        logger.error(f"Failed to set seeds: {e}", exc_info=True)


def setup_logging(log_dir):
    """Sets up file logging handlers."""
    if not log_dir:
        logger.warning("Log directory not specified, only logging to console.")
        return None

    os.makedirs(log_dir, exist_ok=True)

    # Optional cleanup: remove previously created "latest pointer" artifacts at the experiment root.
    # We only remove symlinks (safe) and legacy text pointers.
    try:
        legacy_names = [
            "latest",
            "latest_run.txt",
            "monitor.csv",
            "progress.csv",
            "best_model.zip",
            "final_model.zip",
            "best_episode_details_10.csv",
            "config_used.yml",
            "seed_used.txt",
        ]
        for name in legacy_names:
            p = os.path.join(log_dir, name)
            if os.path.islink(p):
                os.unlink(p)
            elif name == "latest_run.txt" and os.path.exists(p) and os.path.isfile(p):
                os.remove(p)
    except Exception:
        pass

    # Use SECOND precision to ensure each run gets its own directory (avoid overwriting within the same minute).
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(log_dir, timestamp)

    # Create run directory if not exists (may have been created by Java)
    os.makedirs(run_dir, exist_ok=True)

    # Remove previous basic console handler to avoid duplicate messages
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    # Define new handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handlers = [
        # Per-run logs go into the timestamped run directory to avoid overwriting previous runs.
        logging.FileHandler(os.path.join(run_dir, 'current_run.log'), mode='w'),
        logging.FileHandler(os.path.join(run_dir, 'run.log'), mode='w'),
        logging.StreamHandler(sys.stdout) # Log to console
    ]

    # Apply formatter and add handlers
    root_logger.handlers.clear() # Clear existing handlers first
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(logging.INFO) # Set desired root logging level
    # Adjust levels for specific libraries if needed
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("stable_baselines3").setLevel(logging.INFO)
    logging.getLogger("sb3_contrib").setLevel(logging.INFO)

    logger.info(
        "Logging setup complete. Run directory: %s (run.log=%s)",
        run_dir,
        os.path.join(run_dir, 'run.log')
    )

    return run_dir

def main():
    logger.info("--- DRL Manager Entrypoint Starting ---")

    # --- Parse Command Line Arguments ---
    parser = argparse.ArgumentParser(description='DRL Manager Entrypoint')
    parser.add_argument('--exp', '--experiment', type=str, default=None,
                        help='Experiment ID to run (e.g., experiment_1, experiment_single_dc_local_csv)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yml)')
    args = parser.parse_args()

    # --- Determine Experiment Config ---
    # Priority: command line args > environment variables > defaults
    config_file = args.config or os.getenv("CONFIG_FILE", DEFAULT_CONFIG_FILE)
    experiment_id = args.exp or os.getenv("EXPERIMENT_ID", DEFAULT_EXPERIMENT_ID)

    logger.info(f"Using experiment: {experiment_id}, config: {config_file}")

    # --- Load Configuration ---
    params = load_config(config_file=config_file, experiment_id=experiment_id)
    if params is None:
        logger.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # --- Set Seed ---
    # Use seed from config, fallback to random if specified or missing
    seed_value = params.get("seed", "random")
    if isinstance(seed_value, str) and seed_value.lower() == "random":
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Generated random seed: {seed}")
    else:
        try:
            seed = int(seed_value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid seed value '{seed_value}' in config. Using random seed.")
            seed = random.randint(0, 2**32 - 1)
    params['seed'] = seed # Store the actual seed used back into params
    set_seed_globally(seed)

    # --- Setup Logging Directory and Handlers ---
    log_dir = None
    if params.get("save_experiment", False):
        base_log_dir = params.get("base_log_dir", "logs")
        exp_type_dir = params.get("experiment_type_dir", "DefaultType")
        # Use experiment_name from config, fallback to experiment_id
        exp_name = params.get("experiment_name", experiment_id)
        experiment_dir = os.path.join(base_log_dir, exp_type_dir, exp_name)
        run_log_dir = setup_logging(experiment_dir) # Setup file handlers etc. (returns per-run dir)
        # Persist both: a stable experiment directory and a per-run output directory.
        params['experiment_dir'] = experiment_dir
        params['log_dir'] = run_log_dir
        log_dir = run_log_dir

        # Save config and seed to log directory
        try:
            os.makedirs(log_dir, exist_ok=True)
            config_save_path = os.path.join(log_dir, "config_used.yml")
            seed_save_path = os.path.join(log_dir, "seed_used.txt")
            # Try copying original first
            try:
                 shutil.copy(config_file, config_save_path)
            except Exception:
                 # Fallback to writing loaded params if copy fails
                 with open(config_save_path, 'w') as f:
                      yaml.dump(params, f, default_flow_style=False)
            with open(seed_save_path, 'w') as f:
                 f.write(str(seed))
            logger.info(f"Saved config and seed to {log_dir}")
        except Exception as e:
            logger.error(f"Could not save config/seed to log directory: {e}", exc_info=True)
    else:
        params['log_dir'] = None # Ensure log_dir is None if not saving
        params['experiment_dir'] = None
        logger.info("Experiment saving is disabled.")

    # --- Execute Selected Mode ---
    mode = params.get("mode", DEFAULT_MODE)
    logger.info(f"Selected mode: {mode}")

    # --- Check for Multi-Datacenter Mode ---
    is_multi_dc = params.get("multi_datacenter_enabled", False)
    if is_multi_dc and mode == "train":
        logger.error("Multi-datacenter mode detected!")
        logger.error("Please use 'entrypoint_pettingzoo.py' for multi-DC training with RLlib.")
        logger.error("Example: python entrypoint_pettingzoo.py")
        sys.exit(1)
    else:
        # Map mode to new module structure
        mode_mapping = {
            "train": "src.training.train_single_dc",
            "test": "src.training.test",  # If test module exists
            "transfer": "src.training.transfer"  # If transfer module exists
        }
        mode_module = mode_mapping.get(mode, f"src.training.{mode}")
        func_name = mode

    try:
        # Dynamically import the module corresponding to the mode
        # Modules are now in src/training/ directory
        try:
            logger.info(f"Attempting to import module: {mode_module}")
            module = importlib.import_module(mode_module)
            logger.info(f"Successfully imported module: {mode_module}")
        except ModuleNotFoundError as e:
            logger.error(f"Mode script '{mode_module}.py' not found in src/training/ directory.")
            logger.error(f"Available modes are: 'train', 'transfer', 'test'.")
            logger.error(f"Error details: {e}")
            sys.exit(1)

        # Get the function with the same name as the mode
        logger.info(f"Attempting to get function: {func_name} from module: {mode_module}")
        func = getattr(module, func_name)
        logger.info(f"Successfully found function: {func_name}")

        # Execute the function, passing the parameters
        logger.info(f"Executing function: {func_name}")
        func(params)

    except AttributeError as e:
        logger.error(f"Function '{func_name}' not found within '{mode_module}.py'.")
        logger.error(f"Error details: {e}")
        logger.error(f"Available functions in module: {dir(module)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An error occurred during execution of mode '{mode}': {e}", exc_info=True)
        traceback.print_exc() # Print detailed traceback
        sys.exit(1)

    logger.info(f"--- DRL Manager Entrypoint Finished Mode '{mode}' ---")

if __name__ == "__main__":
    main()
