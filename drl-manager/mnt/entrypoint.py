import os
import sys
import logging
import random
import shutil
import numpy as np
import torch
import importlib
import traceback
import yaml
from datetime import datetime

# Assuming utils are in the same mnt directory structure
try:
    from utils.config_loader import load_config
except ImportError:
    # Handle potential import issues if structure changes
    print("Error: Could not import ConfigLoader. Make sure utils/config_loader.py exists.")
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
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU.
            # Potentially enable deterministic algorithms for reproducibility
            # Note: This can impact performance and might not work with all ops
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.use_deterministic_algorithms(True)
        os.environ['PYTHONHASHSEED'] = str(seed)
        logger.info(f"Global random seeds set to: {seed}")
    except Exception as e:
        logger.error(f"Failed to set seeds: {e}", exc_info=True)


def setup_logging(log_dir):
    """Sets up file logging handlers."""
    if not log_dir:
        logger.warning("Log directory not specified, only logging to console.")
        return

    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp matching Java's format but with MINUTE precision
    timestamp_minute = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir_minute = os.path.join(log_dir, timestamp_minute)

    # Create run directory if not exists (may have been created by Java)
    os.makedirs(run_dir_minute, exist_ok=True)

    # Remove previous basic console handler to avoid duplicate messages
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    # Define new handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handlers = [
        logging.FileHandler(os.path.join(log_dir, 'current_run.log'), mode='w'),
        logging.FileHandler(os.path.join(run_dir_minute, 'run.log'), mode='w'),
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

    logger.info(f"Logging setup complete. Current log: {os.path.join(log_dir, 'current_run.log')}, Run log: {os.path.join(run_dir_minute, 'run.log')}")

def main():
    logger.info("--- DRL Manager Entrypoint Starting ---")

    # --- Determine Experiment Config ---
    # Use environment variables set by docker-compose or run script
    config_file = os.getenv("CONFIG_FILE", DEFAULT_CONFIG_FILE)
    # Default to experiment_1 if not specified
    experiment_id = os.getenv("EXPERIMENT_ID", DEFAULT_EXPERIMENT_ID)

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
        log_dir = os.path.join(base_log_dir, exp_type_dir, exp_name)
        params['log_dir'] = log_dir # Add final log_dir path to params dict
        setup_logging(log_dir) # Setup file handlers etc.

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
        logger.info("Experiment saving is disabled.")

    # --- Execute Selected Mode ---
    mode = params.get("mode", DEFAULT_MODE)
    logger.info(f"Selected mode: {mode}")

    try:
        # Dynamically import the module corresponding to the mode
        # Assumes train.py, test.py, transfer.py are in the same directory (mnt)
        try:
            module = importlib.import_module(mode)
        except ModuleNotFoundError:
            print(
                f"Mode {params['mode']} was not found. Available modes are: 'train', 'transfer', 'test'."
            )
        # Get the function with the same name as the mode
        func = getattr(module, mode)
        # Execute the function, passing the parameters
        func(params)
    except ModuleNotFoundError:
        logger.error(f"Mode script '{mode}.py' not found in mnt directory.")
        sys.exit(1)
    except AttributeError:
        logger.error(f"Function '{mode}' not found within '{mode}.py'.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An error occurred during execution of mode '{mode}': {e}", exc_info=True)
        traceback.print_exc() # Print detailed traceback
        sys.exit(1)

    logger.info(f"--- DRL Manager Entrypoint Finished Mode '{mode}' ---")

if __name__ == "__main__":
    main()
