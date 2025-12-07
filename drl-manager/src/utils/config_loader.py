import yaml
import os
import logging

logger = logging.getLogger(__name__.split('.')[-1])

def load_config(config_file="config.yml", experiment_id="experiment_1"):
    """
    Loads parameters from a YAML config file, merging common and specific sections.

    Args:
        config_file (str): Path to the YAML configuration file.
        experiment_id (str): The key for the specific experiment section (e.g., "experiment_1").

    Returns:
        dict: A dictionary containing the merged parameters, or None if loading fails.
    """
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        return None

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            full_config = yaml.safe_load(file)

        if full_config is None:
            logger.error(f"Configuration file is empty or invalid: {config_file}")
            return None

        # Get common and experiment params safely using get() with default empty dict
        common_params = full_config.get("common", {})
        experiment_params = full_config.get(experiment_id, {})

        # Check if they are actually dictionaries before unpacking
        if not isinstance(common_params, dict):
             logger.error(f"'common' section in {config_file} is not a dictionary.")
             common_params = {}
        if not isinstance(experiment_params, dict):
             logger.warning(f"Experiment section '{experiment_id}' in {config_file} is not a dictionary.")
             # If experiment key doesn't exist, get() returns {}, so this check also handles missing keys
             experiment_params = {}

        # --- Use dictionary unpacking for concise merging ---
        merged_params = {**common_params, **experiment_params}
        # -----------------------------------------------------

        # Add experiment_id itself to the params for potential use
        merged_params['experiment_id'] = experiment_id
        merged_params['experiment_name'] = merged_params.get('experiment_name', experiment_id) # Ensure name exists


        logger.info(f"Loaded configuration for experiment '{experiment_id}' from {config_file}")
        return merged_params

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {e}")
        return None

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
     config_path = "config.yml"

     print(f"Looking for config at: {config_path}")

     test_params = load_config(config_path, "experiment_1")
     if test_params:
         print("\nLoaded Params for experiment_1:")
         from pprint import pprint
         pprint(test_params)
     else:
         print("\nFailed to load params for experiment_1.")

     test_params_2 = load_config(config_path, "experiment_2")
     if test_params_2:
         print("\nLoaded Params for experiment_2:")
         from pprint import pprint
         pprint(test_params_2)
     else:
          print("\nFailed to load params for experiment_2.")

     test_params_missing = load_config(config_path, "experiment_nonexistent")
     if test_params_missing:
          print("\nLoaded Params for experiment_nonexistent (should be common + ID):")
          from pprint import pprint
          pprint(test_params_missing)
     else:
          print("\nFailed to load params for experiment_nonexistent.")
