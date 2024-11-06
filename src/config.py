# %%
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Define path to the configuration file
path_config_settings = Path("configs") / "config_settings.yaml"


def nan_constructor(loader, node):
    return np.nan


# Register the constructor for the 'tag:yaml.org,2002:null' tag
yaml.add_constructor("tag:yaml.org,2002:null", nan_constructor, Loader=yaml.SafeLoader)


# Load config function
def load_config() -> Dict[str, Any]:
    """
    Load configuration settings from config_settings.yaml.
    If the configuration file does not exist, returns an empty dictionary and logs a warning.

    Returns:
        Dict[str, Any]: Configuration settings.
    """
    if not path_config_settings.exists():
        print(
            f"Configuration file not found at {path_config_settings}. Using default settings."
        )
        return {}
    try:
        with path_config_settings.open("r") as file:
            config = yaml.safe_load(file)
            if config is None:
                config = {}
            return config
    except yaml.YAMLError as e:
        print(f"Error parsing the configuration file: {e}. Using default settings.")
        return {}


# Load the configuration once and make it accessible
config = load_config()
# %%
# %%

# import yaml
# from pathlib import Path
# from typing import Dict, Any

# # Define path to the configuration file
# path_config_settings = Path("configs") / "config_settings.yaml"


# def load_config() -> Dict[str, Any]:
#     """
#     Load configuration settings from config_settings.yaml.
#     If the configuration file does not exist, returns an empty dictionary and logs a warning.

#     Returns:
#         Dict[str, Any]: Configuration settings.
#     """
#     if not path_config_settings.exists():
#         print(
#             f"Configuration file not found at {path_config_settings}. Using default settings."
#         )
#         return {}
#     try:
#         with path_config_settings.open("r") as file:
#             config = yaml.safe_load(file)
#             if config is None:
#                 config = {}
#             return config
#     except yaml.YAMLError as e:
#         print(f"Error parsing the configuration file: {e}. Using default settings.")
#         return {}


# # Load the configuration once and make it accessible
# config = load_config()


# # Define paths using configuration with sensible defaults
# def get_path(key: str, default: str) -> Path:
#     """
#     Retrieve a path from the configuration, providing a default if the key is missing.

#     Parameters:
#         key (str): The key path in the configuration dictionary (e.g., "paths.data_raw").
#         default (str): The default path to use if the key is not found.

#     Returns:
#         Path: The resolved filesystem path.
#     """
#     keys = key.split(".")
#     value = config
#     for k in keys:
#         if isinstance(value, dict):
#             value = value.get(k, {})
#         else:
#             value = {}
#     if isinstance(value, dict):
#         return Path(default)
#     return Path(value) if value else Path(default)


# # Usage in the project:
# path_data_raw = get_path("paths.data_raw", "data/raw/data.csv")
# path_data_cleaned = get_path("paths.data_cleaned", "data/processed/df_cleaned.pkl")
# path_key_file = get_path("paths.key_file", "data/raw/key.csv")
# path_results = get_path("paths.results", "results")
# path_scripts = get_path("paths.scripts", "scripts")
# path_src = get_path("paths.src", "src")
# path_logs = get_path("paths.logs", "results/logs")
