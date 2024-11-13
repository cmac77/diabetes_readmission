#%% Base Imports
import sys
from pathlib import Path
import yaml
import numpy as np

#%% Third-Party Imports
from pyprojroot import here  # Ensure pyprojroot is installed

#%% Register NaN Constructor
def nan_constructor(loader, node):
    return np.nan

# Register the constructor for the 'tag:yaml.org,2002:null' tag
yaml.add_constructor("tag:yaml.org,2002:null", nan_constructor, Loader=yaml.SafeLoader)

#%% Local Imports
# Set the project root directory using here() and add it to sys.path
path_root = Path(here())
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

# Define path to the configuration file
path_config_settings = path_root / "configs" / "config_settings.yaml"

# Load Configuration Function
def load_config() -> dict:
    """Load configuration settings from config_settings.yaml."""
    if not path_config_settings.exists():
        print(f"Configuration file not found at {path_config_settings}. Using default settings.")
        return {}
    try:
        with path_config_settings.open("r") as file:
            config = yaml.safe_load(file)
            return config or {}
    except yaml.YAMLError as e:
        print(f"Error parsing the configuration file: {e}. Using default settings.")
        return {}

# Load the configuration once
config = load_config()
