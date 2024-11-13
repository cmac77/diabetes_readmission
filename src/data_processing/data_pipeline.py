#%% src/data_processing/data_pipeline.py

# Standard library imports
import logging
from pathlib import Path
import sys
from typing import Tuple

# Third-party imports
import pandas as pd
from pyprojroot import here

#%% Set the project root directory using here() and add it to sys.path
path_root = here()
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))


#%% Local imports (only after confirming src is on the path)
from src.config import config
from src.utils.logging_utils import setup_logging
from src.data_processing.data_frame_column_cleaner import DataFrameColumnCleaner


# %% Configure Logger
def configure_logger() -> logging.Logger:
    """
    Sets up the logger with the specified configuration.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_path = Path(config.get("paths", {}).get("logs", "results/logs"))
    log_path.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
    log_file = log_path / "data_pipeline.log"
    setup_logging(log_file)

    logger = logging.getLogger("DataPipeline")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if already configured
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = configure_logger()
logger.info("Data pipeline logging is configured.")

# %% Define Paths
path_data_raw: Path = Path(config.get("paths", {}).get("data_raw", "data/raw/data.csv"))
path_data_cleaned: Path = Path(
    config.get("paths", {}).get("data_cleaned", "data/processed/df_cleaned.pkl")
)
path_key_file: Path = Path(config.get("paths", {}).get("key_file", "data/raw/key.csv"))


# %% Load Data
def load_data(path_data_cleaned: Path, path_data_raw: Path) -> pd.DataFrame:
    """
    Load cleaned data if available; otherwise, load raw data, clean it, and save the cleaned version.

    Parameters:
        path_data_cleaned (Path): Path to the cleaned data file.
        path_data_raw (Path): Path to the raw data file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        if path_data_cleaned.exists():
            logger.info("Loading cleaned data from saved file.")
            df_cleaned = pd.read_pickle(path_data_cleaned)
            df_parquet = pd.read_parquet(Path(path_data_cleaned.parent, "df_cleaned.parquet"), engine="pyarrow")
        else:
            logger.info("Loading raw data and starting cleaning process.")
            try:
                df_raw = pd.read_csv(path_data_raw)
                logger.info("Raw data loaded successfully.")
            except FileNotFoundError as e:
                logger.error(f"Raw data file not found at {path_data_raw}")
                raise e

            # Clean data using DataFrameColumnCleaner
            config_cleaner = config.get("config_cleaner", {})
            cleaner = DataFrameColumnCleaner(config=config_cleaner)
            df_cleaned = cleaner.process_dataframe(df_raw.copy())
            logger.info("Data cleaning completed.")

            # Save cleaned data
            try:
                path_data_cleaned.parent.mkdir(parents=True, exist_ok=True)
                df_cleaned.to_pickle(path_data_cleaned)
                df_cleaned.to_parquet(Path(path_data_cleaned.parent, "df_cleaned.parquet"), engine="pyarrow")
                logger.info(f"Cleaned data saved as pkl and parquet at {path_data_cleaned}.")
            except IOError as e:
                logger.error(f"Failed to save cleaned data: {e}")
                raise e
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise e

    return df_cleaned


# %% Load and Process Key File
def load_key(key_file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process the key file for mapping IDs.

    Parameters:
        key_file_path (Path): Path to the key file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Admission Type Mapping
            - Discharge Disposition Mapping
            - Admission Source Mapping
    """
    logger.info("Loading key file.")
    try:
        df_key = pd.read_csv(key_file_path)
    except FileNotFoundError as e:
        logger.error(f"Key file not found at {key_file_path}")
        raise e
    except pd.errors.EmptyDataError as e:
        logger.error("Key file is empty.")
        raise e

    # Process key file with blank rows as delimiters
    try:
        mask = df_key["admission_type_id"].isna()
        blank_indices = mask[mask].index.tolist()

        if len(blank_indices) < 2:
            logger.error("Key file format incorrect; expected two blank rows.")
            raise ValueError("Invalid key file format.")

        # Parse mappings
        id_admission_type = df_key.loc[: blank_indices[0] - 1].reset_index(drop=True)
        id_discharge_disposition = df_key.loc[
            blank_indices[0] + 1 : blank_indices[1] - 1
        ].reset_index(drop=True)
        id_discharge_disposition.columns = id_discharge_disposition.iloc[0]
        id_discharge_disposition = id_discharge_disposition.drop(0).reset_index(
            drop=True
        )
        id_admission_source = df_key.loc[blank_indices[1] + 1 :].reset_index(drop=True)
        id_admission_source.columns = id_admission_source.iloc[0]
        id_admission_source = id_admission_source.drop(0).reset_index(drop=True)

        logger.info("Key file processed successfully.")
    except Exception as e:
        logger.error(f"Error processing key file: {e}")
        raise e

    return id_admission_type, id_discharge_disposition, id_admission_source


# %% Main Function to Execute the Pipeline
def main() -> None:
    """
    Execute the entire data processing pipeline.
    """
    logger.info("Starting data pipeline.")

    try:
        # Load and clean data
        df_cleaned = load_data(
            path_data_cleaned=path_data_cleaned, path_data_raw=path_data_raw
        )
        df_parquet = pd.read_parquet(Path(path_data_cleaned.parent, "df_cleaned.parquet"), engine="pyarrow")


        # Process the key file
        id_admission_type, id_discharge_disposition, id_admission_source = load_key(
            key_file_path=path_key_file
        )

        logger.info("Data pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        raise


# %%
# Run main function if script is executed
if __name__ == "__main__":
    main()
# %%
