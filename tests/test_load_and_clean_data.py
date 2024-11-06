# tests/test_load_and_clean_data.py
import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


import pickle
import os
import pandas as pd
from data_pipeline import load_and_clean_data
from config import CLEANED_DATA_PATH


def test_load_and_clean_data():
    # Run the function
    df = load_and_clean_data()

    # Check if the result is a DataFrame
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"

    # Check if cleaned data is saved
    assert os.path.exists(CLEANED_DATA_PATH), "Cleaned data file should be saved"

    # Check that the DataFrame is not empty
    assert not df.empty, "DataFrame should not be empty"
