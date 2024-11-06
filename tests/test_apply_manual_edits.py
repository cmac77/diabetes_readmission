# tests/test_apply_manual_edits.py

import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from data_pipeline import apply_manual_edits


def test_apply_manual_edits():
    # Sample data
    df = pd.DataFrame({"col1": [" 1", " 2", " 3"], "col2": ["a ", "b ", "c "]})

    # Mocking a manual edit with known outcomes (requires manual confirmation)
    df_edited = apply_manual_edits(df)

    # Check that the function returns a DataFrame
    assert isinstance(df_edited, pd.DataFrame), "Output should be a DataFrame"
