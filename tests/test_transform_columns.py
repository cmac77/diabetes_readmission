# tests/test_transform_columns.py
import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from data_pipeline import transform_columns


def test_transform_columns():
    # Sample data with known mappings
    df = pd.DataFrame(
        {
            "A1Cresult": [0, 1, 2, None],
            "max_glu_serum": ["Norm", "High", None, "High"],
            "readmitted": ["NO", "SHORT", "LONG", "NO"],
            "weight": [100, 200, None, 300],
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [101, 102, 103, 104],
        }
    )

    # Run the function
    df_transformed, columns_numerical, columns_categorical = transform_columns(df)

    # Assertions
    assert "weight" not in df_transformed.columns, "'weight' column should be dropped"
    assert (
        "encounter_id" not in df_transformed.columns
    ), "'encounter_id' should be dropped"
    assert (
        "patient_nbr" not in df_transformed.columns
    ), "'patient_nbr' should be dropped"
    assert (
        df_transformed["A1Cresult"].isna().sum() == 0
    ), "NaNs in 'A1Cresult' should be filled"
    assert (
        df_transformed["max_glu_serum"].isna().sum() == 0
    ), "NaNs in 'max_glu_serum' should be filled"
