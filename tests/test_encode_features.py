# tests/test_encode_features.py
import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


import pandas as pd
from data_pipeline import encode_features


def test_encode_features():
    # Sample data
    df = pd.DataFrame(
        {
            "admission_type_id": [1, 2, 3],
            "discharge_disposition_id": [1, 2, 1],
            "admission_source_id": [1, 1, 2],
        }
    )
    columns_categorical = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]
    columns_numerical = []

    # Run the function
    df_encoded = encode_features(df, columns_categorical, columns_numerical)

    # Assert that the resulting DataFrame has dummies
    assert df_encoded.shape[1] > len(
        columns_categorical
    ), "Columns should be expanded with one-hot encoding"
    assert (
        "admission_type_id_2" in df_encoded.columns
    ), "Encoded column 'admission_type_id_2' should be present"
