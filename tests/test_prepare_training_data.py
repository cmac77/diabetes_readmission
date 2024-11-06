# tests/test_prepare_training_data.py
import sys
from pathlib import Path

# Add the `src` directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from data_pipeline import prepare_training_data


def test_prepare_training_data():
    # Sample data with a target column
    df = pd.DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "readmitted": [0, 1, 0]}
    )

    # Run the function
    X, y = prepare_training_data(df)

    # Assertions
    assert X.shape[1] == 2, "X should have 2 features"
    assert len(y) == 3, "y should have 3 labels"
    assert y[1] == 1, "Second target value should match input data"
