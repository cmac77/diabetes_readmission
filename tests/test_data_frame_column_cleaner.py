# tests/test_data_frame_column_cleaner.py

import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_frame_column_cleaner import DataFrameColumnCleaner

# Sample configurations for testing
default_config = None

custom_config = {
    "remove_columns": ["unnecessary_column"],
    "handle_missing_values": {"strategy": "fill", "fill_value": "missing"},
    "apply_remapping": {
        "column_mappings": {
            "status": {"active": 1, "inactive": 0, "unknown": np.nan},
        },
        "global_mappings": {
            "N/A": np.nan,
            "Unknown": "Unknown Value",
        },
    },
    "whitespace_handling": {"mode": "normalize"},
    "replace_text": {
        "modes": ["nonalphanumeric", "digits"],
        "replacements": [" ", ""],
    },
    "standardize_text": {"modes": ["lowercase", "remove_accents"]},
    "identify_string_numbers": {"enabled": True},
    "standardize_number_strings": {
        "modes": ["thousands-separators", "percent-to-decimal"]
    },
    "convert_words_to_numbers": {"enabled": True},
    "standardize_numbers": {
        "default": {"modes": ["decimal", "handle_inf"], "inf_replacement": np.nan}
    },
    "column_types": {
        "categorical": ["category_column"],
        "integer": ["status"],
        "float": ["percentage"],
        "boolean": ["is_active"],
        "datetime": ["date"],
        "string": ["description"],
        "timedelta": ["duration"],
    },
}


# Sample DataFrames for testing
@pytest.fixture
def sample_df():
    data = {
        "unnecessary_column": ["remove1", "remove2", "remove3"],
        "status": ["active", "inactive", "unknown"],
        "value": [100, 200, np.nan],
        "description": ["This is a test!", "Another test123", "N/A"],
        "percentage": ["50%", "75%", "100%"],
        "category_column": ["A", "B", "C"],
        "is_active": [True, False, True],
        "date": ["2021-01-01", "2021-02-01", "2021-03-01"],
        "duration": ["1 days", "2 days", "3 days"],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def cleaner_default():
    return DataFrameColumnCleaner()


@pytest.fixture
def cleaner_custom():
    return DataFrameColumnCleaner(config=custom_config)


# Sample configurations for testing `cast_column_dtype`
cast_column_dtype_config = {
    "column_dtypes": {
        "categorical": {"columns": ["category_column"], "dtype": "category"},
        "integer": {"columns": ["integer_column"], "dtype": "Int64"},
        "float": {"columns": ["float_column"], "dtype": "float64"},
        "boolean": {"columns": ["boolean_column"], "dtype": "boolean"},
        "datetime": {"columns": ["date_column"], "dtype": "datetime64[ns]"},
        "string": {"columns": ["string_column"], "dtype": "string"},
        "timedelta": {"columns": ["timedelta_column"], "dtype": "timedelta64[ns]"},
    }
}


# Sample DataFrame for testing casting
@pytest.fixture
def sample_df_cast():
    data = {
        "category_column": ["A", "B", "C"],
        "integer_column": [1, 2, 3],
        "float_column": [1.0, 2.5, 3.75],
        "boolean_column": [True, False, True],
        "date_column": ["2021-01-01", "2021-02-01", "2021-03-01"],
        "string_column": ["hello", "world", "pytest"],
        "timedelta_column": ["1 days", "2 days", "3 days"],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def cleaner_cast():
    return DataFrameColumnCleaner(config=cast_column_dtype_config)


# Test Initialization
def test_initialization_default(cleaner_default):
    assert cleaner_default.config["remove_columns"] == []
    assert cleaner_default.config["handle_missing_values"]["strategy"] is None


def test_initialization_custom(cleaner_custom):
    assert cleaner_custom.config["remove_columns"] == ["unnecessary_column"]
    assert cleaner_custom.config["handle_missing_values"]["strategy"] == "fill"
    assert cleaner_custom.config["handle_missing_values"]["fill_value"] == "missing"


def test_initialization_invalid_config():
    with pytest.raises(ValueError):
        DataFrameColumnCleaner(config="invalid_config")


# Test remove_columns
def test_remove_columns(cleaner_custom, sample_df):
    cleaned_df = cleaner_custom.remove_columns(sample_df)
    assert "unnecessary_column" not in cleaned_df.columns
    # Ensure other columns are intact
    assert "status" in cleaned_df.columns


def test_remove_columns_no_columns(cleaner_default, sample_df):
    cleaned_df = cleaner_default.remove_columns(sample_df)
    # All columns should remain unchanged
    assert list(cleaned_df.columns) == list(sample_df.columns)


# Test handle_missing_values
def test_handle_missing_values_fill(cleaner_custom, sample_df):
    cleaned_df = cleaner_custom.handle_missing_values(
        sample_df,
        strategy=cleaner_custom.config["handle_missing_values"]["strategy"],
        fill_value=cleaner_custom.config["handle_missing_values"]["fill_value"],
    )
    # 'value' was NaN and should be filled with 'Missing', then lowercased to 'missing'
    assert cleaned_df.at[2, "value"] == "missing"
    # Verify that 'value' dtype has changed to object
    assert cleaned_df["value"].dtype == object


def test_handle_missing_values_drop(cleaner_default, sample_df):
    # Configure cleaner to drop missing values
    cleaner_default.config["handle_missing_values"] = {
        "strategy": "drop",
        "fill_value": None,
    }
    cleaned_df = cleaner_default.handle_missing_values(
        sample_df,
        strategy=cleaner_default.config["handle_missing_values"]["strategy"],
        fill_value=cleaner_default.config["handle_missing_values"]["fill_value"],
    )
    # Original sample_df has 1 NaN in 'value'
    # Rows with NaN in any column should be dropped
    assert cleaned_df.shape[0] == 2  # Original 3 rows - 1 with NaN


def test_handle_missing_values_invalid_strategy(cleaner_default, sample_df):
    with pytest.raises(ValueError):
        cleaner_default.handle_missing_values(sample_df, strategy="invalid_strategy")


# Test remove_whitespace
def test_remove_whitespace_normalize(cleaner_default):
    series = pd.Series(["  Test  ", "\tAnother\tTest\n", "NoWhitespace"])
    expected = pd.Series(["Test", "Another Test", "NoWhitespace"])
    result = cleaner_default.remove_whitespace(series, mode="normalize")
    pd.testing.assert_series_equal(result, expected)


def test_remove_whitespace_leading(cleaner_default):
    series = pd.Series(["  Leading", "  Spaces", "NoChange"])
    expected = pd.Series(["Leading", "Spaces", "NoChange"])
    result = cleaner_default.remove_whitespace(series, mode="leading")
    pd.testing.assert_series_equal(result, expected)


def test_remove_whitespace_trailing(cleaner_default):
    series = pd.Series(["Trailing  ", "Spaces   ", "NoChange"])
    expected = pd.Series(["Trailing", "Spaces", "NoChange"])
    result = cleaner_default.remove_whitespace(series, mode="trailing")
    pd.testing.assert_series_equal(result, expected)


def test_remove_whitespace_extra(cleaner_default):
    series = pd.Series(
        ["This  has  extra spaces", "NoExtraSpaces", "Multiple   spaces"]
    )
    expected = pd.Series(["This has extra spaces", "NoExtraSpaces", "Multiple spaces"])
    result = cleaner_default.remove_whitespace(series, mode="extra")
    pd.testing.assert_series_equal(result, expected)


def test_remove_whitespace_invalid_mode(cleaner_default):
    series = pd.Series(["Test", "Another Test"])
    with pytest.raises(ValueError):
        cleaner_default.remove_whitespace(series, mode="invalid_mode")


# Test replace_text
def test_replace_text_nonalphanumeric(cleaner_default):
    series = pd.Series(["Hello, World!", "Test@123", "No$Symbols"])
    expected = pd.Series(["Hello  World ", "Test 123", "No Symbols"])
    result = cleaner_default.replace_text(
        series, modes=["nonalphanumeric"], replacements=[" "]
    )
    pd.testing.assert_series_equal(result, expected)


def test_replace_text_digits(cleaner_default):
    series = pd.Series(["abc123", "def456", "ghi789"])
    expected = pd.Series(["abc", "def", "ghi"])
    result = cleaner_default.replace_text(series, modes=["digits"], replacements=[""])
    pd.testing.assert_series_equal(result, expected)


def test_replace_text_multiple_modes(cleaner_default):
    series = pd.Series(["Hello, World! 123", "Test@456", "No$789Symbols"])
    expected = pd.Series(["Hello  World  ", "Test ", "No Symbols"])
    result = cleaner_default.replace_text(
        series, modes=["nonalphanumeric", "digits"], replacements=[" ", ""]
    )
    pd.testing.assert_series_equal(result, expected)


def test_replace_text_stopwords(cleaner_custom):
    series = pd.Series(["This is a test", "Another test case", "No stopwords here"])
    expected = pd.Series(["This   test", "Another test case", "No stopwords "])
    result = cleaner_custom.replace_text(series, modes=["stopwords"], replacements=[""])
    pd.testing.assert_series_equal(result, expected)


def test_replace_text_invalid_mode(cleaner_default):
    series = pd.Series(["Test", "Another Test"])
    # Unsupported mode should be skipped with a warning, but no exception
    result = cleaner_default.replace_text(
        series, modes=["unsupported_mode"], replacements=[""]
    )
    pd.testing.assert_series_equal(result, series)


def test_replace_text_replacement_length_mismatch(cleaner_default):
    series = pd.Series(["Test1", "Test2"])
    # Provide a replacements list with length != 1 and != len(modes)
    with pytest.raises(ValueError):
        cleaner_default.replace_text(
            series, modes=["nonalphanumeric", "digits"], replacements=[" ", "", "extra"]
        )  # replacements length=3, modes length=2


# Test standardize_text
def test_standardize_text_lowercase(cleaner_default):
    series = pd.Series(["Hello World", "TEST", "MixedCase"])
    expected = pd.Series(["hello world", "test", "mixedcase"])
    result = cleaner_default.standardize_text(series, modes=["lowercase"])
    pd.testing.assert_series_equal(result, expected)


def test_standardize_text_remove_accents(cleaner_default):
    series = pd.Series(["Café", "naïve", "résumé"])
    expected = pd.Series(["Cafe", "naive", "resume"])
    result = cleaner_default.standardize_text(series, modes=["remove_accents"])
    pd.testing.assert_series_equal(result, expected)


def test_standardize_text_multiple_modes(cleaner_default):
    series = pd.Series(["Café", "  Naïve Test  ", "Résumé"])
    # First normalize whitespace
    series = cleaner_default.remove_whitespace(series, mode="normalize")
    # Then apply standardize_text
    expected = pd.Series(["cafe", "naive test", "resume"])
    result = cleaner_default.standardize_text(
        series, modes=["lowercase", "remove_accents"]
    )
    pd.testing.assert_series_equal(result, expected)


def test_standardize_text_expand_contractions(cleaner_default):
    series = pd.Series(["I'm happy", "they're going", "can't stop"])
    expected = pd.Series(["I am happy", "they are going", "cannot stop"])
    result = cleaner_default.standardize_text(series, modes=["expand_contractions"])
    pd.testing.assert_series_equal(result, expected)


def test_standardize_text_invalid_mode(cleaner_default):
    series = pd.Series(["Test", "Another Test"])
    with pytest.raises(ValueError):
        cleaner_default.standardize_text(series, modes=["invalid_mode"])


# Test identify_string_numbers
def test_identify_string_numbers(cleaner_default):
    series = pd.Series(["123", "45.67", "89%", "1.2e3", "1,000", "Not a number"])
    expected = pd.Series([True, True, True, True, True, False])
    result = cleaner_default.identify_string_numbers(series)
    pd.testing.assert_series_equal(result, expected)


def test_identify_string_numbers_empty(cleaner_default):
    series = pd.Series(["", "   ", "NaN", "Infinity"])
    expected = pd.Series([False, False, False, False])
    result = cleaner_default.identify_string_numbers(series)
    pd.testing.assert_series_equal(result, expected)


# Test standardize_number_strings
def test_standardize_number_strings_thousands_separators(cleaner_default):
    series = pd.Series(["1,000", "2,500", "1000"])
    expected = pd.Series(["1000", "2500", "1000"])
    result = cleaner_default.standardize_number_strings(
        series, modes=["thousands-separators"]
    )
    pd.testing.assert_series_equal(result, expected)


def test_standardize_number_strings_percent_to_decimal(cleaner_default):
    series = pd.Series(["50%", "75%", "100%"])
    expected = pd.Series(["0.5", "0.75", "1.0"])
    result = cleaner_default.standardize_number_strings(
        series, modes=["percent-to-decimal"]
    )
    pd.testing.assert_series_equal(result, expected)


def test_standardize_number_strings_multiple_modes(cleaner_default):
    series = pd.Series(["1,000%", "2,500%", "1000%"])
    expected = pd.Series(["10.0", "25.0", "10.0"])
    result = cleaner_default.standardize_number_strings(
        series, modes=["thousands-separators", "percent-to-decimal"]
    )
    pd.testing.assert_series_equal(result, expected)


def test_standardize_number_strings_invalid_mode(cleaner_default):
    series = pd.Series(["1000", "2000"])
    with pytest.raises(ValueError):
        cleaner_default.standardize_number_strings(series, modes=["invalid_mode"])


# Test convert_words_to_numbers
def test_convert_words_to_numbers(cleaner_default):
    series = pd.Series(["one", "two", "three", "twenty one", "invalid"])
    expected = pd.Series([1, 2, 3, 21, "invalid"])
    result = cleaner_default.convert_words_to_numbers(series)
    pd.testing.assert_series_equal(result, expected)


def test_convert_words_to_numbers_non_string(cleaner_default):
    series = pd.Series([1, "two", 3.5, "four"])
    expected = pd.Series([1, 2, 3.5, 4])
    result = cleaner_default.convert_words_to_numbers(series)
    pd.testing.assert_series_equal(result, expected)


# Test standardize_numbers
def test_standardize_numbers_decimal(cleaner_default):
    series = pd.Series(["100", "200.5", "300"])
    expected = pd.Series([100.0, 200.5, 300.0])
    result = cleaner_default.standardize_numbers(series, modes=["decimal"])
    pd.testing.assert_series_equal(result, expected)


def test_standardize_numbers_handle_inf(cleaner_default):
    series = pd.Series([float("inf"), -float("inf"), np.nan, 100])
    expected = pd.Series([np.nan, np.nan, np.nan, 100])
    result = cleaner_default.standardize_numbers(
        series, modes=["handle_inf"], inf_replacement=np.nan
    )
    pd.testing.assert_series_equal(result, expected)


def test_standardize_numbers_multiple_modes(cleaner_default):
    series = pd.Series(["100", "200.5", "300", float("inf"), -float("inf"), np.nan])
    expected = pd.Series([100.0, 200.5, 300.0, np.nan, np.nan, np.nan])
    result = cleaner_default.standardize_numbers(
        series, modes=["decimal", "handle_inf"], inf_replacement=np.nan
    )
    pd.testing.assert_series_equal(result, expected)


def test_standardize_numbers_invalid_mode(cleaner_default):
    series = pd.Series(["100", "200"])
    with pytest.raises(ValueError):
        cleaner_default.standardize_numbers(series, modes=["invalid_mode"])


# Test apply_remapping
def test_apply_remapping_column_specific(cleaner_custom, sample_df):
    cleaned_df = cleaner_custom.apply_remapping(sample_df)
    # Original sample_df has 'status' as ["active", "inactive", "unknown"]
    # After remapping, expect [1, 0, pd.NA]
    assert cleaned_df.at[0, "status"] == 1
    assert cleaned_df.at[1, "status"] == 0
    assert pd.isna(cleaned_df.at[2, "status"])


def test_apply_remapping_global(cleaner_custom, sample_df):
    cleaned_df = cleaner_custom.apply_remapping(sample_df)
    # Original sample_df has 'description' as ["This is a test!", "Another test123", "N/A"]
    # After remapping, expect [original_value, original_value, np.nan]
    assert pd.isna(cleaned_df.at[2, "description"])


def test_apply_remapping_both_sections(cleaner_custom, sample_df):
    cleaned_df = cleaner_custom.apply_remapping(sample_df)
    # Column-specific
    assert cleaned_df["status"].tolist() == [1, 0, pd.NA]
    # Global
    assert pd.isna(cleaned_df["description"].iloc[2])
    # Non-mapped values remain unchanged
    assert cleaned_df["description"].iloc[0] == "This is a test!"


def test_apply_remapping_nonexistent_column(cleaner_custom, sample_df):
    # Add a column that is not in the DataFrame mappings
    sample_df["nonexistent"] = ["value1", "value2", "value3"]
    cleaned_df = cleaner_custom.apply_remapping(sample_df)
    # Ensure no error and column remains unchanged
    assert "nonexistent" in cleaned_df.columns
    assert cleaned_df["nonexistent"].tolist() == ["value1", "value2", "value3"]


def test_apply_remapping_no_mappings(cleaner_default, sample_df):
    cleaned_df = cleaner_default.apply_remapping(sample_df)
    # No mappings applied, DataFrame should remain unchanged
    pd.testing.assert_frame_equal(cleaned_df, sample_df)


# Test cast_column_dtype functionality
def test_cast_column_dtype(cleaner_cast, sample_df_cast):
    # Apply dtype casting
    cleaned_df = cleaner_cast.cast_column_dtype(sample_df_cast)

    # Verify each column's dtype is as expected
    assert isinstance(
        cleaned_df["category_column"].dtype, pd.CategoricalDtype
    )  # Updated categorical check
    assert pd.api.types.is_integer_dtype(
        cleaned_df["integer_column"]
    )  # General check for integer compatibility
    assert pd.api.types.is_float_dtype(cleaned_df["float_column"])
    assert pd.api.types.is_bool_dtype(cleaned_df["boolean_column"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date_column"])
    assert pd.api.types.is_string_dtype(cleaned_df["string_column"])
    assert pd.api.types.is_timedelta64_dtype(cleaned_df["timedelta_column"])


def test_cast_column_dtype_infer_dtype(cleaner_cast, sample_df_cast):
    # Modify config to exclude `dtype` for certain categories to test inference
    cleaner_cast.config["column_dtypes"]["integer"].pop("dtype")
    cleaner_cast.config["column_dtypes"]["float"].pop("dtype")
    cleaner_cast.config["column_dtypes"]["boolean"].pop("dtype")

    # Apply dtype casting with inference
    cleaned_df = cleaner_cast.cast_column_dtype(sample_df_cast)

    # Check that inferred dtypes are as expected
    assert pd.api.types.is_integer_dtype(cleaned_df["integer_column"])
    assert pd.api.types.is_float_dtype(cleaned_df["float_column"])
    assert pd.api.types.is_bool_dtype(cleaned_df["boolean_column"])


def test_cast_column_dtype_missing_category(cleaner_cast, sample_df_cast):
    # Remove "string" and "timedelta" categories to test missing categories
    cleaner_cast.config["column_dtypes"].pop("string")
    cleaner_cast.config["column_dtypes"].pop("timedelta")

    # Apply dtype casting
    cleaned_df = cleaner_cast.cast_column_dtype(sample_df_cast)

    # Verify that columns in missing categories remain unchanged
    assert cleaned_df["string_column"].dtype == object  # Original dtype remains
    assert cleaned_df["timedelta_column"].dtype == object  # Original dtype remains


# Test process_dataframe
def test_process_dataframe(cleaner_custom, sample_df):
    cleaned_df = cleaner_custom.process_dataframe(sample_df)
    # Remove columns
    assert "unnecessary_column" not in cleaned_df.columns
    # Handle missing values
    assert "value" in cleaned_df.columns
    # 'value' should have 'missing' where NaN if fill strategy is applied and lowercased
    assert cleaned_df.at[2, "value"] == "missing"
    # Apply remapping
    assert cleaned_df.at[0, "status"] == 1
    assert cleaned_df.at[1, "status"] == 0
    assert pd.isna(cleaned_df.at[2, "status"])
    # Global remapping
    assert pd.isna(cleaned_df.at[2, "description"])
    # Whitespace handling
    assert cleaned_df.at[0, "description"] == "this is a test"
    # Replace text
    assert cleaned_df.at[1, "description"] == "another test"
    # Standardize text
    assert cleaned_df.at[0, "description"] == "this is a test"
    # Identify string numbers and standardize number strings
    assert cleaned_df.at[0, "percentage"] == 0.5
    # Convert words to numbers
    # Not directly tested here, ensure no errors
    # Standardize numbers
    # 'value' should be "missing"
    assert cleaned_df.at[0, "value"] is pd.NA
    # Check data types
    assert pd.api.types.is_float_dtype(cleaned_df["status"].dtype)
    assert pd.api.types.is_string_dtype(cleaned_df["description"].dtype)


if __name__ == "__main__":
    import pytest

    pytest.main(["-v", "tests/test_data_frame_column_cleaner.py"])
