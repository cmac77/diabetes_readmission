#%% src/data_processing/data_frame_column_cleaner.py

# Standard library imports
import logging
import re
import unicodedata
import sys
from typing import Any, Dict, List, Optional, Literal, Union

# Third-party imports
import numpy as np
import pandas as pd
from pandas._libs.missing import NAType
from pandas import NaT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from word2number import w2n
from pyprojroot import here

#%% Ensure path_root is added to sys.path
path_root = here()

if str(path_root) not in sys.path:
    sys.path.append(str(path_root))

# Local imports
from src.config import config 


#%% Define a compatible type for fill values, including NAType and NaTType
FillValueType = Union[float, int, str, pd.Timestamp, pd.Timedelta, NAType, type(NaT)]


# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataFrameColumnCleaner:
    """
    A robust DataFrame cleaner tailored for healthcare analytics.

    This class offers comprehensive data cleaning functionalities, including handling missing values,
    removing unwanted characters, converting data types, and more. It integrates seamlessly with
    external configuration files for flexibility and is structured to support future enhancements
    like parallel processing.

    Attributes:
        config (Dict[str, Any]): Configuration settings loaded from external sources or defaults.
        stop_words (set): Set of English stop words for text cleaning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DataFrameColumnCleaner with provided or default configurations.

        Args:
            config (Optional[Dict[str, Any]]): Custom configuration settings.
                If None, default settings are used.

        Raises:
            ValueError: If the custom configuration is not a dictionary.
        """
        if config is not None and not isinstance(config, dict):
            logger.error("Custom configuration must be a dictionary.")
            raise ValueError("Custom configuration must be a dictionary.")

        self.config = self._load_config(config)
        self.stop_words = set(ENGLISH_STOP_WORDS)
        logger.debug(
            "DataFrameColumnCleaner initialized with configuration: %s", self.config
        )

    @staticmethod
    def _load_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Loads configuration by merging custom configurations with defaults.

        Args:
            config (Optional[Dict[str, Any]]): Custom configuration settings.

        Returns:
            Dict[str, Any]: Merged configuration settings.
        """
        default_config = {
            "remove_columns": [],
            "handle_missing_values": {"strategy": None, "fill_value": None},
            "apply_remapping": {"column_mappings": {}, "global_mappings": {}},
            "whitespace_handling": {"mode": "normalize"},
            "replace_text": {"modes": [], "replacements": [""]},
            "standardize_text": {"modes": ["lowercase", "unicode"]},
            "identify_string_numbers": {"enabled": True},
            "standardize_number_strings": {
                "modes": [
                    "thousands-separators",
                    "comma-separators",
                    "parentheses",
                    "scientific-notation",
                    "percent-to-decimal",
                ]
            },
            "convert_words_to_numbers": {"enabled": True},
            "standardize_numbers": {
                "default": {"modes": ["handle_inf"], "inf_replacement": pd.NA}
            },
            "column_dtypes": {
                "categorical": {
                    "columns": [],
                    "dtype": "category",  # Default for categorical data
                },
                "integer": {
                    "columns": [],
                    "dtype": "Int64",  # Nullable integer by default
                },
                "float": {
                    "columns": [],
                    "dtype": "float64",  # Default for floating-point numbers
                },
                "boolean": {
                    "columns": [],
                    "dtype": "boolean",  # Nullable boolean by default
                },
                "datetime": {
                    "columns": [],
                    "dtype": "datetime64[ns]",  # Default for datetime
                },
                "string": {"columns": [], "dtype": "string"},  # Default for text data
                "timedelta": {
                    "columns": [],
                    "dtype": "timedelta64[ns]",  # Default for time duration
                },
                "complex": {
                    "columns": [],
                    "dtype": "complex128",  # Default for complex numbers
                },
                "sparse": {
                    "columns": [],
                    "dtype": None,  # Infers based on content, typically `Sparse[int]` or `Sparse[float]`
                },
                "period": {
                    "columns": [],
                    "dtype": None,  # Infers based on frequency (e.g., 'M' for monthly, 'Q' for quarterly)
                },
                "interval": {
                    "columns": [],
                    "dtype": None,  # Infers based on contents, e.g., `Interval[int64]`
                },
            },
        }

        if config:
            for key, value in config.items():
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
            logger.debug("Custom configuration applied.")
        else:
            logger.debug("Using default configuration.")

        return default_config

    def remove_whitespace(
        self, series: pd.Series, mode: str = "normalize"
    ) -> pd.Series:
        """
        Removes or normalizes whitespace in a Pandas Series based on the specified mode.

        Args:
            series (pd.Series): The series to process.
            mode (str): The whitespace removal mode.

        Returns:
            pd.Series: Series with whitespace processed as per the mode.
        """
        # Create an independent copy of the series to avoid SettingWithCopyWarning
        series = series.copy()

        if not pd.api.types.is_string_dtype(series):
            # Convert non-missing values to strings only, if necessary
            series = series.astype("string")

        # Create a mask for non-missing values
        non_missing_mask = ~series.isna()
        series_non_missing = series[non_missing_mask]

        whitespace_patterns = {
            "leading": r"^\s+",
            "trailing": r"\s+$",
            "extra": r"\s+",
            "tabs_to_spaces": r"\t",
            "non_breaking": "\u00A0",
            "within_numbers": r"(?<=\d) (?=\d)",
        }

        try:
            if mode == "leading":
                series_non_missing = series_non_missing.str.replace(
                    whitespace_patterns["leading"], "", regex=True
                )

            elif mode == "trailing":
                series_non_missing = series_non_missing.str.replace(
                    whitespace_patterns["trailing"], "", regex=True
                )

            elif mode == "both":
                series_non_missing = series_non_missing.str.strip()

            elif mode == "extra":
                series_non_missing = series_non_missing.str.replace(
                    whitespace_patterns["extra"], " ", regex=True
                )

            elif mode == "all":
                series_non_missing = series_non_missing.str.strip().str.replace(
                    whitespace_patterns["extra"], " ", regex=True
                )

            elif mode == "within_numbers":
                series_non_missing = series_non_missing.str.replace(
                    whitespace_patterns["within_numbers"], "", regex=True
                )

            elif mode == "tabs_to_spaces":
                series_non_missing = series_non_missing.str.replace(
                    whitespace_patterns["tabs_to_spaces"], " ", regex=True
                )

            elif mode == "non_breaking":
                series_non_missing = series_non_missing.str.replace(
                    whitespace_patterns["non_breaking"], " ", regex=False
                )

            elif mode == "normalize":
                series_non_missing = (
                    series_non_missing.str.replace(
                        whitespace_patterns["tabs_to_spaces"], " ", regex=True
                    )
                    .str.replace(whitespace_patterns["non_breaking"], " ", regex=False)
                    .str.replace(whitespace_patterns["extra"], " ", regex=True)
                    .str.strip()
                )
            else:
                logger.warning(
                    f"Unsupported whitespace mode '{mode}'. No changes applied."
                )
                raise ValueError(f"Unsupported whitespace mode '{mode}'.")

            # Combine the processed non-missing values back with missing values using .loc to avoid SettingWithCopyWarning
            series.loc[non_missing_mask] = series_non_missing

        except Exception as e:
            logger.error(f"Error in remove_whitespace with mode '{mode}': {e}")
            raise

        return series

    def replace_text(
        self, series: pd.Series, modes: List[str], replacements: List[str] = [""]
    ) -> pd.Series:
        """
        Replaces text in a Pandas Series based on specified modes and replacements.

        Args:
            series (pd.Series): The series to process.
            modes (List[str]): List of text replacement modes. Supported modes include:
                - 'nonalphanumeric': Remove non-alphanumeric characters.
                - 'alphanumeric': Retain only alphanumeric characters.
                - 'punctuation': Remove punctuation.
                - 'digits': Remove all digits.
                - 'stopwords': Remove English stop words.
            replacements (List[str]): Corresponding replacement strings. If a single
                replacement is provided, it is applied to all modes.

        Returns:
            pd.Series: Series with text replaced as per the modes.

        Raises:
            ValueError: If the number of replacements does not match the number of modes.
        """
        text_patterns = {
            "nonalphanumeric": r"[^a-zA-Z0-9\s]",
            "alphanumeric": r"[a-zA-Z0-9]",
            "punctuation": r"[^\w\s]",
            "digits": r"\d",
        }

        if len(replacements) == 1:
            replacements *= len(modes)
        elif len(replacements) != len(modes):
            logger.error(
                "Replacements length must match modes length or be a single string."
            )
            raise ValueError(
                "Number of replacements must match the number of modes or be a single replacement."
            )

        try:
            for mode, replacement in zip(modes, replacements):
                if mode in text_patterns:
                    series = series.str.replace(
                        text_patterns[mode], replacement, regex=True
                    )

                elif mode == "stopwords":
                    stopwords_pattern = (
                        r"\b(?:" + "|".join(map(re.escape, self.stop_words)) + r")\b"
                    )
                    series = series.str.replace(
                        stopwords_pattern, replacement, regex=True
                    )

                else:
                    logger.warning(f"Unsupported replace_text mode '{mode}'. Skipping.")
        except Exception as e:
            logger.error(f"Error in replace_text: {e}")
            raise

        return series

    def standardize_text(
        self, series: pd.Series, modes: List[str] = ["lowercase"]
    ) -> pd.Series:
        """
        Standardizes text in a Pandas Series based on specified modes.

        Args:
            series (pd.Series): The series to process.
            modes (List[str]): List of text standardization modes. Supported modes include:
                - 'lowercase': Convert text to lowercase.
                - 'unicode': Normalize Unicode characters.
                - 'remove_accents': Remove accentuated characters.
                - 'expand_contractions': Expand common English contractions.
                - 'normalize_special_chars': Replace special characters with their descriptions.

        Returns:
            pd.Series: Series with standardized text.

        Raises:
            ValueError: If an unsupported mode is provided.
        """
        contractions = {
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
            r"\bI'm\b": "I am",
            r"\bthey're\b": "they are",
            r"\bthere's\b": "there is",
            # Add more contractions as needed
        }

        try:
            for mode in modes:
                if mode == "lowercase":
                    series = series.str.lower()

                elif mode == "unicode":
                    series = series.apply(
                        lambda x: (
                            unicodedata.normalize("NFKD", x)
                            if isinstance(x, str)
                            else x
                        )
                    )

                elif mode == "remove_accents":
                    series = series.apply(
                        lambda x: (
                            "".join(
                                c
                                for c in unicodedata.normalize("NFKD", x)
                                if not unicodedata.combining(c)
                            )
                            if isinstance(x, str)
                            else x
                        )
                    )

                elif mode == "expand_contractions":
                    for contraction, expanded in contractions.items():
                        series = series.str.replace(contraction, expanded, regex=True)

                elif mode == "normalize_special_chars":
                    special_chars = {
                        "°": " degrees",
                        "€": " euro",
                        "£": " pound",
                        # Add more special characters as needed
                    }
                    for char, replacement in special_chars.items():
                        series = series.str.replace(char, replacement, regex=False)

                else:
                    logger.warning(
                        f"Unsupported standardize_text mode '{mode}'. Skipping."
                    )
                    raise ValueError(f"Unsupported standardize_text mode '{mode}'.")

        except Exception as e:
            logger.error(f"Error in standardize_text: {e}")
            raise

        return series

    def identify_string_numbers(self, series: pd.Series) -> pd.Series:
        """
        Identifies potential numeric string representations in a Pandas Series.

        Args:
            series (pd.Series): The series to analyze.

        Returns:
            pd.Series: Boolean series indicating presence of numeric strings.
        """
        patterns = [
            r"^[+-]?\d*\.?\d+$",  # General numbers
            r"^\d+%$",  # Percentages
            r"^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$",  # Scientific notation
            r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$",  # Thousand separators
        ]

        try:
            combined_pattern = "|".join(f"(?:{pattern})" for pattern in patterns)
            return series.str.contains(combined_pattern, na=False, regex=True)
        except Exception as e:
            logger.error(f"Error in identify_string_numbers: {e}")
            raise

    def standardize_number_strings(
        self, series: pd.Series, modes: List[str] = ["thousands-separators"]
    ) -> pd.Series:
        """
        Standardizes numeric string representations based on specified modes.

        Args:
            series (pd.Series): The series to process.
            modes (List[str]): List of standardization modes. Supported modes include:
                - 'thousands-separators': Remove commas used as thousand separators.
                - 'comma-separators': Alias for 'thousands-separators'.
                - 'parentheses': Convert parentheses to negative signs.
                - 'scientific-notation': Convert scientific notation to float.
                - 'percent-to-decimal': Convert percentages to decimal numbers.
                - 'currency-symbols': Remove currency symbols like $, €, £.
                - 'fractional-numbers': Convert fractional strings to float.

        Returns:
            pd.Series: Series with standardized numeric strings.

        Raises:
            ValueError: If an unsupported mode is provided.
        """
        try:
            for mode in modes:
                if mode in ["thousands-separators", "comma-separators"]:
                    series = series.str.replace(",", "", regex=False)

                elif mode == "parentheses":
                    series = series.str.replace(
                        r"\((\d+(\.\d+)?)\)", r"-\1", regex=True
                    )

                elif mode == "scientific-notation":
                    series = series.apply(
                        lambda x: (
                            "{:.10f}".format(float(x))
                            if re.match(r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$", str(x))
                            else x
                        )
                    )

                elif mode == "percent-to-decimal":
                    series = series.str.replace(
                        r"(\d+)%", lambda m: str(float(m.group(1)) / 100), regex=True
                    )

                elif mode == "currency-symbols":
                    series = series.str.replace(r"[$€£]", "", regex=True)

                elif mode == "fractional-numbers":
                    series = series.str.replace(
                        r"(\d+)/(\d+)",
                        lambda m: str(float(m.group(1)) / float(m.group(2))),
                        regex=True,
                    )

                else:
                    logger.warning(
                        f"Unsupported standardize_number_strings mode '{mode}'. Skipping."
                    )
                    raise ValueError(
                        f"Unsupported standardize_number_strings mode '{mode}'."
                    )

        except Exception as e:
            logger.error(f"Error in standardize_number_strings: {e}")
            raise

        return series

    def convert_words_to_numbers(self, series: pd.Series) -> pd.Series:
        """
        Converts number words in a Pandas Series to their numeric representations.

        Args:
            series (pd.Series): The series to process.

        Returns:
            pd.Series: Series with word-based numbers converted to numeric format.

        Notes:
            Utilizes the `word2number` library which may not handle complex or colloquial expressions.

        Raises:
            Exception: Propagates any unexpected errors during conversion.
        """

        def convert_word_to_num(text: str) -> Any:
            try:
                return w2n.word_to_num(text)
            except ValueError:
                return text

        try:
            return series.apply(
                lambda x: convert_word_to_num(x) if isinstance(x, str) else x
            )
        except Exception as e:
            logger.error(f"Error in convert_words_to_numbers: {e}")
            raise

    def standardize_numbers(
        self,
        series: pd.Series,
        modes: List[str] = ["decimal"],
        decimal_places: Optional[int] = None,
        inf_replacement: Any = pd.NA,
    ) -> pd.Series:
        """
        Standardizes numerical representations based on specified modes.

        Args:
            series (pd.Series): The series to process.
            modes (List[str]): List of standardization modes. Supported modes include:
                - 'decimal': Convert to float.
                - 'integer': Convert to integer if applicable.
                - 'rounded': Round to a specified number of decimal places.
                - 'fixed-precision': Format to fixed decimal places as strings.
                - 'handle_inf': Replace infinite and NaN values with a specified replacement.
            decimal_places (Optional[int]): Decimal places for rounding or fixed precision.
                Required if modes include 'rounded' or 'fixed-precision'.
            inf_replacement (Any): Replacement value for infinite or NaN values.

        Returns:
            pd.Series: Series with standardized numerical representations.

        Raises:
            ValueError: If required parameters for a mode are missing.
        """
        try:
            for mode in modes:
                if mode == "decimal":
                    series = series.apply(
                        lambda x: (
                            float(x)
                            if isinstance(x, (int, float, str))
                            and self._is_convertible_to_float(x)
                            else x
                        )
                    )

                elif mode == "integer":
                    series = series.apply(
                        lambda x: (
                            int(float(x))
                            if isinstance(x, (int, float, str))
                            and self._is_convertible_to_float(x)
                            and float(x).is_integer()
                            else x
                        )
                    )

                elif mode == "rounded":
                    if decimal_places is None:
                        logger.error(
                            "Decimal places must be provided for 'rounded' mode."
                        )
                        raise ValueError(
                            "decimal_places must be provided when mode is 'rounded'."
                        )
                    series = series.apply(
                        lambda x: (
                            round(float(x), decimal_places)
                            if isinstance(x, (int, float, str))
                            and self._is_convertible_to_float(x)
                            else x
                        )
                    )

                elif mode == "fixed-precision":
                    if decimal_places is None:
                        logger.error(
                            "Decimal places must be provided for 'fixed-precision' mode."
                        )
                        raise ValueError(
                            "decimal_places must be provided when mode is 'fixed-precision'."
                        )
                    series = series.apply(
                        lambda x: (
                            f"{float(x):.{decimal_places}f}"
                            if isinstance(x, (int, float, str))
                            and self._is_convertible_to_float(x)
                            else x
                        )
                    )

                elif mode == "handle_inf":
                    series = series.replace(
                        [float("inf"), float("-inf"), float("nan")], inf_replacement
                    )

                else:
                    logger.warning(
                        f"Unsupported standardize_numbers mode '{mode}'. Skipping."
                    )
                    raise ValueError(f"Unsupported standardize_numbers mode '{mode}'.")

        except Exception as e:
            logger.error(f"Error in standardize_numbers: {e}")
            raise

        return series

    @staticmethod
    def _is_convertible_to_float(value: Any) -> bool:
        """
        Checks if a value can be converted to float.

        Args:
            value (Any): The value to check.

        Returns:
            bool: True if convertible to float, else False.
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes specified columns from the DataFrame based on configuration.

        Args:
            df (pd.DataFrame): The DataFrame from which columns will be removed.

        Returns:
            pd.DataFrame: DataFrame with specified columns removed.
        """
        columns_to_remove = self.config.get("remove_columns", [])

        if not columns_to_remove:
            logger.info("No columns specified for removal.")
            return df

        columns_in_df = [col for col in columns_to_remove if col in df.columns]
        if columns_in_df:
            df = df.drop(columns=columns_in_df)
            logger.info(f"Removed columns: {columns_in_df}")
        else:
            logger.info("No specified columns found in DataFrame to remove.")

        return df

    def get_missing_value_for_dtype(self, col_dtype):
        """
        Determines the appropriate missing value based on the column's data type.

        Args:
            col_dtype: Data type of the column.

        Returns:
            Missing value representation suitable for the given data type.
        """
        if pd.api.types.is_numeric_dtype(col_dtype):
            return np.nan
        elif pd.api.types.is_string_dtype(col_dtype) or isinstance(
            col_dtype, pd.CategoricalDtype
        ):
            return pd.NA
        elif pd.api.types.is_bool_dtype(col_dtype):
            return pd.NA
        elif pd.api.types.is_datetime64_any_dtype(col_dtype):
            return pd.NaT
        else:
            return pd.NA  # Default for other dtypes

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Optional[Literal["drop", "fill"]] = None,
        fill_value: Any = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Handles missing values in the DataFrame by dropping or filling them.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            strategy (Optional[Literal["drop", "fill"]]): Strategy to handle missing values.
                - 'drop': Remove rows with missing values.
                - 'fill': Replace missing values with a specified fill value or dtype-specific value.
            fill_value (Any): Value to fill missing entries with (if strategy is "fill").
                If None, dtype-specific fill values are used.
            columns (Optional[List[str]]): Specific columns to apply the strategy to.
                If None, applies to all columns.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.

        Raises:
            ValueError: If an invalid strategy is provided or required parameters are missing.
        """
        if strategy not in {"drop", "fill"} and not pd.isna(strategy):
            logger.error(f"Invalid strategy '{strategy}' provided.")
            raise ValueError(
                f"Invalid strategy '{strategy}'. Choose 'drop', 'fill', or None."
            )

        columns_to_process = columns if columns else df.columns.tolist()

        if strategy == "drop":
            initial_rows = df.shape[0]
            df = df.dropna(subset=columns_to_process)
            # Additionally drop rows where specified columns have empty strings
            empty_rows_mask = df[columns_to_process].astype(str).eq("").any(axis=1)
            df = df[~empty_rows_mask]
            final_rows = df.shape[0]
            dropped_rows = initial_rows - final_rows
            logger.info(f"Dropped {dropped_rows} rows with missing or empty values.")

        elif strategy == "fill":
            if fill_value is not None:
                cast_fill_value: FillValueType = fill_value
                for col in columns_to_process:
                    if col in df.columns:
                        # Apply fillna with user-defined fill value
                        df[col] = df[col].fillna(cast_fill_value)

                        # Apply replace only if the column is object or string dtype
                        if pd.api.types.is_object_dtype(
                            df[col]
                        ) or pd.api.types.is_string_dtype(df[col]):
                            df[col] = df[col].replace("", cast_fill_value)
                logger.info(
                    f"Filled missing and empty values in columns {columns_to_process} with '{fill_value}'."
                )
            else:
                # Use dtype-specific fill values
                for col in columns_to_process:
                    if col in df.columns:
                        col_dtype = df[col].dtype
                        dtype_fill_value: FillValueType = (
                            self.get_missing_value_for_dtype(col_dtype)
                        )

                        # Apply fillna with dtype-specific fill value
                        df[col] = df[col].fillna(dtype_fill_value)

                        # Only replace empty strings in columns that can handle string replacements
                        if pd.api.types.is_object_dtype(
                            df[col]
                        ) or pd.api.types.is_string_dtype(df[col]):
                            df[col] = df[col].replace("", dtype_fill_value)

                logger.info(
                    f"Filled missing and empty values in columns {columns_to_process} with dtype-specific values."
                )

        else:
            logger.info(
                "No missing value handling strategy provided. Missing values are retained."
            )
            return df

        return df

    def apply_remapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies column-specific and global remappings to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: DataFrame with remapped values.
        """
        remapping_config = self.config.get("apply_remapping", {})
        column_mappings = remapping_config.get("column_mappings", {})
        global_mappings = remapping_config.get("global_mappings", {})

        # Precompute dtypes for efficiency
        dtypes = df.dtypes

        # ------------------------------
        # 1. Apply Column-Specific Mappings
        # ------------------------------
        for col, mapping in column_mappings.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            col_dtype = dtypes[col]

            # Prepare adjusted mapping: handle missing values based on dtype
            adjusted_mapping = {}
            for key, value in mapping.items():
                adjusted_value = (
                    self.get_missing_value_for_dtype(col_dtype)
                    if pd.isna(value)
                    else value
                )
                adjusted_mapping[key] = adjusted_value

            # Determine if the column is intended to be numeric after mapping
            # Check if all non-missing mapped values are numeric
            mapped_values = [v for v in adjusted_mapping.values() if not pd.isna(v)]
            is_numeric_mapping = all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in mapped_values
            )

            # If the mapping is numeric, convert the column to a nullable integer type
            target_dtype = (
                "Int64"
                if is_numeric_mapping
                else (
                    pd.StringDtype()
                    if pd.api.types.is_string_dtype(col_dtype)
                    else col_dtype
                )
            )

            # Perform replacement
            try:
                df[col] = df[col].replace(adjusted_mapping)
            except Exception as e:
                logger.error(f"Error applying column mapping for '{col}': {e}")
                continue

            # Cast to target dtype
            try:
                df[col] = df[col].astype(target_dtype)
            except Exception as e:
                logger.error(
                    f"Error casting column '{col}' to dtype '{target_dtype}': {e}"
                )
                # Attempt to retain original dtype if casting fails
                df[col] = df[col].astype(col_dtype)

            logger.info(
                f"Applied column-specific remapping to '{col}': {adjusted_mapping} and cast to '{target_dtype}'"
            )

        # ------------------------------
        # 2. Apply Global Mappings
        # ------------------------------
        # Prepare global mapping with handling for missing values
        global_adjusted_mapping = {}
        for key, value in global_mappings.items():
            global_adjusted_mapping[key] = np.nan if pd.isna(value) else value

        # Replace global mappings except for missing values
        non_missing_keys = {
            k: v for k, v in global_adjusted_mapping.items() if not pd.isna(v)
        }
        if non_missing_keys:
            df.replace(non_missing_keys, inplace=True)

        # Handle missing value replacements based on dtype
        missing_keys = [k for k, v in global_adjusted_mapping.items() if pd.isna(v)]
        if missing_keys:
            # Create a mask for all missing keys across the DataFrame
            mask = df.isin(missing_keys)
            for col in df.columns:
                if mask[col].any():
                    missing_value = self.get_missing_value_for_dtype(dtypes[col])
                    df[col] = df[col].mask(mask[col], missing_value)

        logger.info(f"Applied global remapping with adjusted values: {global_mappings}")
        return df

    def cast_column_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Casts columns in the DataFrame to data types specified in the configuration.
        Infers specific dtypes when none are provided and uses default 64-bit types
        where possible.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: DataFrame with columns cast to specified data types.
        """
        column_dtypes_config = self.config.get("column_dtypes", {})

        # Default 64-bit types for each category, where applicable
        default_dtype_mapping = {
            "integer": "Int64",
            "float": "float64",
            "boolean": "boolean",
            "datetime": "datetime64[ns]",
            "string": "string",
            "timedelta": "timedelta64[ns]",
            "categorical": "category",
            "complex": "complex128",
            "sparse": None,  # Inferred based on column values
            "period": None,  # Inferred frequency-based, e.g., period[M], period[Q]
            "interval": None,  # Requires continuous intervals, e.g., interval[int64]
        }

        for dtype_category, settings in column_dtypes_config.items():
            # Retrieve columns and dtype (if specified) from settings
            columns = settings.get("columns", [])
            target_dtype = settings.get("dtype")

            for col in columns:
                if col in df.columns:
                    current_dtype = df[col].dtype

                    # Infer dtype if not provided in config
                    if target_dtype in {None, np.nan, ""}:
                        inferred_dtype = self._infer_dtype(df[col], dtype_category)
                        target_dtype = (
                            inferred_dtype
                            if inferred_dtype
                            else default_dtype_mapping.get(dtype_category)
                        )
                    else:
                        target_dtype = target_dtype

                    # Skip casting if column already aligns with target dtype
                    is_consistent = self._is_dtype_consistent(
                        current_dtype, dtype_category
                    )
                    if is_consistent:
                        logger.info(
                            f"Column '{col}' dtype '{current_dtype}' already aligns with '{dtype_category}'. No casting applied."
                        )
                        continue

                    # Attempt casting
                    try:
                        df[col] = self._cast_column(
                            df[col], target_dtype, dtype_category
                        )
                        logger.info(f"Casted column '{col}' to '{target_dtype}'.")
                    except Exception as e:
                        logger.warning(
                            f"Failed to cast column '{col}' to '{target_dtype}': {e}"
                        )

        return df

    def _infer_dtype(self, series: pd.Series, dtype_category: str):
        """
        Infers a specific dtype based on the data in the series and the target dtype category.

        Args:
            series (pd.Series): The column to analyze.
            dtype_category (str): The broad dtype category for inference.

        Returns:
            str or None: The inferred dtype, or None if unable to determine.
        """
        if dtype_category == "integer":
            # Check if all non-missing values are whole numbers
            if pd.api.types.is_integer_dtype(series) or all(
                series.dropna().apply(lambda x: isinstance(x, (int, np.integer)))
            ):
                return "Int64"  # Nullable integer by default
            else:
                return "float64"  # Fall back to float if mixed integers and decimals

        elif dtype_category == "float":
            return "float64"
        elif dtype_category == "boolean":
            return "boolean"
        elif dtype_category == "datetime":
            return "datetime64[ns]"
        elif dtype_category == "timedelta":
            return "timedelta64[ns]"
        elif dtype_category == "categorical":
            return "category"
        elif dtype_category == "complex":
            return "complex128"
        elif dtype_category == "sparse":
            # Determine sparse type based on data
            if pd.api.types.is_integer_dtype(series):
                return pd.SparseDtype("Int64")
            elif pd.api.types.is_float_dtype(series):
                return pd.SparseDtype("float64")
        elif dtype_category == "period":
            # Infer period frequency if possible (e.g., monthly, quarterly)
            return (
                series.dt.to_period().freq
                if pd.api.types.is_datetime64_any_dtype(series)
                else None
            )
        elif dtype_category == "interval":
            return (
                pd.IntervalDtype("int64")
                if pd.api.types.is_integer_dtype(series)
                else pd.IntervalDtype("float64")
            )

        return None  # Fallback if no specific dtype can be inferred

    def _is_dtype_consistent(self, current_dtype, dtype_category: str):
        """
        Checks if the current dtype aligns with the specified dtype category.

        Args:
            current_dtype (str): The column's current dtype.
            dtype_category (str): The target dtype category.

        Returns:
            bool: True if the dtype is consistent, False otherwise.
        """
        return (
            isinstance(current_dtype, pd.CategoricalDtype)
            and dtype_category == "categorical"
            or pd.api.types.is_integer_dtype(current_dtype)
            and dtype_category == "integer"
            or pd.api.types.is_float_dtype(current_dtype)
            and dtype_category == "float"
            or pd.api.types.is_bool_dtype(current_dtype)
            and dtype_category == "boolean"
            or pd.api.types.is_datetime64_any_dtype(current_dtype)
            and dtype_category == "datetime"
            or pd.api.types.is_string_dtype(current_dtype)
            and dtype_category == "string"
            or pd.api.types.is_timedelta64_dtype(current_dtype)
            and dtype_category == "timedelta"
            or pd.api.types.is_complex_dtype(current_dtype)
            and dtype_category == "complex"
            or isinstance(current_dtype, pd.SparseDtype)
            and dtype_category == "sparse"
        )

    def _cast_column(self, series: pd.Series, target_dtype, dtype_category: str):
        """
        Casts the series to the target dtype based on the dtype category.

        Args:
            series (pd.Series): The column to cast.
            target_dtype (str): The dtype to cast to.
            dtype_category (str): The broad dtype category.

        Returns:
            pd.Series: The series cast to the target dtype.
        """
        if dtype_category == "datetime":
            return pd.to_datetime(series, errors="coerce")
        elif dtype_category == "timedelta":
            return pd.to_timedelta(series, errors="coerce")
        elif dtype_category == "period" and pd.api.types.is_datetime64_any_dtype(
            series
        ):
            return series.dt.to_period(target_dtype)
        elif dtype_category == "interval":
            return pd.cut(series, bins=10)  # Example: create intervals based on bins

        return series.astype(target_dtype)

    # %%

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full data cleaning pipeline on the DataFrame.

        Sequence of Execution:
            1. Remove specified columns.
            2. Handle missing values.
            3. Apply remappings.
            4. Identify string-based numbers.
            5. Standardize number-like strings.
            6. Convert word-based numbers to numeric format.
            7. Standardize numerical representations.
            8. Replace specified text patterns.
            9. Remove or normalize whitespace.
            10. Standardize text.
            11. Recast column dtypes if spefified in config file.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: Fully processed DataFrame.

        Raises:
            Exception: Propagates any unexpected errors during the processing pipeline.
        """
        logger.info("Starting full DataFrame cleaning process.")

        try:
            # 1. Remove specified columns
            df = self.remove_columns(df)

            # 2. Handle missing values
            missing_config = self.config.get("handle_missing_values", {})
            strategy = missing_config.get("strategy")
            fill_value = missing_config.get("fill_value")
            df = self.handle_missing_values(
                df, strategy=strategy, fill_value=fill_value
            )

            # 3. Apply remappings
            df = self.apply_remapping(df)

            # 4. Identify string-based numbers
            if self.config.get("identify_string_numbers", {}).get("enabled", True):
                numeric_mask = df.select_dtypes(include=["object", "string"]).apply(
                    self.identify_string_numbers
                )
                logger.info("Identified string-based numbers.")
            else:
                numeric_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
                logger.info("String number identification disabled.")

            # 5. Standardize number-like strings
            number_string_config = self.config.get("standardize_number_strings", {})
            number_string_modes = number_string_config.get("modes", [])
            if number_string_modes:
                for col in df.columns:
                    if col in numeric_mask and numeric_mask[col].any():
                        df[col] = self.standardize_number_strings(
                            df[col], modes=number_string_modes
                        )
                logger.info("Number-like string standardization completed.")
            else:
                logger.info(
                    "No number string standardization modes specified. Skipping."
                )

            # 6. Convert word-based numbers to numeric format
            if self.config.get("convert_words_to_numbers", {}).get("enabled", True):
                for col in df.columns:
                    if df[col].dtype == "object" or df[col].dtype == "string":
                        df[col] = self.convert_words_to_numbers(df[col])
                logger.info("Converted word-based numbers to numeric format.")
            else:
                logger.info("Word-to-number conversion disabled.")

            # 7. Standardize numerical representations
            number_config = self.config.get("standardize_numbers", {})

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype in [
                    "string",
                    "object",
                ]:
                    # Retrieve the column-specific config if it exists; otherwise, use the default
                    col_config = number_config.get(
                        col, number_config.get("default", {})
                    )

                    # Extract modes and inf_replacement for the column from its config
                    number_modes = col_config.get("modes", [])
                    inf_replacement = col_config.get("inf_replacement", pd.NA)

                    # Only apply standardization if there are modes specified
                    if number_modes:
                        df[col] = self.standardize_numbers(
                            df[col],
                            modes=number_modes,
                            inf_replacement=inf_replacement,
                        )

            logger.info("Numerical standardization completed.")

            # 8. Replace specified text patterns
            string_columns = df.select_dtypes(include=["object", "string"]).columns
            replace_config = self.config.get("replace_text", {})
            replace_modes = replace_config.get("modes", [])
            replacements = replace_config.get("replacements", [""])
            if replace_modes:
                for col in string_columns:
                    df[col] = self.replace_text(
                        df[col], modes=replace_modes, replacements=replacements
                    )
                logger.info("Text replacement completed.")
            else:
                logger.info("No text replacement modes specified. Skipping.")

            # 9. Standardize text
            text_config = self.config.get("standardize_text", {})
            text_modes = text_config.get("modes", [])
            if text_modes:
                for col in string_columns:
                    df[col] = self.standardize_text(df[col], modes=text_modes)
                logger.info("Text standardization completed.")
            else:
                logger.info("No text standardization modes specified. Skipping.")

            # 10. Remove or normalize whitespace in all string columns
            whitespace_config = self.config.get("whitespace_handling", {})
            whitespace_mode = whitespace_config.get("mode", "normalize")
            for col in string_columns:
                df[col] = self.remove_whitespace(df[col], mode=whitespace_mode)
            logger.info("Whitespace handling completed.")

            # 11. Recast column dtypes if explicitly specified in the config yaml file
            df = self.cast_column_dtype(df)
            logger.info("Column dtype handling complete.")

            logger.info("DataFrame cleaning process completed successfully.")

            return df

        except Exception as e:
            logger.error(f"Error during DataFrame processing: {e}")
            raise
