"""
Data validation and type conversion utilities.

Provides safe type conversions, schema validation, and column normalization
for pandas DataFrames in ETL pipelines.
"""
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of a validation or conversion operation."""
    success: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.success = False


class DataValidator:
    """
    Validates and converts DataFrame columns with configurable type handling.

    Features:
        - Safe numeric conversion (handles mixed types, currency symbols)
        - Integer conversion with overflow detection
        - Multi-format datetime parsing
        - Column name normalization
        - Schema-driven transformations

    Example:
        >>> validator = DataValidator()
        >>> df['price'] = validator.to_numeric(df['price'])
        >>> df['created_at'] = validator.to_datetime(df['created_at'])
        >>> df = validator.normalize_columns(df)
    """

    # Default datetime formats to try (order matters - more specific first)
    DEFAULT_DATE_FORMATS = [
        '%Y-%m-%d %H:%M:%S.%f',  # ISO with microseconds
        '%Y-%m-%d %H:%M:%S',     # ISO with time
        '%Y-%m-%dT%H:%M:%S.%fZ', # ISO 8601 with Z
        '%Y-%m-%dT%H:%M:%SZ',    # ISO 8601 with Z (no ms)
        '%Y-%m-%dT%H:%M:%S',     # ISO 8601
        '%Y-%m-%d',              # ISO date only
        '%m/%d/%Y %I:%M:%S %p',  # US 12-hour with seconds
        '%m/%d/%Y %I:%M %p',     # US 12-hour
        '%m/%d/%y %I:%M %p',     # US 12-hour short year
        '%m/%d/%Y %H:%M:%S',     # US 24-hour with seconds
        '%m/%d/%Y',              # US date
        '%m/%d/%y',              # US date short year
        '%d/%m/%Y',              # European date
        '%d-%m-%Y',              # European with dashes
        '%Y%m%d',                # Compact date
    ]

    def __init__(
        self,
        date_formats: Optional[list[str]] = None,
        strict: bool = False
    ):
        """
        Initialize the validator.

        Args:
            date_formats: Custom datetime formats to try. If None, uses defaults.
            strict: If True, raise errors instead of coercing invalid values.
        """
        self.date_formats = date_formats or self.DEFAULT_DATE_FORMATS
        self.strict = strict
        self._last_result: Optional[ValidationResult] = None

    @property
    def last_result(self) -> Optional[ValidationResult]:
        """Get the result of the last validation operation."""
        return self._last_result

    def to_numeric(
        self,
        series: pd.Series,
        allow_negative: bool = True
    ) -> pd.Series:
        """
        Convert series to numeric, handling mixed types and cleaning.

        Handles:
            - Currency symbols ($, €, £)
            - Thousands separators (commas)
            - Percentage signs
            - Whitespace

        Args:
            series: Input series to convert.
            allow_negative: Whether to allow negative values.

        Returns:
            Numeric series with invalid values as NaN.
        """
        self._last_result = ValidationResult(success=True)

        # Already numeric - return as-is
        if pd.api.types.is_numeric_dtype(series):
            return series

        cleaned = series.copy()

        def clean_value(x):
            if pd.isna(x):
                return x

            s = str(x).strip()

            # Remove currency symbols and whitespace
            s = re.sub(r'[$€£¥₹\s]', '', s)

            # Remove thousands separators (commas not followed by decimals)
            s = re.sub(r',(?=\d{3}(?:[,.]|$))', '', s)

            # Handle percentages (convert 50% to 0.5)
            if s.endswith('%'):
                s = s[:-1]
                try:
                    return float(s) / 100
                except ValueError:
                    return None

            # Keep only valid numeric characters
            valid_chars = '0123456789.'
            if allow_negative:
                valid_chars += '-'

            cleaned_str = ''.join(c for c in s if c in valid_chars)

            # Handle multiple decimal points or dashes
            if cleaned_str.count('.') > 1 or cleaned_str.count('-') > 1:
                self._last_result.add_warning(f"Invalid numeric format: {x}")
                return None

            # Dash must be at start
            if '-' in cleaned_str and not cleaned_str.startswith('-'):
                self._last_result.add_warning(f"Invalid negative format: {x}")
                return None

            return cleaned_str if cleaned_str else None

        cleaned = cleaned.apply(clean_value)
        result = pd.to_numeric(cleaned, errors='coerce')

        # Count NaN values introduced
        new_nans = result.isna().sum() - series.isna().sum()
        if new_nans > 0:
            self._last_result.add_warning(
                f"{new_nans} values could not be converted to numeric"
            )

        return result

    def to_integer(
        self,
        series: pd.Series,
        warn_truncation: bool = True
    ) -> pd.Series:
        """
        Convert series to nullable integer with overflow detection.

        Args:
            series: Input series to convert.
            warn_truncation: Warn if decimal values are truncated.

        Returns:
            Integer series using pandas nullable Int64 dtype.
        """
        self._last_result = ValidationResult(success=True)

        # First convert to numeric
        numeric = self.to_numeric(series)

        def safe_int(x):
            if pd.isna(x):
                return pd.NA

            try:
                float_val = float(x)

                # Check for overflow
                if float_val > np.iinfo(np.int64).max:
                    self._last_result.add_error(
                        f"Integer overflow: {x} exceeds int64 max"
                    )
                    return pd.NA
                if float_val < np.iinfo(np.int64).min:
                    self._last_result.add_error(
                        f"Integer underflow: {x} below int64 min"
                    )
                    return pd.NA

                return np.int64(float_val)

            except (ValueError, TypeError, OverflowError):
                return pd.NA

        # Check for decimal truncation
        if warn_truncation:
            numeric_clean = pd.to_numeric(numeric, errors='coerce')
            has_decimals = ((numeric_clean % 1) != 0).any()
            if has_decimals:
                self._last_result.add_warning(
                    "Decimal values will be truncated to integers"
                )

        converted = numeric.apply(safe_int)
        return pd.Series(converted, dtype=pd.Int64Dtype())

    def to_datetime(
        self,
        series: pd.Series,
        formats: Optional[list[str]] = None,
        unit: str = 'us'
    ) -> pd.Series:
        """
        Convert series to datetime, trying multiple formats.

        Args:
            series: Input series to convert.
            formats: Specific formats to try. If None, uses instance defaults.
            unit: Datetime precision ('s', 'ms', 'us', 'ns').

        Returns:
            Datetime series with specified precision.
        """
        self._last_result = ValidationResult(success=True)
        formats_to_try = formats or self.date_formats

        for date_format in formats_to_try:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    result = pd.to_datetime(
                        series,
                        format=date_format,
                        errors='raise'
                    )
                    return result.astype(f'datetime64[{unit}]')
            except (ValueError, TypeError):
                continue

        # Fall back to pandas inference
        self._last_result.add_warning(
            "No exact format match - using pandas datetime inference"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = pd.to_datetime(series, errors='coerce')

            # Count failed conversions
            new_nans = result.isna().sum() - series.isna().sum()
            if new_nans > 0:
                self._last_result.add_warning(
                    f"{new_nans} values could not be parsed as datetime"
                )

            return result.astype(f'datetime64[{unit}]')

    def to_boolean(
        self,
        series: pd.Series,
        true_values: Optional[list] = None,
        false_values: Optional[list] = None
    ) -> pd.Series:
        """
        Convert series to nullable boolean.

        Args:
            series: Input series to convert.
            true_values: Values to interpret as True.
            false_values: Values to interpret as False.

        Returns:
            Boolean series using pandas nullable boolean dtype.
        """
        self._last_result = ValidationResult(success=True)

        true_vals = true_values or [
            True, 'true', 'True', 'TRUE', 'yes', 'Yes', 'YES',
            '1', 1, 1.0, 'y', 'Y', 'on', 'On', 'ON'
        ]
        false_vals = false_values or [
            False, 'false', 'False', 'FALSE', 'no', 'No', 'NO',
            '0', 0, 0.0, 'n', 'N', 'off', 'Off', 'OFF'
        ]

        def convert_bool(x):
            if pd.isna(x):
                return pd.NA
            if x in true_vals:
                return True
            if x in false_vals:
                return False

            # Try string matching
            s = str(x).strip().lower()
            if s in [str(v).lower() for v in true_vals]:
                return True
            if s in [str(v).lower() for v in false_vals]:
                return False

            self._last_result.add_warning(f"Unknown boolean value: {x}")
            return pd.NA

        converted = series.apply(convert_bool)
        return pd.Series(converted, dtype=pd.BooleanDtype())

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names for consistency.

        Transformations:
            - Remove special characters (keep alphanumeric and underscore)
            - Replace spaces/hyphens with underscores
            - Remove leading/trailing underscores
            - Remove leading dollar signs
            - Convert to lowercase
            - Collapse multiple underscores

        Args:
            df: DataFrame with columns to normalize.

        Returns:
            DataFrame with normalized column names.
        """
        self._last_result = ValidationResult(success=True)
        df = df.copy()

        original_names = df.columns.tolist()

        # Apply transformations (order matters: convert hyphens before removing special chars)
        df.columns = df.columns.str.replace(r'[\s-]+', '_', regex=True)
        df.columns = df.columns.str.replace(r'[^\w]', '', regex=True)
        df.columns = df.columns.str.replace(r'_+', '_', regex=True)
        df.columns = df.columns.str.strip('_')
        df.columns = df.columns.str.lstrip('$')
        df.columns = df.columns.str.lower()

        # Check for duplicates after normalization
        new_names = df.columns.tolist()
        if len(new_names) != len(set(new_names)):
            duplicates = [n for n in new_names if new_names.count(n) > 1]
            self._last_result.add_error(
                f"Column normalization created duplicates: {set(duplicates)}"
            )

        # Track renames
        for old, new in zip(original_names, new_names):
            if old != new:
                self._last_result.add_warning(f"Renamed column: '{old}' -> '{new}'")

        return df

    def apply_schema(
        self,
        df: pd.DataFrame,
        schema: dict[str, dict],
        drop_extra: bool = False
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Apply schema-driven transformations to DataFrame.

        Schema format:
            {
                "column_name": {
                    "source": "original_column",  # Optional, defaults to key
                    "type": "integer",            # Required: integer, float, string, datetime, boolean
                    "required": True,             # Optional, defaults to False
                    "rename": "new_name"          # Optional output column name
                }
            }

        Args:
            df: Input DataFrame.
            schema: Column schema definitions.
            drop_extra: If True, drop columns not in schema.

        Returns:
            Tuple of (transformed DataFrame, ValidationResult).
        """
        result = ValidationResult(success=True)
        output_df = pd.DataFrame()

        # Build column name mapping (case-insensitive)
        col_map = {c.lower(): c for c in df.columns}

        for col_name, col_spec in schema.items():
            source = col_spec.get('source', col_name)
            dtype = col_spec.get('type', 'string').lower()
            required = col_spec.get('required', False)
            output_name = col_spec.get('rename', col_name)

            # Find source column (case-insensitive)
            source_lower = source.lower()
            if source_lower not in col_map:
                if required:
                    result.add_error(f"Required column missing: {source}")
                else:
                    result.add_warning(f"Optional column missing: {source}")
                continue

            actual_source = col_map[source_lower]
            series = df[actual_source].copy()

            # Apply type conversion
            try:
                if dtype in ('integer', 'int', 'bigint'):
                    series = self.to_integer(series)
                elif dtype in ('float', 'decimal', 'numeric', 'double'):
                    series = self.to_numeric(series)
                elif dtype in ('datetime', 'date', 'timestamp'):
                    series = self.to_datetime(series)
                elif dtype in ('boolean', 'bool'):
                    series = self.to_boolean(series)
                elif dtype in ('string', 'str', 'varchar', 'text'):
                    series = series.astype(str).replace('nan', pd.NA)
                else:
                    result.add_warning(f"Unknown type '{dtype}' for {col_name}")

                # Capture conversion warnings
                if self._last_result and self._last_result.warnings:
                    for w in self._last_result.warnings:
                        result.add_warning(f"{col_name}: {w}")

            except Exception as e:
                result.add_error(f"Error converting {col_name}: {e}")
                continue

            output_df[output_name] = series

        # Optionally include extra columns
        if not drop_extra:
            for col in df.columns:
                if col.lower() not in [s.get('source', k).lower() for k, s in schema.items()]:
                    output_df[col] = df[col]

        self._last_result = result
        return output_df, result

    def validate_not_null(
        self,
        df: pd.DataFrame,
        columns: list[str]
    ) -> ValidationResult:
        """
        Validate that specified columns have no null values.

        Args:
            df: DataFrame to validate.
            columns: Column names to check.

        Returns:
            ValidationResult with any null value errors.
        """
        result = ValidationResult(success=True)

        for col in columns:
            if col not in df.columns:
                result.add_error(f"Column not found: {col}")
                continue

            null_count = df[col].isna().sum()
            if null_count > 0:
                result.add_error(
                    f"Column '{col}' has {null_count} null values"
                )

        self._last_result = result
        return result

    def validate_unique(
        self,
        df: pd.DataFrame,
        columns: Union[str, list[str]]
    ) -> ValidationResult:
        """
        Validate that columns (individually or combined) are unique.

        Args:
            df: DataFrame to validate.
            columns: Single column or list of columns for composite uniqueness.

        Returns:
            ValidationResult with any duplicate errors.
        """
        result = ValidationResult(success=True)

        if isinstance(columns, str):
            columns = [columns]

        # Check columns exist
        for col in columns:
            if col not in df.columns:
                result.add_error(f"Column not found: {col}")
                return result

        # Check uniqueness
        duplicates = df.duplicated(subset=columns, keep=False)
        dup_count = duplicates.sum()

        if dup_count > 0:
            result.add_error(
                f"Found {dup_count} duplicate rows for columns: {columns}"
            )

        self._last_result = result
        return result
