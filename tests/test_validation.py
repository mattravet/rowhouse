"""Tests for the validation module."""
import pytest
import pandas as pd
import numpy as np

from validation import DataValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_initial_state(self):
        result = ValidationResult(success=True)
        assert result.success is True
        assert result.warnings == []
        assert result.errors == []

    def test_add_warning(self):
        result = ValidationResult(success=True)
        result.add_warning("test warning")
        assert result.warnings == ["test warning"]
        assert result.success is True  # Warnings don't change success

    def test_add_error(self):
        result = ValidationResult(success=True)
        result.add_error("test error")
        assert result.errors == ["test error"]
        assert result.success is False  # Errors set success to False


class TestToNumeric:
    """Tests for numeric conversion."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_already_numeric(self, validator):
        series = pd.Series([1.0, 2.0, 3.0])
        result = validator.to_numeric(series)
        pd.testing.assert_series_equal(result, series)

    def test_string_numbers(self, validator):
        series = pd.Series(['1', '2', '3.5'])
        result = validator.to_numeric(series)
        expected = pd.Series([1.0, 2.0, 3.5])
        pd.testing.assert_series_equal(result, expected)

    def test_currency_symbols(self, validator):
        series = pd.Series(['$100', '€50', '£75.50'])
        result = validator.to_numeric(series)
        expected = pd.Series([100.0, 50.0, 75.5])
        pd.testing.assert_series_equal(result, expected)

    def test_thousands_separators(self, validator):
        series = pd.Series(['1,000', '1,000,000', '1,234.56'])
        result = validator.to_numeric(series)
        expected = pd.Series([1000.0, 1000000.0, 1234.56])
        pd.testing.assert_series_equal(result, expected)

    def test_percentages(self, validator):
        series = pd.Series(['50%', '100%', '25.5%'])
        result = validator.to_numeric(series)
        expected = pd.Series([0.5, 1.0, 0.255])
        pd.testing.assert_series_equal(result, expected)

    def test_negative_values(self, validator):
        series = pd.Series(['-100', '-50.5', '25'])
        result = validator.to_numeric(series)
        expected = pd.Series([-100.0, -50.5, 25.0])
        pd.testing.assert_series_equal(result, expected)

    def test_disallow_negative(self, validator):
        series = pd.Series(['-100', '50'])
        result = validator.to_numeric(series, allow_negative=False)
        # Dash stripped, result is numeric (may be int or float)
        assert result.iloc[0] == 100
        assert result.iloc[1] == 50

    def test_invalid_values_become_nan(self, validator):
        series = pd.Series(['abc', '100', 'xyz'])
        result = validator.to_numeric(series)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 100.0
        assert np.isnan(result.iloc[2])

    def test_preserves_nan(self, validator):
        series = pd.Series([1.0, np.nan, 3.0])
        result = validator.to_numeric(series)
        assert result.iloc[0] == 1.0
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 3.0

    def test_whitespace_handling(self, validator):
        series = pd.Series(['  100  ', ' 50.5 '])
        result = validator.to_numeric(series)
        expected = pd.Series([100.0, 50.5])
        pd.testing.assert_series_equal(result, expected)


class TestToInteger:
    """Tests for integer conversion."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_basic_conversion(self, validator):
        series = pd.Series([1.0, 2.0, 3.0])
        result = validator.to_integer(series)
        expected = pd.Series([1, 2, 3], dtype=pd.Int64Dtype())
        pd.testing.assert_series_equal(result, expected)

    def test_from_strings(self, validator):
        series = pd.Series(['1', '2', '3'])
        result = validator.to_integer(series)
        expected = pd.Series([1, 2, 3], dtype=pd.Int64Dtype())
        pd.testing.assert_series_equal(result, expected)

    def test_truncation_warning(self, validator):
        series = pd.Series([1.5, 2.7, 3.9])
        result = validator.to_integer(series)
        assert 'truncated' in validator.last_result.warnings[0].lower()

    def test_invalid_becomes_na(self, validator):
        series = pd.Series(['abc', '100', None])
        result = validator.to_integer(series)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 100
        assert pd.isna(result.iloc[2])

    def test_overflow_detection(self, validator):
        series = pd.Series([10**20])  # Too large for int64
        result = validator.to_integer(series)
        assert pd.isna(result.iloc[0])
        assert len(validator.last_result.errors) > 0


class TestToDatetime:
    """Tests for datetime conversion."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_iso_format(self, validator):
        series = pd.Series(['2024-01-15', '2024-02-20'])
        result = validator.to_datetime(series)
        assert result.iloc[0] == pd.Timestamp('2024-01-15')
        assert result.iloc[1] == pd.Timestamp('2024-02-20')

    def test_iso_with_time(self, validator):
        series = pd.Series(['2024-01-15 10:30:00', '2024-02-20 14:45:30'])
        result = validator.to_datetime(series)
        assert result.iloc[0] == pd.Timestamp('2024-01-15 10:30:00')

    def test_us_format(self, validator):
        series = pd.Series(['01/15/2024', '02/20/2024'])
        result = validator.to_datetime(series)
        assert result.iloc[0] == pd.Timestamp('2024-01-15')

    def test_us_12hour_format(self, validator):
        series = pd.Series(['01/15/24 10:30 AM', '02/20/24 02:45 PM'])
        result = validator.to_datetime(series)
        assert result.iloc[0].hour == 10
        assert result.iloc[1].hour == 14

    def test_custom_formats(self, validator):
        custom_formats = ['%d.%m.%Y']  # German format
        result = validator.to_datetime(
            pd.Series(['15.01.2024']),
            formats=custom_formats
        )
        assert result.iloc[0] == pd.Timestamp('2024-01-15')

    def test_invalid_becomes_nat(self, validator):
        series = pd.Series(['2024-01-15', 'not-a-date', '2024-02-20'])
        result = validator.to_datetime(series)
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.notna(result.iloc[2])

    def test_datetime_precision(self, validator):
        series = pd.Series(['2024-01-15 10:30:00.123456'])
        result = validator.to_datetime(series, unit='us')
        assert result.dtype == 'datetime64[us]'


class TestToBoolean:
    """Tests for boolean conversion."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_true_values(self, validator):
        series = pd.Series(['true', 'True', 'TRUE', 'yes', 'YES', '1', 'on'])
        result = validator.to_boolean(series)
        assert all(result == True)

    def test_false_values(self, validator):
        series = pd.Series(['false', 'False', 'FALSE', 'no', 'NO', '0', 'off'])
        result = validator.to_boolean(series)
        assert all(result == False)

    def test_mixed_values(self, validator):
        series = pd.Series(['yes', 'no', True, False, 1, 0])
        result = validator.to_boolean(series)
        expected = pd.Series([True, False, True, False, True, False], dtype=pd.BooleanDtype())
        pd.testing.assert_series_equal(result, expected)

    def test_invalid_becomes_na(self, validator):
        series = pd.Series(['yes', 'maybe', 'no'])
        result = validator.to_boolean(series)
        assert result.iloc[0] == True
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == False

    def test_custom_values(self, validator):
        series = pd.Series(['active', 'inactive'])
        result = validator.to_boolean(
            series,
            true_values=['active'],
            false_values=['inactive']
        )
        expected = pd.Series([True, False], dtype=pd.BooleanDtype())
        pd.testing.assert_series_equal(result, expected)


class TestNormalizeColumns:
    """Tests for column name normalization."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_lowercase(self, validator):
        df = pd.DataFrame({'Column_Name': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['column_name']

    def test_spaces_to_underscores(self, validator):
        df = pd.DataFrame({'Column Name': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['column_name']

    def test_hyphens_to_underscores(self, validator):
        df = pd.DataFrame({'column-name': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['column_name']

    def test_remove_special_chars(self, validator):
        df = pd.DataFrame({'column@name!': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['columnname']

    def test_remove_leading_dollar(self, validator):
        df = pd.DataFrame({'$revenue': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['revenue']

    def test_collapse_underscores(self, validator):
        df = pd.DataFrame({'column__name': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['column_name']

    def test_strip_leading_trailing_underscores(self, validator):
        df = pd.DataFrame({'_column_': [1]})
        result = validator.normalize_columns(df)
        assert list(result.columns) == ['column']

    def test_duplicate_detection(self, validator):
        df = pd.DataFrame({'Column A': [1], 'column_a': [2]})
        result = validator.normalize_columns(df)
        assert 'duplicates' in validator.last_result.errors[0].lower()


class TestApplySchema:
    """Tests for schema-driven transformations."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_basic_schema(self, validator):
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'value': ['100.5', '200.5', '300.5']
        })
        schema = {
            'id': {'type': 'integer'},
            'value': {'type': 'float'}
        }
        result, validation = validator.apply_schema(df, schema)

        assert result['id'].dtype == pd.Int64Dtype()
        assert result['value'].dtype == np.float64

    def test_column_rename(self, validator):
        df = pd.DataFrame({'old_name': [1]})
        schema = {
            'old_name': {'type': 'integer', 'rename': 'new_name'}
        }
        result, _ = validator.apply_schema(df, schema)
        assert 'new_name' in result.columns
        assert 'old_name' not in result.columns

    def test_source_mapping(self, validator):
        df = pd.DataFrame({'source_col': ['2024-01-15']})
        schema = {
            'date_field': {'source': 'source_col', 'type': 'datetime'}
        }
        result, _ = validator.apply_schema(df, schema)
        assert 'date_field' in result.columns

    def test_case_insensitive_source(self, validator):
        df = pd.DataFrame({'COLUMN_NAME': [1]})
        schema = {
            'output': {'source': 'column_name', 'type': 'integer'}
        }
        result, _ = validator.apply_schema(df, schema)
        assert 'output' in result.columns

    def test_required_missing_column(self, validator):
        df = pd.DataFrame({'other': [1]})
        schema = {
            'required_col': {'type': 'integer', 'required': True}
        }
        result, validation = validator.apply_schema(df, schema)
        assert not validation.success
        assert 'required' in validation.errors[0].lower()

    def test_optional_missing_column(self, validator):
        df = pd.DataFrame({'other': [1]})
        schema = {
            'optional_col': {'type': 'integer', 'required': False}
        }
        result, validation = validator.apply_schema(df, schema)
        assert validation.success
        assert len(validation.warnings) > 0

    def test_drop_extra_columns(self, validator):
        df = pd.DataFrame({'keep': [1], 'drop': [2]})
        schema = {'keep': {'type': 'integer'}}
        result, _ = validator.apply_schema(df, schema, drop_extra=True)
        assert 'keep' in result.columns
        assert 'drop' not in result.columns

    def test_keep_extra_columns(self, validator):
        df = pd.DataFrame({'keep': [1], 'extra': [2]})
        schema = {'keep': {'type': 'integer'}}
        result, _ = validator.apply_schema(df, schema, drop_extra=False)
        assert 'keep' in result.columns
        assert 'extra' in result.columns


class TestValidateNotNull:
    """Tests for null validation."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_no_nulls(self, validator):
        df = pd.DataFrame({'col': [1, 2, 3]})
        result = validator.validate_not_null(df, ['col'])
        assert result.success

    def test_with_nulls(self, validator):
        df = pd.DataFrame({'col': [1, None, 3]})
        result = validator.validate_not_null(df, ['col'])
        assert not result.success
        assert '1 null' in result.errors[0]

    def test_missing_column(self, validator):
        df = pd.DataFrame({'other': [1]})
        result = validator.validate_not_null(df, ['missing'])
        assert not result.success
        assert 'not found' in result.errors[0].lower()


class TestValidateUnique:
    """Tests for uniqueness validation."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_unique_values(self, validator):
        df = pd.DataFrame({'id': [1, 2, 3]})
        result = validator.validate_unique(df, 'id')
        assert result.success

    def test_duplicate_values(self, validator):
        df = pd.DataFrame({'id': [1, 1, 2]})
        result = validator.validate_unique(df, 'id')
        assert not result.success
        assert 'duplicate' in result.errors[0].lower()

    def test_composite_uniqueness(self, validator):
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': [1, 2, 1]
        })
        result = validator.validate_unique(df, ['a', 'b'])
        assert result.success

    def test_composite_duplicates(self, validator):
        df = pd.DataFrame({
            'a': [1, 1, 1],
            'b': [1, 1, 2]
        })
        result = validator.validate_unique(df, ['a', 'b'])
        assert not result.success
