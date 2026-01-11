import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional

from validation import DataValidator

logger = logging.getLogger(__name__)

class JsonProcessor:
    def __init__(self, split_path: List[str], config: Dict, coerce: bool = False):
        """
        Initialize JsonProcessor.

        Args:
            split_path: Path to the field used to route messages to configs.
            config: Configuration dict mapping message types to field definitions.
            coerce: If True, use smart coercion for type conversion (handles
                   currency symbols, multiple date formats, etc.). Can also be
                   set per-field with {"coerce": True} in field config.
        """
        self.split_path = split_path
        self.config = config
        self.filepath = None
        self.last_modified = None
        self.default_coerce = coerce
        self._validator = DataValidator()
        
    def set_file_metadata(self, filepath: str, last_modified: str):
        """Set file metadata before processing"""
        self.filepath = filepath
        self.last_modified = last_modified

    def _extract_path(self, source):
        path = source.split('.')
        is_in_array = any('[]' in part for part in path)
        path = [part.replace('[]', '') for part in path]
        return path, is_in_array

    def _traverse(self, data, path_parts, row_so_far, config):
        # Handle top level cartesian product
        if not path_parts:
            # Extract top-level keys, stripping [] from array notation
            top_level_keys = set(
                field['source'].split('.')[0].replace('[]', '')
                for field in config['fields']
            )
            partial_results = {key: [] for key in top_level_keys}

            for key in top_level_keys:
                if key in data:
                    key_data = data[key]
                    # Check if this top-level key is used as an array in any field
                    is_array_key = any(
                        field['source'].startswith(f'{key}[]')
                        for field in config['fields']
                    )

                    if is_array_key and isinstance(key_data, list):
                        # Explode array at top level
                        for item in key_data:
                            partial_result = self._traverse(item, [key], {}, config)
                            if partial_result:
                                partial_results[key].extend(partial_result)
                    else:
                        partial_result = self._traverse(key_data, [key], {}, config)
                        if partial_result:
                            partial_results[key].extend(partial_result)

            # Combine partial results (cartesian product)
            rows = [{}]
            for key in top_level_keys:
                if not partial_results[key]:
                    # If a branch has no results, use empty dict to preserve other branches
                    partial_results[key] = [{}]
                new_rows = []
                for row in rows:
                    for partial_row in partial_results[key]:
                        combined = row.copy()
                        combined.update(partial_row)
                        new_rows.append(combined)
                rows = new_rows

            return rows

        rows = []
        current_row = row_so_far.copy()

        # Track which nested paths we need to recurse into
        # Use a dict to avoid duplicate recursion for the same path
        paths_to_recurse = {}

        # First pass: extract all leaf values at current level
        for field in config['fields']:
            path, is_in_array = self._extract_path(field['source'])

            if path[:len(path_parts)] != path_parts:
                continue

            remaining = path[len(path_parts):]
            if not remaining:
                continue

            next_part = remaining[0]
            key = next_part

            # Handle leaf nodes - extract value directly
            if len(remaining) == 1:
                if isinstance(data, dict) and key in data:
                    current_row[field['alias']] = data[key]
            # Track nested structures for recursion
            elif isinstance(data, dict) and key in data:
                next_path_parts = tuple(path_parts + [key])
                if next_path_parts not in paths_to_recurse:
                    paths_to_recurse[next_path_parts] = {
                        'key': key,
                        'data': data[key],
                        'has_array': is_in_array
                    }
                # If any field at this path needs array handling, mark it
                if is_in_array:
                    paths_to_recurse[next_path_parts]['has_array'] = True

        # Second pass: recurse into nested structures
        # Separate array paths from non-array dict paths
        array_paths = []
        dict_paths = []

        for next_path_parts, info in paths_to_recurse.items():
            next_data = info['data']
            is_in_array = info['has_array']

            if is_in_array and isinstance(next_data, list):
                array_paths.append((next_path_parts, info))
            elif isinstance(next_data, dict):
                dict_paths.append((next_path_parts, info))

        # Collect all sub-results for cartesian product
        all_sub_results = []
        empty_array_encountered = False

        # For non-array dict paths: traverse and collect results
        # Note: dict paths may return multiple rows if they contain arrays
        for next_path_parts, info in dict_paths:
            sub_rows = self._traverse(info['data'], list(next_path_parts), {}, config)
            if sub_rows:
                all_sub_results.append(sub_rows)

        # For array paths: traverse each item and collect results
        for next_path_parts, info in array_paths:
            next_data = info['data']
            sub_rows_for_array = []
            for item in next_data:
                sub_rows = self._traverse(item, list(next_path_parts), {}, config)
                if sub_rows:
                    sub_rows_for_array.extend(sub_rows)
            if sub_rows_for_array:
                all_sub_results.append(sub_rows_for_array)
            elif len(next_data) == 0:
                # Array was empty - this branch produces no rows
                empty_array_encountered = True
                break

        # Combine results via cartesian product
        if empty_array_encountered:
            # Empty array kills the branch - no rows produced
            pass
        elif all_sub_results:
            # Cartesian product of all sub-results with current_row
            result_rows = [current_row.copy()] if current_row else [{}]
            for sub_result in all_sub_results:
                new_rows = []
                for base_row in result_rows:
                    for sub_row in sub_result:
                        combined = base_row.copy()
                        combined.update(sub_row)
                        new_rows.append(combined)
                result_rows = new_rows
            rows.extend(result_rows)
        elif current_row:
            # No nested results, just use current row
            rows.append(current_row)

        return rows
        
    def split_messages(self, raw_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Match the original split_messages function logic"""
        split_lists = {}
        for message in raw_data:
            try:
                # Get split value using the same logic as original
                split_val = message.get(self.split_path[0])
                for i in range(len(self.split_path) - 1):
                    split_val = split_val.get(self.split_path[i+1])
                
                # Add to existing list or create new one
                if split_val in split_lists:
                    split_lists[split_val].append(message)
                else:
                    split_lists[split_val] = [message]
            except AttributeError:
                logger.error("Split path not found in message")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
        return split_lists

    def process_messages(self, raw_data: List[Dict]) -> Dict[str, pd.DataFrame]:
        split_messages = self.split_messages(raw_data)
        return self._create_dataframes(split_messages)

    def _create_dataframes(self, split_messages: Dict) -> Dict[str, pd.DataFrame]:
        dataframes = {}
        
        if not self.filepath:
            raise ValueError("Filepath not set in JsonProcessor")
        
        for key, config in self.config.items():
            if str(key) not in split_messages:
                continue
                        
            # Create DataFrame using traverse
            rows = []
            for data in split_messages[str(key)]:
                message_rows = self._traverse(data, [], {}, config)
                if message_rows:
                    rows.extend(message_rows)
            
            if rows:
                df = pd.DataFrame(rows)
                
                # Replace empty strings with NaN
                df = df.replace('', np.nan)
    
                df['s3_source_path'] = self.filepath
                df['s3_last_modified_utc']= pd.to_datetime(self.last_modified).tz_localize(None)
                df['s3_last_modified_utc']= df['s3_last_modified_utc'].astype('datetime64[us]')

                # Add empty columns for missing fields from config
                for field in config['fields']:
                    if field['alias'] not in df.columns:
                        df[field['alias']] = np.nan
                
                # Apply data type enforcement
                df = self._enforce_datatypes(df, config['fields'])
                
                # Order columns according to config
                ordered_columns = [field['alias'] for field in config['fields']]
                # Add the metadata columns at the end
                ordered_columns.extend(['s3_source_path', 's3_last_modified_utc'])
                # Reorder columns to match config order
                df = df[ordered_columns]
    
                dataframes[key] = df
                
        return dataframes


    def _enforce_datatypes(self, df: pd.DataFrame, fields: List[Dict]) -> pd.DataFrame:
        """
        Enforce data types on DataFrame columns based on configuration.

        When coerce=True (globally or per-field), uses smart coercion that handles:
        - Currency symbols ($, €, £) and thousands separators for numeric types
        - Multiple date/time formats for timestamp/date types
        - Integer overflow detection
        - Flexible boolean parsing (yes/no, true/false, 1/0, on/off)
        """
        type_map = {
            'string': 'string[pyarrow]',
            'integer': 'Int64',
            'float': 'Float64',
            'boolean': 'boolean',
            'timestamp': 'datetime64[us]',
            'date': 'datetime64[us]'
        }

        for field in fields:
            col_name = field['alias']
            if 'type' not in field:
                continue

            data_type = field['type'].lower()
            # Check per-field coerce setting, fall back to global default
            should_coerce = field.get('coerce', self.default_coerce)

            if col_name not in df.columns:
                continue

            try:
                # Convert 'nan' strings to proper nulls first
                df[col_name] = df[col_name].replace({'nan': None, 'NaN': None, 'NAN': None})

                if should_coerce:
                    # Use DataValidator for smart coercion
                    df[col_name] = self._coerce_column(df[col_name], data_type, col_name)
                else:
                    # Standard pandas conversion
                    if data_type in ['timestamp', 'date']:
                        df[col_name] = pd.to_datetime(
                            df[col_name], utc=True, format='mixed', dayfirst=False
                        ).dt.tz_localize(None)
                        df[col_name] = df[col_name].astype('datetime64[us]')
                    elif data_type in type_map:
                        df[col_name] = df[col_name].astype(type_map[data_type])
                    else:
                        logger.warning(
                            f"Unknown data type '{data_type}' for column '{col_name}'. "
                            "Defaulting to string."
                        )
                        df[col_name] = df[col_name].astype('string[pyarrow]')

            except Exception as e:
                logger.error(f"Error enforcing data type for column '{col_name}': {str(e)}")

        return df

    def _coerce_column(self, series: pd.Series, data_type: str, col_name: str) -> pd.Series:
        """
        Apply smart coercion to a column using DataValidator.

        Handles messy real-world data like:
        - "$1,234.56" -> 1234.56 (float)
        - "50%" -> 0.5 (float)
        - "01/15/2024" or "2024-01-15" -> datetime
        - "yes", "true", "1" -> True (boolean)
        """
        if data_type == 'integer':
            result = self._validator.to_integer(series)
            # Log any warnings/errors from validation
            if self._validator.last_result:
                for warn in self._validator.last_result.warnings:
                    logger.warning(f"Column '{col_name}': {warn}")
                for err in self._validator.last_result.errors:
                    logger.warning(f"Column '{col_name}': {err}")
            return result

        elif data_type == 'float':
            result = self._validator.to_numeric(series)
            if self._validator.last_result:
                for warn in self._validator.last_result.warnings:
                    logger.warning(f"Column '{col_name}': {warn}")
            return result.astype('Float64')

        elif data_type in ['timestamp', 'date']:
            result = self._validator.to_datetime(series, unit='us')
            if self._validator.last_result:
                for warn in self._validator.last_result.warnings:
                    logger.warning(f"Column '{col_name}': {warn}")
            return result

        elif data_type == 'boolean':
            result = self._validator.to_boolean(series)
            if self._validator.last_result:
                for warn in self._validator.last_result.warnings:
                    logger.warning(f"Column '{col_name}': {warn}")
            return result

        elif data_type == 'string':
            return series.astype('string[pyarrow]')

        else:
            logger.warning(
                f"Unknown data type '{data_type}' for column '{col_name}'. "
                "Defaulting to string."
            )
            return series.astype('string[pyarrow]')

