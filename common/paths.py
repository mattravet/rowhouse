"""
Path extraction utilities for JSON documents.

Extracts all paths from nested JSON structures, handling arrays and nested objects.
Used by both unfurl (for field mapping) and inspect (for structure analysis).
"""
from typing import Any


class PathExtractor:
    """
    Extracts all unique paths from JSON documents.

    Paths use dot notation for nested objects and [] for arrays:
        - "header.action" - nested object
        - "body.items[].sku" - array of objects
        - "data.tags[]" - array of primitives

    Example:
        >>> extractor = PathExtractor()
        >>> doc = {"header": {"action": "test"}, "items": [{"sku": "A"}]}
        >>> paths = extractor.extract(doc)
        >>> print(paths)
        {'header.action', 'items[].sku'}
    """

    def __init__(self, include_array_indices: bool = False):
        """
        Initialize PathExtractor.

        Args:
            include_array_indices: If True, include array index in path (items[0].sku).
                                  If False (default), use [] notation (items[].sku).
        """
        self.include_array_indices = include_array_indices

    def extract(self, document: dict) -> set[str]:
        """
        Extract all paths from a JSON document.

        Args:
            document: JSON document as a dict.

        Returns:
            Set of all paths in the document.
        """
        paths = set()
        self._extract_recursive(document, "", paths)
        return paths

    def _extract_recursive(self, data: Any, current_path: str, paths: set) -> None:
        """Recursively extract paths from nested structures."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, (dict, list)):
                    self._extract_recursive(value, new_path, paths)
                else:
                    # Leaf value - add the path
                    paths.add(new_path)

        elif isinstance(data, list):
            if not data:
                # Empty array - still record the path
                paths.add(f"{current_path}[]")
                return

            # Check if array contains dicts, primitives, or mixed
            has_dicts = any(isinstance(item, dict) for item in data)
            has_primitives = any(not isinstance(item, (dict, list)) for item in data)

            if has_primitives and not has_dicts:
                # Array of primitives
                paths.add(f"{current_path}[]")
            else:
                # Array of objects (or mixed) - recurse into each
                for i, item in enumerate(data):
                    if self.include_array_indices:
                        array_path = f"{current_path}[{i}]"
                    else:
                        array_path = f"{current_path}[]"

                    if isinstance(item, dict):
                        self._extract_recursive(item, array_path, paths)
                    elif isinstance(item, list):
                        self._extract_recursive(item, array_path, paths)
                    else:
                        # Primitive in array
                        paths.add(f"{current_path}[]")

    def extract_with_values(self, document: dict) -> dict[str, list[Any]]:
        """
        Extract paths along with their values.

        Useful for understanding the data at each path.

        Args:
            document: JSON document as a dict.

        Returns:
            Dict mapping paths to lists of values found at that path.
        """
        path_values: dict[str, list[Any]] = {}
        self._extract_values_recursive(document, "", path_values)
        return path_values

    def _extract_values_recursive(
        self, data: Any, current_path: str, path_values: dict[str, list[Any]]
    ) -> None:
        """Recursively extract paths and their values."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, (dict, list)):
                    self._extract_values_recursive(value, new_path, path_values)
                else:
                    # Leaf value
                    if new_path not in path_values:
                        path_values[new_path] = []
                    path_values[new_path].append(value)

        elif isinstance(data, list):
            array_path = f"{current_path}[]"

            for item in data:
                if isinstance(item, dict):
                    self._extract_values_recursive(item, array_path, path_values)
                elif isinstance(item, list):
                    self._extract_values_recursive(item, array_path, path_values)
                else:
                    # Primitive in array
                    if array_path not in path_values:
                        path_values[array_path] = []
                    path_values[array_path].append(item)

    def get_value_at_path(self, document: dict, path: str) -> Any:
        """
        Get the value at a specific path in a document.

        Args:
            document: JSON document.
            path: Dot-notation path (e.g., "header.action").

        Returns:
            Value at the path, or None if not found.
        """
        parts = path.replace("[]", "").split(".")
        current = document

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                # For arrays, we'd need index - return None for now
                return None
            else:
                return None

        return current


def extract_paths(document: dict, include_array_indices: bool = False) -> set[str]:
    """
    Convenience function to extract paths from a document.

    Args:
        document: JSON document as a dict.
        include_array_indices: If True, include array indices in paths.

    Returns:
        Set of all paths in the document.
    """
    extractor = PathExtractor(include_array_indices=include_array_indices)
    return extractor.extract(document)
