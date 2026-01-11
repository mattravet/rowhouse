"""
Rowhouse - Data tools for building data pipelines.

Subpackages:
    - rowhouse.unfurl: Unfurl nested JSON into flat tabular data
    - rowhouse.discover: JSON structure analysis and splitter discovery
    - rowhouse.validation: Data validation and type conversion
    - rowhouse.aws: AWS utilities (S3, etc.)

Example:
    >>> from rowhouse.discover import StructureAnalyzer
    >>> from rowhouse.unfurl import JsonProcessor
    >>> from rowhouse.aws import S3Handler
    >>> from rowhouse.validation import DataValidator
"""
__version__ = "0.1.0"
