"""
Rowhouse Discover - JSON structure analysis and splitter discovery.

Analyzes collections of JSON documents to discover structural patterns
and identify the best field to use as a splitter for JsonProcessor.

Example:
    >>> from rowhouse.discover import StructureAnalyzer
    >>>
    >>> analyzer = StructureAnalyzer()
    >>> results = analyzer.find_splitters(documents)
    >>> print(results[0])
    SplitterResult(field='header.action', score=3.76, values=5, coverage=100.0%)
    >>>
    >>> # Use with JsonProcessor
    >>> processor = JsonProcessor(split_path=results[0].field.split('.'), config)
"""
from .analyzer import StructureAnalyzer, SplitterResult, StructureSummary
from .similarity import (
    SimilarityStrategy,
    JaccardPathSimilarity,
    WeightedJaccardSimilarity,
    ExactMatchSimilarity,
)

__all__ = [
    'StructureAnalyzer',
    'SplitterResult',
    'StructureSummary',
    'SimilarityStrategy',
    'JaccardPathSimilarity',
    'WeightedJaccardSimilarity',
    'ExactMatchSimilarity',
]
