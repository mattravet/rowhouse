"""
Similarity strategies for comparing JSON document structures.

Provides pluggable similarity calculations for the StructureAnalyzer.
"""
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class SimilarityStrategy(Protocol):
    """Protocol for similarity calculation strategies."""

    def similarity(self, paths_a: set[str], paths_b: set[str]) -> float:
        """
        Calculate similarity between two path sets.

        Args:
            paths_a: Path set from first document.
            paths_b: Path set from second document.

        Returns:
            Similarity score between 0 and 1.
        """
        ...


class JaccardPathSimilarity:
    """
    Jaccard similarity on path sets.

    Jaccard Index = |A ∩ B| / |A ∪ B|

    Range: 0 (completely different) to 1 (identical)

    Example:
        >>> sim = JaccardPathSimilarity()
        >>> paths_a = {"header.action", "body.items[].sku"}
        >>> paths_b = {"header.action", "body.user.name"}
        >>> sim.similarity(paths_a, paths_b)
        0.333...  # 1 intersection / 3 union
    """

    def similarity(self, paths_a: set[str], paths_b: set[str]) -> float:
        """Calculate Jaccard similarity between two path sets."""
        if not paths_a and not paths_b:
            return 1.0  # Both empty = identical

        intersection = len(paths_a & paths_b)
        union = len(paths_a | paths_b)

        if union == 0:
            return 1.0

        return intersection / union


class WeightedJaccardSimilarity:
    """
    Weighted Jaccard similarity with depth decay.

    Deeper paths contribute less to the similarity score.
    Useful when structural differences near the root matter more.

    Args:
        depth_decay: Factor to multiply weight by for each level of depth.
                    0.8 means depth 2 has 0.64 weight, depth 3 has 0.51, etc.
    """

    def __init__(self, depth_decay: float = 0.8):
        self.depth_decay = depth_decay

    def _path_weight(self, path: str) -> float:
        """Calculate weight based on path depth."""
        depth = path.count(".") + path.count("[]")
        return self.depth_decay ** depth

    def similarity(self, paths_a: set[str], paths_b: set[str]) -> float:
        """Calculate weighted Jaccard similarity."""
        if not paths_a and not paths_b:
            return 1.0

        all_paths = paths_a | paths_b
        if not all_paths:
            return 1.0

        intersection_weight = sum(
            self._path_weight(p) for p in paths_a & paths_b
        )
        union_weight = sum(
            self._path_weight(p) for p in all_paths
        )

        if union_weight == 0:
            return 1.0

        return intersection_weight / union_weight


class ExactMatchSimilarity:
    """
    Binary similarity - 1 if path sets are identical, 0 otherwise.

    Useful for strict structural matching.
    """

    def similarity(self, paths_a: set[str], paths_b: set[str]) -> float:
        """Return 1.0 if identical, 0.0 otherwise."""
        return 1.0 if paths_a == paths_b else 0.0
