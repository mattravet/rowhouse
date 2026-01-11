"""
Structure analyzer for JSON documents.

Analyzes collections of JSON documents to discover structural patterns
and identify splitter fields that determine document structure.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from itertools import combinations
from collections import defaultdict

from common.paths import PathExtractor
from .similarity import SimilarityStrategy, JaccardPathSimilarity


@dataclass
class SplitterResult:
    """Result of evaluating a candidate splitter field."""

    field: str
    """Path to the splitter field (e.g., 'header.action')"""

    score: float
    """Splitter score: within-group similarity / between-group similarity.
    Higher scores indicate better splitters."""

    distinct_values: int
    """Number of distinct values for this field."""

    coverage: float
    """Fraction of documents that have this field (0-1)."""

    value_counts: dict[str, int] = field(default_factory=dict)
    """Count of documents for each distinct value."""

    within_similarity: float = 0.0
    """Average similarity within groups."""

    between_similarity: float = 0.0
    """Average similarity between groups."""

    def __repr__(self) -> str:
        return (
            f"SplitterResult(field='{self.field}', score={self.score:.2f}, "
            f"values={self.distinct_values}, coverage={self.coverage:.1%})"
        )


@dataclass
class StructureSummary:
    """Summary of document structure for a specific splitter value."""

    value: str
    """The splitter value."""

    count: int
    """Number of documents with this value."""

    unique_paths: set[str]
    """All unique paths in documents with this value."""

    common_paths: set[str]
    """Paths that appear in ALL documents with this value."""

    sample_paths: list[str]
    """Representative sample of paths (for display)."""


class StructureAnalyzer:
    """
    Analyzes JSON document structure and discovers splitter fields.

    A splitter field is one whose value determines the structure of the rest
    of the document. For example, if documents with header.action="OrderCreated"
    always have body.items[], while documents with header.action="UserCreated"
    always have body.user.*, then header.action is a good splitter.

    Example:
        >>> analyzer = StructureAnalyzer()
        >>> results = analyzer.find_splitters(documents)
        >>> print(results[0])
        SplitterResult(field='header.action', score=3.76, values=5, coverage=100.0%)

        >>> # Use the best splitter with JsonProcessor
        >>> best = results[0]
        >>> processor = JsonProcessor(split_path=best.field.split('.'), config)
    """

    def __init__(
        self,
        similarity_strategy: Optional[SimilarityStrategy] = None,
        path_extractor: Optional[PathExtractor] = None,
    ):
        """
        Initialize StructureAnalyzer.

        Args:
            similarity_strategy: Strategy for comparing document structures.
                               Defaults to JaccardPathSimilarity.
            path_extractor: Extractor for getting paths from documents.
                          Defaults to standard PathExtractor.
        """
        self.similarity = similarity_strategy or JaccardPathSimilarity()
        self.extractor = path_extractor or PathExtractor()

    def find_splitters(
        self,
        documents: list[dict],
        # Grouping options
        grouping_field: Optional[str] = None,
        grouping_fields: Optional[list[str]] = None,
        grouping_fn: Optional[Callable[[dict], Any]] = None,
        # Auto-detection settings
        auto_detect: bool = True,
        max_cardinality: int = 50,
        min_coverage: float = 0.5,
        max_depth: int = 3,
    ) -> list[SplitterResult]:
        """
        Find and evaluate splitter fields in a document collection.

        Args:
            documents: List of JSON documents to analyze.
            grouping_field: Specific field to evaluate as splitter.
            grouping_fields: Multiple fields to combine as composite splitter.
            grouping_fn: Custom function to extract grouping key from document.
            auto_detect: If True (default), automatically find candidate splitters.
            max_cardinality: Skip fields with more than this many distinct values.
            min_coverage: Field must appear in at least this fraction of docs.
            max_depth: Only consider fields at this depth or shallower.

        Returns:
            List of SplitterResult, sorted by score (best first).
        """
        if not documents:
            return []

        # Extract paths for all documents
        doc_paths = [self.extractor.extract(doc) for doc in documents]

        # Determine which fields to evaluate
        if grouping_field:
            candidates = [grouping_field]
        elif grouping_fields:
            # Composite key - combine fields
            candidates = [".".join(grouping_fields)]
            grouping_fn = self._make_composite_fn(grouping_fields)
        elif grouping_fn:
            # Custom function provided - evaluate it directly
            return [self._evaluate_grouping(
                documents, doc_paths, grouping_fn, "custom"
            )]
        elif auto_detect:
            candidates = self._find_candidate_fields(
                documents, doc_paths, max_cardinality, min_coverage, max_depth
            )
        else:
            return []

        # Evaluate each candidate
        results = []
        for candidate in candidates:
            fn = self._make_field_fn(candidate)
            result = self._evaluate_grouping(documents, doc_paths, fn, candidate)
            if result.score > 0:
                results.append(result)

        # Sort by score (best first)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _find_candidate_fields(
        self,
        documents: list[dict],
        doc_paths: list[set[str]],
        max_cardinality: int,
        min_coverage: float,
        max_depth: int,
    ) -> list[str]:
        """Find fields that could be good splitters."""
        # Get all leaf paths (non-array) that are shallow enough
        all_paths: set[str] = set()
        for paths in doc_paths:
            for path in paths:
                # Skip array paths - we want scalar values
                if "[]" not in path:
                    depth = path.count(".") + 1
                    if depth <= max_depth:
                        all_paths.add(path)

        # Filter by coverage and cardinality
        candidates = []
        for path in all_paths:
            values = []
            present_count = 0

            for doc in documents:
                value = self.extractor.get_value_at_path(doc, path)
                if value is not None:
                    present_count += 1
                    values.append(value)

            coverage = present_count / len(documents)
            if coverage < min_coverage:
                continue

            distinct = len(set(str(v) for v in values))
            if distinct > max_cardinality:
                continue

            # Must have at least 2 distinct values to be a splitter
            if distinct >= 2:
                candidates.append(path)

        return candidates

    def _make_field_fn(self, field_path: str) -> Callable[[dict], Any]:
        """Create a grouping function for a field path."""
        def fn(doc: dict) -> Any:
            return self.extractor.get_value_at_path(doc, field_path)
        return fn

    def _make_composite_fn(self, field_paths: list[str]) -> Callable[[dict], Any]:
        """Create a grouping function for multiple fields."""
        def fn(doc: dict) -> Any:
            values = tuple(
                self.extractor.get_value_at_path(doc, p)
                for p in field_paths
            )
            return values
        return fn

    def _evaluate_grouping(
        self,
        documents: list[dict],
        doc_paths: list[set[str]],
        grouping_fn: Callable[[dict], Any],
        field_name: str,
    ) -> SplitterResult:
        """Evaluate a specific grouping as a potential splitter."""
        # Group documents by the grouping function
        groups: dict[Any, list[int]] = defaultdict(list)
        present_count = 0

        for i, doc in enumerate(documents):
            value = grouping_fn(doc)
            if value is not None:
                groups[str(value)].append(i)
                present_count += 1

        coverage = present_count / len(documents) if documents else 0
        distinct_values = len(groups)
        value_counts = {k: len(v) for k, v in groups.items()}

        # Calculate within-group similarity
        within_sims = []
        for indices in groups.values():
            if len(indices) >= 2:
                group_paths = [doc_paths[i] for i in indices]
                for p1, p2 in combinations(group_paths, 2):
                    within_sims.append(self.similarity.similarity(p1, p2))

        within_avg = sum(within_sims) / len(within_sims) if within_sims else 1.0

        # Calculate between-group similarity
        between_sims = []
        group_keys = list(groups.keys())
        if len(group_keys) >= 2:
            for g1, g2 in combinations(group_keys, 2):
                # Sample one from each group for comparison
                idx1 = groups[g1][0]
                idx2 = groups[g2][0]
                between_sims.append(
                    self.similarity.similarity(doc_paths[idx1], doc_paths[idx2])
                )

        between_avg = sum(between_sims) / len(between_sims) if between_sims else 0.001

        # Score = within / between (higher = better splitter)
        score = within_avg / between_avg if between_avg > 0 else within_avg

        return SplitterResult(
            field=field_name,
            score=score,
            distinct_values=distinct_values,
            coverage=coverage,
            value_counts=value_counts,
            within_similarity=within_avg,
            between_similarity=between_avg,
        )

    def describe(
        self,
        documents: list[dict],
        splitter: Optional[str] = None,
        top_n: int = 5,
    ) -> str:
        """
        Generate a human-readable structure summary.

        Args:
            documents: List of JSON documents.
            splitter: Specific splitter to use. If None, auto-detects.
            top_n: Number of top splitters to show.

        Returns:
            Formatted string describing the structure.
        """
        if not documents:
            return "No documents to analyze."

        doc_paths = [self.extractor.extract(doc) for doc in documents]
        all_paths = set().union(*doc_paths)

        lines = [
            f"Documents analyzed: {len(documents):,}",
            f"Unique paths: {len(all_paths)}",
            "",
        ]

        # Find splitters
        results = self.find_splitters(documents)
        if results:
            lines.append("Candidate splitters:")
            for i, r in enumerate(results[:top_n]):
                marker = " ← RECOMMENDED" if i == 0 else ""
                lines.append(
                    f"  {r.field} ({r.distinct_values} values, "
                    f"score: {r.score:.2f}){marker}"
                )
            lines.append("")

            # Show structure by best splitter
            best = results[0] if not splitter else next(
                (r for r in results if r.field == splitter), results[0]
            )
            lines.append(f"Structure by {best.field}:")

            grouping_fn = self._make_field_fn(best.field)
            groups: dict[str, list[set[str]]] = defaultdict(list)

            for doc, paths in zip(documents, doc_paths):
                value = grouping_fn(doc)
                if value is not None:
                    groups[str(value)].append(paths)

            for value, path_sets in sorted(groups.items(), key=lambda x: -len(x[1])):
                common = set.intersection(*path_sets) if path_sets else set()
                # Get paths unique to this group (not in other groups)
                other_paths = set()
                for other_value, other_sets in groups.items():
                    if other_value != value:
                        other_paths.update(*other_sets)
                unique_to_group = set.union(*path_sets) - other_paths if path_sets else set()

                # Show compact summary
                sample = sorted(unique_to_group)[:3]
                sample_str = ", ".join(sample)
                if len(unique_to_group) > 3:
                    sample_str += f" (+{len(unique_to_group) - 3} more)"

                lines.append(f'  "{value}" ({len(path_sets)} docs): {sample_str or "(same as others)"}')

        else:
            lines.append("No candidate splitters found.")

        return "\n".join(lines)

    def get_structure_by_value(
        self,
        documents: list[dict],
        splitter: str,
    ) -> dict[str, StructureSummary]:
        """
        Get detailed structure information grouped by splitter value.

        Args:
            documents: List of JSON documents.
            splitter: Field path to use as splitter.

        Returns:
            Dict mapping splitter values to StructureSummary.
        """
        doc_paths = [self.extractor.extract(doc) for doc in documents]
        grouping_fn = self._make_field_fn(splitter)

        groups: dict[str, list[set[str]]] = defaultdict(list)
        for doc, paths in zip(documents, doc_paths):
            value = grouping_fn(doc)
            if value is not None:
                groups[str(value)].append(paths)

        summaries = {}
        for value, path_sets in groups.items():
            all_paths = set.union(*path_sets) if path_sets else set()
            common_paths = set.intersection(*path_sets) if path_sets else set()

            summaries[value] = StructureSummary(
                value=value,
                count=len(path_sets),
                unique_paths=all_paths,
                common_paths=common_paths,
                sample_paths=sorted(all_paths)[:10],
            )

        return summaries
