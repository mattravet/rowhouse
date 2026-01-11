"""Tests for the discover module - structure analysis and splitter discovery."""
import pytest

from common.paths import PathExtractor, extract_paths
from discover import (
    StructureAnalyzer,
    SplitterResult,
    JaccardPathSimilarity,
    WeightedJaccardSimilarity,
    ExactMatchSimilarity,
)


class TestPathExtractor:
    """Tests for path extraction from JSON documents."""

    @pytest.fixture
    def extractor(self):
        return PathExtractor()

    def test_simple_object(self, extractor):
        doc = {"name": "test", "value": 42}
        paths = extractor.extract(doc)
        assert paths == {"name", "value"}

    def test_nested_object(self, extractor):
        doc = {"header": {"action": "test", "id": "123"}}
        paths = extractor.extract(doc)
        assert paths == {"header.action", "header.id"}

    def test_deep_nesting(self, extractor):
        doc = {"a": {"b": {"c": {"d": "value"}}}}
        paths = extractor.extract(doc)
        assert paths == {"a.b.c.d"}

    def test_array_of_primitives(self, extractor):
        doc = {"tags": ["a", "b", "c"]}
        paths = extractor.extract(doc)
        assert paths == {"tags[]"}

    def test_array_of_objects(self, extractor):
        doc = {"items": [{"sku": "A", "price": 10}, {"sku": "B", "price": 20}]}
        paths = extractor.extract(doc)
        assert paths == {"items[].sku", "items[].price"}

    def test_nested_arrays(self, extractor):
        doc = {
            "orders": [
                {"id": "1", "items": [{"sku": "A"}, {"sku": "B"}]},
                {"id": "2", "items": [{"sku": "C"}]}
            ]
        }
        paths = extractor.extract(doc)
        assert paths == {"orders[].id", "orders[].items[].sku"}

    def test_mixed_structure(self, extractor):
        doc = {
            "header": {"action": "OrderCreated"},
            "body": {
                "customer": {"name": "Alice"},
                "items": [{"sku": "A", "qty": 1}]
            }
        }
        paths = extractor.extract(doc)
        expected = {
            "header.action",
            "body.customer.name",
            "body.items[].sku",
            "body.items[].qty"
        }
        assert paths == expected

    def test_empty_array(self, extractor):
        doc = {"items": []}
        paths = extractor.extract(doc)
        assert paths == {"items[]"}

    def test_empty_object(self, extractor):
        doc = {"data": {}}
        paths = extractor.extract(doc)
        assert paths == set()

    def test_convenience_function(self):
        doc = {"a": {"b": "value"}}
        paths = extract_paths(doc)
        assert paths == {"a.b"}

    def test_get_value_at_path(self, extractor):
        doc = {"header": {"action": "test"}, "body": {"value": 42}}
        assert extractor.get_value_at_path(doc, "header.action") == "test"
        assert extractor.get_value_at_path(doc, "body.value") == 42
        assert extractor.get_value_at_path(doc, "missing") is None
        assert extractor.get_value_at_path(doc, "header.missing") is None


class TestJaccardSimilarity:
    """Tests for Jaccard similarity calculation."""

    @pytest.fixture
    def sim(self):
        return JaccardPathSimilarity()

    def test_identical_sets(self, sim):
        paths = {"a", "b", "c"}
        assert sim.similarity(paths, paths) == 1.0

    def test_completely_different(self, sim):
        paths_a = {"a", "b"}
        paths_b = {"c", "d"}
        assert sim.similarity(paths_a, paths_b) == 0.0

    def test_partial_overlap(self, sim):
        paths_a = {"a", "b", "c"}
        paths_b = {"b", "c", "d"}
        # Intersection: {b, c} = 2
        # Union: {a, b, c, d} = 4
        # Jaccard = 2/4 = 0.5
        assert sim.similarity(paths_a, paths_b) == 0.5

    def test_subset(self, sim):
        paths_a = {"a", "b"}
        paths_b = {"a", "b", "c", "d"}
        # Intersection: {a, b} = 2
        # Union: {a, b, c, d} = 4
        # Jaccard = 2/4 = 0.5
        assert sim.similarity(paths_a, paths_b) == 0.5

    def test_empty_sets(self, sim):
        assert sim.similarity(set(), set()) == 1.0

    def test_one_empty(self, sim):
        assert sim.similarity({"a"}, set()) == 0.0


class TestWeightedJaccardSimilarity:
    """Tests for weighted Jaccard similarity."""

    def test_deeper_paths_matter_less(self):
        sim = WeightedJaccardSimilarity(depth_decay=0.5)

        # Shallow difference
        paths_a = {"a", "b"}
        paths_b = {"a", "c"}

        # Deep difference
        paths_c = {"a", "x.y.z"}
        paths_d = {"a", "x.y.w"}

        shallow_sim = sim.similarity(paths_a, paths_b)
        deep_sim = sim.similarity(paths_c, paths_d)

        # Deep differences should result in higher similarity
        # because the differing paths have lower weight
        assert deep_sim > shallow_sim


class TestExactMatchSimilarity:
    """Tests for exact match similarity."""

    @pytest.fixture
    def sim(self):
        return ExactMatchSimilarity()

    def test_identical(self, sim):
        paths = {"a", "b", "c"}
        assert sim.similarity(paths, paths) == 1.0

    def test_different(self, sim):
        paths_a = {"a", "b"}
        paths_b = {"a", "b", "c"}
        assert sim.similarity(paths_a, paths_b) == 0.0


class TestStructureAnalyzer:
    """Tests for the main StructureAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return StructureAnalyzer()

    @pytest.fixture
    def sample_documents(self):
        """Documents with clear structural differences based on header.action."""
        return [
            # OrderCreated documents - have items array
            {"header": {"action": "OrderCreated"}, "body": {"items": [{"sku": "A"}]}},
            {"header": {"action": "OrderCreated"}, "body": {"items": [{"sku": "B"}]}},
            {"header": {"action": "OrderCreated"}, "body": {"items": [{"sku": "C"}]}},
            # UserCreated documents - have user object
            {"header": {"action": "UserCreated"}, "body": {"user": {"name": "Alice"}}},
            {"header": {"action": "UserCreated"}, "body": {"user": {"name": "Bob"}}},
            # EventLogged documents - have event object
            {"header": {"action": "EventLogged"}, "body": {"event": {"type": "click"}}},
            {"header": {"action": "EventLogged"}, "body": {"event": {"type": "view"}}},
        ]

    def test_find_splitters_auto_detect(self, analyzer, sample_documents):
        """Auto-detect should find header.action as best splitter."""
        results = analyzer.find_splitters(sample_documents)

        assert len(results) > 0
        best = results[0]
        assert best.field == "header.action"
        assert best.distinct_values == 3
        assert best.score > 1.0  # Good splitters have score > 1

    def test_find_splitters_explicit_field(self, analyzer, sample_documents):
        """Explicit field evaluation should work."""
        results = analyzer.find_splitters(
            sample_documents,
            grouping_field="header.action"
        )

        assert len(results) == 1
        assert results[0].field == "header.action"

    def test_find_splitters_custom_function(self, analyzer, sample_documents):
        """Custom grouping function should work."""
        results = analyzer.find_splitters(
            sample_documents,
            grouping_fn=lambda d: d.get("header", {}).get("action")
        )

        assert len(results) == 1
        assert results[0].field == "custom"
        assert results[0].distinct_values == 3

    def test_splitter_result_contents(self, analyzer, sample_documents):
        """SplitterResult should contain expected data."""
        results = analyzer.find_splitters(sample_documents)
        best = results[0]

        assert isinstance(best.score, float)
        assert best.coverage == 1.0  # All docs have header.action
        assert best.value_counts == {
            "OrderCreated": 3,
            "UserCreated": 2,
            "EventLogged": 2
        }

    def test_describe_output(self, analyzer, sample_documents):
        """describe() should return formatted string."""
        output = analyzer.describe(sample_documents)

        assert "Documents analyzed: 7" in output
        assert "header.action" in output
        assert "RECOMMENDED" in output

    def test_empty_documents(self, analyzer):
        """Should handle empty document list."""
        results = analyzer.find_splitters([])
        assert results == []

        output = analyzer.describe([])
        assert "No documents" in output

    def test_uniform_structure(self, analyzer):
        """Documents with identical structure - no good splitter."""
        docs = [
            {"type": "A", "value": 1},
            {"type": "B", "value": 2},
            {"type": "C", "value": 3},
        ]
        results = analyzer.find_splitters(docs)

        # May find type as candidate, but score should be ~1 (no differentiation)
        if results:
            # Score near 1 means within-group ≈ between-group similarity
            assert results[0].score < 2.0

    def test_get_structure_by_value(self, analyzer, sample_documents):
        """get_structure_by_value should return detailed summaries."""
        summaries = analyzer.get_structure_by_value(
            sample_documents,
            splitter="header.action"
        )

        assert "OrderCreated" in summaries
        assert "UserCreated" in summaries

        order_summary = summaries["OrderCreated"]
        assert order_summary.count == 3
        assert "body.items[].sku" in order_summary.unique_paths

        user_summary = summaries["UserCreated"]
        assert user_summary.count == 2
        assert "body.user.name" in user_summary.unique_paths


class TestSplitterDetectionAccuracy:
    """Tests to verify splitter detection works on realistic data."""

    @pytest.fixture
    def analyzer(self):
        return StructureAnalyzer()

    def test_ocpp_like_messages(self, analyzer):
        """Test with OCPP-like charge point messages."""
        docs = [
            # MeterValues - have meterValue array
            {
                "header": {"action": "MeterValues"},
                "body": {"meterValue": [{"timestamp": "2024-01-01", "sampledValue": [{"value": "100"}]}]}
            },
            {
                "header": {"action": "MeterValues"},
                "body": {"meterValue": [{"timestamp": "2024-01-02", "sampledValue": [{"value": "200"}]}]}
            },
            # StatusNotification - have status/errorCode
            {
                "header": {"action": "StatusNotification"},
                "body": {"status": "Available", "errorCode": "NoError"}
            },
            {
                "header": {"action": "StatusNotification"},
                "body": {"status": "Charging", "errorCode": "NoError"}
            },
            # Heartbeat - minimal body
            {
                "header": {"action": "Heartbeat"},
                "body": {}
            },
            {
                "header": {"action": "Heartbeat"},
                "body": {}
            },
        ]

        results = analyzer.find_splitters(docs)

        assert len(results) > 0
        assert results[0].field == "header.action"
        assert results[0].score > 1.5  # Should clearly differentiate

    def test_high_cardinality_excluded(self, analyzer):
        """Fields with too many distinct values should be excluded."""
        docs = [
            {"id": f"unique-{i}", "type": "A" if i < 5 else "B", "value": i}
            for i in range(10)
        ]

        results = analyzer.find_splitters(docs, max_cardinality=5)

        # 'id' has 10 distinct values, should be excluded
        # 'type' has 2 values, should be included
        field_names = [r.field for r in results]
        assert "id" not in field_names
        assert "type" in field_names
