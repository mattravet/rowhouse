"""
Tests that demonstrate the processed_paths bug in JsonProcessor.

Bug: When multiple fields share a common path prefix (e.g., body.level1.x and body.level1.y.z),
the first field's traversal marks the shared prefix as "processed", blocking subsequent fields.
"""
import pytest
import pandas as pd


class TestProcessedPathsBug:
    """Demonstrate the sibling field blocking bug"""

    @pytest.fixture
    def sibling_fields_config(self):
        """Config where fields share a common path prefix"""
        return {
            "SiblingTest": {
                "table_name": "sibling_test",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    # These three fields share the prefix "body.level1"
                    {"source": "body.level1.shallow", "alias": "shallow", "type": "string"},
                    {"source": "body.level1.nested.deep", "alias": "deep", "type": "string"},
                ]
            }
        }

    def test_sibling_fields_at_same_level_BUG(self, sibling_fields_config, processor_factory):
        """
        BUG: body.level1.shallow should be extracted, but isn't.

        When body.level1.nested.deep is processed first (or vice versa),
        the shared prefix "body.level1" is marked as processed,
        blocking extraction of body.level1.shallow.
        """
        processor = processor_factory(sibling_fields_config)

        messages = [{
            "header": {"action": "SiblingTest"},
            "body": {
                "level1": {
                    "shallow": "SHOULD_BE_FOUND",
                    "nested": {
                        "deep": "ALSO_FOUND"
                    }
                }
            }
        }]

        result = processor.process_messages(messages)

        assert "SiblingTest" in result
        df = result["SiblingTest"]
        assert len(df) == 1

        # This passes - the deep value is found
        deep_val = df.iloc[0]["deep"]
        assert str(deep_val) == "ALSO_FOUND", f"Expected 'ALSO_FOUND', got {deep_val}"

        # BUG: This should pass but fails - shallow is blocked by processed_paths
        shallow_val = df.iloc[0]["shallow"]
        # The following assertion will FAIL due to the bug:
        assert str(shallow_val) == "SHOULD_BE_FOUND", f"BUG: shallow was {shallow_val} (expected 'SHOULD_BE_FOUND')"

    @pytest.fixture
    def three_sibling_fields_config(self):
        """More complex case with three sibling fields at different depths"""
        return {
            "ThreeSiblings": {
                "table_name": "three_siblings",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    # All three share prefix "body.data"
                    {"source": "body.data.alpha", "alias": "alpha", "type": "string"},
                    {"source": "body.data.beta.value", "alias": "beta_value", "type": "string"},
                    {"source": "body.data.gamma.nested.deep", "alias": "gamma_deep", "type": "string"},
                ]
            }
        }

    def test_three_sibling_fields_all_extracted(self, three_sibling_fields_config, processor_factory):
        """
        Deeply nested sibling fields at different depths are all correctly extracted.

        Tests extraction of:
        - body.data.alpha (2 levels from body)
        - body.data.beta.value (3 levels from body)
        - body.data.gamma.nested.deep (4 levels from body)

        All sibling non-array branches are merged into a single row.
        """
        processor = processor_factory(three_sibling_fields_config)

        messages = [{
            "header": {"action": "ThreeSiblings"},
            "body": {
                "data": {
                    "alpha": "A",
                    "beta": {"value": "B"},
                    "gamma": {"nested": {"deep": "C"}}
                }
            }
        }]

        result = processor.process_messages(messages)

        assert "ThreeSiblings" in result
        df = result["ThreeSiblings"]

        # Should be 1 row with all fields merged
        assert len(df) == 1

        alpha = df.iloc[0]["alpha"]
        beta_value = df.iloc[0]["beta_value"]
        gamma_deep = df.iloc[0]["gamma_deep"]

        # All three fields should be extracted
        assert str(alpha) == "A", f"alpha should be 'A', got {alpha}"
        assert str(beta_value) == "B", f"beta_value should be 'B', got {beta_value}"
        assert str(gamma_deep) == "C", f"gamma_deep should be 'C', got {gamma_deep}"


class TestFieldOrderDependence:
    """Show that the bug's behavior depends on field order in config"""

    def test_field_order_affects_extraction(self, processor_factory):
        """
        When deep field is listed FIRST, shallow field is blocked.
        """
        config_deep_first = {
            "DeepFirst": {
                "table_name": "deep_first",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "body.path.to.deep.value", "alias": "deep_value", "type": "string"},  # Listed first
                    {"source": "body.path.to.shallow", "alias": "shallow_value", "type": "string"},  # Listed second
                ]
            }
        }

        processor = processor_factory(config_deep_first)

        messages = [{
            "header": {"action": "DeepFirst"},
            "body": {
                "path": {
                    "to": {
                        "shallow": "SHALLOW",
                        "deep": {"value": "DEEP"}
                    }
                }
            }
        }]

        result = processor.process_messages(messages)
        df = result["DeepFirst"]

        deep_val = str(df.iloc[0]["deep_value"])
        shallow_val = df.iloc[0]["shallow_value"]

        print(f"Deep first config: deep={deep_val}, shallow={shallow_val}")

        # Deep should be found since it's processed first
        assert deep_val == "DEEP"

        # BUG: Shallow is blocked because body.path.to is already in processed_paths
        # This SHOULD pass but won't:
        assert str(shallow_val) == "SHALLOW", f"BUG: shallow_value was {shallow_val}"

    def test_field_order_reversed(self, processor_factory):
        """
        When shallow field is listed FIRST, it gets extracted.
        """
        config_shallow_first = {
            "ShallowFirst": {
                "table_name": "shallow_first",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "body.path.to.shallow", "alias": "shallow_value", "type": "string"},  # Listed first
                    {"source": "body.path.to.deep.value", "alias": "deep_value", "type": "string"},  # Listed second
                ]
            }
        }

        processor = processor_factory(config_shallow_first)

        messages = [{
            "header": {"action": "ShallowFirst"},
            "body": {
                "path": {
                    "to": {
                        "shallow": "SHALLOW",
                        "deep": {"value": "DEEP"}
                    }
                }
            }
        }]

        result = processor.process_messages(messages)
        df = result["ShallowFirst"]

        shallow_val = str(df.iloc[0]["shallow_value"])
        deep_val = df.iloc[0]["deep_value"]

        print(f"Shallow first config: shallow={shallow_val}, deep={deep_val}")

        # Shallow should be found since it's a leaf at body.path.to
        # (processed before we try to recurse into 'deep')
        assert shallow_val == "SHALLOW"

        # But now deep MIGHT be blocked... depends on implementation details
        # Let's see what happens:
        print(f"Deep value after shallow: {deep_val}")


class TestWorkingCases:
    """Document cases that DO work (no shared path prefix issue)"""

    def test_separate_branches_work(self, processor_factory):
        """Fields under completely separate top-level keys work fine"""
        config = {
            "SeparateBranches": {
                "table_name": "separate",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "header.id", "alias": "id", "type": "string"},
                    {"source": "body.name", "alias": "name", "type": "string"},
                    {"source": "body.value", "alias": "value", "type": "integer"},
                ]
            }
        }

        processor = processor_factory(config)

        messages = [{
            "header": {"action": "SeparateBranches", "id": "123"},
            "body": {"name": "test", "value": 42}
        }]

        result = processor.process_messages(messages)
        df = result["SeparateBranches"]

        # All fields are direct children, no shared nested prefix
        assert df.iloc[0]["id"] == "123"
        assert df.iloc[0]["name"] == "test"
        assert df.iloc[0]["value"] == 42

    def test_arrays_with_sibling_fields_work(self, processor_factory):
        """Arrays with multiple fields at the same level work"""
        config = {
            "ArraySiblings": {
                "table_name": "array_siblings",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "body.items[].name", "alias": "name", "type": "string"},
                    {"source": "body.items[].price", "alias": "price", "type": "float"},
                    {"source": "body.items[].quantity", "alias": "quantity", "type": "integer"},
                ]
            }
        }

        processor = processor_factory(config)

        messages = [{
            "header": {"action": "ArraySiblings"},
            "body": {
                "items": [
                    {"name": "A", "price": 10.0, "quantity": 1},
                    {"name": "B", "price": 20.0, "quantity": 2}
                ]
            }
        }]

        result = processor.process_messages(messages)
        df = result["ArraySiblings"]

        assert len(df) == 2
        # All three sibling fields within each array element are captured
        assert set(df["name"]) == {"A", "B"}
        assert set(df["price"]) == {10.0, 20.0}
        assert set(df["quantity"]) == {1, 2}
