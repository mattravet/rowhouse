"""
Tests for JsonProcessor with increasingly complex nested JSON structures.

Test hierarchy:
1. Simple nested objects (2 levels)
2. Deep nested objects (4+ levels, no arrays)
3. Single-level arrays
4. Nested arrays (2 levels)
5. Triple-nested arrays (3+ levels)
6. Mixed nesting (objects + arrays combined)
7. Edge cases (empty arrays, missing fields, null values)
"""
import pytest
import pandas as pd
import numpy as np


class TestSimpleNestedObjects:
    """Test basic nested object traversal"""

    def test_simple_two_level_nesting(self, simple_config, processor_factory):
        """header.id and body.name should be extracted correctly"""
        processor = processor_factory(simple_config)

        messages = [{
            "header": {"id": "123", "action": "TestAction"},
            "body": {"name": "test_item", "value": 42}
        }]

        result = processor.process_messages(messages)

        assert "TestAction" in result
        df = result["TestAction"]
        assert len(df) == 1
        assert df.iloc[0]["header_id"] == "123"
        assert df.iloc[0]["name"] == "test_item"
        assert df.iloc[0]["value"] == 42


class TestDeepNestedObjects:
    """Test deeply nested object traversal (4+ levels, no arrays)"""

    def test_four_level_deep_nesting(self, nested_object_config, processor_factory):
        """Should extract values from 4 levels deep"""
        processor = processor_factory(nested_object_config)

        messages = [{
            "header": {"action": "DeepNested"},
            "level1": {
                "topValue": "top",
                "level2": {
                    "shallowValue": "shallow",
                    "level3": {
                        "midValue": "mid",
                        "level4": {
                            "deepValue": "deep_secret"
                        }
                    }
                }
            }
        }]

        result = processor.process_messages(messages)

        assert "DeepNested" in result
        df = result["DeepNested"]
        assert len(df) == 1
        assert df.iloc[0]["deep_value"] == "deep_secret"
        assert df.iloc[0]["mid_value"] == "mid"
        assert df.iloc[0]["shallow_value"] == "shallow"
        assert df.iloc[0]["top_value"] == "top"

    def test_partial_deep_nesting(self, nested_object_config, processor_factory):
        """Should handle missing intermediate levels gracefully"""
        processor = processor_factory(nested_object_config)

        messages = [{
            "header": {"action": "DeepNested"},
            "level1": {
                "topValue": "top",
                "level2": {
                    "shallowValue": "shallow"
                    # level3 missing entirely
                }
            }
        }]

        result = processor.process_messages(messages)

        assert "DeepNested" in result
        df = result["DeepNested"]
        assert len(df) == 1
        assert df.iloc[0]["top_value"] == "top"
        assert df.iloc[0]["shallow_value"] == "shallow"
        # Missing nested values should be NaN
        assert pd.isna(df.iloc[0]["mid_value"])
        assert pd.isna(df.iloc[0]["deep_value"])


class TestSingleArrayExplosion:
    """Test single-level array explosion"""

    def test_array_creates_multiple_rows(self, single_array_config, processor_factory):
        """Each array element should create a separate row"""
        processor = processor_factory(single_array_config)

        messages = [{
            "header": {"action": "SingleArray", "id": "order-001"},
            "items": [
                {"name": "Widget A", "quantity": 5},
                {"name": "Widget B", "quantity": 10},
                {"name": "Widget C", "quantity": 3}
            ]
        }]

        result = processor.process_messages(messages)

        assert "SingleArray" in result
        df = result["SingleArray"]
        assert len(df) == 3

        # All rows should have same header values
        assert all(df["header_id"] == "order-001")
        assert all(df["action"] == "SingleArray")

        # Each row should have different item values
        assert set(df["item_name"]) == {"Widget A", "Widget B", "Widget C"}
        assert set(df["quantity"]) == {5, 10, 3}

    def test_empty_array(self, single_array_config, processor_factory):
        """Empty array produces a row with header values but NA for array fields"""
        processor = processor_factory(single_array_config)

        messages = [{
            "header": {"action": "SingleArray", "id": "order-empty"},
            "items": []
        }]

        result = processor.process_messages(messages)

        # Empty array still produces a row from header, with NA for array fields
        assert "SingleArray" in result
        df = result["SingleArray"]
        assert len(df) == 1
        assert df.iloc[0]["header_id"] == "order-empty"
        # Array fields will be NA since the array was empty
        import pandas as pd
        assert pd.isna(df.iloc[0]["item_name"])


class TestNestedArrays:
    """Test nested arrays (array within array)"""

    def test_two_level_nested_arrays(self, nested_array_config, processor_factory):
        """orders[].items[] should create cartesian product of rows"""
        processor = processor_factory(nested_array_config)

        messages = [{
            "header": {"action": "NestedArray", "id": "batch-001"},
            "orders": [
                {
                    "orderId": "ORD-1",
                    "items": [
                        {"sku": "SKU-A", "price": 10.99},
                        {"sku": "SKU-B", "price": 20.50}
                    ]
                },
                {
                    "orderId": "ORD-2",
                    "items": [
                        {"sku": "SKU-C", "price": 5.00}
                    ]
                }
            ]
        }]

        result = processor.process_messages(messages)

        assert "NestedArray" in result
        df = result["NestedArray"]

        # Should have 3 rows: 2 from ORD-1 + 1 from ORD-2
        assert len(df) == 3

        # Check ORD-1 rows
        ord1_rows = df[df["order_id"] == "ORD-1"]
        assert len(ord1_rows) == 2
        assert set(ord1_rows["sku"]) == {"SKU-A", "SKU-B"}

        # Check ORD-2 rows
        ord2_rows = df[df["order_id"] == "ORD-2"]
        assert len(ord2_rows) == 1
        assert ord2_rows.iloc[0]["sku"] == "SKU-C"

    def test_nested_array_with_empty_inner_array(self, nested_array_config, processor_factory):
        """Order with empty items array should not create rows"""
        processor = processor_factory(nested_array_config)

        messages = [{
            "header": {"action": "NestedArray", "id": "batch-002"},
            "orders": [
                {
                    "orderId": "ORD-EMPTY",
                    "items": []  # Empty inner array
                },
                {
                    "orderId": "ORD-FULL",
                    "items": [{"sku": "SKU-X", "price": 15.00}]
                }
            ]
        }]

        result = processor.process_messages(messages)

        assert "NestedArray" in result
        df = result["NestedArray"]

        # Only ORD-FULL should create a row
        assert len(df) == 1
        assert df.iloc[0]["order_id"] == "ORD-FULL"


class TestTripleNestedArrays:
    """Test triple-nested arrays (3+ levels deep)"""

    def test_three_level_nested_arrays(self, triple_nested_array_config, processor_factory):
        """warehouses[].aisles[].shelves[].items[] - 4 levels of arrays"""
        processor = processor_factory(triple_nested_array_config)

        messages = [{
            "header": {"action": "TripleNested"},
            "warehouses": [
                {
                    "warehouseId": "WH-1",
                    "aisles": [
                        {
                            "aisleId": "A1",
                            "shelves": [
                                {
                                    "shelfId": "S1",
                                    "items": [
                                        {"itemId": "ITEM-001", "quantity": 10},
                                        {"itemId": "ITEM-002", "quantity": 5}
                                    ]
                                },
                                {
                                    "shelfId": "S2",
                                    "items": [
                                        {"itemId": "ITEM-003", "quantity": 20}
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "warehouseId": "WH-2",
                    "aisles": [
                        {
                            "aisleId": "B1",
                            "shelves": [
                                {
                                    "shelfId": "S3",
                                    "items": [
                                        {"itemId": "ITEM-004", "quantity": 15}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }]

        result = processor.process_messages(messages)

        assert "TripleNested" in result
        df = result["TripleNested"]

        # Total: WH-1/A1/S1 has 2 items, WH-1/A1/S2 has 1 item, WH-2/B1/S3 has 1 item = 4 rows
        assert len(df) == 4

        # Verify hierarchy is preserved
        wh1_rows = df[df["warehouse_id"] == "WH-1"]
        assert len(wh1_rows) == 3
        assert all(wh1_rows["aisle_id"] == "A1")

        wh2_rows = df[df["warehouse_id"] == "WH-2"]
        assert len(wh2_rows) == 1
        assert wh2_rows.iloc[0]["aisle_id"] == "B1"
        assert wh2_rows.iloc[0]["shelf_id"] == "S3"
        assert wh2_rows.iloc[0]["item_id"] == "ITEM-004"

    def test_deeply_nested_with_varying_depths(self, triple_nested_array_config, processor_factory):
        """Test warehouses with different nesting depths"""
        processor = processor_factory(triple_nested_array_config)

        messages = [{
            "header": {"action": "TripleNested"},
            "warehouses": [
                {
                    "warehouseId": "WH-DEEP",
                    "aisles": [
                        {
                            "aisleId": "AISLE-1",
                            "shelves": [
                                {
                                    "shelfId": "SHELF-1",
                                    "items": [
                                        {"itemId": "DEEP-ITEM-1", "quantity": 100},
                                        {"itemId": "DEEP-ITEM-2", "quantity": 200},
                                        {"itemId": "DEEP-ITEM-3", "quantity": 300}
                                    ]
                                }
                            ]
                        },
                        {
                            "aisleId": "AISLE-2",
                            "shelves": [
                                {
                                    "shelfId": "SHELF-2",
                                    "items": [{"itemId": "DEEP-ITEM-4", "quantity": 400}]
                                },
                                {
                                    "shelfId": "SHELF-3",
                                    "items": [{"itemId": "DEEP-ITEM-5", "quantity": 500}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }]

        result = processor.process_messages(messages)

        assert "TripleNested" in result
        df = result["TripleNested"]

        # AISLE-1/SHELF-1: 3 items + AISLE-2/SHELF-2: 1 item + AISLE-2/SHELF-3: 1 item = 5 rows
        assert len(df) == 5

        # All should be from same warehouse
        assert all(df["warehouse_id"] == "WH-DEEP")

        # Verify aisle distribution
        aisle1_rows = df[df["aisle_id"] == "AISLE-1"]
        aisle2_rows = df[df["aisle_id"] == "AISLE-2"]
        assert len(aisle1_rows) == 3
        assert len(aisle2_rows) == 2


class TestMixedNesting:
    """Test mixed nesting patterns (objects and arrays combined)"""

    def test_nested_objects_with_nested_arrays(self, mixed_nesting_config, processor_factory):
        """metadata.source.system.name + data.records[].attributes.tags[]"""
        processor = processor_factory(mixed_nesting_config)

        messages = [{
            "header": {"action": "MixedNesting"},
            "metadata": {
                "source": {
                    "system": {
                        "name": "InventorySystem",
                        "version": "2.5.1"
                    }
                }
            },
            "data": {
                "records": [
                    {
                        "id": "REC-001",
                        "attributes": {
                            "category": "electronics",
                            "tags": [
                                {"name": "popular", "weight": 0.9},
                                {"name": "new", "weight": 0.7}
                            ]
                        }
                    },
                    {
                        "id": "REC-002",
                        "attributes": {
                            "category": "clothing",
                            "tags": [
                                {"name": "sale", "weight": 0.8}
                            ]
                        }
                    }
                ]
            }
        }]

        result = processor.process_messages(messages)

        assert "MixedNesting" in result
        df = result["MixedNesting"]

        # REC-001 has 2 tags, REC-002 has 1 tag = 3 rows
        assert len(df) == 3

        # Deeply nested object values should be same across all rows
        assert all(df["system_name"] == "InventorySystem")
        assert all(df["system_version"] == "2.5.1")

        # Check record distribution
        rec1_rows = df[df["record_id"] == "REC-001"]
        rec2_rows = df[df["record_id"] == "REC-002"]

        assert len(rec1_rows) == 2
        assert all(rec1_rows["category"] == "electronics")
        assert set(rec1_rows["tag_name"]) == {"popular", "new"}

        assert len(rec2_rows) == 1
        assert rec2_rows.iloc[0]["category"] == "clothing"
        assert rec2_rows.iloc[0]["tag_name"] == "sale"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_null_values_in_nested_path(self, simple_config, processor_factory):
        """Null values in the middle of a path should not crash"""
        processor = processor_factory(simple_config)

        messages = [{
            "header": {"id": "123", "action": "TestAction"},
            "body": None  # Entire body is null
        }]

        result = processor.process_messages(messages)

        assert "TestAction" in result
        df = result["TestAction"]
        assert len(df) == 1
        assert df.iloc[0]["header_id"] == "123"
        assert pd.isna(df.iloc[0]["name"])

    def test_missing_optional_fields(self, simple_config, processor_factory):
        """Missing fields should result in NaN, not errors"""
        processor = processor_factory(simple_config)

        messages = [{
            "header": {"id": "123", "action": "TestAction"},
            "body": {"name": "test"}  # value field missing
        }]

        result = processor.process_messages(messages)

        assert "TestAction" in result
        df = result["TestAction"]
        assert len(df) == 1
        assert df.iloc[0]["name"] == "test"
        assert pd.isna(df.iloc[0]["value"])

    def test_multiple_messages_same_action(self, simple_config, processor_factory):
        """Multiple messages with same action should be combined"""
        processor = processor_factory(simple_config)

        messages = [
            {
                "header": {"id": "1", "action": "TestAction"},
                "body": {"name": "first", "value": 1}
            },
            {
                "header": {"id": "2", "action": "TestAction"},
                "body": {"name": "second", "value": 2}
            },
            {
                "header": {"id": "3", "action": "TestAction"},
                "body": {"name": "third", "value": 3}
            }
        ]

        result = processor.process_messages(messages)

        assert "TestAction" in result
        df = result["TestAction"]
        assert len(df) == 3
        assert set(df["header_id"]) == {"1", "2", "3"}

    def test_unicode_and_special_characters(self, simple_config, processor_factory):
        """Unicode and special characters should be handled correctly"""
        processor = processor_factory(simple_config)

        messages = [{
            "header": {"id": "日本語-123", "action": "TestAction"},
            "body": {"name": "Ümläüt Tëst 🎉", "value": 42}
        }]

        result = processor.process_messages(messages)

        assert "TestAction" in result
        df = result["TestAction"]
        assert df.iloc[0]["header_id"] == "日本語-123"
        assert df.iloc[0]["name"] == "Ümläüt Tëst 🎉"

    def test_large_nested_arrays(self, nested_array_config, processor_factory):
        """Performance test with larger nested arrays"""
        processor = processor_factory(nested_array_config)

        # Create message with 10 orders, each with 10 items = 100 rows
        messages = [{
            "header": {"action": "NestedArray", "id": "large-batch"},
            "orders": [
                {
                    "orderId": f"ORD-{i}",
                    "items": [
                        {"sku": f"SKU-{i}-{j}", "price": float(i * 10 + j)}
                        for j in range(10)
                    ]
                }
                for i in range(10)
            ]
        }]

        result = processor.process_messages(messages)

        assert "NestedArray" in result
        df = result["NestedArray"]
        assert len(df) == 100

        # Verify some random samples
        ord5_rows = df[df["order_id"] == "ORD-5"]
        assert len(ord5_rows) == 10


class TestMessageSplitting:
    """Test the message splitting by action"""

    def test_different_actions_split_correctly(self, processor_factory):
        """Messages with different actions go to different tables"""
        config = {
            "ActionA": {
                "table_name": "table_a",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.valueA", "alias": "value_a", "type": "string"}
                ]
            },
            "ActionB": {
                "table_name": "table_b",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.valueB", "alias": "value_b", "type": "string"}
                ]
            }
        }

        processor = processor_factory(config)

        messages = [
            {"header": {"action": "ActionA"}, "data": {"valueA": "from_a"}},
            {"header": {"action": "ActionB"}, "data": {"valueB": "from_b"}},
            {"header": {"action": "ActionA"}, "data": {"valueA": "from_a_2"}},
        ]

        result = processor.process_messages(messages)

        assert "ActionA" in result
        assert "ActionB" in result
        assert len(result["ActionA"]) == 2
        assert len(result["ActionB"]) == 1


class TestCoercion:
    """Test smart type coercion for messy real-world data"""

    def test_currency_coercion_global(self, processor_factory):
        """Currency symbols should be stripped when coerce=True globally"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.price", "alias": "price", "type": "float"},
                    {"source": "data.cost", "alias": "cost", "type": "float"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {
                "price": "$1,234.56",
                "cost": "€50.00"
            }
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert df.iloc[0]["price"] == 1234.56
        assert df.iloc[0]["cost"] == 50.0

    def test_currency_coercion_per_field(self, processor_factory):
        """Per-field coerce=True should only affect that field"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.price", "alias": "price", "type": "float", "coerce": True},
                    {"source": "data.quantity", "alias": "quantity", "type": "integer"},
                ]
            }
        }

        # Global coerce is False, but price has coerce=True
        processor = processor_factory(config, coerce=False)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {
                "price": "$99.99",
                "quantity": 5
            }
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert df.iloc[0]["price"] == 99.99
        assert df.iloc[0]["quantity"] == 5

    def test_percentage_coercion(self, processor_factory):
        """Percentages should be converted to decimals"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.rate", "alias": "rate", "type": "float"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {"rate": "25%"}
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert df.iloc[0]["rate"] == 0.25

    def test_multiple_date_formats(self, processor_factory):
        """Various date formats should all be parsed correctly"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.date_iso", "alias": "date_iso", "type": "timestamp"},
                    {"source": "data.date_us", "alias": "date_us", "type": "timestamp"},
                    {"source": "data.date_time", "alias": "date_time", "type": "timestamp"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {
                "date_iso": "2024-01-15",
                "date_us": "01/15/2024",
                "date_time": "2024-01-15 14:30:00"
            }
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        # All should parse to January 15, 2024
        assert df.iloc[0]["date_iso"].year == 2024
        assert df.iloc[0]["date_iso"].month == 1
        assert df.iloc[0]["date_iso"].day == 15

        assert df.iloc[0]["date_us"].year == 2024
        assert df.iloc[0]["date_us"].month == 1
        assert df.iloc[0]["date_us"].day == 15

        assert df.iloc[0]["date_time"].hour == 14
        assert df.iloc[0]["date_time"].minute == 30

    def test_boolean_coercion(self, processor_factory):
        """Various boolean representations should be parsed"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.flag1", "alias": "flag1", "type": "boolean"},
                    {"source": "data.flag2", "alias": "flag2", "type": "boolean"},
                    {"source": "data.flag3", "alias": "flag3", "type": "boolean"},
                    {"source": "data.flag4", "alias": "flag4", "type": "boolean"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {
                "flag1": "yes",
                "flag2": "no",
                "flag3": "1",
                "flag4": "false"
            }
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert df.iloc[0]["flag1"] == True
        assert df.iloc[0]["flag2"] == False
        assert df.iloc[0]["flag3"] == True
        assert df.iloc[0]["flag4"] == False

    def test_integer_with_thousands_separator(self, processor_factory):
        """Integers with thousands separators should be parsed"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.count", "alias": "count", "type": "integer"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {"count": "1,000,000"}
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert df.iloc[0]["count"] == 1000000

    def test_invalid_values_become_null(self, processor_factory):
        """Invalid values should become null, not crash"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.price", "alias": "price", "type": "float"},
                    {"source": "data.date", "alias": "date", "type": "timestamp"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {
                "price": "not a number",
                "date": "invalid date"
            }
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert pd.isna(df.iloc[0]["price"])
        assert pd.isna(df.iloc[0]["date"])

    def test_coercion_with_nested_arrays(self, processor_factory):
        """Coercion should work with nested array explosion"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "items[].name", "alias": "name", "type": "string"},
                    {"source": "items[].price", "alias": "price", "type": "float"},
                    {"source": "items[].in_stock", "alias": "in_stock", "type": "boolean"},
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "items": [
                {"name": "Widget", "price": "$29.99", "in_stock": "yes"},
                {"name": "Gadget", "price": "€49.99", "in_stock": "no"},
                {"name": "Gizmo", "price": "£19.99", "in_stock": "1"}
            ]
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert len(df) == 3
        assert list(df["price"]) == [29.99, 49.99, 19.99]
        assert list(df["in_stock"]) == [True, False, True]

    def test_mixed_coerce_flags(self, processor_factory):
        """Mix of global and per-field coerce settings"""
        config = {
            "TestAction": {
                "table_name": "test_coerce",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.price", "alias": "price", "type": "float"},  # Uses global
                    {"source": "data.quantity", "alias": "quantity", "type": "integer", "coerce": False},  # Override
                ]
            }
        }

        processor = processor_factory(config, coerce=True)

        messages = [{
            "header": {"action": "TestAction"},
            "data": {
                "price": "$100",
                "quantity": 5  # Clean integer, no coercion needed
            }
        }]

        result = processor.process_messages(messages)
        df = result["TestAction"]

        assert df.iloc[0]["price"] == 100.0
        assert df.iloc[0]["quantity"] == 5
