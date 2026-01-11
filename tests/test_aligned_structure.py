"""
Tests aligned with the original processor's expected structure.

The processor was designed for messages with:
- header.* for metadata fields
- body.* for data fields
- Arrays within body (e.g., body.items[], body.meterValue[])

These tests verify the processor works with its intended structure.
"""
import pytest
import pandas as pd


class TestAlignedStructure:
    """Tests using header/body structure that matches original config"""

    @pytest.fixture
    def body_array_config(self):
        """Config matching original pattern: body.items[]"""
        return {
            "TestAction": {
                "table_name": "test_body_array",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "header.id", "alias": "header_id", "type": "string"},
                    {"source": "body.items[].name", "alias": "item_name", "type": "string"},
                    {"source": "body.items[].quantity", "alias": "quantity", "type": "integer"},
                ]
            }
        }

    @pytest.fixture
    def nested_body_array_config(self):
        """Config matching MeterValues pattern: body.meterValue[].sampledValue[]"""
        return {
            "NestedBodyArray": {
                "table_name": "test_nested_body",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "header.id", "alias": "header_id", "type": "string"},
                    {"source": "body.orders[].orderId", "alias": "order_id", "type": "string"},
                    {"source": "body.orders[].items[].sku", "alias": "sku", "type": "string"},
                    {"source": "body.orders[].items[].price", "alias": "price", "type": "float"},
                ]
            }
        }

    @pytest.fixture
    def deep_body_object_config(self):
        """Deep nesting within body"""
        return {
            "DeepBody": {
                "table_name": "test_deep_body",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "body.level1.level2.level3.value", "alias": "deep_value", "type": "string"},
                    {"source": "body.level1.level2.mid", "alias": "mid_value", "type": "string"},
                    {"source": "body.level1.top", "alias": "top_value", "type": "string"},
                ]
            }
        }

    def test_single_array_in_body(self, body_array_config, processor_factory):
        """body.items[] should create multiple rows"""
        processor = processor_factory(body_array_config)

        messages = [{
            "header": {"action": "TestAction", "id": "order-001"},
            "body": {
                "items": [
                    {"name": "Widget A", "quantity": 5},
                    {"name": "Widget B", "quantity": 10},
                    {"name": "Widget C", "quantity": 3}
                ]
            }
        }]

        result = processor.process_messages(messages)

        assert "TestAction" in result, f"Result keys: {result.keys()}"
        df = result["TestAction"]
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

        # All rows should have same header values
        assert all(df["header_id"] == "order-001")
        assert set(df["item_name"]) == {"Widget A", "Widget B", "Widget C"}

    def test_nested_arrays_in_body(self, nested_body_array_config, processor_factory):
        """body.orders[].items[] - nested arrays within body"""
        processor = processor_factory(nested_body_array_config)

        messages = [{
            "header": {"action": "NestedBodyArray", "id": "batch-001"},
            "body": {
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
            }
        }]

        result = processor.process_messages(messages)

        assert "NestedBodyArray" in result, f"Result keys: {result.keys()}"
        df = result["NestedBodyArray"]
        # ORD-1: 2 items + ORD-2: 1 item = 3 rows
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}. Data:\n{df}"

    def test_deep_objects_in_body(self, deep_body_object_config, processor_factory):
        """body.level1.level2.level3.value - deep object nesting"""
        processor = processor_factory(deep_body_object_config)

        messages = [{
            "header": {"action": "DeepBody"},
            "body": {
                "level1": {
                    "top": "top_val",
                    "level2": {
                        "mid": "mid_val",
                        "level3": {
                            "value": "deep_val"
                        }
                    }
                }
            }
        }]

        result = processor.process_messages(messages)

        assert "DeepBody" in result, f"Result keys: {result.keys()}"
        df = result["DeepBody"]
        assert len(df) == 1

        # Check each value (handling NA carefully)
        deep_val = df.iloc[0]["deep_value"]
        mid_val = df.iloc[0]["mid_value"]
        top_val = df.iloc[0]["top_value"]

        assert deep_val == "deep_val" or str(deep_val) == "deep_val", f"deep_value was {deep_val}"
        assert mid_val == "mid_val" or str(mid_val) == "mid_val", f"mid_value was {mid_val}"
        assert top_val == "top_val" or str(top_val) == "top_val", f"top_value was {top_val}"


class TestOriginalConfigPattern:
    """Test using pattern from actual output_config.json"""

    @pytest.fixture
    def meter_values_config(self):
        """Replica of MeterValues config from output_config.json"""
        return {
            "MeterValues": {
                "table_name": "meter_values",
                "fields": [
                    {"source": "header.locationID", "alias": "location_id", "type": "string"},
                    {"source": "header.chargerID", "alias": "charger_id", "type": "string"},
                    {"source": "header.uniqueID", "alias": "unique_id", "type": "string"},
                    {"source": "body.connectorId", "alias": "connector_id", "type": "integer"},
                    {"source": "body.transactionId", "alias": "transaction_id", "type": "integer"},
                    {"source": "body.meterValue[].timestamp", "alias": "timestamp_utc", "type": "timestamp"},
                    {"source": "body.meterValue[].sampledValue[].unit", "alias": "unit", "type": "string"},
                    {"source": "body.meterValue[].sampledValue[].value", "alias": "value", "type": "string"},
                    {"source": "body.meterValue[].sampledValue[].measurand", "alias": "measurand", "type": "string"},
                ]
            }
        }

    def test_meter_values_structure(self, meter_values_config, processor_factory):
        """Test exact structure expected by MeterValues config"""
        processor = processor_factory(meter_values_config)

        messages = [{
            "header": {
                "action": "MeterValues",
                "locationID": "LOC-001",
                "chargerID": "CHG-001",
                "uniqueID": "UID-001"
            },
            "body": {
                "connectorId": 1,
                "transactionId": 12345,
                "meterValue": [
                    {
                        "timestamp": "2024-01-15T10:00:00Z",
                        "sampledValue": [
                            {"unit": "Wh", "value": "1000", "measurand": "Energy.Active.Import.Register"},
                            {"unit": "A", "value": "16", "measurand": "Current.Import"}
                        ]
                    },
                    {
                        "timestamp": "2024-01-15T10:05:00Z",
                        "sampledValue": [
                            {"unit": "Wh", "value": "1100", "measurand": "Energy.Active.Import.Register"}
                        ]
                    }
                ]
            }
        }]

        result = processor.process_messages(messages)

        assert "MeterValues" in result, f"Result keys: {result.keys()}"
        df = result["MeterValues"]

        # First meterValue has 2 sampledValues, second has 1 = 3 total rows
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}. Data:\n{df}"

        # All rows should have same header values
        assert all(df["location_id"] == "LOC-001")
        assert all(df["charger_id"] == "CHG-001")
        assert all(df["connector_id"] == 1)

        # Check measurand distribution
        measurands = df["measurand"].tolist()
        assert "Energy.Active.Import.Register" in measurands
        assert "Current.Import" in measurands

    def test_config_key_pattern(self, meter_values_config, processor_factory):
        """
        Test that the config lookup works with actual config.json pattern.

        The config uses action value (e.g., "MeterValues") as the key,
        and split_path=['header', 'action'] extracts this from messages.
        """
        processor = processor_factory(meter_values_config)

        # Message with action that matches config key
        messages = [{
            "header": {"action": "MeterValues", "locationID": "L1", "chargerID": "C1", "uniqueID": "U1"},
            "body": {
                "connectorId": 1,
                "transactionId": 100,
                "meterValue": [{"timestamp": "2024-01-15T00:00:00Z", "sampledValue": [{"unit": "W", "value": "50", "measurand": "Power"}]}]
            }
        }]

        result = processor.process_messages(messages)

        # Should be keyed by the action value
        assert "MeterValues" in result


class TestDebugTraversal:
    """Diagnostic tests to understand traversal behavior"""

    def test_print_traversal_steps(self, processor_factory, capsys):
        """Debug test to see what's happening during traversal"""
        config = {
            "Debug": {
                "table_name": "debug",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "body.items[].name", "alias": "name", "type": "string"},
                ]
            }
        }
        processor = processor_factory(config)

        messages = [{
            "header": {"action": "Debug"},
            "body": {"items": [{"name": "A"}, {"name": "B"}]}
        }]

        result = processor.process_messages(messages)

        print(f"\n=== Debug Output ===")
        print(f"Result keys: {result.keys()}")
        if "Debug" in result:
            print(f"DataFrame:\n{result['Debug']}")
        else:
            print("No 'Debug' key in result!")
            print(f"Full result: {result}")
