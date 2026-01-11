"""Pytest configuration and fixtures for unfurl tests"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unfurl import JsonProcessor


@pytest.fixture
def simple_config():
    """Config for simple nested object traversal"""
    return {
        "TestAction": {
            "table_name": "test_simple",
            "fields": [
                {"source": "header.id", "alias": "header_id", "type": "string"},
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "body.name", "alias": "name", "type": "string"},
                {"source": "body.value", "alias": "value", "type": "integer"},
            ]
        }
    }


@pytest.fixture
def nested_object_config():
    """Config for deeply nested object traversal (no arrays)"""
    return {
        "DeepNested": {
            "table_name": "test_deep_nested",
            "fields": [
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "level1.level2.level3.level4.deepValue", "alias": "deep_value", "type": "string"},
                {"source": "level1.level2.level3.midValue", "alias": "mid_value", "type": "string"},
                {"source": "level1.level2.shallowValue", "alias": "shallow_value", "type": "string"},
                {"source": "level1.topValue", "alias": "top_value", "type": "string"},
            ]
        }
    }


@pytest.fixture
def single_array_config():
    """Config for single-level array explosion"""
    return {
        "SingleArray": {
            "table_name": "test_single_array",
            "fields": [
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "header.id", "alias": "header_id", "type": "string"},
                {"source": "items[].name", "alias": "item_name", "type": "string"},
                {"source": "items[].quantity", "alias": "quantity", "type": "integer"},
            ]
        }
    }


@pytest.fixture
def nested_array_config():
    """Config for nested arrays (array within array)"""
    return {
        "NestedArray": {
            "table_name": "test_nested_array",
            "fields": [
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "header.id", "alias": "header_id", "type": "string"},
                {"source": "orders[].orderId", "alias": "order_id", "type": "string"},
                {"source": "orders[].items[].sku", "alias": "sku", "type": "string"},
                {"source": "orders[].items[].price", "alias": "price", "type": "float"},
            ]
        }
    }


@pytest.fixture
def triple_nested_array_config():
    """Config for triple-nested arrays (3 levels deep)"""
    return {
        "TripleNested": {
            "table_name": "test_triple_nested",
            "fields": [
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "warehouses[].warehouseId", "alias": "warehouse_id", "type": "string"},
                {"source": "warehouses[].aisles[].aisleId", "alias": "aisle_id", "type": "string"},
                {"source": "warehouses[].aisles[].shelves[].shelfId", "alias": "shelf_id", "type": "string"},
                {"source": "warehouses[].aisles[].shelves[].items[].itemId", "alias": "item_id", "type": "string"},
                {"source": "warehouses[].aisles[].shelves[].items[].quantity", "alias": "quantity", "type": "integer"},
            ]
        }
    }


@pytest.fixture
def mixed_nesting_config():
    """Config mixing nested objects and arrays"""
    return {
        "MixedNesting": {
            "table_name": "test_mixed",
            "fields": [
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "metadata.source.system.name", "alias": "system_name", "type": "string"},
                {"source": "metadata.source.system.version", "alias": "system_version", "type": "string"},
                {"source": "data.records[].id", "alias": "record_id", "type": "string"},
                {"source": "data.records[].attributes.category", "alias": "category", "type": "string"},
                {"source": "data.records[].attributes.tags[].name", "alias": "tag_name", "type": "string"},
                {"source": "data.records[].attributes.tags[].weight", "alias": "tag_weight", "type": "float"},
            ]
        }
    }


@pytest.fixture
def processor_factory():
    """Factory to create JsonProcessor with custom config"""
    def _create_processor(config, split_path=None, coerce=False):
        if split_path is None:
            split_path = ['header', 'action']
        processor = JsonProcessor(split_path=split_path, config=config, coerce=coerce)
        processor.set_file_metadata("test/2024/01/15/00/test.json.gz", "2024-01-15T00:00:00Z")
        return processor
    return _create_processor
