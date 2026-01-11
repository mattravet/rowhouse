"""
EXTREME nesting tests - pushing the JsonProcessor to its limits.

These tests explore:
- 6+ levels of pure object nesting
- 5+ levels of nested arrays
- Cartesian explosion scenarios (arrays at multiple parallel branches)
- Pathological edge cases
"""
import pytest
import pandas as pd


class TestExtremeObjectNesting:
    """Test object nesting at extreme depths (6+ levels)"""

    @pytest.fixture
    def six_level_config(self):
        return {
            "DeepDive": {
                "table_name": "extreme_depth",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "a.b.c.d.e.f.value", "alias": "depth_6", "type": "string"},
                    {"source": "a.b.c.d.e.f.g.value", "alias": "depth_7", "type": "string"},
                    {"source": "a.b.c.d.e.f.g.h.value", "alias": "depth_8", "type": "string"},
                    {"source": "a.b.c.d.e.f.g.h.i.value", "alias": "depth_9", "type": "string"},
                    {"source": "a.b.c.d.e.f.g.h.i.j.value", "alias": "depth_10", "type": "string"},
                ]
            }
        }

    def test_ten_level_deep_object(self, six_level_config, processor_factory):
        """Navigate 10 levels deep into nested objects"""
        processor = processor_factory(six_level_config)

        messages = [{
            "header": {"action": "DeepDive"},
            "a": {"b": {"c": {"d": {"e": {"f": {
                "value": "level_6",
                "g": {
                    "value": "level_7",
                    "h": {
                        "value": "level_8",
                        "i": {
                            "value": "level_9",
                            "j": {
                                "value": "level_10_reached!"
                            }
                        }
                    }
                }
            }}}}}}
        }]

        result = processor.process_messages(messages)

        assert "DeepDive" in result
        df = result["DeepDive"]
        assert len(df) == 1
        assert df.iloc[0]["depth_6"] == "level_6"
        assert df.iloc[0]["depth_7"] == "level_7"
        assert df.iloc[0]["depth_8"] == "level_8"
        assert df.iloc[0]["depth_9"] == "level_9"
        assert df.iloc[0]["depth_10"] == "level_10_reached!"


class TestExtremeArrayNesting:
    """Test deeply nested arrays (5+ levels)"""

    @pytest.fixture
    def five_level_array_config(self):
        """5 levels of nested arrays - this will create massive row explosion"""
        return {
            "ArrayInception": {
                "table_name": "array_inception",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "l1[].id", "alias": "l1_id", "type": "string"},
                    {"source": "l1[].l2[].id", "alias": "l2_id", "type": "string"},
                    {"source": "l1[].l2[].l3[].id", "alias": "l3_id", "type": "string"},
                    {"source": "l1[].l2[].l3[].l4[].id", "alias": "l4_id", "type": "string"},
                    {"source": "l1[].l2[].l3[].l4[].l5[].id", "alias": "l5_id", "type": "string"},
                    {"source": "l1[].l2[].l3[].l4[].l5[].payload", "alias": "payload", "type": "string"},
                ]
            }
        }

    def test_five_level_nested_arrays(self, five_level_array_config, processor_factory):
        """
        5 levels of arrays, each with 2 elements = 2^5 = 32 rows
        This tests the cartesian product explosion.
        """
        processor = processor_factory(five_level_array_config)

        messages = [{
            "header": {"action": "ArrayInception"},
            "l1": [
                {
                    "id": "L1-A",
                    "l2": [
                        {
                            "id": "L2-A1",
                            "l3": [
                                {
                                    "id": "L3-A1a",
                                    "l4": [
                                        {
                                            "id": "L4-A1a1",
                                            "l5": [
                                                {"id": "L5-deep-1", "payload": "payload-1"},
                                                {"id": "L5-deep-2", "payload": "payload-2"}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                },
                {
                    "id": "L1-B",
                    "l2": [
                        {
                            "id": "L2-B1",
                            "l3": [
                                {
                                    "id": "L3-B1a",
                                    "l4": [
                                        {
                                            "id": "L4-B1a1",
                                            "l5": [
                                                {"id": "L5-deep-3", "payload": "payload-3"}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }]

        result = processor.process_messages(messages)

        assert "ArrayInception" in result
        df = result["ArrayInception"]

        # L1-A branch: 2 items at L5 = 2 rows
        # L1-B branch: 1 item at L5 = 1 row
        # Total = 3 rows
        assert len(df) == 3

        # Verify the hierarchy is preserved
        l1a_rows = df[df["l1_id"] == "L1-A"]
        l1b_rows = df[df["l1_id"] == "L1-B"]

        assert len(l1a_rows) == 2
        assert len(l1b_rows) == 1
        assert set(l1a_rows["l5_id"]) == {"L5-deep-1", "L5-deep-2"}
        assert l1b_rows.iloc[0]["l5_id"] == "L5-deep-3"

    def test_asymmetric_deep_nesting(self, five_level_array_config, processor_factory):
        """Test where different branches have different depths"""
        processor = processor_factory(five_level_array_config)

        messages = [{
            "header": {"action": "ArrayInception"},
            "l1": [
                {
                    "id": "FULL-DEPTH",
                    "l2": [{
                        "id": "L2",
                        "l3": [{
                            "id": "L3",
                            "l4": [{
                                "id": "L4",
                                "l5": [
                                    {"id": "LEAF-1", "payload": "complete-path"},
                                    {"id": "LEAF-2", "payload": "complete-path-2"},
                                    {"id": "LEAF-3", "payload": "complete-path-3"}
                                ]
                            }]
                        }]
                    }]
                }
            ]
        }]

        result = processor.process_messages(messages)

        assert "ArrayInception" in result
        df = result["ArrayInception"]
        assert len(df) == 3
        assert all(df["l1_id"] == "FULL-DEPTH")
        assert set(df["payload"]) == {"complete-path", "complete-path-2", "complete-path-3"}


class TestCartesianExplosion:
    """Test parallel arrays causing cartesian product explosion"""

    @pytest.fixture
    def parallel_arrays_config(self):
        """Config with arrays at the same level (siblings)"""
        return {
            "ParallelArrays": {
                "table_name": "parallel_explosion",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "header.id", "alias": "header_id", "type": "string"},
                    {"source": "products[].sku", "alias": "product_sku", "type": "string"},
                    {"source": "products[].name", "alias": "product_name", "type": "string"},
                    {"source": "customers[].customerId", "alias": "customer_id", "type": "string"},
                    {"source": "customers[].region", "alias": "customer_region", "type": "string"},
                ]
            }
        }

    def test_sibling_arrays_cartesian_product(self, parallel_arrays_config, processor_factory):
        """
        Two parallel arrays at root level.
        products (3 items) x customers (2 items) = 6 row cartesian product
        """
        processor = processor_factory(parallel_arrays_config)

        messages = [{
            "header": {"action": "ParallelArrays", "id": "CART-001"},
            "products": [
                {"sku": "PROD-A", "name": "Widget"},
                {"sku": "PROD-B", "name": "Gadget"},
                {"sku": "PROD-C", "name": "Gizmo"}
            ],
            "customers": [
                {"customerId": "CUST-1", "region": "NA"},
                {"customerId": "CUST-2", "region": "EU"}
            ]
        }]

        result = processor.process_messages(messages)

        assert "ParallelArrays" in result
        df = result["ParallelArrays"]

        # Should get cartesian product: 3 products x 2 customers = 6 rows
        assert len(df) == 6

        # Every product should appear with every customer
        for sku in ["PROD-A", "PROD-B", "PROD-C"]:
            sku_rows = df[df["product_sku"] == sku]
            assert len(sku_rows) == 2
            assert set(sku_rows["customer_id"]) == {"CUST-1", "CUST-2"}


class TestMixedExtremeNesting:
    """Combine deep objects with deep arrays"""

    @pytest.fixture
    def mixed_extreme_config(self):
        """Deep objects containing deep arrays"""
        return {
            "MixedExtreme": {
                "table_name": "mixed_extreme",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    # Deep object path
                    {"source": "meta.system.config.settings.name", "alias": "config_name", "type": "string"},
                    {"source": "meta.system.config.settings.version", "alias": "config_version", "type": "string"},
                    # Deep array path within objects
                    {"source": "data.regions[].regionId", "alias": "region_id", "type": "string"},
                    {"source": "data.regions[].zones[].zoneId", "alias": "zone_id", "type": "string"},
                    {"source": "data.regions[].zones[].servers[].serverId", "alias": "server_id", "type": "string"},
                    {"source": "data.regions[].zones[].servers[].metrics.cpu", "alias": "cpu", "type": "float"},
                    {"source": "data.regions[].zones[].servers[].metrics.memory", "alias": "memory", "type": "float"},
                ]
            }
        }

    def test_deep_objects_with_deep_arrays(self, mixed_extreme_config, processor_factory):
        """
        Structure:
        meta.system.config.settings (4 levels deep object)
        + data.regions[].zones[].servers[] (3 levels of arrays)
        + servers[].metrics.cpu (2 more object levels)
        """
        processor = processor_factory(mixed_extreme_config)

        messages = [{
            "header": {"action": "MixedExtreme"},
            "meta": {
                "system": {
                    "config": {
                        "settings": {
                            "name": "ProductionCluster",
                            "version": "3.14.159"
                        }
                    }
                }
            },
            "data": {
                "regions": [
                    {
                        "regionId": "us-east-1",
                        "zones": [
                            {
                                "zoneId": "us-east-1a",
                                "servers": [
                                    {
                                        "serverId": "srv-001",
                                        "metrics": {"cpu": 45.5, "memory": 72.3}
                                    },
                                    {
                                        "serverId": "srv-002",
                                        "metrics": {"cpu": 88.1, "memory": 91.0}
                                    }
                                ]
                            },
                            {
                                "zoneId": "us-east-1b",
                                "servers": [
                                    {
                                        "serverId": "srv-003",
                                        "metrics": {"cpu": 23.4, "memory": 55.6}
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "regionId": "eu-west-1",
                        "zones": [
                            {
                                "zoneId": "eu-west-1a",
                                "servers": [
                                    {
                                        "serverId": "srv-eu-001",
                                        "metrics": {"cpu": 67.8, "memory": 45.2}
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }]

        result = processor.process_messages(messages)

        assert "MixedExtreme" in result
        df = result["MixedExtreme"]

        # us-east-1a: 2 servers + us-east-1b: 1 server + eu-west-1a: 1 server = 4 rows
        assert len(df) == 4

        # Deep object values should be same for all rows
        assert all(df["config_name"] == "ProductionCluster")
        assert all(df["config_version"] == "3.14.159")

        # Check specific server
        srv001 = df[df["server_id"] == "srv-001"].iloc[0]
        assert srv001["region_id"] == "us-east-1"
        assert srv001["zone_id"] == "us-east-1a"
        assert srv001["cpu"] == 45.5
        assert srv001["memory"] == 72.3

        # Check EU server
        srv_eu = df[df["server_id"] == "srv-eu-001"].iloc[0]
        assert srv_eu["region_id"] == "eu-west-1"
        assert srv_eu["zone_id"] == "eu-west-1a"


class TestPathologicalCases:
    """Edge cases that might break naive implementations"""

    @pytest.fixture
    def pathological_config(self):
        return {
            "Pathological": {
                "table_name": "pathological",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "data.value", "alias": "value", "type": "string"},
                    {"source": "data.items[].x", "alias": "x", "type": "string"},
                ]
            }
        }

    def test_single_element_arrays(self, pathological_config, processor_factory):
        """Arrays with exactly one element"""
        processor = processor_factory(pathological_config)

        messages = [{
            "header": {"action": "Pathological"},
            "data": {
                "value": "singleton",
                "items": [{"x": "only-one"}]
            }
        }]

        result = processor.process_messages(messages)

        assert "Pathological" in result
        df = result["Pathological"]
        assert len(df) == 1
        assert df.iloc[0]["value"] == "singleton"
        assert df.iloc[0]["x"] == "only-one"

    def test_deeply_nested_empty_arrays(self, processor_factory):
        """Empty arrays at various nesting depths still produce rows with NA values"""
        config = {
            "EmptyDeep": {
                "table_name": "empty_deep",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "a[].b[].c[].value", "alias": "deep_value", "type": "string"},
                ]
            }
        }
        processor = processor_factory(config)

        # Empty at different levels
        messages = [
            {
                "header": {"action": "EmptyDeep"},
                "a": []  # Empty at level 1
            },
            {
                "header": {"action": "EmptyDeep"},
                "a": [{"b": []}]  # Empty at level 2
            },
            {
                "header": {"action": "EmptyDeep"},
                "a": [{"b": [{"c": []}]}]  # Empty at level 3
            }
        ]

        result = processor.process_messages(messages)

        # Messages with empty arrays still produce rows (from header data)
        # but with NA for the array-derived fields
        assert "EmptyDeep" in result
        df = result["EmptyDeep"]
        assert len(df) == 3  # One row per message
        assert all(df["action"] == "EmptyDeep")
        # All deep_value should be NA since arrays were empty
        import pandas as pd
        assert all(pd.isna(df["deep_value"]))

    def test_same_field_name_at_different_levels(self, processor_factory):
        """
        Field 'id' appears at multiple nesting levels.

        When sibling branches have both arrays and non-array nested objects,
        the processor does cartesian product: the nested dict values are merged
        with each array row. This gives complete data on every row.
        """
        config = {
            "SameName": {
                "table_name": "same_name",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "header.id", "alias": "header_id", "type": "string"},
                    {"source": "body.id", "alias": "body_id", "type": "string"},
                    {"source": "body.nested.id", "alias": "nested_id", "type": "string"},
                    {"source": "body.items[].id", "alias": "item_id", "type": "string"},
                ]
            }
        }
        processor = processor_factory(config)

        messages = [{
            "header": {"action": "SameName", "id": "HEADER-ID"},
            "body": {
                "id": "BODY-ID",
                "nested": {
                    "id": "NESTED-ID"
                },
                "items": [
                    {"id": "ITEM-1"},
                    {"id": "ITEM-2"}
                ]
            }
        }]

        result = processor.process_messages(messages)

        assert "SameName" in result
        df = result["SameName"]

        # Cartesian product: nested (1 row) × items (2 rows) = 2 rows
        # Each row has BOTH nested_id AND item_id
        assert len(df) == 2

        # All rows should have same header/body/nested IDs
        assert all(df["header_id"] == "HEADER-ID")
        assert all(df["body_id"] == "BODY-ID")
        assert all(df["nested_id"] == "NESTED-ID")

        # Each row has a different item_id
        assert set(df["item_id"]) == {"ITEM-1", "ITEM-2"}

    def test_very_long_field_names(self, processor_factory):
        """Extremely long field names in paths"""
        config = {
            "LongNames": {
                "table_name": "long_names",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {
                        "source": "thisIsAnExtremelyLongFieldNameThatGoesOnAndOn.andAnotherVeryLongFieldName.yetAnotherLongOne.value",
                        "alias": "long_path_value",
                        "type": "string"
                    },
                ]
            }
        }
        processor = processor_factory(config)

        messages = [{
            "header": {"action": "LongNames"},
            "thisIsAnExtremelyLongFieldNameThatGoesOnAndOn": {
                "andAnotherVeryLongFieldName": {
                    "yetAnotherLongOne": {
                        "value": "found-it!"
                    }
                }
            }
        }]

        result = processor.process_messages(messages)

        assert "LongNames" in result
        df = result["LongNames"]
        assert len(df) == 1
        assert df.iloc[0]["long_path_value"] == "found-it!"


class TestStressTest:
    """Performance and stress tests"""

    def test_wide_arrays(self, processor_factory):
        """Single array with many elements"""
        config = {
            "Wide": {
                "table_name": "wide",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "items[].id", "alias": "item_id", "type": "string"},
                    {"source": "items[].value", "alias": "value", "type": "integer"},
                ]
            }
        }
        processor = processor_factory(config)

        # 1000 items in single array
        messages = [{
            "header": {"action": "Wide"},
            "items": [{"id": f"item-{i}", "value": i} for i in range(1000)]
        }]

        result = processor.process_messages(messages)

        assert "Wide" in result
        df = result["Wide"]
        assert len(df) == 1000

    def test_deep_and_wide(self, processor_factory):
        """Combination of deep nesting and wide arrays"""
        config = {
            "DeepWide": {
                "table_name": "deep_wide",
                "fields": [
                    {"source": "header.action", "alias": "action", "type": "string"},
                    {"source": "level1[].level2[].level3[].id", "alias": "deep_id", "type": "string"},
                ]
            }
        }
        processor = processor_factory(config)

        # 5 x 5 x 5 = 125 leaf nodes
        messages = [{
            "header": {"action": "DeepWide"},
            "level1": [
                {
                    "level2": [
                        {
                            "level3": [{"id": f"leaf-{i}-{j}-{k}"} for k in range(5)]
                        } for j in range(5)
                    ]
                } for i in range(5)
            ]
        }]

        result = processor.process_messages(messages)

        assert "DeepWide" in result
        df = result["DeepWide"]
        assert len(df) == 125
