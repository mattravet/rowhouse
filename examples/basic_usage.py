#!/usr/bin/env python3
"""
Basic usage example for Unfurl (part of Rowhouse).

This example shows how to flatten nested JSON into tabular data.
The split_path parameter tells JsonProcessor which field routes
documents to different configurations.

TIP: If you don't know the right split_path for new data, use
     rowhouse.discover to analyze the JSON structure first.
     See examples/discover_example.py for details.

Run from the rowhouse directory:
    python examples/basic_usage.py
"""
import sys
import os

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unfurl import JsonProcessor


def main():
    # Define configuration for message types you want to process
    config = {
        "OrderCreated": {
            "table_name": "orders",
            "fields": [
                {"source": "header.action", "alias": "action", "type": "string"},
                {"source": "header.orderId", "alias": "order_id", "type": "string"},
                {"source": "body.customer.name", "alias": "customer_name", "type": "string"},
                {"source": "body.customer.email", "alias": "customer_email", "type": "string"},
                {"source": "body.items[].sku", "alias": "sku", "type": "string"},
                {"source": "body.items[].name", "alias": "item_name", "type": "string"},
                {"source": "body.items[].quantity", "alias": "quantity", "type": "integer"},
                {"source": "body.items[].price", "alias": "price", "type": "float"},
            ]
        }
    }

    # Sample nested JSON data
    messages = [
        {
            "header": {
                "action": "OrderCreated",
                "orderId": "ORD-001",
                "timestamp": "2024-01-15T10:30:00Z"
            },
            "body": {
                "customer": {
                    "name": "Alice Smith",
                    "email": "alice@example.com"
                },
                "items": [
                    {"sku": "WIDGET-A", "name": "Blue Widget", "quantity": 2, "price": 29.99},
                    {"sku": "GADGET-B", "name": "Red Gadget", "quantity": 1, "price": 49.99},
                    {"sku": "GIZMO-C", "name": "Green Gizmo", "quantity": 3, "price": 19.99}
                ]
            }
        },
        {
            "header": {
                "action": "OrderCreated",
                "orderId": "ORD-002",
                "timestamp": "2024-01-15T11:00:00Z"
            },
            "body": {
                "customer": {
                    "name": "Bob Jones",
                    "email": "bob@example.com"
                },
                "items": [
                    {"sku": "WIDGET-A", "name": "Blue Widget", "quantity": 5, "price": 29.99}
                ]
            }
        }
    ]

    # Create processor
    # split_path tells the processor how to route messages to different configs
    processor = JsonProcessor(
        split_path=['header', 'action'],
        config=config
    )

    # Set metadata (required before processing)
    processor.set_file_metadata("example.json", "2024-01-15T12:00:00Z")

    # Process messages
    result = processor.process_messages(messages)

    # Print results
    print("=" * 60)
    print("UNFURL - JSON Flattening Example")
    print("=" * 60)

    for table_name, df in result.items():
        print(f"\nTable: {table_name}")
        print("-" * 40)
        # Drop metadata columns for cleaner display
        display_cols = [c for c in df.columns if not c.startswith('s3_')]
        print(df[display_cols].to_string(index=False))
        print(f"\nTotal rows: {len(df)}")

    # Example: Save to Parquet
    # for table_name, df in result.items():
    #     df.to_parquet(f"{table_name}.parquet")


if __name__ == "__main__":
    main()
