#!/usr/bin/env python3
"""
Discover Example - Finding the Right Split Path

This example demonstrates how to use rowhouse.discover to analyze
unfamiliar JSON data and automatically find the best "splitter" field
for use with JsonProcessor.

The Problem:
    When you receive new JSON data, you need to know which field determines
    the structure of each document. This field (the "splitter") routes
    documents to different processing configurations.

The Solution:
    StructureAnalyzer examines your documents and scores candidate fields
    based on how well they predict document structure using Jaccard similarity.

Run from the rowhouse directory:
    python examples/discover_example.py
"""
import sys
import os
import json

# Add parent directory to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discover import StructureAnalyzer
from unfurl import JsonProcessor


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_json_sample(data, label, max_items=2):
    """Print a sample of JSON data."""
    print(f"\n{label}:")
    print("-" * 50)
    for i, doc in enumerate(data[:max_items]):
        print(json.dumps(doc, indent=2))
        if i < max_items - 1:
            print()


def main():
    # =========================================================================
    # STEP 1: THE RAW DATA (Before)
    # =========================================================================
    # Imagine you've received this JSON from a new data source.
    # Different document types have different structures.

    raw_documents = [
        # Order events have items array
        {
            "header": {"action": "OrderCreated", "id": "evt-001"},
            "body": {
                "orderId": "ORD-100",
                "customer": {"name": "Alice", "email": "alice@example.com"},
                "items": [
                    {"sku": "WIDGET-A", "qty": 2, "price": 29.99},
                    {"sku": "GADGET-B", "qty": 1, "price": 49.99}
                ]
            }
        },
        {
            "header": {"action": "OrderCreated", "id": "evt-002"},
            "body": {
                "orderId": "ORD-101",
                "customer": {"name": "Bob", "email": "bob@example.com"},
                "items": [
                    {"sku": "GIZMO-C", "qty": 3, "price": 19.99}
                ]
            }
        },
        # User events have preferences array
        {
            "header": {"action": "UserCreated", "id": "evt-003"},
            "body": {
                "userId": "USR-200",
                "profile": {"name": "Charlie", "role": "admin"},
                "preferences": [
                    {"key": "theme", "value": "dark"},
                    {"key": "notifications", "value": "email"}
                ]
            }
        },
        {
            "header": {"action": "UserCreated", "id": "evt-004"},
            "body": {
                "userId": "USR-201",
                "profile": {"name": "Diana", "role": "user"},
                "preferences": [
                    {"key": "theme", "value": "light"}
                ]
            }
        },
        # Payment events have flat structure
        {
            "header": {"action": "PaymentProcessed", "id": "evt-005"},
            "body": {
                "paymentId": "PAY-300",
                "amount": 129.97,
                "currency": "USD",
                "status": "completed"
            }
        },
        {
            "header": {"action": "PaymentProcessed", "id": "evt-006"},
            "body": {
                "paymentId": "PAY-301",
                "amount": 59.97,
                "currency": "USD",
                "status": "completed"
            }
        }
    ]

    print_section("STEP 1: RAW NESTED JSON (Before)")
    print(f"\nYou have {len(raw_documents)} documents from a new data source.")
    print("Each document type has a DIFFERENT structure:")
    print_json_sample(raw_documents, "Sample documents")
    print("\n... and 4 more documents")

    print("\nNotice:")
    print("  - OrderCreated has: body.items[], body.customer.*")
    print("  - UserCreated has:  body.preferences[], body.profile.*")
    print("  - PaymentProcessed has: body.paymentId, body.amount (flat)")
    print("\nHow do we know which field determines the structure?")

    # =========================================================================
    # STEP 2: DISCOVER THE SPLITTER
    # =========================================================================

    print_section("STEP 2: ANALYZE WITH DISCOVER")

    analyzer = StructureAnalyzer()

    # Find the best splitter field automatically
    results = analyzer.find_splitters(raw_documents)

    print("\nStructureAnalyzer examines all documents and scores candidate fields")
    print("based on how well they predict document structure.\n")

    print("Candidate Splitters Found:")
    print("-" * 50)
    for result in results:
        recommended = " <-- RECOMMENDED" if result == results[0] else ""
        print(f"  {result.field}")
        print(f"    Score: {result.score:.2f} (higher = better)")
        print(f"    Values: {result.distinct_values} distinct")
        print(f"    Coverage: {result.coverage:.0%} of documents{recommended}")
        print()

    # Show the human-readable summary
    print("\nHuman-Readable Summary:")
    print("-" * 50)
    print(analyzer.describe(raw_documents))

    # =========================================================================
    # STEP 3: HOW THE SCORING WORKS
    # =========================================================================

    print_section("STEP 3: HOW JACCARD SCORING WORKS")

    print("""
The score measures how well a field's values predict document structure.

For each candidate field (like 'header.action'):
  1. Group documents by field value (OrderCreated, UserCreated, etc.)
  2. Extract "path sets" from each document:
     - OrderCreated docs have: {header.action, body.orderId, body.items[].sku, ...}
     - UserCreated docs have:  {header.action, body.userId, body.preferences[].key, ...}

  3. Calculate Jaccard similarity within and between groups:
     - Within-group:  How similar are OrderCreated docs to each other? (high = good)
     - Between-group: How similar are OrderCreated docs to UserCreated? (low = good)

  4. Score = within-group similarity / between-group similarity
     - High score means: same values have similar structure, different values differ

Why 'header.action' wins:
  - Documents with same action have nearly identical paths (similarity ~0.95)
  - Documents with different actions share few paths (similarity ~0.25)
  - Score = 0.95 / 0.25 = 3.8 (strong predictor!)
""")

    # =========================================================================
    # STEP 4: USE THE SPLITTER WITH JSONPROCESSOR
    # =========================================================================

    print_section("STEP 4: PROCESS WITH JSONPROCESSOR (After)")

    best = results[0]
    split_path = best.field.split('.')

    print(f"\nUsing discovered split_path: {split_path}")

    # Define configs for each message type
    config = {
        "OrderCreated": {
            "table_name": "orders",
            "fields": [
                {"source": "header.id", "alias": "event_id", "type": "string"},
                {"source": "body.orderId", "alias": "order_id", "type": "string"},
                {"source": "body.customer.name", "alias": "customer", "type": "string"},
                {"source": "body.items[].sku", "alias": "sku", "type": "string"},
                {"source": "body.items[].qty", "alias": "quantity", "type": "integer"},
                {"source": "body.items[].price", "alias": "price", "type": "float"},
            ]
        },
        "UserCreated": {
            "table_name": "users",
            "fields": [
                {"source": "header.id", "alias": "event_id", "type": "string"},
                {"source": "body.userId", "alias": "user_id", "type": "string"},
                {"source": "body.profile.name", "alias": "name", "type": "string"},
                {"source": "body.profile.role", "alias": "role", "type": "string"},
                {"source": "body.preferences[].key", "alias": "pref_key", "type": "string"},
                {"source": "body.preferences[].value", "alias": "pref_value", "type": "string"},
            ]
        },
        "PaymentProcessed": {
            "table_name": "payments",
            "fields": [
                {"source": "header.id", "alias": "event_id", "type": "string"},
                {"source": "body.paymentId", "alias": "payment_id", "type": "string"},
                {"source": "body.amount", "alias": "amount", "type": "float"},
                {"source": "body.currency", "alias": "currency", "type": "string"},
                {"source": "body.status", "alias": "status", "type": "string"},
            ]
        }
    }

    # Process with the discovered split_path
    processor = JsonProcessor(split_path=split_path, config=config)
    processor.set_file_metadata("example.json", "2024-01-15T12:00:00Z")
    result = processor.process_messages(raw_documents)

    # Show the flattened results
    for table_name, df in result.items():
        print(f"\n{table_name} -> Flat DataFrame:")
        print("-" * 50)
        display_cols = [c for c in df.columns if not c.startswith('s3_')]
        print(df[display_cols].to_string(index=False))
        print(f"({len(df)} rows)")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_section("SUMMARY: BEFORE vs AFTER")

    print("""
BEFORE (Nested JSON):
---------------------
6 documents with varying structures:
  - Nested objects (body.customer.name)
  - Nested arrays (body.items[].sku)
  - Different fields per document type

AFTER (Flat DataFrames):
------------------------""")

    total_rows = sum(len(df) for df in result.values())
    for table_name, df in result.items():
        print(f"  {table_name}: {len(df)} rows x {len(df.columns)} columns")

    print(f"""
Total: {total_rows} rows across {len(result)} tables

The discover module found 'header.action' as the best splitter,
enabling JsonProcessor to route each document to the correct
configuration and flatten it appropriately.
""")


if __name__ == "__main__":
    main()
