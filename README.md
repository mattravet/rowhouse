# Rowhouse

Data tools for building ETL pipelines and data processing workflows.

## Modules

### Unfurl (`rowhouse.unfurl`)

Unfurl deeply nested JSON into flat tabular data.

```python
from rowhouse.unfurl import JsonProcessor

config = {
    "OrderCreated": {
        "table_name": "orders",
        "fields": [
            {"source": "header.orderId", "alias": "order_id", "type": "string"},
            {"source": "body.items[].sku", "alias": "sku", "type": "string"},
            {"source": "body.items[].price", "alias": "price", "type": "float"},
        ]
    }
}

processor = JsonProcessor(['header', 'action'], config)
processor.set_file_metadata("data.json", "2024-01-15")
result = processor.process_messages(messages)

df = result["OrderCreated"]  # Flat DataFrame with one row per item
```

**Features:**
- Arbitrary nesting depth (10+ levels)
- Nested array explosion (`orders[].items[].variants[]`)
- Cartesian product handling
- Config-driven field mappings
- Type enforcement (string, integer, float, timestamp, etc.)
- **Smart coercion** for messy real-world data (see below)

#### Smart Coercion for Messy Data

Enable `coerce=True` to handle user-generated data that would otherwise break type conversion:

```python
# Global coercion - applies to all fields
processor = JsonProcessor(['header', 'action'], config, coerce=True)

# Or per-field coercion
config = {
    "OrderCreated": {
        "table_name": "orders",
        "fields": [
            {"source": "body.price", "alias": "price", "type": "float", "coerce": True},
            {"source": "body.quantity", "alias": "quantity", "type": "integer"},  # No coercion
        ]
    }
}
```

**What coercion handles:**
- Currency: `"$1,234.56"` → `1234.56`, `"€50"` → `50.0`
- Percentages: `"25%"` → `0.25`
- Thousands separators: `"1,000,000"` → `1000000`
- Multiple date formats: `"01/15/2024"`, `"2024-01-15"`, `"Jan 15, 2024"`
- Boolean variants: `"yes"`, `"no"`, `"1"`, `"0"`, `"true"`, `"false"`
- Invalid values become `NULL` instead of crashing

### Discover (`rowhouse.discover`)

Analyze JSON structure and automatically find the best splitter field for JsonProcessor.

```python
from rowhouse.discover import StructureAnalyzer

analyzer = StructureAnalyzer()

# Auto-detect the best splitter field
results = analyzer.find_splitters(documents)
print(results[0])
# SplitterResult(field='header.action', score=3.76, values=5, coverage=100.0%)

# Use it with JsonProcessor
best = results[0]
processor = JsonProcessor(split_path=best.field.split('.'), config)

# Get a human-readable summary
print(analyzer.describe(documents))
# Documents analyzed: 1,000
# Unique paths: 47
# Candidate splitters:
#   header.action (5 values, score: 3.76) ← RECOMMENDED
#   header.version (3 values, score: 1.2)
#
# Structure by header.action:
#   "OrderCreated" (412 docs): body.items[], body.customer.*
#   "UserCreated" (301 docs): body.user.*, body.preferences[]
```

**Features:**
- Automatic splitter detection using Jaccard similarity
- Configurable grouping (explicit field, custom function, auto-detect)
- Pluggable similarity strategies
- Terminal-friendly `describe()` output

### Validation (`rowhouse.validation`)

Data validation and type conversion utilities for DataFrames.

```python
from rowhouse.validation import DataValidator

validator = DataValidator()

# Safe type conversions
df['price'] = validator.to_numeric(df['price'])      # Handles $, €, commas, %
df['quantity'] = validator.to_integer(df['quantity']) # Overflow detection
df['created'] = validator.to_datetime(df['created']) # Multi-format parsing

# Column name normalization
df = validator.normalize_columns(df)  # "Column Name!" -> "column_name"

# Schema-driven transformation
schema = {
    "user_id": {"source": "UserID", "type": "integer", "required": True},
    "amount": {"type": "float"},
    "created_at": {"type": "datetime", "rename": "created_date"}
}
df, result = validator.apply_schema(df, schema)

# Validation checks
result = validator.validate_not_null(df, ['user_id', 'amount'])
result = validator.validate_unique(df, ['user_id'])
```

**Features:**
- Safe numeric conversion (handles currency, percentages, thousands separators)
- Integer conversion with overflow detection
- Multi-format datetime parsing (15+ formats)
- Column name normalization
- Schema-driven transformations
- Null and uniqueness validation

### AWS (`rowhouse.aws`)

Utilities for working with AWS services.

```python
from rowhouse.aws import S3Handler

handler = S3Handler("my-bucket")

# Read
data = handler.read_gzipped_json("path/to/file.json.gz")
data = handler.read_json("path/to/file.json")

# Write
handler.write_parquet(df, "output/data.parquet")
handler.write_json({"key": "value"}, "output/data.json")

# List & metadata
keys = handler.list_objects(prefix="data/2024/")
meta = handler.get_object_metadata("path/to/file.json")
```

## Installation

```bash
# Core (unfurl only)
pip install rowhouse

# With AWS support
pip install rowhouse[aws]

# Everything
pip install rowhouse[all]
```

## Quick Example: JSON to Parquet Pipeline

```python
from rowhouse.unfurl import JsonProcessor
from rowhouse.aws import S3Handler

# Read nested JSON from S3
input_handler = S3Handler("input-bucket")
data = input_handler.read_gzipped_json("raw/events.json.gz")

# Flatten with unfurl
processor = JsonProcessor(['header', 'action'], config)
processor.set_file_metadata("events.json.gz", "2024-01-15")
result = processor.process_messages(data)

# Write Parquet to S3
output_handler = S3Handler("output-bucket")
for table_name, df in result.items():
    output_handler.write_parquet(df, f"processed/{table_name}.parquet")
```

See `examples/unfurl_lambda.py` for a complete AWS Lambda example.

## Project Structure

```
rowhouse/
├── unfurl/              # JSON flattening
│   └── json_processor.py
├── discover/            # Structure analysis
│   ├── analyzer.py
│   └── similarity.py
├── validation/          # Data validation
│   └── validator.py
├── common/              # Shared utilities
│   └── paths.py
├── aws/                 # AWS utilities
│   └── s3.py
├── examples/
│   ├── basic_usage.py
│   ├── unfurl_lambda.py
│   └── sample_config.json
└── tests/               # 131 tests
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT
