"""
AWS Lambda handler for JSON to Parquet conversion using Rowhouse.

This example shows how to use rowhouse.unfurl with rowhouse.aws
to process gzipped JSON files from S3 and output Parquet files.

Environment variables:
    INPUT_BUCKET: S3 bucket containing source JSON files
    OUTPUT_BUCKET: S3 bucket for output Parquet files
    CONFIG_PATH: Path to field mapping configuration (default: config.json)

TIP: Use rowhouse.discover to analyze new JSON sources and find the
     right split_path. See examples/discover_example.py for details.

Requirements:
    - rowhouse (this package)
"""
import os
import json
import logging

from rowhouse.aws import S3Handler
from rowhouse.unfurl import JsonProcessor

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def extract_files_from_event(event: dict) -> list:
    """
    Extract file paths from Lambda event.
    Handles S3 event notifications (direct or via SQS/SNS).
    """
    files = []

    # Try SQS/SNS wrapped S3 event
    try:
        body = json.loads(event['Records'][0]['body'])
        if body.get('Type') == 'Notification':
            s3_event = json.loads(body['Message'])
            file_path = s3_event['Records'][0]['s3']['object']['key']
        else:
            file_path = body['Records'][0]['s3']['object']['key']
        files.append(file_path)
    except (KeyError, IndexError, json.JSONDecodeError):
        pass

    # Try direct S3 event
    if not files:
        try:
            file_path = event['Records'][0]['s3']['object']['key']
            files.append(file_path)
        except (KeyError, IndexError):
            pass

    # Try Step Function format
    if not files:
        try:
            for item in event.get('Items', []):
                if 'Key' in item:
                    files.append(item['Key'])
        except (KeyError, TypeError):
            pass

    if not files:
        raise ValueError(f"Could not extract files from event: {event}")

    return files


def load_config() -> dict:
    """Load field mapping configuration."""
    config_path = os.environ.get('CONFIG_PATH', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def format_output_key(input_key: str, table_name: str) -> str:
    """Format output S3 key based on input path and table name."""
    parts = input_key.split('/')
    # Extract date from path (adjust based on your key structure)
    date = '-'.join(parts[1:4]) if len(parts) > 4 else 'unknown'
    filename = parts[-1].replace('.json.gz', '.parquet').replace('.json', '.parquet')
    return f"processed/{table_name}/load_date={date}/{filename}"


def lambda_handler(event, context):
    """Main Lambda entry point."""
    logger.info("Starting JSON to Parquet conversion")

    input_bucket = os.environ.get('INPUT_BUCKET')
    output_bucket = os.environ.get('OUTPUT_BUCKET')

    if not input_bucket or not output_bucket:
        raise ValueError("INPUT_BUCKET and OUTPUT_BUCKET environment variables required")

    try:
        config = load_config()
        input_handler = S3Handler(bucket=input_bucket)
        output_handler = S3Handler(bucket=output_bucket)

        # split_path routes documents to configs based on this field's value
        # Use rowhouse.discover.StructureAnalyzer to find the best split_path
        processor = JsonProcessor(
            split_path=['header', 'action'],  # Customize for your JSON structure
            config=config
        )

        files = extract_files_from_event(event)
        results = []

        for file_key in files:
            try:
                result = process_file(
                    file_key, input_handler, output_handler,
                    processor, config
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_key}: {e}")
                continue

        return {
            'statusCode': 200,
            'processed': len(results),
            'results': results
        }

    except Exception as e:
        logger.error(f"Lambda error: {e}")
        raise


def process_file(file_key: str, input_handler: S3Handler,
                 output_handler: S3Handler, processor, config: dict) -> dict:
    """Process a single JSON file and output Parquet."""
    logger.info(f"Processing: {file_key}")

    # Get file metadata
    metadata = input_handler.get_object_metadata(file_key)
    last_modified = metadata['LastModified'].isoformat()
    processor.set_file_metadata(file_key, last_modified)

    # Read, process, and write
    raw_data = input_handler.read_gzipped_json(file_key)
    dataframes = processor.process_messages(raw_data)

    tables_written = 0
    for key, df in dataframes.items():
        if len(df) > 0:
            table_name = config[key]['table_name']
            output_key = format_output_key(file_key, table_name)
            output_handler.write_parquet(df, output_key)
            tables_written += 1
            logger.info(f"Wrote {len(df)} rows to {output_key}")

    return {
        'file': file_key,
        'tables_written': tables_written,
        'total_tables': len(dataframes)
    }
