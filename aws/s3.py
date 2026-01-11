"""S3 utilities for reading and writing data."""
import boto3
import botocore
import gzip
import json
import logging
from io import BytesIO
from typing import Any, List, Dict, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger(__name__)


class S3Handler:
    """
    Utility class for S3 operations.

    Handles common patterns like reading gzipped JSON and writing Parquet.

    Args:
        bucket: S3 bucket name
        validate: Whether to validate bucket exists on init (default: True)

    Example:
        >>> handler = S3Handler("my-bucket")
        >>> data = handler.read_gzipped_json("path/to/file.json.gz")
        >>> handler.write_parquet(df, "output/data.parquet")
    """

    def __init__(self, bucket: str, validate: bool = True):
        if not bucket:
            raise ValueError("bucket cannot be empty")

        self.bucket = bucket
        self.s3_client = boto3.client('s3')

        if validate:
            self._validate_bucket()

    def _validate_bucket(self):
        """Validate bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket '{self.bucket}' does not exist")
            elif error_code == '403':
                raise ValueError(f"No permission to access bucket '{self.bucket}'")
            raise

    def read_json(self, key: str) -> Any:
        """
        Read JSON file from S3.

        Args:
            key: S3 object key

        Returns:
            Parsed JSON data
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            logger.error(f"Error reading JSON from s3://{self.bucket}/{key}: {e}")
            raise

    def read_gzipped_json(self, key: str) -> Any:
        """
        Read and decompress gzipped JSON from S3.

        Args:
            key: S3 object key

        Returns:
            Parsed JSON data
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            gzipped_data = response['Body'].read()
            json_data = json.loads(gzip.decompress(gzipped_data))
            return json_data
        except Exception as e:
            logger.error(f"Error reading gzipped JSON from s3://{self.bucket}/{key}: {e}")
            raise

    def write_json(self, data: Any, key: str, indent: Optional[int] = None):
        """
        Write JSON to S3.

        Args:
            data: Data to serialize as JSON
            key: S3 object key
            indent: JSON indentation (default: None for compact)
        """
        if not key:
            raise ValueError("key cannot be empty")

        try:
            body = json.dumps(data, indent=indent, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body.encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"Wrote JSON to s3://{self.bucket}/{key}")
        except Exception as e:
            logger.error(f"Error writing JSON to s3://{self.bucket}/{key}: {e}")
            raise

    def write_parquet(self, df: 'pd.DataFrame', key: str):
        """
        Write DataFrame to S3 as Parquet.

        Requires pandas and pyarrow to be installed.

        Args:
            df: Pandas DataFrame
            key: S3 object key
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for write_parquet")
        if df.empty:
            raise ValueError("Cannot write empty DataFrame")
        if not key:
            raise ValueError("key cannot be empty")

        try:
            buffer = BytesIO()
            df.to_parquet(buffer, engine="pyarrow")
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buffer.getvalue()
            )
            logger.info(f"Wrote Parquet ({len(df)} rows) to s3://{self.bucket}/{key}")
        except Exception as e:
            logger.error(f"Error writing Parquet to s3://{self.bucket}/{key}: {e}")
            raise

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[str]:
        """
        List objects in bucket with optional prefix.

        Args:
            prefix: Key prefix to filter by
            max_keys: Maximum number of keys to return

        Returns:
            List of object keys
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            keys = []

            for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            ):
                for obj in page.get('Contents', []):
                    keys.append(obj['Key'])

            return keys
        except Exception as e:
            logger.error(f"Error listing objects in s3://{self.bucket}/{prefix}: {e}")
            raise

    def get_object_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get object metadata without downloading the object.

        Args:
            key: S3 object key

        Returns:
            Dict with metadata including LastModified, ContentLength, etc.
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return {
                'LastModified': response['LastModified'],
                'ContentLength': response['ContentLength'],
                'ContentType': response.get('ContentType'),
                'ETag': response['ETag'],
                'Metadata': response.get('Metadata', {})
            }
        except Exception as e:
            logger.error(f"Error getting metadata for s3://{self.bucket}/{key}: {e}")
            raise
