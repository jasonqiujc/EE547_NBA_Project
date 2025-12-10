#!/usr/bin/env python3
# upload_raw_to_s3.py
"""
Local utility for uploading raw CSV files to S3.

Behavior:
  - Scan all CSV files under LOCAL_DATA_DIR
  - Upload to: s3://S3_BUCKET/S3_PREFIX/raw/<filename>.csv

Requirements:
  - AWS credentials configured (aws configure or environment variables)
  - config_aws.py defines AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR
"""

from pathlib import Path
import boto3
from botocore.exceptions import ClientError

from config_aws import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR


def upload_one_file(s3_client, local_path: Path, s3_key: str) -> None:
    """Upload a single file to S3."""
    print(f"Uploading {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    try:
        s3_client.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )
    except ClientError as e:
        print(f"[ERROR] Failed to upload {local_path}: {e}")


def main():
    # Ensure local directory exists
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Gather CSV files recursively
    csv_files = sorted(LOCAL_DATA_DIR.glob("**/*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found under {LOCAL_DATA_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files under {LOCAL_DATA_DIR}")

    # Initialize S3 client (credentials from env or ~/.aws/credentials)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    for local_path in csv_files:
        filename = local_path.name
        s3_key = f"{S3_PREFIX}raw/{filename}"
        upload_one_file(s3, local_path, s3_key)

    print("Done uploading all CSVs.")


if __name__ == "__main__":
    main()
