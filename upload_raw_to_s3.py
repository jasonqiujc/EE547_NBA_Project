#!/usr/bin/env python3
# upload_raw_to_s3.py
"""
本地使用：
  - 扫描 data/ 目录里的所有 CSV
  - 上传到 S3:  s3://S3_BUCKET/S3_PREFIX/raw/<filename>.csv

要求：
  - 已经配置好 AWS 凭证（本机 aws configure，或者环境变量）
  - config_aws.py 中设置好 AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR
"""

from pathlib import Path
import boto3
from botocore.exceptions import ClientError

from config_aws import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR


def upload_one_file(s3_client, local_path: Path, s3_key: str) -> None:
    """上传单个文件到 S3."""
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
    # 确保 data 目录存在
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 找到所有 csv（包括 data/ 子目录）
    csv_files = sorted(LOCAL_DATA_DIR.glob("**/*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found under {LOCAL_DATA_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files under {LOCAL_DATA_DIR}")

    # 初始化 S3 client（凭证从环境 / ~/.aws/credentials 取）
    s3 = boto3.client("s3", region_name=AWS_REGION)

    for local_path in csv_files:
        # 只取文件名，放在 raw/ 下面。如果想保留子目录结构，可以自己改这里
        filename = local_path.name
        s3_key = f"{S3_PREFIX}raw/{filename}"  # 例如 datasets/nba_project/raw/player_logs_clean_xxx.csv
        upload_one_file(s3, local_path, s3_key)

    print("Done uploading all CSVs.")


if __name__ == "__main__":
    main()
