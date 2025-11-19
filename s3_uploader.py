# s3_uploader.py

from pathlib import Path
from typing import Iterable, List

import boto3

from config_aws import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR


def get_s3_client():
    # 不写 Access Key，使用 EC2 实例角色
    return boto3.client("s3", region_name=AWS_REGION)


def upload_file_to_s3(local_path: Path, s3_key: str):
    s3 = get_s3_client()
    local_path = Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(f"本地文件不存在: {local_path}")

    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    print(f"[OK] {local_path} -> s3://{S3_BUCKET}/{s3_key}")


def upload_files_to_s3(files: Iterable[Path]) -> List[str]:
    uploaded_keys = []
    for local_path in files:
        local_path = Path(local_path)
        relative = local_path.relative_to(LOCAL_DATA_DIR)
        s3_key = f"{S3_PREFIX}{relative.as_posix()}"

        upload_file_to_s3(local_path, s3_key)
        uploaded_keys.append(s3_key)

    return uploaded_keys
