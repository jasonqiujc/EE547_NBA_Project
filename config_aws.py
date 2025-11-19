# config_aws.py

from pathlib import Path
import os

AWS_REGION = os.environ.get("AWS_REGION", "us-west-1")
S3_BUCKET = "ee547-nba-project-yourname"  # 改成你的 bucket 名字
S3_PREFIX = "datasets/nba_project/"       # 在 S3 里的前缀
LOCAL_DATA_DIR = Path("data")             # 本地数据目录

LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
