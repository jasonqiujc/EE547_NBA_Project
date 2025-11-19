# config_aws.py
from pathlib import Path
import os

# AWS region for your project (modify if needed)
AWS_REGION = os.environ.get("AWS_REGION", "us-west-1")

# Replace with your actual S3 bucket name
S3_BUCKET = "ee547-nba-project"

# S3 folder prefix
S3_PREFIX = "datasets/nba_project/"

# Local folder to store generated CSV files
LOCAL_DATA_DIR = Path("data")
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

