#!/usr/bin/env python3
# daily_crawl_and_upload.py
"""
每天增量爬虫：
  - 抓取昨天所有球员比赛日志（player level）
  - 保存到本地 data/ 目录
  - 同时上传到 S3 的 raw/ 目录
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import boto3
from nba_api.stats.endpoints import leaguegamelog
from botocore.exceptions import ClientError

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX


def fetch_yesterday_logs() -> pd.DataFrame:
    """用 nba_api 抓取昨天所有球员比赛日志。"""
    # 昨天（用 UTC，也可以换成美东时间）
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Fetching logs for {yesterday} ...")

    resp = leaguegamelog.LeagueGameLog(
        player_or_team_abbreviation="P",   # P = player logs
        season="2024-25",                  # TODO: 换成当前赛季
        season_type_all_star="Regular Season",
        date_from_nullable=yesterday,
        date_to_nullable=yesterday,
        timeout=30,
    )

    df = resp.get_data_frames()[0]
    return df, yesterday


def upload_to_s3(local_path: Path, date_str: str) -> None:
    """上传本地 CSV 到 S3 的 raw/ 目录。"""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    filename = local_path.name
    # 例如：nba_project/raw/daily_player_logs_20251118.csv
    s3_key = f"{S3_PREFIX}raw/{filename}"

    print(f"Uploading {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )
    except ClientError as e:
        print(f"[ERROR] Failed to upload {local_path}: {e}")


def main():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df, date_str = fetch_yesterday_logs()

    if df.empty:
        print("No games found for yesterday, nothing to do.")
        return

    # 本地文件名
    fname = f"player_logs_daily_{date_str.replace('-', '')}.csv"
    local_path = LOCAL_DATA_DIR / fname

    print(f"Saving to local file: {local_path}")
    df.to_csv(local_path, index=False)

    # 上传到 S3
    upload_to_s3(local_path, date_str)

    print("Done daily crawl + upload.")


if __name__ == "__main__":
    main()
