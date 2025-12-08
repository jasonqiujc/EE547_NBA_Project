#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import boto3

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX

def sync_api_files():
    # 使用 ET（和你的爬虫保持一致）
    today_et = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    yesterday_et = today_et - timedelta(days=1)

    y_str = yesterday_et.strftime("%Y%m%d")
    t_str = today_et.strftime("%Y%m%d")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 昨日比赛
    key_games = f"{S3_PREFIX}raw/games_yesterday_{y_str}.csv"
    local_games = LOCAL_DATA_DIR / f"games_yesterday_{y_str}.csv"
    print(f"Downloading {key_games} -> {local_games}")
    s3.download_file(S3_BUCKET, key_games, str(local_games))

    # 今日赛程
    key_schedule = f"{S3_PREFIX}raw/schedule_{t_str}.csv"
    local_schedule = LOCAL_DATA_DIR / f"schedule_{t_str}.csv"
    print(f"Downloading {key_schedule} -> {local_schedule}")
    s3.download_file(S3_BUCKET, key_schedule, str(local_schedule))


if __name__ == "__main__":
    sync_api_files()
