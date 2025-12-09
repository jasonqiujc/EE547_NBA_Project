#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sync the minimal set of CSV files needed by the API server
from S3 down to the EC2 instance.

Files:
  - Yesterday's game results:
      games_yesterday_YYYYMMDD.csv
  - Today's + next 4 days' schedules:
      schedule_YYYYMMDD.csv for each date
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX

NBA_TZ = ZoneInfo("America/Los_Angeles")


def sync_api_files(days_ahead: int = 5) -> None:
    """
    Sync yesterday's games file and today + next N days' schedule files
    from S3 into LOCAL_DATA_DIR.

    All dates are computed in Los Angeles time to stay consistent with
    daily_crawl_and_upload.py and api_server.py.
    """
    today_la = datetime.now(NBA_TZ).date()
    yesterday_la = today_la - timedelta(days=1)

    y_str = yesterday_la.strftime("%Y%m%d")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------- 1) Yesterday games -------- #
    key_games = f"{S3_PREFIX}raw/games_yesterday_{y_str}.csv"
    local_games = LOCAL_DATA_DIR / f"games_yesterday_{y_str}.csv"
    try:
        print(f"Downloading {key_games} -> {local_games}")
        s3.download_file(S3_BUCKET, key_games, str(local_games))
    except ClientError as e:
        print(f"[sync_api_files] WARNING: failed to download {key_games}: {e}")

    # -------- 2) Today + next N days schedule files -------- #
    for offset in range(days_ahead):
        d = today_la + timedelta(days=offset)
        t_str = d.strftime("%Y%m%d")

        key_schedule = f"{S3_PREFIX}raw/schedule_{t_str}.csv"
        local_schedule = LOCAL_DATA_DIR / f"schedule_{t_str}.csv"

        try:
            print(f"Downloading {key_schedule} -> {local_schedule}")
            s3.download_file(S3_BUCKET, key_schedule, str(local_schedule))
        except ClientError as e:
            print(f"[sync_api_files] WARNING: failed to download {key_schedule}: {e}")


if __name__ == "__main__":
    sync_api_files()

