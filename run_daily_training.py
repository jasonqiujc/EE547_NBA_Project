#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_daily_training.py

Daily training controller script for EC2.

Workflow:
  0. Rebuild master player_logs_all.csv:
       - Base: latest player_logs_clean_all_3seasons_plus_current_*.csv (from S3)
       - Incremental: yesterday's player_logs_daily_YYYYMMDD.csv (if exists)
       - Output: upload/overwrite raw/player_logs_all.csv on S3

  1. Run build_team_features():
       - Uses player_logs_all.csv as single source
       - Generates team_game_features.csv
       - Upload handled inside build_team_features()

  2. Run train_model():
       - Train PyTorch score model using team features
       - Save model_latest.pth locally
       - Upload model_latest.pth to S3
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Union

import boto3
import pandas as pd

from config_aws import S3_BUCKET, S3_PREFIX, AWS_REGION
from build_team_features import build_team_features
from train_model import train_model


# ===============================================================
# Step 0: Rebuild master player_logs_all.csv
# ===============================================================

def update_master_player_logs() -> str:
    """
    Rebuild player_logs_all.csv by:
      - Loading the latest clean_all CSV from S3
      - Appending yesterday's daily logs (if available)
      - Removing duplicates
      - Uploading the new master CSV back to S3
    """

    print("========== [run_daily_training] Rebuilding master player logs ==========")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Determine yesterday's date in LA timezone
    now_la = datetime.now(ZoneInfo("America/Los_Angeles"))
    yesterday = (now_la - timedelta(days=1)).date()
    daily_fname = f"player_logs_daily_{yesterday.strftime('%Y%m%d')}.csv"

    prefix = f"{S3_PREFIX}raw/"
    master_key = prefix + "player_logs_all.csv"
    daily_key = prefix + daily_fname

    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(exist_ok=True)

    # Find the base clean_all file
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    contents = resp.get("Contents", [])

    clean_all_key = None
    for obj in contents:
        fname = obj["Key"].split("/")[-1]
        if fname.startswith("player_logs_clean_all_3seasons_plus_current"):
            clean_all_key = obj["Key"]
            break

    if clean_all_key is None:
        raise RuntimeError(
            "ERROR: Missing player_logs_clean_all_3seasons_plus_current_*.csv in raw/. "
            "Cannot rebuild master."
        )

    print(f"[update_master_player_logs] Base clean_all file: {clean_all_key}")

    clean_all_local = tmp_dir / "clean_all.csv"
    s3.download_file(S3_BUCKET, clean_all_key, str(clean_all_local))

    df_all = pd.read_csv(clean_all_local)
    print(f"[update_master_player_logs] Loaded clean_all rows: {len(df_all)}")

    # Append daily logs if available
    try:
        daily_local = tmp_dir / daily_fname
        s3.download_file(S3_BUCKET, daily_key, str(daily_local))
        df_daily = pd.read_csv(daily_local)
        print(f"[update_master_player_logs] Loaded daily: {len(df_daily)} rows.")
        df_all = pd.concat([df_all, df_daily], ignore_index=True)
    except Exception:
        print(f"[update_master_player_logs] No daily file for {daily_fname}, skipping append.")

    # Deduplicate
    if {"GAME_ID", "PLAYER_ID"}.issubset(df_all.columns):
        before = len(df_all)
        df_all.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], inplace=True)
        print(f"[update_master_player_logs] Removed {before - len(df_all)} duplicate rows.")
    else:
        print("[update_master_player_logs] WARNING: GAME_ID/PLAYER_ID missing; cannot deduplicate.")

    print(f"[update_master_player_logs] Final master rows: {len(df_all)}")

    # Save and upload master file
    master_local = tmp_dir / "player_logs_all.csv"
    df_all.to_csv(master_local, index=False)

    s3.upload_file(str(master_local), S3_BUCKET, master_key)
    print(f"[update_master_player_logs] Uploaded master to s3://{S3_BUCKET}/{master_key}")

    return str(master_local)


# ===============================================================
# Step 1 & 2: Feature building and model training
# ===============================================================

def _normalize_feature_paths(feature_paths) -> List[Union[str, Path]]:
    """
    Ensure build_team_features() return value is a list of paths.
    """
    if feature_paths is None:
        raise ValueError("build_team_features() returned None; expected path(s).")
    if isinstance(feature_paths, (str, Path)):
        return [feature_paths]
    return list(feature_paths)


def main():
    # Step 0: Rebuild master CSV
    master_local_path = update_master_player_logs()
    print(f"[run_daily_training] master_local_path = {master_local_path}")

    # Step 1: Build team features
    print("\n========== [run_daily_training] Step 1: Build team features ==========")
    feature_paths = build_team_features()
    feature_paths = _normalize_feature_paths(feature_paths)
    print("[run_daily_training] Feature files:", feature_paths)

    # Step 2: Train PyTorch score model
    print("\n========== [run_daily_training] Step 2: Train score model ==========")
    score_model_s3_key = train_model(feature_paths)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Training complete.")
    print(f"New score model uploaded to: s3://{S3_BUCKET}/{score_model_s3_key}")
    print(f"Latest model:               s3://{S3_BUCKET}/{S3_PREFIX}models/model_latest.pth")


if __name__ == "__main__":
    main()

