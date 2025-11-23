#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_daily_training.py (Enhanced)

EC2 上每天运行的训练总控脚本。

新增功能（重要！）：
  - 自动从 S3 合并历史主表 player_logs_all.csv 和最新增量 daily_xxx.csv
  - 得到新的主表 player_logs_all.csv 并上传回 S3（覆盖旧表）

随后流程：
  1. build_team_features.build_team_features()
  2. train_model.train_model()
  3. train_score_model.train_score_model()
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Union

import boto3
import pandas as pd

from config_aws import S3_BUCKET, S3_PREFIX, AWS_REGION
from build_team_features import build_team_features
from train_model import train_model
from train_score_model import train_score_model


# ===============================================================
#  新增：合并历史主表 + 昨天增量
# ===============================================================

def update_master_player_logs():
    """从 S3 拉历史主表 + 昨天增量，合并成新的主表并上传回 S3。"""
    print("========== [run_daily_training] Updating master player logs ==========")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # 昨天日期
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    daily_fname = f"player_logs_daily_{yesterday.replace('-', '')}.csv"

    # S3 key
    master_key = f"{S3_PREFIX}raw/player_logs_all.csv"
    daily_key = f"{S3_PREFIX}raw/{daily_fname}"

    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(exist_ok=True)

    master_local = tmp_dir / "player_logs_all.csv"
    daily_local = tmp_dir / daily_fname

    # ----------- 下载历史主表 -----------
    try:
        s3.download_file(S3_BUCKET, master_key, str(master_local))
        df_all = pd.read_csv(master_local)
        print(f"Loaded master table from S3: {len(df_all)} rows.")
    except Exception:
        print("[INFO] Master table not found. Starting fresh.")
        df_all = pd.DataFrame()

    # ----------- 下载昨日增量 -----------
    try:
        s3.download_file(S3_BUCKET, daily_key, str(daily_local))
        df_daily = pd.read_csv(daily_local)
        print(f"Loaded daily increment {daily_fname}: {len(df_daily)} rows.")
    except Exception:
        print(f"[WARN] Missing daily file {daily_key}, skip update.")
        return str(master_local)  # 用旧主表继续 pipeline

    # ----------- 合并并去重 -----------
    df_new = pd.concat([df_all, df_daily], ignore_index=True)

    # 按你的数据结构，GAME_ID + PLAYER_ID 是唯一键
    if {"GAME_ID", "PLAYER_ID"}.issubset(df_new.columns):
        df_new.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], inplace=True)

    print(f"Merged new master: {len(df_new)} rows total.")

    # ----------- 保存并上传新主表 -----------
    df_new.to_csv(master_local, index=False)
    s3.upload_file(str(master_local), S3_BUCKET, master_key)
    print(f"Updated master table uploaded to s3://{S3_BUCKET}/{master_key}")

    return str(master_local)  # 返回新主表的本地路径


# ===============================================================
#  原有部分：构建特征 & 模型训练
# ===============================================================

def _normalize_feature_paths(feature_paths) -> List[Union[str, Path]]:
    if feature_paths is None:
        raise ValueError("build_team_features() returned None, expected path(s).")
    if isinstance(feature_paths, (str, Path)):
        return [feature_paths]
    return list(feature_paths)


def main():

    # ----------- 新增：先更新主表 -----------
    master_local_path = update_master_player_logs()

    print("\n========== [run_daily_training] Step 1: Build team features ==========")
    feature_paths = build_team_features()
    feature_paths = _normalize_feature_paths(feature_paths)

    print("[run_daily_training] Feature files:", feature_paths)

    # --------- Step 2: 胜率模型 ----------
    print("\n========== [run_daily_training] Step 2: Train win model ==========")
    win_model_s3_key = train_model(feature_paths)

    # --------- Step 3: 比分模型 ----------
    print("\n========== [run_daily_training] Step 3: Train score model ==========")
    score_model_s3_key = train_score_model(feature_paths)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Training complete.")
    print(f"New win model uploaded to:   s3://{S3_BUCKET}/{win_model_s3_key}")
    print(f"New score model uploaded to: s3://{S3_BUCKET}/{score_model_s3_key}")
    print(f"Latest win model at:         s3://{S3_BUCKET}/{S3_PREFIX}models/model_latest.pkl")
    print(f"Latest score model at:       s3://{S3_BUCKET}/{S3_PREFIX}models/score_model_latest.pkl")


if __name__ == "__main__":
    main()
