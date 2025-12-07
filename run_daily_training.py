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
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Union

import boto3
import pandas as pd

from config_aws import S3_BUCKET, S3_PREFIX, AWS_REGION
from build_team_features import build_team_features
from build_team_features import build_team_features
from train_model import train_model 


# ===============================================================
#  新增：合并历史主表 + 昨天增量
# ===============================================================


def update_master_player_logs():
    """
    每天重新构建 player_logs_all.csv：

    基础：最新的 player_logs_clean_all_3seasons_plus_current_*.csv
    增量：昨天的 daily CSV（如果存在）

    输出：
        s3://bucket/.../raw/player_logs_all.csv  (覆盖)
    """

    print("========== [run_daily_training] Rebuilding master player logs ==========")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # ---- 时间计算（洛杉矶时间） ----
    now_la = datetime.now(ZoneInfo("America/Los_Angeles"))
    yesterday = (now_la - timedelta(days=1)).date()
    daily_fname = f"player_logs_daily_{yesterday.strftime('%Y%m%d')}.csv"

    # S3 keys
    prefix = f"{S3_PREFIX}raw/"
    master_key = prefix + "player_logs_all.csv"
    daily_key = prefix + daily_fname

    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(exist_ok=True)

    # ---- 第一步：找到 clean_all 文件 ----
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    contents = resp.get("Contents", [])

    clean_all_key = None
    for obj in contents:
        fname = obj["Key"].split("/")[-1]
        if fname.startswith("player_logs_clean_all_3seasons_plus_current"):
            clean_all_key = obj["Key"]
            break

    if clean_all_key is None:
        raise RuntimeError("ERROR: Cannot find clean_all CSV in S3 raw/. Cannot rebuild master.")

    print(f"[update_master_player_logs] Base clean_all file: {clean_all_key}")

    clean_all_local = tmp_dir / "clean_all.csv"
    s3.download_file(S3_BUCKET, clean_all_key, str(clean_all_local))

    df_all = pd.read_csv(clean_all_local)
    print(f"[update_master_player_logs] Loaded clean_all rows: {len(df_all)}")

    # ---- 第二步：如果存在 daily，就添加进去 ----
    try:
        daily_local = tmp_dir / daily_fname
        s3.download_file(S3_BUCKET, daily_key, str(daily_local))
        df_daily = pd.read_csv(daily_local)
        print(f"[update_master_player_logs] Loaded daily: {len(df_daily)} rows.")

        df_all = pd.concat([df_all, df_daily], ignore_index=True)
    except Exception:
        print(f"[update_master_player_logs] No daily file for {daily_fname}, skip daily append.")

    # ---- 去重 ----
    if {"GAME_ID", "PLAYER_ID"}.issubset(df_all.columns):
        before = len(df_all)
        df_all.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"], inplace=True)
        print(f"[update_master_player_logs] Removed {before - len(df_all)} duplicate rows.")

    print(f"[update_master_player_logs] Final master rows: {len(df_all)}")

    # ---- 第三步：保存并上传为 player_logs_all.csv ----
    master_local = tmp_dir / "player_logs_all.csv"
    df_all.to_csv(master_local, index=False)

    s3.upload_file(str(master_local), S3_BUCKET, master_key)
    print(f"[update_master_player_logs] Uploaded master to s3://{S3_BUCKET}/{master_key}")

    return str(master_local)


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
    # 先更新主表（不动）
    master_local_path = update_master_player_logs()

    print("\n========== [run_daily_training] Step 1: Build team features ==========")
    feature_paths = build_team_features()
    feature_paths = _normalize_feature_paths(feature_paths)
    print("[run_daily_training] Feature files:", feature_paths)

    # --------- Step 2: 训练 PyTorch 比分模型 ----------
    print("\n========== [run_daily_training] Step 2: Train score model (PyTorch) ==========")
    score_model_s3_key = train_model(feature_paths)   # train_model 返回 .pth 的 key

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Training complete.")
    print(f"New score model uploaded to: s3://{S3_BUCKET}/{score_model_s3_key}")
    print(f"Latest score model at:       s3://{S3_BUCKET}/{S3_PREFIX}models/model_latest.pth")



if __name__ == "__main__":
    main()
