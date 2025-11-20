#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_daily_training.py

在 EC2 上定时运行的训练总控脚本。

流程：
  1. build_team_features.build_team_features()
     - 从 S3 raw/ 读取历史 + 每日增量
     - 生成特征 CSV 并保存在本地 data/
  2. train_model.train_model(feature_paths)
     - 训练“胜率模型”（分类）
     - 保存到 data/models/model_latest.pkl
     - 上传到 S3 的 models/model_latest.pkl
  3. train_score_model.train_score_model(feature_paths)
     - 训练“比分模型”（回归，预测 PTS_FOR）
     - 保存到 data/models/score_model_latest.pkl
     - 上传到 S3 的 models/score_model_latest.pkl
"""

from datetime import datetime
from pathlib import Path
from typing import List, Union

from config_aws import S3_BUCKET, S3_PREFIX
from build_team_features import build_team_features
from train_model import train_model
from train_score_model import train_score_model


def _normalize_feature_paths(feature_paths) -> List[Union[str, Path]]:
    """
    兼容几种常见返回格式：
      - 单个 Path/str
      - Path/str 列表
    """
    if feature_paths is None:
        raise ValueError("build_team_features() returned None, expected path(s).")

    if isinstance(feature_paths, (str, Path)):
        return [feature_paths]

    # 假设是可迭代列表
    return list(feature_paths)


def main():
    print("========== [run_daily_training] Step 1: Build team features ==========")

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

