#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_daily_training.py

在 EC2 上定时运行的训练总控脚本。

流程：
  1. 调用 build_team_features.build_team_features()
     - 从 S3 raw/ 读取历史 + 每日增量
     - 生成特征 CSV 并保存在本地 data/ 或上传到 S3
     - 返回特征 CSV 的本地路径列表
  2. 调用 train_model.train_model(feature_paths)
     - 训练模型
     - 保存到本地 data/models/
     - 上传到 S3 的 models/ 目录

你可以把这个脚本放到 crontab 里每天跑一次。
"""

from datetime import datetime
from pathlib import Path
from typing import List, Union

from config_aws import S3_BUCKET, S3_PREFIX
from build_team_features import build_team_features
from train_model import train_model


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

    # 你自己的 build_team_features 函数应该：
    #   - 读 S3 raw/ 下所有 CSV（或指定时间窗）
    #   - 生成特征 CSV
    #   - 返回特征 CSV 本地路径 或 路径列表
    feature_paths = build_team_features()

    feature_paths = _normalize_feature_paths(feature_paths)
    print("[run_daily_training] Feature files:", feature_paths)

    print("\n========== [run_daily_training] Step 2: Train model ==========")
    model_s3_key = train_model(feature_paths)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Training complete.")
    print(f"New model uploaded to: s3://{S3_BUCKET}/{model_s3_key}")
    print(f"Latest model at:       s3://{S3_BUCKET}/{S3_PREFIX}models/model_latest.pkl")


if __name__ == "__main__":
    main()
