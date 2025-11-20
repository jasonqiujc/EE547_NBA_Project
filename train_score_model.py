#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_score_model.py

功能：
  - 读取 team_game_features.csv
  - 用和胜率模型一样的特征列，去预测该队本场的得分 PTS_FOR（回归）
  - 训练一个 RandomForestRegressor
  - 本地保存 score_model_latest.pkl
  - 上传到 S3:
      - models/score_model_YYYYMMDD.pkl
      - models/score_model_latest.pkl (覆盖)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Union

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from config_aws import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR
from train_model import FEATURE_COLUMNS  # 复用你现有的特征列

# 标签列：本队本场得分
LABEL_COL = "PTS_FOR"

# 使用和胜率模型相同的特征列
EXPLICIT_FEATURE_COLS: List[str] = FEATURE_COLUMNS

# 模型保存目录（和原来一样）
MODEL_DIR = LOCAL_DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_features(feature_paths: List[Union[str, Path]]) -> pd.DataFrame:
    dfs = []
    for p in feature_paths:
        path = Path(p)
        print(f"[train_score_model] Loading features from {path}")
        df = pd.read_csv(path)
        dfs.append(df)
    if not dfs:
        raise ValueError("No feature files provided to train_score_model().")
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[train_score_model] Total feature rows: {len(all_df)}")
    return all_df


def _select_X_y(df: pd.DataFrame):
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Label column '{LABEL_COL}' not found in features. "
            f"Available columns: {list(df.columns)[:20]} ..."
        )

    y = df[LABEL_COL]

    missing = [c for c in EXPLICIT_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Some feature columns for score model are missing in data: {missing}"
        )

    X = df[EXPLICIT_FEATURE_COLS]
    print(f"[train_score_model] Using feature columns: {EXPLICIT_FEATURE_COLS}")
    return X, y


def _upload_to_s3(local_path: Path, s3_key: str) -> None:
    print(f"[train_score_model] Uploading {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )
    except ClientError as e:
        print(f"[ERROR] Failed to upload score model to S3: {e}")
        raise


def train_score_model(feature_paths: List[Union[str, Path]]) -> str:
    """
    训练“得分预测”模型并上传到 S3。
    返回 timestamped 模型在 S3 上的 key。
    """
    df = _load_features(feature_paths)
    X, y = _select_X_y(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"[train_score_model] Train size: {len(X_train)}, Val size: {len(X_val)}")

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    print("[train_score_model] Fitting score model ...")
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"[train_score_model] Validation MAE: {mae:.3f}")
    print(f"[train_score_model] Validation R^2: {r2:.3f}")

    ts = datetime.utcnow().strftime("%Y%m%d")
    local_model_ts = MODEL_DIR / f"score_model_{ts}.pkl"
    local_model_latest = MODEL_DIR / "score_model_latest.pkl"

    print(f"[train_score_model] Saving timestamped score model: {local_model_ts}")
    joblib.dump(reg, local_model_ts)

    print(f"[train_score_model] Saving latest score model: {local_model_latest}")
    joblib.dump(reg, local_model_latest)

    base_prefix = f"{S3_PREFIX}models/"
    ts_key = f"{base_prefix}score_model_{ts}.pkl"
    latest_key = f"{base_prefix}score_model_latest.pkl"

    _upload_to_s3(local_model_ts, ts_key)
    _upload_to_s3(local_model_latest, latest_key)

    print(f"[train_score_model] Done. Timestamped score model at s3://{S3_BUCKET}/{ts_key}")
    print(f"[train_score_model] Latest score model at     s3://{S3_BUCKET}/{latest_key}")

    return ts_key


if __name__ == "__main__":
    print(
        "[INFO] train_score_model.py is intended to be called from "
        "run_daily_training.py with feature_paths."
    )
