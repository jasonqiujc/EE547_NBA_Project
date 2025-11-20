#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py

功能：
  - 读取由 build_team_features.py 生成的特征 CSV
  - 拼成一个 DataFrame
  - 切分训练/验证集
  - 训练一个简单的模型（默认 RandomForest）
  - 在本地保存 model_YYYYMMDD.pkl
  - 上传到 S3:
      - models/model_YYYYMMDD.pkl
      - models/model_latest.pkl (覆盖)

依赖：
  - pandas, numpy, scikit-learn, boto3, joblib
  - config_aws.py 中定义：
      AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from config_aws import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR

# ================== 配置区域（你可以根据自己数据改） ==================

# 标签列
LABEL_COL = "WIN"  # 比如你可以用 "WIN" 或 "HOME_WIN" 等

# 训练 / 预测统一使用的特征列
# 这些名字要和 build_team_features 里生成的列一致
FEATURE_COLUMNS: List[str] = [
    "roll5_PTS_FOR",
    "roll5_PTS_AGAINST",
    "roll5_point_diff",
    "roll10_PTS_FOR",
    "roll10_point_diff",
    "roll10_win_rate",
    "season_win_rate",
]

# 如果你想明确指定哪些列是特征，就用这个列表；
# 这里直接用 FEATURE_COLUMNS，方便 api_server 复用。
EXPLICIT_FEATURE_COLS: List[str] | None = FEATURE_COLUMNS

# 模型保存的本地目录
MODEL_DIR = LOCAL_DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ================== 辅助函数 ==================


def _load_features(feature_paths: List[Union[str, Path]]) -> pd.DataFrame:
    """从多个 CSV 路径加载特征并拼接。"""
    dfs = []
    for p in feature_paths:
        path = Path(p)
        print(f"[train_model] Loading features from {path}")
        df = pd.read_csv(path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No feature files provided to train_model().")

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"[train_model] Total feature rows: {len(all_df)}")
    return all_df


def _select_X_y(df: pd.DataFrame):
    """根据配置，从 df 中切出 X, y。"""
    if LABEL_COL not in df.columns:
        raise KeyError(
            f"Label column '{LABEL_COL}' not found in features. "
            f"Available columns: {list(df.columns)[:20]} ..."
        )

    y = df[LABEL_COL]

    if EXPLICIT_FEATURE_COLS is not None:
        missing = [c for c in EXPLICIT_FEATURE_COLS if c not in df.columns]
        if missing:
            raise KeyError(
                f"Some feature columns specified in EXPLICIT_FEATURE_COLS "
                f"are missing in data: {missing}"
            )
        X = df[EXPLICIT_FEATURE_COLS]
        print(f"[train_model] Using explicit feature columns: {EXPLICIT_FEATURE_COLS}")
    else:
        # 默认策略：只用数值列，并且排除标签列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if LABEL_COL in numeric_cols:
            numeric_cols.remove(LABEL_COL)
        X = df[numeric_cols]
        print(f"[train_model] Using numeric feature columns: {numeric_cols[:20]} ...")

    return X, y


def _upload_to_s3(local_path: Path, s3_key: str) -> None:
    """上传模型文件到 S3."""
    print(f"[train_model] Uploading {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )
    except ClientError as e:
        print(f"[ERROR] Failed to upload model to S3: {e}")
        raise


# ================== 主函数：训练模型 ==================


def train_model(feature_paths: List[Union[str, Path]]) -> str:
    """
    训练模型并上传到 S3。

    参数：
      feature_paths: 特征 CSV 的本地路径列表。

    返回：
      timestamped 模型的 S3 key，例如：'nba_project/models/model_20251119.pkl'
    """
    # 1. 读特征
    df = _load_features(feature_paths)

    # 2. 切 X, y
    X, y = _select_X_y(df)

    # 3. train/val 切分
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    print(f"[train_model] Train size: {len(X_train)}, Val size: {len(X_val)}")

    # 4. 模型（这里先用 RandomForest，你可以换成 XGBoost/LogReg）
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    print("[train_model] Fitting model ...")
    clf.fit(X_train, y_train)

    # 5. 简单验证指标
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"[train_model] Validation Accuracy: {acc:.4f}")

    # 如果是二分类且有 predict_proba，可以算一下 AUC
    if hasattr(clf, "predict_proba") and len(np.unique(y_val)) == 2:
        y_proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        print(f"[train_model] Validation ROC-AUC: {auc:.4f}")

    # 6. 保存模型（本地）
    ts = datetime.utcnow().strftime("%Y%m%d")
    local_model_ts = MODEL_DIR / f"model_{ts}.pkl"
    local_model_latest = MODEL_DIR / "model_latest.pkl"

    print(f"[train_model] Saving timestamped model: {local_model_ts}")
    joblib.dump(clf, local_model_ts)

    print(f"[train_model] Saving latest model: {local_model_latest}")
    joblib.dump(clf, local_model_latest)

    # 7. 上传到 S3
    base_prefix = f"{S3_PREFIX}models/"

    ts_key = f"{base_prefix}model_{ts}.pkl"
    latest_key = f"{base_prefix}model_latest.pkl"

    _upload_to_s3(local_model_ts, ts_key)
    _upload_to_s3(local_model_latest, latest_key)

    print(f"[train_model] Done. Timestamped model at s3://{S3_BUCKET}/{ts_key}")
    print(f"[train_model] Latest model at     s3://{S3_BUCKET}/{latest_key}")

    # 返回 timestamped 的 S3 key，方便上层记录
    return ts_key


if __name__ == "__main__":
    print(
        "[INFO] train_model.py is intended to be called from "
        "run_daily_training.py with feature_paths."
    )

