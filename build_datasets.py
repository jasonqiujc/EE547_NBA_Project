#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a classification model to predict WIN for each team-game row,
based on team-level rolling features produced by build_team_features.py.

Input:
    - LOCAL_DATA_DIR / "team_game_features.csv"

Output:
    - LOCAL_DATA_DIR / "model_YYYYMMDD.pkl"
    - S3:  s3://{S3_BUCKET}/{S3_PREFIX}models/model_YYYYMMDD.pkl
    - S3:  s3://{S3_BUCKET}/{S3_PREFIX}models/model_latest.pkl  (always latest)

This file also defines FEATURE_COLUMNS so that api_server.py can import it.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import boto3
from sklearn.ensemble import RandomForestClassifier
import joblib

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX


# ---------------- Config ---------------- #

TEAM_FEATURES_FILENAME = "team_game_features.csv"

# 训练使用的特征列（需要和 build_team_features 生成的列名对应）
# 如果你之后在 build_team_features 里增加了新的 rolling 特征，
# 可以在这里补充到这个列表中。
FEATURE_COLUMNS = [
    "roll5_PTS_FOR",
    "roll5_PTS_AGAINST",
    "roll5_point_diff",
    "roll10_PTS_FOR",
    "roll10_point_diff",
    "roll10_win_rate",
    "season_win_rate",
]


# ---------------- Helpers ---------------- #

def _now_str() -> str:
    """Return a timestamp prefix '[YYYY-MM-DD HH:MM:SS]' for logs."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def load_team_features(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load team-level game features CSV.

    Parameters
    ----------
    path : Path or str or None
        If None, loads from LOCAL_DATA_DIR / TEAM_FEATURES_FILENAME.

    Returns
    -------
    df : pd.DataFrame
    """
    if path is None:
        path = LOCAL_DATA_DIR / TEAM_FEATURES_FILENAME
    else:
        path = Path(path)

    print(f"{_now_str()} Loading team features from: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Team features file not found: {path}")

    df = pd.read_csv(path)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    return df


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X, y for training.

    Assumes:
      - Target column is 'WIN' (1 if team won, 0 otherwise).
      - FEATURE_COLUMNS exist in df.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in team features: {missing}")

    if "WIN" not in df.columns:
        raise ValueError("Team features must contain 'WIN' column as label.")

    # 只保留有完整特征的样本
    df_model = df.dropna(subset=FEATURE_COLUMNS + ["WIN"]).copy()

    print(f"{_now_str()} Training samples after dropna: {len(df_model)}")
    if len(df_model) == 0:
        raise ValueError("No valid samples for training after dropna.")

    X = df_model[FEATURE_COLUMNS].astype(float)
    y = df_model["WIN"].astype(int)

    return X, y


def build_model() -> RandomForestClassifier:
    """
    Construct the RandomForest model with reasonable defaults.
    """
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    return clf


def save_model_local(clf: RandomForestClassifier) -> Path:
    """
    Save model locally under LOCAL_DATA_DIR / model_YYYYMMDD.pkl

    Returns
    -------
    local_path : Path
    """
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"model_{date_str}.pkl"
    local_path = LOCAL_DATA_DIR / filename

    print(f"{_now_str()} Saving model locally to: {local_path}")
    with open(local_path, "wb") as f:
        joblib.dump(clf, f)

    return local_path


def upload_model_to_s3(local_path: Path) -> tuple[str, str]:
    """
    Upload local model file to S3, both dated version and 'latest'.

    Returns
    -------
    s3_uri_dated : str
    s3_uri_latest : str
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Dated key is derived from filename
    filename = local_path.name  # e.g., 'model_20251120.pkl'
    key_dated = f"{S3_PREFIX}models/{filename}"
    key_latest = f"{S3_PREFIX}models/model_latest.pkl"

    print(f"{_now_str()} Uploading to S3 (dated): s3://{S3_BUCKET}/{key_dated}")
    s3.upload_file(str(local_path), S3_BUCKET, key_dated)

    print(f"{_now_str()} Uploading to S3 (latest): s3://{S3_BUCKET}/{key_latest}")
    s3.upload_file(str(local_path), S3_BUCKET, key_latest)

    s3_uri_dated = f"s3://{S3_BUCKET}/{key_dated}"
    s3_uri_latest = f"s3://{S3_BUCKET}/{key_latest}"
    return s3_uri_dated, s3_uri_latest


# ---------------- Public API ---------------- #

def train_model(team_features_path: Path | str | None = None) -> str:
    """
    Main entry for training pipeline.

    Parameters
    ----------
    team_features_path : optional Path or str
        If None, uses LOCAL_DATA_DIR / TEAM_FEATURES_FILENAME.

    Returns
    -------
    s3_uri_latest : str
        S3 URI of the 'model_latest.pkl'.
    """
    print(f"{_now_str()} Start training...")

    # 1) Load data
    df = load_team_features(team_features_path)

    # 2) Prepare X, y
    X, y = prepare_dataset(df)

    print(f"{_now_str()} Feature shape: X={X.shape}, y={y.shape}")

    # 3) Build model
    clf = build_model()

    # 4) Fit
    clf.fit(X, y)
    print(f"{_now_str()} Model fitting done.")

    # 5) Save locally
    local_model_path = save_model_local(clf)

    # 6) Upload to S3
    s3_uri_dated, s3_uri_latest = upload_model_to_s3(local_model_path)

    print(f"{_now_str()} Training complete.")
    print(f"New model uploaded to: {s3_uri_dated}")
    print(f"Latest model at:       {s3_uri_latest}")

    return s3_uri_latest


if __name__ == "__main__":
    train_model()
