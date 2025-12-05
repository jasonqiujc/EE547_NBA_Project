#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model_pytorch.py

功能：
  - 加载多个 feature CSV
  - 构建训练/验证集
  - 训练一个 PyTorch MLP 回归模型用于预测两队得分
  - 保存 best model 到本地
  - 上传模型到 S3:
        models/model_YYYYMMDD.pth
        models/model_latest.pth
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config_aws import AWS_REGION, S3_BUCKET, S3_PREFIX, LOCAL_DATA_DIR

# ------------------------------------------------------
# 配置
# ------------------------------------------------------
FEATURE_COLUMNS = [
    "roll5_PTS_FOR",
    "roll5_PTS_AGAINST",
    "roll5_point_diff",
    "roll10_PTS_FOR",
    "roll10_point_diff",
    "roll10_win_rate",
    "season_win_rate",
]

SCORE_COLUMNS = ["HOME_SCORE", "AWAY_SCORE"]  # 你 CSV 中存最终比分的两列名

MODEL_DIR = LOCAL_DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# Dataset
# ------------------------------------------------------
class NBADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        self.X = df[FEATURE_COLUMNS].values.astype("float32")
        self.y = df[SCORE_COLUMNS].values.astype("float32")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------------
# MLP Model
# ------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()
        layers = []

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h

        layers.append(nn.Linear(input_dim, 2))  # 输出：两队得分
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------
# Load multiple CSVs
# ------------------------------------------------------
def load_feature_files(paths: List[Union[str, Path]]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"[INFO] Loaded: {p}")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Total rows: {len(df_all)}")
    return df_all

# ------------------------------------------------------
# S3 Upload
# ------------------------------------------------------
def upload_to_s3(local_path: Path, s3_key: str):
    print(f"[S3] Uploading {local_path} → s3://{S3_BUCKET}/{s3_key}")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )
    except ClientError as e:
        print(f"[ERROR] Upload failed: {e}")
        raise

# ------------------------------------------------------
# Train Model
# ------------------------------------------------------
def train_model(feature_paths: List[Union[str, Path]]) -> str:
    # 1. 加载 CSV
    df = load_feature_files(feature_paths)

    # 2. 构建 Dataset
    dataset = NBADataset(df)
    num_samples = len(dataset)

    # 3. 划分训练 / 验证
    val_size = int(0.1 * num_samples)
    train_size = num_samples - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. 初始化模型
    model = MLPRegressor(input_dim=len(FEATURE_COLUMNS)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")

    # 时间戳模型文件名
    ts = datetime.now().strftime("%Y%m%d")
    local_ts = MODEL_DIR / f"model_{ts}.pth"
    local_latest = MODEL_DIR / "model_latest.pth"

    # 5. Train Loop
    for epoch in range(1, 501):  # 训练 500 epoch
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"[Epoch {epoch}] Train={train_loss:.4f} | Val={val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), local_ts)
            torch.save(model.state_dict(), local_latest)
            print(f"  💾 Saved best model → {local_ts}")

    # 6. 上传到 S3
    base_prefix = f"{S3_PREFIX}models/"
    key_ts = f"{base_prefix}model_{ts}.pth"
    key_latest = f"{base_prefix}model_latest.pth"

    upload_to_s3(local_ts, key_ts)
    upload_to_s3(local_latest, key_latest)

    print(f"[DONE] Best model uploaded: {key_ts}")
    print(f"[DONE] Latest model uploaded: {key_latest}")

    return key_ts


if __name__ == "__main__":
    print("train_model_pytorch.py should be called by run_daily_training.py")


