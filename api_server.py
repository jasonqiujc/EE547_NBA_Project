#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple FastAPI server exposing two main APIs:

1) POST /train
   - Trigger the daily training pipeline (run_daily_training.main).

2) POST /predict
   - Use the latest model from S3 (model_latest.pkl) to predict
     win probability for a given matchup.

Extra:
   GET /health  - health check & model status.
"""

import os
import io
import joblib
import boto3
import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config_aws import (
    LOCAL_DATA_DIR,
    AWS_REGION,
    S3_BUCKET,
    S3_PREFIX,
)

# 你自己的训练脚本（我们会从这里调用 main）
import run_daily_training
from train_model import FEATURE_COLUMNS  # 假设你在 train_model 里有这个列表


# ---------------- Config ---------------- #

MODEL_KEY_LATEST = f"{S3_PREFIX}models/model_latest.pkl"
LOCAL_MODEL_PATH = LOCAL_DATA_DIR / "model_latest.pkl"
LOCAL_FEATURES_PATH = LOCAL_DATA_DIR / "team_game_features.csv"

app = FastAPI(title="NBA Win Predictor API")


# ---------------- Pydantic Models ---------------- #

class TrainResponse(BaseModel):
    status: str
    started_at: datetime


class PredictRequest(BaseModel):
    home_team: str = Field(..., description="主队缩写，如 'LAL'")
    away_team: str = Field(..., description="客队缩写，如 'BOS'")
    game_date: Optional[str] = Field(
        None, description="比赛日期 YYYY-MM-DD；为空则用最新一场特征"
    )


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    game_date: str
    home_win_prob: float
    away_win_prob: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    model_version: Optional[str] = None
    last_error: Optional[str] = None


# ---------------- Model / Data Helpers ---------------- #

_s3_client = boto3.client("s3", region_name=AWS_REGION)
_model_cache = {"clf": None, "version": None, "last_error": None}


def download_latest_model() -> None:
    """Download model_latest.pkl from S3 to LOCAL_MODEL_PATH."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[api] Downloading latest model from s3://{S3_BUCKET}/{MODEL_KEY_LATEST}")
        _s3_client.download_file(
            Bucket=S3_BUCKET,
            Key=MODEL_KEY_LATEST,
            Filename=str(LOCAL_MODEL_PATH),
        )
    except Exception as e:
        msg = f"Failed to download latest model: {e}"
        print("[api]", msg)
        _model_cache["last_error"] = msg
        raise


def load_model_into_cache() -> None:
    """Load model from LOCAL_MODEL_PATH into memory."""
    try:
        if not LOCAL_MODEL_PATH.exists():
            download_latest_model()
        with open(LOCAL_MODEL_PATH, "rb") as f:
            clf = joblib.load(f)
        _model_cache["clf"] = clf
        _model_cache["version"] = LOCAL_MODEL_PATH.name
        _model_cache["last_error"] = None
        print("[api] Model loaded into cache.")
    except Exception as e:
        tb = traceback.format_exc()
        print("[api] Failed to load model:", tb)
        _model_cache["clf"] = None
        _model_cache["last_error"] = str(e)


def ensure_model_loaded():
    """Make sure model is in memory; if失效则重新加载。"""
    if _model_cache["clf"] is None:
        load_model_into_cache()
    if _model_cache["clf"] is None:
        raise RuntimeError(_model_cache["last_error"] or "Model not available")


def load_team_features() -> pd.DataFrame:
    """Load team_game_features.csv (由 run_daily_training 生成)."""
    if not LOCAL_FEATURES_PATH.exists():
        raise RuntimeError(
            f"Team features not found at {LOCAL_FEATURES_PATH}. "
            f"Make sure run_daily_training.py has been executed."
        )
    df = pd.read_csv(LOCAL_FEATURES_PATH)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def make_feature_row_for_matchup(
    df_feat: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    简单版本：取给定日期之前，每队最近一场比赛的特征，拼成一行做输入。

    注意：这里的实现只是一个“作业级别 demo”，具体特征拼接方式
    要跟你 train_model.py 里用的保持一致。如果你在 train_model 里有一个
    build_X_for_game(...) 之类的函数，最好直接复用。
    """
    df = df_feat.copy()
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str)

    if game_date is not None:
        cutoff = pd.to_datetime(game_date)
        df = df[df["GAME_DATE"] < cutoff]

    home_hist = df[df["TEAM_ABBREVIATION"] == home_team].sort_values("GAME_DATE")
    away_hist = df[df["TEAM_ABBREVIATION"] == away_team].sort_values("GAME_DATE")

    if home_hist.empty or away_hist.empty:
        raise ValueError("Not enough history for one of the teams.")

    home_last = home_hist.iloc[-1]
    away_last = away_hist.iloc[-1]

    # 示例：简单地把两队相同名称的 rolling 特征做差值作为输入
    # FEATURE_COLUMNS 应该与你训练时的 X 列名一致
    feat = {}
    for col in FEATURE_COLUMNS:
        if col.startswith("home_"):
            base = col[len("home_") :]
            feat[col] = home_last.get(base)
        elif col.startswith("away_"):
            base = col[len("away_") :]
            feat[col] = away_last.get(base)
        else:
            # 如果你在 train_model 里用的是“主队减客队”的风格，在这里实现
            feat[col] = home_last.get(col) - away_last.get(col)

    X = pd.DataFrame([feat])
    return X


# ---------------- FastAPI Events ---------------- #

@app.on_event("startup")
def on_startup():
    """服务启动时尝试加载一次模型."""
    print("[api] Startup: loading model...")
    try:
        load_model_into_cache()
    except Exception:
        # 失败也无所谓，之后请求里再尝试
        pass


# ---------------- Endpoints ---------------- #

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if _model_cache["clf"] is not None else "degraded",
        model_loaded=_model_cache["clf"] is not None,
        model_path=str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else None,
        model_version=_model_cache["version"],
        last_error=_model_cache["last_error"],
    )


@app.post("/train", response_model=TrainResponse)
def train(background_tasks: BackgroundTasks):
    """
    手动触发一次完整的 daily training。
    使用 FastAPI 的 BackgroundTasks，让请求立即返回，训练在后台跑。
    """

    def _run_training_job():
        print("[api] Background training job started.")
        try:
            run_daily_training.main()
            # 训练完成后刷新本地模型
            download_latest_model()
            load_model_into_cache()
            print("[api] Background training job finished.")
        except Exception:
            print("[api] Training job failed:\n", traceback.format_exc())

    background_tasks.add_task(_run_training_job)

    return TrainResponse(
        status="started",
        started_at=datetime.utcnow(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    使用最新模型预测主队获胜概率。
    """
    try:
        ensure_model_loaded()
        clf = _model_cache["clf"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    try:
        df_feat = load_team_features()
        X = make_feature_row_for_matchup(
            df_feat,
            home_team=req.home_team.upper(),
            away_team=req.away_team.upper(),
            game_date=req.game_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build features: {e}")

    try:
        proba = clf.predict_proba(X)[0]
        # 假设 clf.classes_ = [0, 1]，1 表示主队赢
        if list(clf.classes_).index(1) == 0:
            home_prob = float(proba[0])
        else:
            home_prob = float(proba[1])

        return PredictResponse(
            home_team=req.home_team.upper(),
            away_team=req.away_team.upper(),
            game_date=req.game_date or "latest_history",
            home_win_prob=home_prob,
            away_win_prob=1.0 - home_prob,
            model_version=_model_cache["version"] or "unknown",
        )
    except Exception as e:
        tb = traceback.format_exc()
        print("[api] Prediction failed:", tb)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
