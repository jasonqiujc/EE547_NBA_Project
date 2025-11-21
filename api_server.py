#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple FastAPI server exposing two main APIs:

1) POST /train
   - Trigger the daily training pipeline (run_daily_training.main).

2) POST /predict
   - Use the latest models from S3 to predict:
       - win probability for a given matchup
       - predicted scores for both teams

Extra:
   GET /health  - health check & model status.
"""

import traceback
from datetime import datetime
from typing import Optional, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config_aws import AWS_REGION, LOCAL_DATA_DIR, S3_BUCKET, S3_PREFIX
import run_daily_training
from train_model import FEATURE_COLUMNS  # 复用训练时的特征列

# ---------------- Config ---------------- #

# 胜率模型（分类）
MODEL_KEY_LATEST = f"{S3_PREFIX}models/model_latest.pkl"
LOCAL_MODEL_PATH = LOCAL_DATA_DIR / "model_latest.pkl"

# 比分模型（回归）
SCORE_MODEL_KEY_LATEST = f"{S3_PREFIX}models/score_model_latest.pkl"
LOCAL_SCORE_MODEL_PATH = LOCAL_DATA_DIR / "score_model_latest.pkl"

# 特征文件
LOCAL_FEATURES_PATH = LOCAL_DATA_DIR / "team_game_features.csv"

app = FastAPI(title="NBA Win & Score Predictor API")

# 开发阶段先全开 CORS，方便前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 之后可以改成 ["http://localhost:3000", "https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    predicted_home_points: float
    predicted_away_points: float
    model_version: str          # 胜率模型版本
    score_model_version: str    # 得分模型版本


class HealthResponse(BaseModel):
    status: str
    win_model_loaded: bool
    score_model_loaded: bool
    win_model_path: Optional[str] = None
    score_model_path: Optional[str] = None
    win_model_version: Optional[str] = None
    score_model_version: Optional[str] = None
    last_error: Optional[str] = None


# ---------------- Model / Data Helpers ---------------- #

_s3_client = boto3.client("s3", region_name=AWS_REGION)

_model_cache = {
    "clf": None,             # 胜率模型
    "clf_version": None,
    "score_reg": None,       # 得分模型
    "score_version": None,
    "last_error": None,
}


def download_latest_model() -> None:
    """Download model_latest.pkl from S3 to LOCAL_MODEL_PATH."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[api] Downloading latest win model from s3://{S3_BUCKET}/{MODEL_KEY_LATEST}")
        _s3_client.download_file(
            Bucket=S3_BUCKET,
            Key=MODEL_KEY_LATEST,
            Filename=str(LOCAL_MODEL_PATH),
        )
    except Exception as e:
        msg = f"Failed to download latest win model: {e}"
        print("[api]", msg)
        _model_cache["last_error"] = msg
        raise


def download_latest_score_model() -> None:
    """Download score_model_latest.pkl from S3 to LOCAL_SCORE_MODEL_PATH."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[api] Downloading latest score model from s3://{S3_BUCKET}/{SCORE_MODEL_KEY_LATEST}")
        _s3_client.download_file(
            Bucket=S3_BUCKET,
            Key=SCORE_MODEL_KEY_LATEST,
            Filename=str(LOCAL_SCORE_MODEL_PATH),
        )
    except Exception as e:
        msg = f"Failed to download latest score model: {e}"
        print("[api]", msg)
        _model_cache["last_error"] = msg
        raise


def load_model_into_cache() -> None:
    """Load win model from LOCAL_MODEL_PATH into memory."""
    try:
        if not LOCAL_MODEL_PATH.exists():
            download_latest_model()
        with open(LOCAL_MODEL_PATH, "rb") as f:
            clf = joblib.load(f)
        _model_cache["clf"] = clf
        _model_cache["clf_version"] = LOCAL_MODEL_PATH.name
        _model_cache["last_error"] = None
        print("[api] Win model loaded into cache.")
    except Exception as e:
        tb = traceback.format_exc()
        print("[api] Failed to load win model:", tb)
        _model_cache["clf"] = None
        _model_cache["last_error"] = str(e)


def load_score_model_into_cache() -> None:
    """Load score model from LOCAL_SCORE_MODEL_PATH into memory."""
    try:
        if not LOCAL_SCORE_MODEL_PATH.exists():
            download_latest_score_model()
        with open(LOCAL_SCORE_MODEL_PATH, "rb") as f:
            reg = joblib.load(f)
        _model_cache["score_reg"] = reg
        _model_cache["score_version"] = LOCAL_SCORE_MODEL_PATH.name
        _model_cache["last_error"] = None
        print("[api] Score model loaded into cache.")
    except Exception as e:
        tb = traceback.format_exc()
        print("[api] Failed to load score model:", tb)
        _model_cache["score_reg"] = None
        _model_cache["last_error"] = str(e)


def ensure_model_loaded():
    """确保胜率模型在内存中。"""
    if _model_cache["clf"] is None:
        load_model_into_cache()
    if _model_cache["clf"] is None:
        raise RuntimeError(_model_cache["last_error"] or "Win model not available")


def ensure_score_model_loaded():
    """确保得分模型在内存中。"""
    if _model_cache["score_reg"] is None:
        load_score_model_into_cache()
    if _model_cache["score_reg"] is None:
        raise RuntimeError(_model_cache["last_error"] or "Score model not available")


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
    为“胜率模型”准备一行特征：
      - 使用 FEATURE_COLUMNS 构造主队减客队的差值特征。
    """
    df = df_feat.copy()
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str)

    if game_date is not None:
        cutoff = pd.to_datetime(game_date)
        df = df[df["GAME_DATE"] < cutoff]

    home_hist = df[df["TEAM_ABBREVIATION"] == home_team].sort_values("GAME_DATE")
    away_hist = df[df["TEAM_ABBREVIATION"] == away_team].sort_values("GAME_DATE")

    if home_hist.empty or away_hist.empty:
        raise ValueError("Not enough history for one of the teams (win model).")

    home_last = home_hist.iloc[-1]
    away_last = away_hist.iloc[-1]

    feat = {}
    # 这里假设 FEATURE_COLUMNS 本身就是 team-level 的 rolling 特征
    # 我们用主队特征减客队特征作为最终输入
    for col in FEATURE_COLUMNS:
        feat[col] = home_last.get(col) - away_last.get(col)

    X = pd.DataFrame([feat])
    return X


def make_feature_rows_for_score(
    df_feat: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    为“得分模型”准备输入：
      - X_home: 主队特征（一行）
      - X_away: 客队特征（一行）
    使用的列和 train_score_model 里一致（FEATURE_COLUMNS）。
    """
    df = df_feat.copy()
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str)

    if game_date is not None:
        cutoff = pd.to_datetime(game_date)
        df = df[df["GAME_DATE"] < cutoff]

    home_hist = df[df["TEAM_ABBREVIATION"] == home_team].sort_values("GAME_DATE")
    away_hist = df[df["TEAM_ABBREVIATION"] == away_team].sort_values("GAME_DATE")

    if home_hist.empty or away_hist.empty:
        raise ValueError("Not enough history for one of the teams (score model).")

    home_last = home_hist.iloc[-1]
    away_last = away_hist.iloc[-1]

    X_home = pd.DataFrame([home_last[FEATURE_COLUMNS].to_dict()])
    X_away = pd.DataFrame([away_last[FEATURE_COLUMNS].to_dict()])

    return X_home, X_away


# ---------------- FastAPI Events ---------------- #

@app.on_event("startup")
def on_startup():
    """服务启动时尝试加载一次两个模型."""
    print("[api] Startup: loading models...")
    try:
        load_model_into_cache()
    except Exception:
        pass
    try:
        load_score_model_into_cache()
    except Exception:
        pass


# ---------------- Endpoints ---------------- #

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if (_model_cache["clf"] is not None and _model_cache["score_reg"] is not None) else "degraded",
        win_model_loaded=_model_cache["clf"] is not None,
        score_model_loaded=_model_cache["score_reg"] is not None,
        win_model_path=str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else None,
        score_model_path=str(LOCAL_SCORE_MODEL_PATH) if LOCAL_SCORE_MODEL_PATH.exists() else None,
        win_model_version=_model_cache["clf_version"],
        score_model_version=_model_cache["score_version"],
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
            # 训练完成后刷新本地两个模型
            download_latest_model()
            load_model_into_cache()
            download_latest_score_model()
            load_score_model_into_cache()
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
    使用最新模型预测主队获胜概率 + 双方得分。
    """
    try:
        ensure_model_loaded()
        ensure_score_model_loaded()
        clf = _model_cache["clf"]
        score_reg = _model_cache["score_reg"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    try:
        df_feat = load_team_features()
        home = req.home_team.upper()
        away = req.away_team.upper()

        # 1) 胜率模型输入（主队减客队的差值特征）
        X_win = make_feature_row_for_matchup(
            df_feat,
            home_team=home,
            away_team=away,
            game_date=req.game_date,
        )

        # 2) 得分模型输入（主队 / 客队各一行）
        X_home_score, X_away_score = make_feature_rows_for_score(
            df_feat,
            home_team=home,
            away_team=away,
            game_date=req.game_date,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build features: {e}")

    try:
        # ---------- 1) 胜率 ----------
        proba = clf.predict_proba(X_win)[0]
        # 假设 clf.classes_ = [0, 1]，1 表示主队赢
        if list(clf.classes_).index(1) == 0:
            home_prob = float(proba[0])
        else:
            home_prob = float(proba[1])

        # ---------- 2) 得分 ----------
        pred_home_pts = float(score_reg.predict(X_home_score)[0])
        pred_away_pts = float(score_reg.predict(X_away_score)[0])

        return PredictResponse(
            home_team=home,
            away_team=away,
            game_date=req.game_date or "latest_history",
            home_win_prob=home_prob,
            away_win_prob=1.0 - home_prob,
            predicted_home_points=pred_home_pts,
            predicted_away_points=pred_away_pts,
            model_version=_model_cache["clf_version"] or "unknown",
            score_model_version=_model_cache["score_version"] or "unknown",
        )
    except Exception as e:
        tb = traceback.format_exc()
        print("[api] Prediction failed:", tb)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
