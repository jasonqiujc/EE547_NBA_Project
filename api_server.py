#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI server for NBA dashboard.

提供给前端的数据：
  - 昨天的赛程 + 比分结果
  - 未来 5 天的赛程 + 预测结果（主队/客队胜率 + 预测分差）
  - 单场比赛预测：给定 game_date + home_team + away_team，返回预测结果

约定：
  - 昨天比赛结果文件：
      LOCAL_DATA_DIR / f"games_yesterday_YYYYMMDD.csv"
  - 赛程文件（今天 + 未来 4 天）：
      LOCAL_DATA_DIR / f"schedule_YYYYMMDD.csv"

预测部分：
  - 目前是“假预测”（根据 hash + 随机数生成稳定的概率和分差），
    未来你可以在 fake_predict_single_game / predict_for_games 里接入真实模型。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX
import boto3


# -------------------- FastAPI app -------------------- #

app = FastAPI(title="NBA Dashboard API", version="1.0.0")

# CORS：开发阶段允许所有来源（前端才能在浏览器里调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产环境建议改成你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Pydantic models -------------------- #

class GameResult(BaseModel):
    game_date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int


class GamePrediction(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None

    home_score: Optional[int] = None
    away_score: Optional[int] = None

    home_win_prob: float
    away_win_prob: float
    predicted_point_diff: float  # home_score - away_score


# ------ 单场预测用的请求 / 响应模型 ------ #

class PredictionRequest(BaseModel):
    game_date: str      # "2025-11-24"
    home_team: str      # "LAL"
    away_team: str      # "BOS"


class PredictionResponse(BaseModel):
    game_date: str
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    predicted_point_diff: float   # home - away


# -------------------- Helpers: dates & files -------------------- #

def _today_et() -> datetime:
    """简单版：用本地时间近似 ET。如果你严格要 ET，可以用 pytz / zoneinfo。"""
    return datetime.now()


def _yesterday_et() -> datetime:
    return _today_et() - timedelta(days=1)


def _date_to_str(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


# -------------------- Helpers: model loading -------------------- #

_model = None
_model_loaded_from: Optional[str] = None


def load_model() -> Optional[object]:
    """
    加载训练好的模型（懒加载，全局缓存一次）。

    约定：
      1）优先从本地 LOCAL_DATA_DIR / "model_latest.pkl" 读取；
      2）如果本地没有，尝试从 S3: {S3_PREFIX}models/model_latest.pkl 下载再加载；
      3）如果都失败，返回 None（API 会用 Dummy 预测而不是报 500）。
    """
    global _model, _model_loaded_from

    if _model is not None:
        return _model

    local_model_path = LOCAL_DATA_DIR / "model_latest.pkl"

    # 1) 尝试本地
    if local_model_path.exists():
        _model = joblib.load(local_model_path)
        _model_loaded_from = "local"
        print(f"[api_server] Loaded model from local: {local_model_path}")
        return _model

    # 2) 尝试从 S3 下载
    try:
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        key = f"{S3_PREFIX}models/model_latest.pkl"
        print(f"[api_server] Downloading model from S3: s3://{S3_BUCKET}/{key}")
        s3_client.download_file(S3_BUCKET, key, str(local_model_path))
        _model = joblib.load(local_model_path)
        _model_loaded_from = "s3"
        print("[api_server] Model downloaded and loaded successfully.")
        return _model
    except Exception as e:
        print(f"[api_server] WARNING: Failed to load model: {e}")
        _model = None
        _model_loaded_from = None
        return None


# -------------------- Prediction logic (fake for now) -------------------- #

def predict_for_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据简单赛程表（只有 GAME_DATE / HOME_TEAM / AWAY_TEAM）生成“假的预测”。

    必须包含的列：
      - GAME_DATE
      - HOME_TEAM
      - AWAY_TEAM

    输出会多出：
      - game_id（用日期+对阵拼出来的一个字符串）
      - home_win_prob
      - away_win_prob
      - predicted_point_diff
    """
    required_cols = ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]
    missing = [c for c in required_cols if c not in schedule_df.columns]
    if missing:
        raise ValueError(f"[api_server] schedule CSV is missing columns: {missing}")

    # 统一类型
    schedule_df = schedule_df.copy()
    schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    schedule_df["HOME_TEAM"] = schedule_df["HOME_TEAM"].astype(str)
    schedule_df["AWAY_TEAM"] = schedule_df["AWAY_TEAM"].astype(str)

    # 生成一个稳定的“伪 game_id”，保证同一场比赛每次请求都一样
    schedule_df["game_id"] = (
        schedule_df["GAME_DATE"]
        + "_"
        + schedule_df["HOME_TEAM"]
        + "_vs_"
        + schedule_df["AWAY_TEAM"]
    )

    # 下面是完全“假的预测”，但看起来比较合理
    base_rng = np.random.default_rng(seed=2025)
    base_random = base_rng.uniform(0.0, 1.0, size=len(schedule_df))

    home_win_probs = []
    point_diffs = []

    for i, row in schedule_df.iterrows():
        game_key = row["game_id"]
        # 用 hash 让每场比赛预测稳定不变
        game_hash = hash(game_key)
        game_hash_float = (game_hash % 10_000) / 10_000.0  # 映射到 [0,1)

        raw_prob = 0.5 + (game_hash_float - 0.5) * 0.5 + (base_random[i] - 0.5) * 0.1
        home_prob = float(np.clip(raw_prob, 0.4, 0.7))  # 限制在 0.4~0.7 之间

        point_diff = (home_prob - 0.5) * 30.0  # 映射成大约 -15~+15 分

        home_win_probs.append(home_prob)
        point_diffs.append(point_diff)

    schedule_df["home_win_prob"] = home_win_probs
    schedule_df["away_win_prob"] = 1.0 - schedule_df["home_win_prob"]
    schedule_df["predicted_point_diff"] = point_diffs

    return schedule_df


def fake_predict_single_game(game_date: str, home_team: str, away_team: str):
    """
    给单场比赛生成“假的但稳定的”预测。

    返回:
      home_win_prob, away_win_prob, predicted_point_diff
    """
    # 标准化一下输入
    game_date = pd.to_datetime(game_date).strftime("%Y-%m-%d")
    home_team = str(home_team).upper().strip()
    away_team = str(away_team).upper().strip()

    game_key = f"{game_date}_{home_team}_vs_{away_team}"

    # 用 hash 让同一场比赛多次调用结果一致
    game_hash = hash(game_key)
    game_hash_float = (game_hash % 10_000) / 10_000.0  # 映射到 [0,1)

    raw_prob = 0.5 + (game_hash_float - 0.5) * 0.5
    home_prob = float(np.clip(raw_prob, 0.4, 0.7))
    away_prob = 1.0 - home_prob

    point_diff = float((home_prob - 0.5) * 30.0)   # -15 ~ +15

    return home_prob, away_prob, point_diff


# -------------------- Endpoint helpers -------------------- #

def _load_yesterday_games_df() -> pd.DataFrame:
    y = _yesterday_et()
    fname = f"games_yesterday_{_date_to_str(y)}.csv"
    path = LOCAL_DATA_DIR / fname
    print(f"[api_server] Loading yesterday games: {path}")
    df = _read_csv_if_exists(path)
    return df


def _load_upcoming_schedule_df(days: int = 5) -> pd.DataFrame:
    """
    加载今天起往后若干天的赛程（每天一个 schedule_YYYYMMDD.csv），合并成一个 DataFrame。
    """
    today = _today_et().date()
    dfs = []
    for offset in range(days):
        d = today + timedelta(days=offset)
        fname = f"schedule_{d.strftime('%Y%m%d')}.csv"
        path = LOCAL_DATA_DIR / fname
        try:
            print(f"[api_server] Loading schedule: {path}")
            df_day = _read_csv_if_exists(path)
            dfs.append(df_day)
        except FileNotFoundError:
            print(f"[api_server] WARNING: schedule file not found: {path}, skip.")
            continue

    if not dfs:
        raise FileNotFoundError("[api_server] No schedule CSV files found for upcoming days.")

    schedule_df = pd.concat(dfs, ignore_index=True)
    return schedule_df


# -------------------- API endpoints -------------------- #


@app.get("/health")
def health():
    """健康检查：前端或监控可以用这个看 API 是否存活。"""
    model = load_model()
    return {
        "status": "ok",
        "model_loaded_from": _model_loaded_from,
    }


@app.get("/yesterday", response_model=List[GameResult])
def get_yesterday_games():
    """
    返回昨天的赛程 + 比赛结果。

    适配当前的 games_yesterday_YYYYMMDD.csv 结构：
      GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_SCORE, AWAY_SCORE
    """
    try:
        df = _load_yesterday_games_df()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Yesterday games file not found.")

    required_cols = [
        "GAME_DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_SCORE",
        "AWAY_SCORE",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"games_yesterday CSV missing columns: {missing}",
        )

    # 统一日期格式
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    results: List[GameResult] = []
    for _, row in df.iterrows():
        results.append(
            GameResult(
                game_date=row["GAME_DATE"],
                home_team=str(row["HOME_TEAM"]),
                away_team=str(row["AWAY_TEAM"]),
                home_score=int(row["HOME_SCORE"]),
                away_score=int(row["AWAY_SCORE"]),
            )
        )

    return results


@app.get("/upcoming", response_model=List[GamePrediction])
def get_upcoming_with_predictions(days: int = 5):
    """
    返回今天开始未来 N 天的赛程 + 预测。
    赛程 CSV 只需要包含：GAME_DATE / HOME_TEAM / AWAY_TEAM
    """
    try:
        schedule_df = _load_upcoming_schedule_df(days=days)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 做预测（假的）
    try:
        schedule_with_pred = predict_for_games(schedule_df.copy())
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    required_cols = [
        "game_id",
        "GAME_DATE",
        "HOME_TEAM",
        "AWAY_TEAM",
        "home_win_prob",
        "away_win_prob",
        "predicted_point_diff",
    ]
    missing = [c for c in required_cols if c not in schedule_with_pred.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"schedule+prediction missing columns: {missing}")

    preds: List[GamePrediction] = []
    for _, row in schedule_with_pred.iterrows():
        preds.append(
            GamePrediction(
                game_id=str(row["game_id"]),
                game_date=row["GAME_DATE"],
                home_team=row["HOME_TEAM"],
                away_team=row["AWAY_TEAM"],
                home_team_id=None,
                away_team_id=None,
                home_score=None,
                away_score=None,
                home_win_prob=float(row["home_win_prob"]),
                away_win_prob=float(row["away_win_prob"]),
                predicted_point_diff=float(row["predicted_point_diff"]),
            )
        )
    return preds


@app.post("/predict", response_model=PredictionResponse)
def predict_game(req: PredictionRequest):
    """
    给定 game_date + home_team + away_team，返回一场比赛的预测结果。

    示例请求 JSON:
    {
      "game_date": "2025-11-24",
      "home_team": "LAL",
      "away_team": "BOS"
    }
    """
    try:
        home_prob, away_prob, diff = fake_predict_single_game(
            req.game_date, req.home_team, req.away_team
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(
        game_date=pd.to_datetime(req.game_date).strftime("%Y-%m-%d"),
        home_team=req.home_team.upper(),
        away_team=req.away_team.upper(),
        home_win_prob=home_prob,
        away_win_prob=away_prob,
        predicted_point_diff=diff,
    )



