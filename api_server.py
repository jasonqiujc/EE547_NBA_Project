#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI server for NBA dashboard.

提供给前端的数据：
  - 昨天的赛程 + 比分结果
  - 未来 N 天的赛程 + （如果模型可用）预测比分
  - 单场比赛预测：给定 game_date + home_team + away_team，返回预测比分

约定：
  - 昨天比赛结果文件：
      LOCAL_DATA_DIR / f"games_yesterday_YYYYMMDD.csv"
  - 赛程文件（今天 + 未来若干天）：
      LOCAL_DATA_DIR / f"schedule_YYYYMMDD.csv"
  - 特征文件：
      LOCAL_DATA_DIR / "team_game_features.csv"

预测部分：
  - 不再使用任何“假预测”
  - 如果模型或特征不可用：只返回赛程，预测字段为 null
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX
import boto3

# 从 train_model.py 导入模型结构 & 特征列 & 模型目录
from train_model import MLPRegressor, FEATURE_COLUMNS, MODEL_DIR
from zoneinfo import ZoneInfo


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

    # 暂时不返回真实比分（未来可以接入）
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # 模型预测的比分与分差
    pred_home_score: Optional[float] = None
    pred_away_score: Optional[float] = None
    predicted_point_diff: Optional[float] = None  # home - away


# 单场预测用的请求 / 响应模型 #

class PredictionRequest(BaseModel):
    game_date: str      # "2025-12-06"
    home_team: str      # "LAL"
    away_team: str      # "BOS"


class PredictionResponse(BaseModel):
    game_date: str
    home_team: str
    away_team: str

    pred_home_score: Optional[float] = None
    pred_away_score: Optional[float] = None
    predicted_point_diff: Optional[float] = None   # home - away


# -------------------- Helpers: dates & files -------------------- #


def _today_et() -> datetime:
    """使用洛杉矶时间（NBA 常用的太平洋时区）作为 today"""
    return datetime.now(ZoneInfo("America/Los_Angeles"))


def _yesterday_et() -> datetime:
    """昨天 = today - 1 天（同一时区下计算）"""
    return _today_et() - timedelta(days=1)


def _date_to_str(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


# -------------------- Helpers: PyTorch model loading -------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"

_model: Optional[MLPRegressor] = None
_model_loaded_from: Optional[str] = None


def load_model() -> Optional[MLPRegressor]:
    """
    加载 PyTorch 模型（懒加载，全局缓存一次）。

    约定：
      1）优先从本地 MODEL_DIR / "model_latest.pth" 读取；
      2）如果本地没有，尝试从 S3: {S3_PREFIX}models/model_latest.pth 下载再加载；
      3）如果都失败，返回 None（不会 fallback 到假预测）。
    """
    global _model, _model_loaded_from

    if _model is not None:
        return _model

    local_model_path = MODEL_DIR / "model_latest.pth"

    # 1) 尝试本地
    if local_model_path.exists():
        try:
            model = MLPRegressor(input_dim=len(FEATURE_COLUMNS)).to(device)
            state = torch.load(local_model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            _model = model
            _model_loaded_from = "local"
            print(f"[api_server] Loaded PyTorch model from local: {local_model_path}")
            return _model
        except Exception as e:
            print(f"[api_server] ERROR loading local model: {e}")

    # 2) 本地没有或加载失败，就从 S3 下载
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        key = f"{S3_PREFIX}models/model_latest.pth"
        print(f"[api_server] Downloading PyTorch model from S3: s3://{S3_BUCKET}/{key}")
        s3_client.download_file(S3_BUCKET, key, str(local_model_path))

        model = MLPRegressor(input_dim=len(FEATURE_COLUMNS)).to(device)
        state = torch.load(local_model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        _model = model
        _model_loaded_from = "s3"
        print("[api_server] PyTorch model downloaded and loaded successfully.")
        return _model
    except Exception as e:
        print(f"[api_server] WARNING: Failed to load PyTorch model: {e}")
        _model = None
        _model_loaded_from = None
        return None


# -------------------- Helpers: feature + prediction -------------------- #

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


def _load_latest_team_features() -> Optional[pd.DataFrame]:
    """
    从 team_game_features.csv 中读取每支球队的“最新一场比赛”的特征行。

    注意：这是一个简化版逻辑，用于给未来比赛构造输入特征：
      - 对于每支球队，取按 GAME_DATE 排序后的最后一行
      - 假设这行的 FEATURE_COLUMNS 能代表球队当前状态
    """
    feature_file = LOCAL_DATA_DIR / "team_game_features.csv"
    if not feature_file.exists():
        print(f"[api_server] WARNING: team_game_features.csv not found at {feature_file}")
        return None

    df = pd.read_csv(feature_file)
    if "TEAM" not in df.columns or "GAME_DATE" not in df.columns:
        print("[api_server] WARNING: team_game_features.csv missing TEAM or GAME_DATE.")
        return None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # 对每支球队取最近一场的特征
    df_sorted = df.sort_values(["TEAM", "GAME_DATE"])
    latest_df = df_sorted.groupby("TEAM").tail(1).reset_index(drop=True)

    missing = [c for c in FEATURE_COLUMNS if c not in latest_df.columns]
    if missing:
        print(f"[api_server] WARNING: team_game_features.csv missing feature columns: {missing}")
        return None

    return latest_df


def _predict_scores_for_schedule(schedule_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    对赛程表中的每场比赛，用“主队最近一场比赛特征”作为输入，预测比分。
    如果模型或特征不可用，返回 None。
    """
    model = load_model()
    if model is None:
        print("[api_server] WARNING: model not available, skip predictions.")
        return None

    latest_team_feat = _load_latest_team_features()
    if latest_team_feat is None:
        print("[api_server] WARNING: latest team features not available, skip predictions.")
        return None

    # 将 HOME_TEAM 映射到 TEAM 特征
    merged = schedule_df.merge(
        latest_team_feat[["TEAM"] + FEATURE_COLUMNS],
        left_on="HOME_TEAM",
        right_on="TEAM",
        how="left",
    )

    # 如果有球队缺特征，跳过预测
    if merged[FEATURE_COLUMNS].isnull().any().any():
        print("[api_server] WARNING: some teams lack features, skip predictions.")
        return None

    X = (
        merged[FEATURE_COLUMNS]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
        .values.astype("float32")
    )
    X_tensor = torch.from_numpy(X).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    # 将预测写回 DataFrame
    schedule_df = schedule_df.copy()
    schedule_df["pred_home_score"] = preds[:, 0]
    schedule_df["pred_away_score"] = preds[:, 1]
    schedule_df["predicted_point_diff"] = schedule_df["pred_home_score"] - schedule_df["pred_away_score"]

    return schedule_df


def _predict_single_game(home_team: str) -> Optional[dict]:
    """
    使用主队最近一场的特征行来预测一场比赛比分。
    这里只根据 HOME_TEAM，忽略对手，是简化版逻辑。
    """
    model = load_model()
    if model is None:
        print("[api_server] WARNING: model not available for single game.")
        return None

    latest_team_feat = _load_latest_team_features()
    if latest_team_feat is None:
        print("[api_server] WARNING: latest team features not available for single game.")
        return None

    home_team = home_team.upper().strip()
    row = latest_team_feat[latest_team_feat["TEAM"] == home_team]
    if row.empty:
        print(f"[api_server] WARNING: no features for team {home_team}")
        return None

    x = (
        row.iloc[0][FEATURE_COLUMNS]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
        .to_numpy()
        .astype("float32")
        .reshape(1, -1)
    )

    x_tensor = torch.from_numpy(x).to(device)
    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()

    return {
        "pred_home_score": float(preds[0, 0]),
        "pred_away_score": float(preds[0, 1]),
        "predicted_point_diff": float(preds[0, 0] - preds[0, 1]),
    }


# -------------------- API endpoints -------------------- #


@app.on_event("startup")
def _startup():
    """启动时尝试加载一次模型（失败也不影响 API 启动）。"""
    load_model()


@app.get("/health")
def health():
    """健康检查：前端或监控可以用这个看 API 是否存活。"""
    model = load_model()
    return {
        "status": "ok",
        "model_loaded_from": _model_loaded_from,
        "model_available": model is not None,
        "device": device,
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
    返回今天开始未来 N 天的赛程。
    - 如果模型和特征可用：返回预测比分
    - 否则：预测字段为 null
    """
    try:
        schedule_df = _load_upcoming_schedule_df(days=days)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    required_cols = ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]
    missing = [c for c in required_cols if c not in schedule_df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"schedule CSV missing columns: {missing}",
        )

    # 标准化
    schedule_df = schedule_df.copy()
    schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    schedule_df["HOME_TEAM"] = schedule_df["HOME_TEAM"].astype(str)
    schedule_df["AWAY_TEAM"] = schedule_df["AWAY_TEAM"].astype(str)

    schedule_df["game_id"] = (
        schedule_df["GAME_DATE"]
        + "_"
        + schedule_df["HOME_TEAM"]
        + "_vs_"
        + schedule_df["AWAY_TEAM"]
    )

    # 尝试做预测
    schedule_with_pred = _predict_scores_for_schedule(schedule_df)

    results: List[GamePrediction] = []
    if schedule_with_pred is None:
        # 没法预测：只返回赛程
        for _, row in schedule_df.iterrows():
            results.append(
                GamePrediction(
                    game_id=row["game_id"],
                    game_date=row["GAME_DATE"],
                    home_team=row["HOME_TEAM"],
                    away_team=row["AWAY_TEAM"],
                    pred_home_score=None,
                    pred_away_score=None,
                    predicted_point_diff=None,
                )
            )
    else:
        for _, row in schedule_with_pred.iterrows():
            results.append(
                GamePrediction(
                    game_id=row["game_id"],
                    game_date=row["GAME_DATE"],
                    home_team=row["HOME_TEAM"],
                    away_team=row["AWAY_TEAM"],
                    pred_home_score=float(row["pred_home_score"]),
                    pred_away_score=float(row["pred_away_score"]),
                    predicted_point_diff=float(row["predicted_point_diff"]),
                )
            )

    return results


@app.post("/predict", response_model=PredictionResponse)
def predict_game(req: PredictionRequest):
    """
    给定 game_date + home_team + away_team，返回一场比赛的预测比分。

    简化版逻辑：
      - 仅使用 HOME_TEAM 最近一场的特征来预测比分（忽略对手差异）
      - 如果模型或特征不可用，则返回预测字段为 null
    """
    try:
        game_date_norm = pd.to_datetime(req.game_date).strftime("%Y-%m-%d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid game_date: {e}")

    pred = _predict_single_game(req.home_team)
    if pred is None:
        # 无法预测：只返回规范化后的输入信息
        return PredictionResponse(
            game_date=game_date_norm,
            home_team=req.home_team.upper(),
            away_team=req.away_team.upper(),
            pred_home_score=None,
            pred_away_score=None,
            predicted_point_diff=None,
        )

    return PredictionResponse(
        game_date=game_date_norm,
        home_team=req.home_team.upper(),
        away_team=req.away_team.upper(),
        pred_home_score=pred["pred_home_score"],
        pred_away_score=pred["pred_away_score"],
        predicted_point_diff=pred["predicted_point_diff"],
    )
