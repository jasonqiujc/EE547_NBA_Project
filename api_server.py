#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI server for NBA dashboard.

提供给前端的数据：
  - 昨天的赛程 + 比分结果
  - 未来 5 天的赛程 + 预测结果（主队/客队胜率 + 预测分差）

约定：
  - 昨天比赛结果文件：
      LOCAL_DATA_DIR / f"games_yesterday_YYYYMMDD.csv"
  - 赛程文件（今天 + 未来 4 天）：
      LOCAL_DATA_DIR / f"schedule_YYYYMMDD.csv"

预测部分：
  - 这里给出结构化代码，留了一个 TODO 给你把真实模型接进去：
      - load_model()
      - predict_for_games(schedule_df)
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
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_score: Optional[int]
    away_score: Optional[int]


class GamePrediction(BaseModel):
    game_id: str
    game_date: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    # 赛程里可能没有比分（未来比赛），所以是 Optional
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # 模型预测
    home_win_prob: float
    away_win_prob: float
    predicted_point_diff: float  # home_score - away_score


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


# -------------------- Prediction logic (to customize) -------------------- #

# def predict_for_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     给未来赛程加上预测结果。

#     输入：schedule_df，必须至少包含列：
#       - GAME_ID
#       - GAME_DATE
#       - HOME_TEAM_ABBR / HOME_TEAM_ID  （具体根据你自己的 schedule CSV 来）
#       - AWAY_TEAM_ABBR / AWAY_TEAM_ID

#     输出：在 DataFrame 上增加：
#       - home_win_prob
#       - away_win_prob
#       - predicted_point_diff

#     现在先给一个“安全的默认实现”：如果模型加载失败，就输出 0.5 概率 + 0 分差；
#     你可以在这里接入你真正的特征构造 + model.predict_proba。
#     """
#     model = load_model()

#     # 先保证必要的列存在（列名请根据你自己的 CSV 调整）
#     # 这里假设列名为：HOME_TEAM_ABBREVIATION / VISITOR_TEAM_ABBREVIATION / HOME_TEAM_ID / VISITOR_TEAM_ID
#     expected_cols = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION"]
#     missing = [c for c in expected_cols if c not in schedule_df.columns]
#     if missing:
#         raise ValueError(f"[api_server] schedule CSV is missing columns: {missing}")

#     # 如果你已经有一套 “从 team_game_features 构造未来比赛特征” 的函数，
#     # 可以在这里调用，比如：
#     #
#     #   feature_df = build_features_for_schedule(schedule_df)
#     #   proba = model.predict_proba(feature_df[FEATURE_COLUMNS])
#     #
#     # 这里先用 dummy 的方式给个结构，保证 API 能正常返回。
#     n = len(schedule_df)
#     if model is None:
#         # 没有模型 → 全部 50% / 50%，分差 0
#         schedule_df["home_win_prob"] = 0.5
#         schedule_df["away_win_prob"] = 0.5
#         schedule_df["predicted_point_diff"] = 0.0
#         return schedule_df

#     # TODO: 在这里接入真实特征构造
#     # 目前先用一个简单的随机示例（你部署前务必换成真实逻辑）
#     rng = np.random.default_rng(seed=42)
#     home_win = rng.uniform(0.35, 0.65, size=n)  # 稍微偏中间一点
#     schedule_df["home_win_prob"] = home_win
#     schedule_df["away_win_prob"] = 1.0 - home_win
#     schedule_df["predicted_point_diff"] = (home_win - 0.5) * 20.0  # 粗暴映射成分差

#     return schedule_df


def predict_for_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    给未来赛程加上“假的”预测结果（不依赖模型）：

    - home_win_prob:  主队获胜概率（0.4 ~ 0.7 之间）
    - away_win_prob:  客队获胜概率 = 1 - home_win_prob
    - predicted_point_diff:  预测分差 = home_score - away_score，约在 -15 ~ +15 之间

    为了保证每次返回的结果“稳定不变”，我们用：
      - 固定随机种子
      - 再加上 GAME_ID 的 hash 做偏移
    这样相同的比赛每次调用 API 都会得到一致的预测值。
    """

    # 先确保这些列在你的 schedule CSV 里存在，列名可以根据你的实际调整
    required_cols = [
        "GAME_ID",
        "GAME_DATE",
        "HOME_TEAM_ABBREVIATION",
        "VISITOR_TEAM_ABBREVIATION",
        "HOME_TEAM_ID",
        "VISITOR_TEAM_ID",
    ]
    missing = [c for c in required_cols if c not in schedule_df.columns]
    if missing:
        raise ValueError(f"[api_server] schedule CSV is missing columns: {missing}")

    # 确保 GAME_ID 是字符串，方便做 hash
    schedule_df["GAME_ID"] = schedule_df["GAME_ID"].astype(str)

    # 为了让同一场比赛的预测在多次请求中保持一致：
    # 使用固定种子 + GAME_ID 的 hash 作为偏移
    base_rng = np.random.default_rng(seed=2025)
    base_random = base_rng.uniform(0.0, 1.0, size=len(schedule_df))

    home_win_probs = []
    point_diffs = []

    for i, row in schedule_df.iterrows():
        game_id = row["GAME_ID"]

        # 基于 GAME_ID 生成一个小的偏移，这样不同比赛不会一模一样
        game_hash = hash(game_id)  # Python 内置 hash
        # 映射到 [0, 1)
        game_hash_float = (game_hash % 10_000) / 10_000.0

        # 基础概率在 0.4 ~ 0.7 之间波动，看起来比较合理
        # 加一点随机噪声，再截断到 [0.4, 0.7]
        raw_prob = 0.5 + (game_hash_float - 0.5) * 0.5 + (base_random[i] - 0.5) * 0.1
        home_prob = float(np.clip(raw_prob, 0.4, 0.7))

        # 分差大致在 -15~+15 之间
        # home_prob > 0.5 → 正分差，home_prob < 0.5 → 负分差
        point_diff = (home_prob - 0.5) * 30.0  # 0.2 * 30 ≈ 6 分的优势

        home_win_probs.append(home_prob)
        point_diffs.append(point_diff)

    schedule_df["home_win_prob"] = home_win_probs
    schedule_df["away_win_prob"] = 1.0 - schedule_df["home_win_prob"]
    schedule_df["predicted_point_diff"] = point_diffs

    return schedule_df


# -------------------- Endpoint helpers -------------------- #

def _load_yesterday_games_df() -> pd.DataFrame:
    y = _yesterday_et()
    fname = f"games_yesterday_{_date_to_str(y)}.csv"
    path = LOCAL_DATA_DIR / fname
    print(f"[api_server] Loading yesterday games: {path}")
    df = _read_csv_if_exists(path)

    # 这里假设你的 CSV 至少有这些列（根据你的 daily_crawl_and_upload 代码）：
    # GAME_ID, GAME_DATE, HOME_TEAM_ABBREVIATION, VISITOR_TEAM_ABBREVIATION,
    # HOME_TEAM_ID, VISITOR_TEAM_ID, PTS_HOME, PTS_AWAY
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
    """
    try:
        df = _load_yesterday_games_df()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Yesterday games file not found.")

    # 根据你的 CSV 列名映射成统一字段
    # 这里假设列名：
    #   GAME_ID, GAME_DATE,
    #   HOME_TEAM_ABBREVIATION, VISITOR_TEAM_ABBREVIATION,
    #   HOME_TEAM_ID, VISITOR_TEAM_ID,
    #   PTS_HOME, PTS_AWAY
    required_cols = [
        "GAME_ID", "GAME_DATE",
        "HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
        "PTS_HOME", "PTS_AWAY",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"games_yesterday CSV missing columns: {missing}")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    results = []
    for _, row in df.iterrows():
        results.append(
            GameResult(
                game_id=str(row["GAME_ID"]),
                game_date=row["GAME_DATE"],
                home_team=row["HOME_TEAM_ABBREVIATION"],
                away_team=row["VISITOR_TEAM_ABBREVIATION"],
                home_team_id=int(row["HOME_TEAM_ID"]),
                away_team_id=int(row["VISITOR_TEAM_ID"]),
                home_score=int(row["PTS_HOME"]),
                away_score=int(row["PTS_AWAY"]),
            )
        )
    return results


@app.get("/upcoming", response_model=List[GamePrediction])
def get_upcoming_with_predictions(days: int = 5):
    """
    返回今天开始未来 5 天的赛程 + 预测。
    可通过 query 参数调整天数，如 /upcoming?days=3
    """
    try:
        schedule_df = _load_upcoming_schedule_df(days=days)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 做预测（当前是 dummy 实现，需要你之后替换成真实模型逻辑）
    try:
        schedule_with_pred = predict_for_games(schedule_df.copy())
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 统一格式返回
    required_cols = [
        "GAME_ID", "GAME_DATE",
        "HOME_TEAM_ABBREVIATION", "VISITOR_TEAM_ABBREVIATION",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID",
        "home_win_prob", "away_win_prob", "predicted_point_diff",
    ]
    missing = [c for c in required_cols if c not in schedule_with_pred.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"schedule+prediction missing columns: {missing}")

    schedule_with_pred["GAME_DATE"] = pd.to_datetime(
        schedule_with_pred["GAME_DATE"]
    ).dt.strftime("%Y-%m-%d")

    preds: List[GamePrediction] = []
    for _, row in schedule_with_pred.iterrows():
        preds.append(
            GamePrediction(
                game_id=str(row["GAME_ID"]),
                game_date=row["GAME_DATE"],
                home_team=row["HOME_TEAM_ABBREVIATION"],
                away_team=row["VISITOR_TEAM_ABBREVIATION"],
                home_team_id=int(row["HOME_TEAM_ID"]),
                away_team_id=int(row["VISITOR_TEAM_ID"]),
                home_score=None,
                away_score=None,
                home_win_prob=float(row["home_win_prob"]),
                away_win_prob=float(row["away_win_prob"]),
                predicted_point_diff=float(row["predicted_point_diff"]),
            )
        )
    return preds

