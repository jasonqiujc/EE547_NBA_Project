#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI server for NBA dashboard.

Provides:
  - Yesterday’s games with results
  - Upcoming N-day schedule with model predictions (if available)
  - Single-game score prediction

Files expected:
  LOCAL_DATA_DIR / games_yesterday_YYYYMMDD.csv
  LOCAL_DATA_DIR / schedule_YYYYMMDD.csv
  LOCAL_DATA_DIR / team_game_features.csv

Prediction:
  - No fake predictions; returns null if model/features missing.
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

from train_model import MLPRegressor, FEATURE_COLUMNS, MODEL_DIR
from zoneinfo import ZoneInfo


# -------------------- FastAPI -------------------- #

app = FastAPI(title="NBA Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Schemas -------------------- #

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
    pred_home_score: Optional[float] = None
    pred_away_score: Optional[float] = None
    predicted_point_diff: Optional[float] = None


class PredictionRequest(BaseModel):
    game_date: str
    home_team: str
    away_team: str


class PredictionResponse(BaseModel):
    game_date: str
    home_team: str
    away_team: str
    pred_home_score: Optional[float] = None
    pred_away_score: Optional[float] = None
    predicted_point_diff: Optional[float] = None


# -------------------- Date / File Helpers -------------------- #

def _today_et() -> datetime:
    return datetime.now(ZoneInfo("America/Los_Angeles"))


def _yesterday_et() -> datetime:
    return _today_et() - timedelta(days=1)


def _date_to_str(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


# -------------------- Model Loading -------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
_model: Optional[MLPRegressor] = None
_model_loaded_from: Optional[str] = None


def load_model() -> Optional[MLPRegressor]:
    """
    Lazy-load PyTorch model.
    Priority:
      1) Local model_latest.pth
      2) Download from S3
    """
    global _model, _model_loaded_from
    if _model is not None:
        return _model

    local_model_path = MODEL_DIR / "model_latest.pth"

    # Try local model
    if local_model_path.exists():
        try:
            model = MLPRegressor(input_dim=len(FEATURE_COLUMNS)).to(device)
            state = torch.load(local_model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            _model = model
            _model_loaded_from = "local"
            print(f"[api_server] Loaded local model: {local_model_path}")
            return model
        except Exception as e:
            print(f"[api_server] ERROR loading local model: {e}")

    # Try downloading from S3
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        key = f"{S3_PREFIX}models/model_latest.pth"
        print(f"[api_server] Downloading model from s3://{S3_BUCKET}/{key}")
        s3_client.download_file(S3_BUCKET, key, str(local_model_path))

        model = MLPRegressor(input_dim=len(FEATURE_COLUMNS)).to(device)
        state = torch.load(local_model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        _model = model
        _model_loaded_from = "s3"
        print("[api_server] Model downloaded and loaded.")
        return model
    except Exception as e:
        print(f"[api_server] WARNING: Model load failed: {e}")
        return None


# -------------------- Feature Loading / Prediction -------------------- #

def _load_yesterday_games_df() -> pd.DataFrame:
    y = _yesterday_et()
    path = LOCAL_DATA_DIR / f"games_yesterday_{_date_to_str(y)}.csv"
    print(f"[api_server] Loading yesterday games: {path}")
    return _read_csv_if_exists(path)


def _load_upcoming_schedule_df(days: int = 5) -> pd.DataFrame:
    """
    Load schedule_YYYYMMDD.csv for today + N days.
    """
    today = _today_et().date()
    dfs = []

    for offset in range(days):
        d = today + timedelta(days=offset)
        path = LOCAL_DATA_DIR / f"schedule_{d.strftime('%Y%m%d')}.csv"
        try:
            print(f"[api_server] Loading schedule: {path}")
            dfs.append(_read_csv_if_exists(path))
        except FileNotFoundError:
            print(f"[api_server] WARNING: Missing schedule: {path}")

    if not dfs:
        raise FileNotFoundError("No schedule files found.")

    return pd.concat(dfs, ignore_index=True)


def _load_latest_team_features(n_last_games: int = 5) -> Optional[pd.DataFrame]:
    """
    Compute average features of last n games for each team.
    """
    fpath = LOCAL_DATA_DIR / "team_game_features.csv"
    if not fpath.exists():
        print("[api_server] WARNING: team_game_features.csv missing.")
        return None

    df = pd.read_csv(fpath)

    # Auto-detect columns
    team_col = next((c for c in ["TEAM_ABBREVIATION", "TEAM", "TEAM_NAME"] if c in df.columns), None)
    if team_col is None:
        print("[api_server] ERROR: No team column.")
        return None

    if "GAME_DATE" not in df.columns:
        print("[api_server] ERROR: GAME_DATE missing.")
        return None

    df = df.rename(columns={team_col: "TEAM"})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"[api_server] ERROR: Missing feature columns: {missing}")
        return None

    df = df.sort_values(["TEAM", "GAME_DATE"])

    latest = (
        df.groupby("TEAM")
          .tail(n_last_games)
          .groupby("TEAM")[FEATURE_COLUMNS]
          .mean()
          .reset_index()
    )
    return latest


def _predict_scores_for_schedule(schedule_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Predict scores using the home team’s latest features.
    """
    model = load_model()
    if model is None:
        return None

    latest_feat = _load_latest_team_features()
    if latest_feat is None:
        return None

    merged = schedule_df.merge(
        latest_feat[["TEAM"] + FEATURE_COLUMNS],
        left_on="HOME_TEAM",
        right_on="TEAM",
        how="left",
    )

    if merged[FEATURE_COLUMNS].isnull().any().any():
        print("[api_server] WARNING: Missing features.")
        return None

    X = merged[FEATURE_COLUMNS].fillna(0).values.astype("float32")
    X_tensor = torch.from_numpy(X).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    schedule_df = schedule_df.copy()
    schedule_df["pred_home_score"] = preds[:, 0]
    schedule_df["pred_away_score"] = preds[:, 1]
    schedule_df["predicted_point_diff"] = preds[:, 0] - preds[:, 1]

    return schedule_df


def _predict_single_game(home_team: str) -> Optional[dict]:
    """
    Predict a single game using home team’s latest features only.
    """
    model = load_model()
    if model is None:
        return None

    latest_feat = _load_latest_team_features()
    if latest_feat is None:
        return None

    home_team = home_team.upper().strip()
    row = latest_feat[latest_feat["TEAM"] == home_team]
    if row.empty:
        return None

    x = row.iloc[0][FEATURE_COLUMNS].fillna(0).values.astype("float32").reshape(1, -1)
    x_tensor = torch.from_numpy(x).to(device)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()

    return {
        "pred_home_score": float(pred[0, 0]),
        "pred_away_score": float(pred[0, 1]),
        "predicted_point_diff": float(pred[0, 0] - pred[0, 1]),
    }


# -------------------- API Endpoints -------------------- #

@app.on_event("startup")
def _startup():
    load_model()


@app.get("/health")
def health():
    model = load_model()
    return {
        "status": "ok",
        "model_loaded_from": _model_loaded_from,
        "model_available": model is not None,
        "device": device,
    }


@app.get("/yesterday", response_model=List[GameResult])
def get_yesterday_games():
    df = _load_yesterday_games_df()

    required = ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_SCORE", "AWAY_SCORE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns: {missing}")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    return [
        GameResult(
            game_date=row["GAME_DATE"],
            home_team=str(row["HOME_TEAM"]),
            away_team=str(row["AWAY_TEAM"]),
            home_score=int(row["HOME_SCORE"]),
            away_score=int(row["AWAY_SCORE"]),
        )
        for _, row in df.iterrows()
    ]


@app.get("/upcoming", response_model=List[GamePrediction])
def get_upcoming_with_predictions(days: int = 5):
    schedule_df = _load_upcoming_schedule_df(days)

    required = ["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]
    missing = [c for c in required if c not in schedule_df.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing schedule columns: {missing}")

    schedule_df = schedule_df.copy()
    schedule_df["GAME_DATE"] = pd.to_datetime(schedule_df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    schedule_df["game_id"] = (
        schedule_df["GAME_DATE"] + "_" + schedule_df["HOME_TEAM"] + "_vs_" + schedule_df["AWAY_TEAM"]
    )

    schedule_pred = _predict_scores_for_schedule(schedule_df)

    if schedule_pred is None:
        # no prediction
        return [
            GamePrediction(
                game_id=row["game_id"],
                game_date=row["GAME_DATE"],
                home_team=row["HOME_TEAM"],
                away_team=row["AWAY_TEAM"],
            )
            for _, row in schedule_df.iterrows()
        ]

    return [
        GamePrediction(
            game_id=row["game_id"],
            game_date=row["GAME_DATE"],
            home_team=row["HOME_TEAM"],
            away_team=row["AWAY_TEAM"],
            pred_home_score=float(row["pred_home_score"]),
            pred_away_score=float(row["pred_away_score"]),
            predicted_point_diff=float(row["predicted_point_diff"]),
        )
        for _, row in schedule_pred.iterrows()
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict_game(req: PredictionRequest):
    try:
        game_date_norm = pd.to_datetime(req.game_date).strftime("%Y-%m-%d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date: {e}")

    pred = _predict_single_game(req.home_team)

    if pred is None:
        return PredictionResponse(
            game_date=game_date_norm,
            home_team=req.home_team.upper(),
            away_team=req.away_team.upper(),
        )

    return PredictionResponse(
        game_date=game_date_norm,
        home_team=req.home_team.upper(),
        away_team=req.away_team.upper(),
        pred_home_score=pred["pred_home_score"],
        pred_away_score=pred["pred_away_score"],
        predicted_point_diff=pred["predicted_point_diff"],
    )
