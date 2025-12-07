#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build team-level features from cleaned player logs.

Features produced:
  - Aggregate player-level logs into team-game level
  - Add opponent stats (one row per team per game)
  - Add home/away flags, rest_days, back_to_back
  - Rolling averages (5/10 game windows)
  - Season cumulative win rate (no leakage)
  - Output: team_game_features.csv (used by train_model.py)

Usage (manual):
    python build_team_features.py \
        --input data/player_logs_clean_xxx.csv \
        --output data/team_game_features.csv

Usage (EC2 via pipeline):
    from build_team_features import build_team_features
    build_team_features()  # auto-download from S3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX


# ---------------- Basic Loaders ---------------- #

def load_player_logs(input_paths: List[Path]) -> pd.DataFrame:
    """Load one or multiple cleaned player-level CSVs."""
    paths = [Path(p) for p in input_paths]
    dfs: List[pd.DataFrame] = []

    for p in paths:
        print(f"[build_team_features] Loading {p} ...")
        df = pd.read_csv(p)
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        dfs.append(df)

    if not dfs:
        raise ValueError("[build_team_features] No input files found.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[build_team_features] Total player rows loaded: {len(combined)}")
    return combined


# ---------------- Team-Level Aggregation ---------------- #

def _parse_home_away_from_matchup(matchup: str) -> tuple[int, int]:
    """
    Parse home/away flags from MATCHUP text.
      'LAL vs. BOS' → home
      'LAL @ BOS'   → away
    Returns (HOME, AWAY).
    """
    if not isinstance(matchup, str):
        return 0, 0
    matchup = matchup.strip()
    home = 1 if "vs." in matchup else 0
    away = 1 if "@" in matchup else 0
    return home, away


def aggregate_to_team_games(df_players: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player logs into one row per team per game."""
    required_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL",
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "PF", "PLUS_MINUS", "MIN",
        "FG_PCT", "FG3_PCT", "FT_PCT",
    ]
    missing = [c for c in required_cols if c not in df_players.columns]
    if missing:
        raise ValueError(f"[build_team_features] Missing columns: {missing}")

    grp_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL",
    ]

    print("[build_team_features] Aggregating to team-game level ...")
    agg = (
        df_players.groupby(grp_cols, as_index=False)
        .agg(
            PTS=("PTS", "sum"),
            REB=("REB", "sum"),
            AST=("AST", "sum"),
            STL=("STL", "sum"),
            BLK=("BLK", "sum"),
            TOV=("TOV", "sum"),
            OREB=("OREB", "sum"),
            DREB=("DREB", "sum"),
            PF=("PF", "sum"),
            PLUS_MINUS=("PLUS_MINUS", "sum"),
            MIN=("MIN", "sum"),
            FGM=("FGM", "sum"),
            FGA=("FGA", "sum"),
            FG3M=("FG3M", "sum"),
            FG3A=("FG3A", "sum"),
            FTM=("FTM", "sum"),
            FTA=("FTA", "sum"),
        )
    )

    # Recompute team shooting percentages
    agg["FG_PCT"] = agg["FGM"] / agg["FGA"].replace({0: np.nan})
    agg["FG3_PCT"] = agg["FG3M"] / agg["FG3A"].replace({0: np.nan})
    agg["FT_PCT"] = agg["FTM"] / agg["FTA"].replace({0: np.nan})

    # Home/Away flags
    home_flags, away_flags = [], []
    for m in agg["MATCHUP"]:
        h, a = _parse_home_away_from_matchup(m)
        home_flags.append(h)
        away_flags.append(a)
    agg["HOME"] = home_flags
    agg["AWAY"] = away_flags

    agg["WIN"] = (agg["WL"].astype(str).str.strip() == "W").astype(int)
    agg["PTS_FOR"] = agg["PTS"]

    print(f"[build_team_features] Team-game rows: {len(agg)}")
    return agg


# ---------------- Opponent Stats ---------------- #

def attach_opponent(team_game: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent stats (still one row per team per game)."""
    print("[build_team_features] Attaching opponent stats ...")

    base_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "HOME", "AWAY",
        "WIN", "PTS_FOR", "REB", "AST", "STL", "BLK", "TOV",
        "OREB", "DREB", "PF", "PLUS_MINUS",
        "FG_PCT", "FG3_PCT", "FT_PCT",
    ]

    missing = [c for c in base_cols if c not in team_game.columns]
    if missing:
        raise ValueError(f"[build_team_features] Missing columns before merge: {missing}")

    left = team_game[base_cols].copy()

    merged = left.merge(
        left,
        on=["SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE"],
        suffixes=("", "_OPP"),
    )

    merged = merged[merged["TEAM_ID"] != merged["TEAM_ID_OPP"]].copy()

    merged = merged.rename(
        columns={
            "TEAM_ID_OPP": "OPP_TEAM_ID",
            "TEAM_ABBREVIATION_OPP": "OPP_TEAM_ABBREVIATION",
            "WIN_OPP": "OPP_WIN",
            "PTS_FOR_OPP": "PTS_FOR_OPP",
        }
    )

    merged["PTS_AGAINST"] = merged["PTS_FOR_OPP"]
    merged["point_diff"] = merged["PTS_FOR"] - merged["PTS_AGAINST"]

    drop_cols =_
