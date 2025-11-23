#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch and CLEAN player game logs for:
  - The last THREE full NBA seasons
  - The CURRENT season up to a cutoff date (默认 = 昨天)

Produces:
  - One cleaned CSV per full season
  - One cleaned CSV for current partial season
  - One merged CSV for all seasons combined

Designed for automated pipelines using boto3 + S3.
"""

from __future__ import annotations

import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from config_aws import LOCAL_DATA_DIR

# ---------------------- Config ----------------------
INCLUDE_PLAYOFFS = True
SLEEP_BETWEEN_CALLS = 1.5
TIMEOUT = 30

# 只作为文件名前缀，不带目录
OUTPUT_PREFIX = "player_logs_clean"

# 截止到“昨天”的数据
CUTOFF_DATE = date.today() - timedelta(days=1)

# Only keep columns needed for modeling
KEEP_COLS = [
    "SEASON_ID",
    "SEASON_TYPE",
    "GAME_ID",
    "GAME_DATE",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "PLAYER_ID",
    "PLAYER_NAME",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
]


# ---------------- Season helpers -------------------


def current_nba_season_start_year(today: date) -> int:
    """
    Determine NBA season start year based on date.

    Season starts in October.
    If month >= 10 → season_start_year = this year
    Else → season_start_year = last year
    """
    if today.month >= 10:
        return today.year
    else:
        return today.year - 1


def format_season(start_year: int) -> str:
    """Convert e.g., 2022 → '2022-23'"""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def last_n_full_seasons(n: int, today: date) -> List[str]:
    """Return last N *full* seasons (not including current)."""
    cur_start = current_nba_season_start_year(today)
    seasons: List[str] = []
    for i in range(1, n + 1):
        seasons.append(format_season(cur_start - i))
    return seasons


# ---------------- Fetch helpers -------------------


def fetch_logs_for_season(
    season: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Fetch logs for a single season.

    If date_from/date_to are provided, they will be passed to the API.
    """
    frames: List[pd.DataFrame] = []

    season_types = ["Regular Season", "Playoffs"] if INCLUDE_PLAYOFFS else ["Regular Season"]

    for stype in season_types:
        print(f"  - Fetching {season} / {stype} ...")
        resp = leaguegamelog.LeagueGameLog(
            player_or_team_abbreviation="P",
            season=season,
            season_type_all_star=stype,
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            timeout=TIMEOUT,
        )
        df = resp.get_data_frames()[0]
        df["SEASON_STR"] = season
        df["SEASON_TYPE"] = stype
        frames.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Select required columns, normalize GAME_DATE, add SEASON & shooting %."""
    if df.empty:
        return df

    # 只保留需要的原始列
    cols = [c for c in KEEP_COLS if c in df.columns]
    clean = df[cols].copy()

    # 规范日期类型
    if "GAME_DATE" in clean.columns:
        clean["GAME_DATE"] = pd.to_datetime(clean["GAME_DATE"])

    # 从 SEASON_ID 生成更好看的 SEASON（如 '2022-23'）
    if "SEASON_ID" in clean.columns:
        start_year = clean["SEASON_ID"].astype(str).str[:4].astype(int)
        clean["SEASON"] = (
            start_year.astype(str)
            + "-"
            + (start_year + 1).astype(str).str[-2:]
        )

    # 计算命中率列 FG_PCT / FG3_PCT / FT_PCT
    if "FGM" in clean.columns and "FGA" in clean.columns:
        clean["FG_PCT"] = clean["FGM"] / clean["FGA"].replace({0: np.nan})
    if "FG3M" in clean.columns and "FG3A" in clean.columns:
        clean["FG3_PCT"] = clean["FG3M"] / clean["FG3A"].replace({0: np.nan})
    if "FTM" in clean.columns and "FTA" in clean.columns:
        clean["FT_PCT"] = clean["FTM"] / clean["FTA"].replace({0: np.nan})

    return clean


# ---------------- Public API: build_datasets -------------------


def build_datasets() -> List[Path]:
    """
    Full pipeline:
      Fetch → Clean → Save CSVs
    Returns list of generated file paths.
    """
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = CUTOFF_DATE  # 默认为“昨天”
    full_seasons = last_n_full_seasons(3, today)
    current_season = format_season(current_nba_season_start_year(today))

    print(f"Full seasons: {full_seasons}")
    print(f"Current season: {current_season}, cutoff={today.isoformat()}")

    all_frames: List[pd.DataFrame] = []
    output_paths: List[Path] = []

    # --- Full seasons (完整三季) ---
    for s in full_seasons:
        raw = fetch_logs_for_season(s)
        if raw.empty:
            print(f"[WARN] No data for season {s}")
            continue

        clean = clean_logs(raw)
        out_path = LOCAL_DATA_DIR / f"{OUTPUT_PREFIX}_season_{s.replace('-', '')}.csv"
        clean.to_csv(out_path, index=False)
        print(f"[Saved] Season {s}: {out_path}  rows={len(clean)}")

        all_frames.append(clean)
        output_paths.append(out_path)

    # --- Current season partial (本赛季截至 cutoff=昨天) ---
    # 简单稳妥：抓整季，然后在本地用 GAME_DATE 截断
    cur_raw = fetch_logs_for_season(current_season)

    if not cur_raw.empty:
        cur_clean = clean_logs(cur_raw)

        # 只保留 GAME_DATE <= cutoff（也就是昨天）
        if "GAME_DATE" in cur_clean.columns:
            cur_clean = cur_clean[cur_clean["GAME_DATE"].dt.date <= today]

        out_cur = LOCAL_DATA_DIR / (
            f"{OUTPUT_PREFIX}_current_{current_season.replace('-', '')}"
            f"_to_{today.strftime('%Y%m%d')}.csv"
        )
        cur_clean.to_csv(out_cur, index=False)
        print(f"[Saved] Current season: {out_cur}  rows={len(cur_clean)}")

        all_frames.append(cur_clean)
        output_paths.append(out_cur)
    else:
        print(f"[WARN] No data for current season {current_season}")

    # --- Merge all ---
    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        out_all = LOCAL_DATA_DIR / (
            f"{OUTPUT_PREFIX}_all_3seasons_plus_current_to_{today.strftime('%Y%m%d')}.csv"
        )
        merged.to_csv(out_all, index=False)
        print(f"[Saved] Merged all seasons: {out_all}  rows={len(merged)}")

        output_paths.append(out_all)
    else:
        print("[WARN] No data fetched at all.")

    return output_paths


if __name__ == "__main__":
    paths = build_datasets()
    print("\nGenerated files:")
    for p in paths:
        print(" -", p)
