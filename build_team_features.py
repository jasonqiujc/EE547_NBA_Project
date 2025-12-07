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

    # 自连接后会有两行自己 vs 自己，我们只保留“对手那一行”
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

    drop_cols = [c for c in merged.columns if c.endswith("_OPP") and c not in (
        "OPP_TEAM_ID", "OPP_TEAM_ABBREVIATION", "OPP_WIN", "PTS_FOR_OPP"
    )]
    merged = merged.drop(columns=drop_cols)

    print(f"[build_team_features] Rows after opponent merge: {len(merged)}")
    return merged


# ---------------- Schedule Features ---------------- #

def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rest_days and back-to-back indicator."""
    print("[build_team_features] Computing rest-day features ...")
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    grp = df.groupby(["SEASON", "TEAM_ID"], group_keys=False)

    prev_date = grp["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - prev_date).dt.days
    df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)

    return df


# ---------------- Rolling & Season Features ---------------- #

def add_rolling_and_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features (5/10 games) and season-to-date win rate."""
    print("[build_team_features] Computing rolling & season features ...")
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    df["PTS_FOR"] = df["PTS_FOR"].astype(float)
    df["PTS_AGAINST"] = df["PTS_AGAINST"].astype(float)
    df["point_diff"] = df["point_diff"].astype(float)
    df["WIN"] = df["WIN"].astype(int)

    grp_team = df.groupby("TEAM_ID", group_keys=False)

    # Rolling averages
    for col in ["PTS_FOR", "PTS_AGAINST", "point_diff"]:
        df[f"roll5_{col}"] = grp_team[col].apply(lambda s: s.rolling(5, 1).mean()).shift(1)

    for col in ["PTS_FOR", "point_diff"]:
        df[f"roll10_{col}"] = grp_team[col].apply(lambda s: s.rolling(10, 1).mean()).shift(1)

    df["roll10_win_rate"] = grp_team["WIN"].apply(lambda s: s.rolling(10, 1).mean()).shift(1)

    # Season cumulative win rate (no leakage)
    grp_season_team = df.groupby(["SEASON", "TEAM_ID"], group_keys=False)
    df["season_games_played"] = grp_season_team.cumcount()
    cumsum_wins = grp_season_team["WIN"].cumsum()
    df["season_wins_so_far"] = cumsum_wins - df["WIN"]

    df["season_win_rate"] = np.where(
        df["season_games_played"] > 0,
        df["season_wins_so_far"] / df["season_games_played"],
        np.nan,
    )

    return df


# ---------------- S3 Helpers ---------------- #

def _download_raw_from_s3() -> List[Path]:
    """Download all *player-level* CSVs from S3 raw/ and return paths."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = LOCAL_DATA_DIR / "raw_from_s3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = f"{S3_PREFIX}raw/"

    print(f"[build_team_features] Listing S3: s3://{S3_BUCKET}/{prefix}")
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

    contents = resp.get("Contents", [])
    csv_paths: List[Path] = []

    for obj in contents:
        key = obj["Key"]
        if not key.endswith(".csv"):
            continue

        filename = key.split("/")[-1]

        # ✅ 只保留球员级日志：player_logs_ 开头
        #   - player_logs_clean_all_...
        #   - player_logs_clean_season_...
        #   - player_logs_daily_...
        if not filename.startswith("player_logs_"):
            print(f"[build_team_features] Skip non-player file: {filename}")
            continue

        local_path = raw_dir / filename

        if not local_path.exists():
            print(f"[build_team_features] Downloading {key}")
            s3.download_file(S3_BUCKET, key, str(local_path))
        else:
            print(f"[build_team_features] Using local: {local_path}")

        csv_paths.append(local_path)

    if not csv_paths:
        raise RuntimeError("[build_team_features] No player_logs_*.csv in S3 raw/")

    print(f"[build_team_features] Ready {len(csv_paths)} CSVs")
    return csv_paths



def _upload_team_features_to_s3(local_path: Path) -> None:
    """
    Upload the generated team_game_features.csv to S3.

    上传路径：
        s3://{S3_BUCKET}/{S3_PREFIX}raw/team_game_features.csv
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"{S3_PREFIX}raw/team_game_features.csv"
    print(f"[build_team_features] Uploading team features to s3://{S3_BUCKET}/{key}")
    s3.upload_file(str(local_path), S3_BUCKET, key)
    print("[build_team_features] Upload done.")


# ---------------- Public API ---------------- #

def build_team_features(
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> List[Path]:
    """Main function to build team-level features."""
    if input_paths is None:
        csv_paths = _download_raw_from_s3()
    else:
        csv_paths = [Path(p) for p in input_paths]

    df_players = load_player_logs(csv_paths)
    team_game = aggregate_to_team_games(df_players)
    team_with_opp = attach_opponent(team_game)
    team_with_sched = add_schedule_features(team_with_opp)
    team_features = add_rolling_and_season_features(team_with_sched)

    out_path = LOCAL_DATA_DIR / "team_game_features.csv" if output_path is None else Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    team_features.to_csv(out_path, index=False)

    print("\n✅ [build_team_features] Saved to:", out_path)
    print(f"Rows: {len(team_features)}; Cols: {len(team_features.columns)}")

    # 上传到 S3
    try:
        _upload_team_features_to_s3(out_path)
    except Exception as e:
        print(f"[build_team_features] WARNING: Failed to upload team features to S3: {e}")

    return [out_path]


# ---------------- CLI ---------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Build team-level features.")
    p.add_argument("--input", "-i", nargs="+", required=True, help="Input cleaned player CSVs.")
    p.add_argument("--output", "-o", required=True, help="Output CSV path.")
    return p.parse_args()


def main():
    args = parse_args()
    build_team_features(input_paths=args.input, output_path=args.output)


if __name__ == "__main__":
    main()

