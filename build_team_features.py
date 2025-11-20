#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build team-level game features from cleaned player-level logs.

两种用法：

1）本地 / 手动（命令行）：
    python build_team_features.py \
        --input data/player_logs_clean_all_3seasons_plus_current_to_20251119.csv \
        --output data/team_game_features.csv

2）在 EC2 上由 run_daily_training.py 调用（推荐）：
    from build_team_features import build_team_features
    feature_paths = build_team_features()

    - 不传参数时：
        自动从 S3 的 raw/ 目录下载所有 cleaned player CSV，
        构建特征，保存到 data/team_game_features.csv，
        返回 [Path("data/team_game_features.csv")]。
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import boto3

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX


# ---------------- Helpers ---------------- #

def load_player_logs(input_paths: List[Path]) -> pd.DataFrame:
    """Load one or multiple cleaned player-level CSVs into a single DataFrame."""
    paths = [Path(p) for p in input_paths]
    dfs = []
    for p in paths:
        print(f"[build_team_features] Loading {p} ...")
        df = pd.read_csv(p)
        # ensure GAME_DATE is datetime
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        dfs.append(df)
    if not dfs:
        raise ValueError("No input files loaded.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[build_team_features] Loaded total rows: {len(combined)}")
    return combined


def aggregate_to_team_games(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level logs to team-level single-game stats.

    Input columns expected (from cleaned logs):
        SEASON, SEASON_TYPE, GAME_ID, GAME_DATE,
        TEAM_ID, TEAM_ABBREVIATION, WL,
        PTS, REB, AST, STL, BLK, TOV,
        FG_PCT, FG3_PCT, FT_PCT,
        PLUS_MINUS, MIN

    Output: each row = one team in one game.
    """
    required_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "WL",
        "PTS", "REB", "AST", "STL", "BLK", "TOV",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "PLUS_MINUS", "MIN",
    ]
    missing = [c for c in required_cols if c not in df_players.columns]
    if missing:
        raise ValueError(f"Missing required columns in player logs: {missing}")

    # group by team-game; sum counting stats, mean efficiencies
    grp_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "WL",
    ]

    print("[build_team_features] Aggregating player logs to team-game level ...")
    team_game = (
        df_players.groupby(grp_cols, as_index=False)
        .agg(
            team_pts=("PTS", "sum"),
            team_reb=("REB", "sum"),
            team_ast=("AST", "sum"),
            team_stl=("STL", "sum"),
            team_blk=("BLK", "sum"),
            team_tov=("TOV", "sum"),
            team_plusminus=("PLUS_MINUS", "sum"),
            team_min=("MIN", "sum"),
            fg_pct=("FG_PCT", "mean"),
            fg3_pct=("FG3_PCT", "mean"),
            ft_pct=("FT_PCT", "mean"),
        )
    )

    # Create numeric win label at team-level
    team_game["WL"] = team_game["WL"].astype(str).str.strip()
    team_game["win"] = (team_game["WL"] == "W").astype(int)

    print(f"[build_team_features] Team-game rows: {len(team_game)}")
    return team_game


def attach_opponent(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game row, attach opponent stats from the same GAME_ID.
    Output will have two rows per game (one for each team) but with
    both team and opponent columns.
    """
    print("[build_team_features] Attaching opponent stats ...")

    left = team_game.copy()
    right = team_game.copy()

    merged = left.merge(
        right,
        on=["SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE"],
        suffixes=("_team", "_opp"),
    )
    merged = merged[merged["TEAM_ID_team"] != merged["TEAM_ID_opp"]].copy()

    merged = merged.drop_duplicates(
        subset=["SEASON", "SEASON_TYPE", "GAME_ID", "TEAM_ID_team"],
        keep="first",
    )

    merged["point_diff"] = merged["team_pts_team"] - merged["team_pts_opp"]
    merged["win_team"] = merged["win_team"].astype(int)

    merged = merged.rename(
        columns={
            "TEAM_ID_team": "TEAM_ID",
            "TEAM_ABBREVIATION_team": "TEAM_ABBREVIATION",
            "WL_team": "WL",
            "team_pts_team": "PTS_FOR",
            "team_pts_opp": "PTS_AGAINST",
            "win_team": "WIN",
            "TEAM_ID_opp": "OPP_TEAM_ID",
            "TEAM_ABBREVIATION_opp": "OPP_TEAM_ABBREVIATION",
        }
    )

    print(f"[build_team_features] Team-game with opponent rows: {len(merged)}")
    return merged


def compute_rolling_features(df_team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling features per TEAM_ID, based on past games only.
    """
    print("[build_team_features] Computing rolling features ...")

    df = df_team_games.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    df["PTS_FOR"] = df["PTS_FOR"].astype(float)
    df["PTS_AGAINST"] = df["PTS_AGAINST"].astype(float)
    df["point_diff"] = df["point_diff"].astype(float)
    df["WIN"] = df["WIN"].astype(int)

    grp = df.groupby("TEAM_ID", group_keys=False)

    # 5-game rolling averages
    for col in ["PTS_FOR", "PTS_AGAINST", "point_diff"]:
        df[f"roll5_{col}"] = (
            grp[col]
            .apply(lambda s: s.rolling(window=5, min_periods=1).mean())
            .shift(1)
        )

    # 10-game rolling averages
    for col in ["PTS_FOR", "point_diff"]:
        df[f"roll10_{col}"] = (
            grp[col]
            .apply(lambda s: s.rolling(window=10, min_periods=1).mean())
            .shift(1)
        )

    # 10-game rolling win rate
    df["roll10_win_rate"] = (
        grp["WIN"]
        .apply(lambda s: s.rolling(window=10, min_periods=1).mean())
        .shift(1)
    )

    # 赛季到当前为止的胜率
    df["season_games_played"] = grp.cumcount()
    df["season_wins_so_far"] = grp["WIN"].cumsum() - df["WIN"]
    df["season_win_rate"] = np.where(
        df["season_games_played"] > 0,
        df["season_wins_so_far"] / df["season_games_played"],
        np.nan,
    )

    return df


# ---------------- Core function for pipeline ---------------- #

def _download_raw_from_s3() -> List[Path]:
    """
    从 S3 的 raw/ 目录下载所有 cleaned player CSV 到本地 data/raw_from_s3/，
    并返回本地路径列表。
    """
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = LOCAL_DATA_DIR / "raw_from_s3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = f"{S3_PREFIX}raw/"

    print(f"[build_team_features] Listing S3 objects: s3://{S3_BUCKET}/{prefix}")
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

    contents = resp.get("Contents", [])
    csv_paths: List[Path] = []

    for obj in contents:
        key = obj["Key"]
        if not key.endswith(".csv"):
            continue
        filename = key.split("/")[-1]
        local_path = raw_dir / filename

        if not local_path.exists():
            print(f"[build_team_features] Downloading {key} -> {local_path}")
            s3.download_file(S3_BUCKET, key, str(local_path))
        else:
            print(f"[build_team_features] Local file already exists, skip download: {local_path}")

        csv_paths.append(local_path)

    if not csv_paths:
        raise RuntimeError("[build_team_features] No CSV files found in S3 raw/ directory.")

    print(f"[build_team_features] Downloaded/ready {len(csv_paths)} CSV files.")
    return csv_paths


def build_team_features(
    input_paths: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> List[Path]:
    """
    主函数：构建球队级特征。

    - 如果 input_paths 为 None：
        自动从 S3 raw/ 下载所有 cleaned CSV；
    - 否则：
        使用传入的本地 CSV 路径列表。

    返回：
        [Path] 列表，包含生成的特征文件路径（供 train_model 使用）。
    """
    if input_paths is None:
        csv_paths = _download_raw_from_s3()
    else:
        csv_paths = [Path(p) for p in input_paths]

    # 1) load cleaned player logs
    df_players = load_player_logs(csv_paths)

    # 2) aggregate to team-game
    team_game = aggregate_to_team_games(df_players)

    # 3) attach opponent stats
    team_with_opp = attach_opponent(team_game)

    # 4) compute rolling features
    team_features = compute_rolling_features(team_with_opp)

    # 5) save
    if output_path is None:
        out_path = LOCAL_DATA_DIR / "team_game_features.csv"
    else:
        out_path = Path(output_path)

    team_features.to_csv(out_path, index=False)

    print(f"\n✅ [build_team_features] Saved team-level features to: {out_path}")
    print(f"Rows: {len(team_features)}; Columns: {len(team_features.columns)}")

    return [out_path]


# ---------------- CLI ---------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Build team-level game features from cleaned player logs."
    )
    p.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="One or more cleaned player CSV files (nba_playerlogs_clean_*.csv).",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV file for team-level features.",
    )
    return p.parse_args()


def main():
    """
    命令行入口：只用于你手动本地跑：
        python build_team_features.py -i data/xxx.csv -o data/team_game_features.csv
    """
    args = parse_args()
    build_team_features(input_paths=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
