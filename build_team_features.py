#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build team-level game features from cleaned player-level logs.

Input:
    - One or more cleaned CSVs produced by nba_playerlogs_clean_*.py
      Each row = one player in one game.

Output:
    - One team-level features CSV, where each row = one team in one game,
      with rolling features based ONLY on past games.

Example:
    python build_team_features.py \
        --input nba_playerlogs_clean_all_3seasons_plus_current_to_20251117.csv \
        --output team_game_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------- Helpers ---------------- #

def load_player_logs(input_paths):
    """Load one or multiple cleaned player-level CSVs into a single DataFrame."""
    paths = [Path(p) for p in input_paths]
    dfs = []
    for p in paths:
        print(f"Loading {p} ...")
        df = pd.read_csv(p)
        # ensure GAME_DATE is datetime
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        dfs.append(df)
    if not dfs:
        raise ValueError("No input files loaded.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded total rows: {len(combined)}")
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

    print("Aggregating player logs to team-game level ...")
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
    # WL: 'W'/'L' or 'W ' etc.
    team_game["WL"] = team_game["WL"].astype(str).str.strip()
    team_game["win"] = (team_game["WL"] == "W").astype(int)

    print(f"Team-game rows: {len(team_game)}")
    return team_game


def attach_opponent(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game row, attach opponent stats from the same GAME_ID.
    Output will have two rows per game (one for each team) but with
    both team and opponent columns.
    """
    print("Attaching opponent stats ...")

    left = team_game.copy()
    right = team_game.copy()

    # Self-merge on GAME_ID + SEASON + SEASON_TYPE
    merged = left.merge(
        right,
        on=["SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE"],
        suffixes=("_team", "_opp"),
    )

    # keep only rows where team and opponent are different
    merged = merged[merged["TEAM_ID_team"] != merged["TEAM_ID_opp"]].copy()

    # Drop duplicate pairs if any (should naturally have 2 rows per game)
    merged = merged.drop_duplicates(
        subset=["SEASON", "SEASON_TYPE", "GAME_ID", "TEAM_ID_team"],
        keep="first",
    )

    # Create handy columns
    merged["point_diff"] = merged["team_pts_team"] - merged["team_pts_opp"]
    merged["win_team"] = merged["win_team"].astype(int)

    # Rename to cleaner names
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

    print(f"Team-game with opponent rows: {len(merged)}")
    return merged


def compute_rolling_features(df_team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling features per TEAM_ID, based on past games only.
    We:
      - sort by GAME_DATE
      - group by TEAM_ID
      - use rolling window (5, 10) with shift(1) to avoid leak future info.

    Features:
      - rolling_5_pts_for / pts_against / point_diff
      - rolling_10_pts_for / point_diff
      - rolling_10_win_rate
    """

    print("Computing rolling features ...")

    # sort for stable rolling
    df = df_team_games.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    # ensure types
    df["PTS_FOR"] = df["PTS_FOR"].astype(float)
    df["PTS_AGAINST"] = df["PTS_AGAINST"].astype(float)
    df["point_diff"] = df["point_diff"].astype(float)
    df["WIN"] = df["WIN"].astype(int)

    # group by team
    grp = df.groupby("TEAM_ID", group_keys=False)

    # 5-game rolling averages (based on past games)
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

    # 总体赛季胜率（到当前为止）
    df["season_games_played"] = grp.cumcount()
    df["season_wins_so_far"] = grp["WIN"].cumsum() - df["WIN"]
    df["season_win_rate"] = np.where(
        df["season_games_played"] > 0,
        df["season_wins_so_far"] / df["season_games_played"],
        np.nan,
    )

    return df


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
    args = parse_args()

    # 1) load cleaned player logs
    df_players = load_player_logs(args.input)

    # 2) aggregate to team-game
    team_game = aggregate_to_team_games(df_players)

    # 3) attach opponent stats
    team_with_opp = attach_opponent(team_game)

    # 4) compute rolling features
    team_features = compute_rolling_features(team_with_opp)

    # 5) save
    out_path = Path(args.output)
    team_features.to_csv(out_path, index=False)
    print(f"\n✅ Saved team-level features to: {out_path}")
    print(f"Rows: {len(team_features)}; Columns: {len(team_features.columns)}")


if __name__ == "__main__":
    main()
