#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build team-level game features from cleaned player-level logs.

功能：
  - 从 “球员级别 + 已清洗” 的日志（由 build_datasets.py 生成）出发
  - 聚合到 “球队-比赛” 级别
  - 加上对手数据（opponent stats）
  - 加上 home/away、rest_days、back_to_back
  - 计算 rolling 特征（5/10 场）
  - 计算赛季累计胜率 season_win_rate（只用当前比赛之前的比赛避免泄露）
  - 输出 team_game_features.csv，供 train_model.py 使用

用法一：本地 / 手动（命令行）：
    python build_team_features.py \
        --input data/player_logs_clean_all_3seasons_plus_current_to_20251121.csv \
        --output data/team_game_features.csv

用法二：在 EC2 上由 run_daily_training.py 调用（推荐）：
    from build_team_features import build_team_features
    feature_paths = build_team_features()

    - 不传参数时：
        自动从 S3 的 raw/ 目录下载所有 cleaned player CSV，
        构建特征，保存到 LOCAL_DATA_DIR / "team_game_features.csv"，
        返回 [Path(".../team_game_features.csv")]。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import boto3
import numpy as np
import pandas as pd

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX


# ---------------- Basic loaders ---------------- #


def load_player_logs(input_paths: List[Path]) -> pd.DataFrame:
    """Load one or multiple cleaned player-level CSVs into a single DataFrame."""
    paths = [Path(p) for p in input_paths]
    dfs: List[pd.DataFrame] = []

    for p in paths:
        print(f"[build_team_features] Loading {p} ...")
        df = pd.read_csv(p)
        # ensure GAME_DATE is datetime
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        dfs.append(df)

    if not dfs:
        raise ValueError("[build_team_features] No input files loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[build_team_features] Loaded total player rows: {len(combined)}")
    return combined


# ---------------- Team-level aggregation ---------------- #


def _parse_home_away_from_matchup(matchup: str) -> tuple[int, int]:
    """
    从 MATCHUP 文本解析 Home/Away 标记。
      例：
        'LAL vs. BOS' → LAL 主场
        'LAL @ BOS'   → LAL 客场
    返回 (HOME, AWAY)，均为 0/1。
    """
    if not isinstance(matchup, str):
        return 0, 0
    matchup = matchup.strip()
    home = 1 if "vs." in matchup else 0
    away = 1 if "@" in matchup else 0
    return home, away


def aggregate_to_team_games(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player-level logs to team-level single-game stats.

    输入字段（来自 build_datasets 清洗后的 CSV，至少包含）：
        SEASON, SEASON_TYPE, GAME_ID, GAME_DATE,
        TEAM_ID, TEAM_ABBREVIATION, MATCHUP, WL,
        PTS, REB, AST, STL, BLK, TOV,
        FGM, FGA, FG3M, FG3A, FTM, FTA,
        OREB, DREB, PF, PLUS_MINUS, MIN,
        FG_PCT, FG3_PCT, FT_PCT

    输出：每行 = 某队在某场比赛的总体表现。
    """
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
        raise ValueError(f"[build_team_features] Missing required columns in player logs: {missing}")

    # group by team-game; sum counting stats, 重新计算命中率
    grp_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP", "WL",
    ]

    print("[build_team_features] Aggregating player logs to team-game level ...")
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

    # 重新计算球队命中率（比简单平均更准确）
    agg["FG_PCT"] = agg["FGM"] / agg["FGA"].replace({0: np.nan})
    agg["FG3_PCT"] = agg["FG3M"] / agg["FG3A"].replace({0: np.nan})
    agg["FT_PCT"] = agg["FTM"] / agg["FTA"].replace({0: np.nan})

    # 增加 Home/Away
    home_flags = []
    away_flags = []
    for m in agg["MATCHUP"]:
        h, a = _parse_home_away_from_matchup(m)
        home_flags.append(h)
        away_flags.append(a)
    agg["HOME"] = home_flags
    agg["AWAY"] = away_flags

    # 数值胜负标签
    agg["WL"] = agg["WL"].astype(str).str.strip()
    agg["WIN"] = (agg["WL"] == "W").astype(int)

    # 方便后续命名：球队得分
    agg["PTS_FOR"] = agg["PTS"]

    print(f"[build_team_features] Team-game rows: {len(agg)}")
    return agg


# ---------------- Opponent stats ---------------- #


def attach_opponent(team_game: pd.DataFrame) -> pd.DataFrame:
    """
    对每一行 team-game，附上对手同场数据。

    输出仍然是 “每队每场一行”，但多了：
        OPP_TEAM_ID, OPP_TEAM_ABBREVIATION,
        PTS_AGAINST, OPP_WIN, etc.
    """
    print("[build_team_features] Attaching opponent stats ...")

    base_cols = [
        "SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE",
        "TEAM_ID", "TEAM_ABBREVIATION",
        "HOME", "AWAY",
        "WIN", "PTS_FOR", "REB", "AST", "STL", "BLK", "TOV",
        "OREB", "DREB", "PF", "PLUS_MINUS",
        "FG_PCT", "FG3_PCT", "FT_PCT",
    ]
    # 确保需要列存在
    missing = [c for c in base_cols if c not in team_game.columns]
    if missing:
        raise ValueError(f"[build_team_features] Missing columns before attach_opponent: {missing}")

    left = team_game[base_cols].copy()

    merged = left.merge(
        left,
        on=["SEASON", "SEASON_TYPE", "GAME_ID", "GAME_DATE"],
        suffixes=("", "_OPP"),
    )

    # 只保留“我”和“对手”不同的行
    merged = merged[merged["TEAM_ID"] != merged["TEAM_ID_OPP"]].copy()

    # 对手信息重命名
    merged = merged.rename(
        columns={
            "TEAM_ID_OPP": "OPP_TEAM_ID",
            "TEAM_ABBREVIATION_OPP": "OPP_TEAM_ABBREVIATION",
            "WIN_OPP": "OPP_WIN",
            "PTS_FOR_OPP": "PTS_FOR_OPP",
        }
    )

    # 我方失分 = 对手得分
    merged["PTS_AGAINST"] = merged["PTS_FOR_OPP"]

    # 分差
    merged["point_diff"] = merged["PTS_FOR"] - merged["PTS_AGAINST"]

    # 清理不用的中间列
    drop_cols = [c for c in merged.columns if c.endswith("_OPP") and c not in (
        "OPP_TEAM_ID", "OPP_TEAM_ABBREVIATION", "OPP_WIN", "PTS_FOR_OPP"
    )]
    merged = merged.drop(columns=drop_cols)

    print(f"[build_team_features] Team-game with opponent rows: {len(merged)}")
    return merged


# ---------------- Schedule features ---------------- #


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每支球队添加：
      - rest_days: 距离上一次比赛的间隔天数
      - is_back_to_back: 是否背靠背（rest_days == 1）
    """
    print("[build_team_features] Computing schedule features (rest days, back-to-back) ...")
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    # 以赛季为单位计算休息天数更合理
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    grp = df.groupby(["SEASON", "TEAM_ID"], group_keys=False)

    prev_date = grp["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - prev_date).dt.days
    df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)

    return df


# ---------------- Rolling & season features ---------------- #


def add_rolling_and_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每队添加 rolling 特征 + 赛季到当前胜率（不含当场）。
    特别注意：
      - 所有 rolling 特征都用 shift(1) 避免把“当场”的结果泄露到特征里。
      - season_win_rate 只用当前比赛之前的场次。
    """
    print("[build_team_features] Computing rolling & season cumulative features ...")
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()

    # 确保数值类型
    df["PTS_FOR"] = df["PTS_FOR"].astype(float)
    df["PTS_AGAINST"] = df["PTS_AGAINST"].astype(float)
    df["point_diff"] = df["point_diff"].astype(float)
    df["WIN"] = df["WIN"].astype(int)

    # ---------- Rolling features（按 TEAM_ID 滚动，跨赛季也考虑） ----------
    grp_team = df.groupby("TEAM_ID", group_keys=False)

    # 5 场 rolling 平均：PTS_FOR, PTS_AGAINST, point_diff
    for col in ["PTS_FOR", "PTS_AGAINST", "point_diff"]:
        df[f"roll5_{col}"] = (
            grp_team[col]
            .apply(lambda s: s.rolling(window=5, min_periods=1).mean())
            .shift(1)
        )

    # 10 场 rolling 平均：PTS_FOR, point_diff
    for col in ["PTS_FOR", "point_diff"]:
        df[f"roll10_{col}"] = (
            grp_team[col]
            .apply(lambda s: s.rolling(window=10, min_periods=1).mean())
            .shift(1)
        )

    # 10 场 rolling 胜率
    df["roll10_win_rate"] = (
        grp_team["WIN"]
        .apply(lambda s: s.rolling(window=10, min_periods=1).mean())
        .shift(1)
    )

    # ---------- 赛季累计胜率（只用当前比赛之前的场次） ----------
    grp_season_team = df.groupby(["SEASON", "TEAM_ID"], group_keys=False)

    # 当前是该赛季的第几场比赛（0-based）
    df["season_games_played"] = grp_season_team.cumcount()

    # 截止当前之前的累计胜场数：
    #   cumsum(INC 自己) - WIN = “前几场”的胜场
    cumsum_wins = grp_season_team["WIN"].cumsum()
    df["season_wins_so_far"] = cumsum_wins - df["WIN"]

    df["season_win_rate"] = np.where(
        df["season_games_played"] > 0,
        df["season_wins_so_far"] / df["season_games_played"],
        np.nan,
    )

    return df


# ---------------- S3 helper (for EC2 pipeline) ---------------- #


def _download_raw_from_s3() -> List[Path]:
    """
    从 S3 的 raw/ 目录下载所有 cleaned player CSV 到本地 LOCAL_DATA_DIR / "raw_from_s3"，
    并返回本地路径列表。

    约定：
      - daily_crawl_and_upload.py 会把 build_datasets.py 生成的 cleaned CSV
        上传到 s3://{S3_BUCKET}/{S3_PREFIX}raw/ 下面。
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


# ---------------- Public API for pipeline ---------------- #


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

    # 4) schedule features: rest_days, is_back_to_back
    team_with_sched = add_schedule_features(team_with_opp)

    # 5) rolling & season cumulative features
    team_features = add_rolling_and_season_features(team_with_sched)

    # 6) save
    if output_path is None:
        out_path = LOCAL_DATA_DIR / "team_game_features.csv"
    else:
        out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    team_features.to_csv(out_path, index=False)

    print("\n✅ [build_team_features] Saved team-level features to:", out_path)
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
    命令行入口：手动本地跑：
        python build_team_features.py \
          -i data/player_logs_clean_all_3seasons_plus_current_to_20251121.csv \
          -o data/team_game_features.csv
    """
    args = parse_args()
    build_team_features(input_paths=args.input, output_path=args.output)


if __name__ == "__main__":
    main()

