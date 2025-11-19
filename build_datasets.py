#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch and CLEAN player game logs for:
  - The last THREE full NBA seasons (before current season)
  - The CURRENT season up to a cutoff date (e.g., 2025-11-17)

Uses nba_api.stats.endpoints.LeagueGameLog at PLAYER level.
Keeps only useful columns for later modeling.
Saves:
  - one cleaned CSV per season (full seasons)
  - one cleaned CSV for current partial season
  - one merged cleaned CSV for all seasons + current partial
"""

import time
from datetime import datetime, date
from typing import List, Optional

import pandas as pd
from pytz import timezone
from nba_api.stats.endpoints import leaguegamelog

# ---------------------- Config ----------------------
INCLUDE_PLAYOFFS = True               # False = only Regular Season
SLEEP_BETWEEN_CALLS = 1.5             # avoid rate limiting
OUTPUT_PREFIX = "nba_playerlogs_clean"  # 输出文件前缀
TIMEOUT = 30

# 当前赛季“截止日期”：本赛季只抓到这天（含这天）
CUTOFF_DATE = date(2025, 11, 17)      # 你需要的 2025-11-17 之前的所有数据

# ----------------- Season helpers -------------------
def current_nba_season_start_year(d: date) -> int:
    """
    NBA 赛季从每年 10 月开始，到次年 6 月结束。
    若当前日期在 7~9 月，仍算“上一赛季末期”，赛季起始年 = 年份 - 1。
    例如：2025-07-10 → 2024-25 赛季。
    返回赛季起始年，如 2025 表示 2025-26 赛季。
    """
    m = d.month
    if 7 <= m <= 9:
        return d.year - 1
    return d.year if m >= 10 else d.year - 1

def season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

def last_n_full_seasons(n: int) -> List[str]:
    """
    取“上 n 个完整赛季”（不含当前赛季）。
    例如当前是 2025-26，则返回 2024-25, 2023-24, 2022-23。
    """
    et = timezone("US/Eastern")
    today_et = datetime.now(et).date()
    cur_start = current_nba_season_start_year(today_et)
    return [season_str(cur_start - i) for i in range(1, n + 1)]

def current_season_for_cutoff(cutoff: date) -> str:
    """
    根据 CUTOFF_DATE 推断它所在的赛季字符串，如 2025-11-17 → 2025-26。
    """
    start_year = current_nba_season_start_year(cutoff)
    return season_str(start_year)

# ---------------- NBA fetch logic -------------------
def fetch_leaguegamelog_players(
    season: str,
    season_type: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """
    拉取某赛季、某赛段（Regular Season / Playoffs）的所有球员比赛日志。
    可选 date_from/date_to（格式: 'MM/DD/YYYY'），为空则拉整个赛段。
    """
    print(f"Fetching LeagueGameLog: season={season}, season_type={season_type}, "
          f"date_from={date_from}, date_to={date_to}")
    lgl = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,         # "Regular Season" / "Playoffs" / "Pre Season"
        player_or_team_abbreviation="P",          # 球员级别
        date_from_nullable=date_from,
        date_to_nullable=date_to,
        timeout=TIMEOUT,
    )
    df = lgl.get_data_frames()[0]
    if df is None:
        df = pd.DataFrame()
    if not df.empty:
        df["SEASON"] = season
        df["SEASON_TYPE"] = season_type
    return df

# ---------------- Data cleaning -------------------
KEEP_COLS = [
    # 赛季与比赛信息
    "SEASON", "SEASON_TYPE", "SEASON_ID",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    # 球队 & 球员标识
    "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "PLAYER_ID", "PLAYER_NAME",
    # 基础表现
    "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    # 投篮效率
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    # 影响力
    "PLUS_MINUS",
]

def clean_player_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    只保留对你后续建模有用的字段，并做类型处理。
    """
    if df.empty:
        return df

    # 有的版本返回列名可能略有差别，先过滤存在的列
    cols = [c for c in KEEP_COLS if c in df.columns]
    clean = df[cols].copy()

    # 统一 GAME_DATE 为 datetime 类型
    if "GAME_DATE" in clean.columns:
        clean["GAME_DATE"] = pd.to_datetime(clean["GAME_DATE"])

    return clean

# ---------------- Main pipeline -------------------
def main():
    # 1) 上三个完整赛季
    full_seasons = last_n_full_seasons(3)
    print(f"Target full seasons (last 3): {full_seasons}")

    all_clean_frames = []

    for s in full_seasons:
        parts = ["Regular Season"]
        if INCLUDE_PLAYOFFS:
            parts.append("Playoffs")

        per_season_frames = []
        for st in parts:
            df_raw = fetch_leaguegamelog_players(s, st)
            if df_raw.empty:
                print(f"  -> No rows for {s} {st}")
            else:
                df_clean = clean_player_logs(df_raw)
                print(f"  -> {s} {st}: raw={len(df_raw)}, clean={len(df_clean)} (sample:)")
                print(df_clean.head(3))
                per_season_frames.append(df_clean)
            time.sleep(SLEEP_BETWEEN_CALLS)

        if per_season_frames:
            season_df = pd.concat(per_season_frames, ignore_index=True)
            out_path = f"{OUTPUT_PREFIX}_{s.replace('-', '')}.csv"
            season_df.to_csv(out_path, index=False)
            print(f"✅ Saved CLEAN season CSV: {out_path} (rows={len(season_df)})")
            all_clean_frames.append(season_df)
        else:
            print(f"⚠️ Season {s}: no data fetched.")

    # 2) 当前赛季（截至 CUTOFF_DATE）
    cur_season = current_season_for_cutoff(CUTOFF_DATE)
    # 当前赛季通常从 10 月 1 日附近开始，这里简单用 10/01
    season_start_date = date(current_nba_season_start_year(CUTOFF_DATE), 10, 1)
    date_from_str = season_start_date.strftime("%m/%d/%Y")
    date_to_str = CUTOFF_DATE.strftime("%m/%d/%Y")

    print(f"\nCurrent season partial: {cur_season} from {date_from_str} to {date_to_str}")

    parts = ["Regular Season"]
    if INCLUDE_PLAYOFFS:
        parts.append("Playoffs")   # 当前日期一般还没季后赛，可以保留逻辑

    cur_frames = []
    for st in parts:
        df_raw = fetch_leaguegamelog_players(cur_season, st, date_from=date_from_str, date_to=date_to_str)
        if df_raw.empty:
            print(f"  -> No rows for {cur_season} {st} in this range.")
        else:
            df_clean = clean_player_logs(df_raw)
            print(f"  -> {cur_season} {st}: raw={len(df_raw)}, clean={len(df_clean)} (sample:)")
            print(df_clean.head(3))
            cur_frames.append(df_clean)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if cur_frames:
        cur_df = pd.concat(cur_frames, ignore_index=True)
        out_cur = f"{OUTPUT_PREFIX}_{cur_season.replace('-', '')}_to_{CUTOFF_DATE.strftime('%Y%m%d')}.csv"
        cur_df.to_csv(out_cur, index=False)
        print(f"✅ Saved CLEAN current season partial CSV: {out_cur} (rows={len(cur_df)})")
        all_clean_frames.append(cur_df)
    else:
        print(f"⚠️ Current season {cur_season}: no data in given date range.")

    # 3) 合并全部（上三季 + 本季部分）
    if all_clean_frames:
        merged = pd.concat(all_clean_frames, ignore_index=True)
        out_all = f"{OUTPUT_PREFIX}_all_3seasons_plus_current_to_{CUTOFF_DATE.strftime('%Y%m%d')}.csv"
        merged.to_csv(out_all, index=False)
        print(f"\n✅ Saved merged CLEAN CSV: {out_all} (rows={len(merged)})")
    else:
        print("\n⚠️ Nothing fetched at all.")

if __name__ == "__main__":
    main()
