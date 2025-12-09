#!/usr/bin/env python3
# daily_crawl_and_upload.py

"""
Daily incremental crawler:

  1) Fetch yesterday's player game logs (player level)
  2) Aggregate player logs to build yesterday's game results (team level)
  3) Fetch today's + next 4 days' schedule (no scores)
  4) Save everything to LOCAL_DATA_DIR and upload to S3 under raw/
"""

from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
import boto3
from nba_api.stats.endpoints import leaguegamelog, scoreboardv2
from botocore.exceptions import ClientError
from zoneinfo import ZoneInfo  # Python 3.9+

from config_aws import LOCAL_DATA_DIR, AWS_REGION, S3_BUCKET, S3_PREFIX

# ----------------- Global time zone (Los Angeles / Pacific Time) ----------------- #

NBA_TZ = ZoneInfo("America/Los_Angeles")

# ----------------- NBA team id -> abbreviation (for schedule) ----------------- #

TEAM_ID_TO_ABBR: Dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612751: "BKN",
    1610612766: "CHA",
    1610612741: "CHI",
    1610612739: "CLE",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612765: "DET",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612754: "IND",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612763: "MEM",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612740: "NOP",
    1610612752: "NYK",
    1610612760: "OKC",
    1610612753: "ORL",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612764: "WAS",
}


# ----------------- Time helpers: use Los Angeles time ----------------- #

def get_today_yesterday_la() -> Tuple[date, date]:
    """
    Return (today_la, yesterday_la) based on Los Angeles time.

    This is what we use to define "yesterday's games" for crawling.
    """
    now_la = datetime.now(NBA_TZ)
    today_la = now_la.date()
    yesterday_la = today_la - timedelta(days=1)
    return today_la, yesterday_la


# ----------------- Season helper ----------------- #

def get_current_season(d: date) -> str:
    """
    Infer NBA season string like '2024-25' or '2025-26' from a date.

    Rule: NBA season starts in October.
      - Month 10, 11, 12: season start year = current year
      - Month 1..9:       season start year = current year - 1
    """
    if d.month >= 10:
        start_year = d.year
    else:
        start_year = d.year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


# ----------------- Utility: ScoreboardV2 -> dict[name] -> DataFrame ----------------- #

def scoreboard_frames(sb: scoreboardv2.ScoreboardV2):
    """Convert ScoreboardV2 response to a dict of DataFrames keyed by resultSet name."""
    data = sb.get_dict()
    frames = {}
    for rs in data.get("resultSets", []):
        name = rs.get("name")
        headers = rs.get("headers", [])
        rows = rs.get("rowSet", [])
        frames[name] = pd.DataFrame(rows, columns=headers)
    return frames


# ----------------- 1) Yesterday player logs ----------------- #

def fetch_yesterday_player_logs(yesterday_la: date):
    """
    Fetch yesterday's player logs.

    Returns:
      - df: DataFrame of player logs for that date
      - date_str_file: 'YYYY-MM-DD' string used in filenames
    """
    # File naming: YYYY-MM-DD (easy for the pipeline)
    date_str_file = yesterday_la.strftime("%Y-%m-%d")
    # NBA API usually expects MM/DD/YYYY
    date_str_api = yesterday_la.strftime("%m/%d/%Y")

    season_str = get_current_season(yesterday_la)

    print(f"[player logs] season={season_str}, date={date_str_api} (LA-based date) ...")

    resp = leaguegamelog.LeagueGameLog(
        player_or_team_abbreviation="P",
        season=season_str,
        season_type_all_star="Regular Season",
        date_from_nullable=date_str_api,
        date_to_nullable=date_str_api,
        timeout=30,
    )

    df = resp.get_data_frames()[0]
    print(f"[player logs] Raw rows: {len(df)}")

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["GAME_DATE"] = df["GAME_DATE"].dt.strftime("%Y-%m-%d")

    return df, date_str_file


def build_yesterday_games_from_players(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Build team-level game results from yesterday's player logs.

    Output columns:
      - GAME_DATE
      - HOME_TEAM
      - AWAY_TEAM
      - HOME_SCORE
      - AWAY_SCORE
    """
    if df_players.empty:
        print("[games-from-players] input player logs is empty")
        return pd.DataFrame(columns=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_SCORE", "AWAY_SCORE"])

    required = {"GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP", "PTS"}
    missing = required - set(df_players.columns)
    if missing:
        raise ValueError(f"player logs missing columns: {missing}")

    # Aggregate to team-level total points per game + MATCHUP
    team_stats = (
        df_players
        .groupby(["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP"], as_index=False)["PTS"]
        .sum()
    )

    # MATCHUP:
    #   'XXX vs YYY'  -> XXX is home
    #   'XXX @ YYY'   -> XXX is away
    # Safer rule: if contains '@' => away, otherwise home
    team_stats["IS_HOME"] = ~team_stats["MATCHUP"].str.contains("@")

    home = team_stats[team_stats["IS_HOME"]].copy()
    away = team_stats[~team_stats["IS_HOME"]].copy()

    home = home.rename(columns={
        "TEAM_ABBREVIATION": "HOME_TEAM",
        "PTS": "HOME_SCORE",
    })
    away = away.rename(columns={
        "TEAM_ABBREVIATION": "AWAY_TEAM",
        "PTS": "AWAY_SCORE",
    })

    games = home[["GAME_ID", "GAME_DATE", "HOME_TEAM", "HOME_SCORE"]].merge(
        away[["GAME_ID", "AWAY_TEAM", "AWAY_SCORE"]],
        on="GAME_ID",
        how="inner",
    )

    # Normalize date format
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    # One row per game
    games = games[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_SCORE", "AWAY_SCORE"]].drop_duplicates()

    print(f"[games-from-players] built {len(games)} games")
    return games


# ----------------- 2) Schedule (today + next 4 days) ----------------- #

def fetch_schedule_for_date(game_date_la: date) -> pd.DataFrame:
    """
    Fetch schedule for a given date using ScoreboardV2.

    Input date is interpreted as an LA (PT) date; we pass the same calendar
    date to NBA API. This is safe as long as the crawler runs after all
    games of that calendar day have finished.

    Output columns:
      - GAME_DATE   (YYYY-MM-DD, based on GAME_DATE_EST from API)
      - HOME_TEAM
      - AWAY_TEAM
      - HOME_TEAM_ID
      - AWAY_TEAM_ID
      - IS_TBD      (True if matchup is not fully determined)
    """
    date_str_api = game_date_la.strftime("%m/%d/%Y")
    print(f"[schedule] Fetching schedule for {date_str_api} (LA-based date) ...")

    sb = scoreboardv2.ScoreboardV2(
        game_date=date_str_api,
        league_id="00",
        day_offset=0,
        timeout=30,
    )

    frames = scoreboard_frames(sb)
    game_header = frames.get("GameHeader", pd.DataFrame())
    print(f"[schedule] GameHeader rows={len(game_header)}")

    if game_header.empty:
        return pd.DataFrame(columns=[
            "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
            "HOME_TEAM_ID", "AWAY_TEAM_ID", "IS_TBD"
        ])

    g = game_header[["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].copy()

    # Keep original IDs and detect TBD games
    g["HOME_TEAM_ID"] = pd.to_numeric(g["HOME_TEAM_ID"], errors="coerce")
    g["VISITOR_TEAM_ID"] = pd.to_numeric(g["VISITOR_TEAM_ID"], errors="coerce")

    # If either side is NaN, treat as TBD matchup
    g["IS_TBD"] = g["HOME_TEAM_ID"].isna() | g["VISITOR_TEAM_ID"].isna()

    # Map to team abbreviations where possible
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE_EST"]).dt.strftime("%Y-%m-%d")
    g["HOME_TEAM"] = g["HOME_TEAM_ID"].map(TEAM_ID_TO_ABBR)
    g["AWAY_TEAM"] = g["VISITOR_TEAM_ID"].map(TEAM_ID_TO_ABBR)

    # For TBD games, fill missing team abbr with 'TBD'
    g.loc[g["IS_TBD"] & g["HOME_TEAM"].isna(), "HOME_TEAM"] = "TBD"
    g.loc[g["IS_TBD"] & g["AWAY_TEAM"].isna(), "AWAY_TEAM"] = "TBD"

    # Drop truly broken rows (no date at all)
    g = g.dropna(subset=["GAME_DATE"])

    if g.empty:
        print("[schedule] After cleaning, no games left for this date.")
        return pd.DataFrame(columns=[
            "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
            "HOME_TEAM_ID", "AWAY_TEAM_ID", "IS_TBD"
        ])

    result = g[[
        "GAME_DATE", "HOME_TEAM", "AWAY_TEAM",
        "HOME_TEAM_ID", "VISITOR_TEAM_ID", "IS_TBD"
    ]].copy()
    result = result.rename(columns={"VISITOR_TEAM_ID": "AWAY_TEAM_ID"})
    return result


# ----------------- S3 upload helper ----------------- #

def upload_to_s3(local_path: Path, s3_key: str) -> None:
    """Upload a local file to S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    print(f"[upload] {local_path} -> s3://{S3_BUCKET}/{s3_key}")
    try:
        s3.upload_file(
            Filename=str(local_path),
            Bucket=S3_BUCKET,
            Key=s3_key,
        )
    except ClientError as e:
        print(f"[ERROR] Failed to upload {local_path}: {e}")


# ----------------- main ----------------- #

def main():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    today_la, yesterday_la = get_today_yesterday_la()
    print(f"[time] LA today={today_la}, yesterday={yesterday_la}")

    # --------------- 1) Yesterday player logs ---------------- #
    df_players, y_str_file = fetch_yesterday_player_logs(yesterday_la)

    if df_players.empty:
        print("[player logs] No games found for yesterday -> no incremental player_logs file.")
    else:
        # Save & upload yesterday's incremental player logs
        fname_players = f"player_logs_daily_{y_str_file.replace('-', '')}.csv"
        local_players = LOCAL_DATA_DIR / fname_players
        print(f"[player logs] Saving to {local_players}")
        df_players.to_csv(local_players, index=False)

        s3_key_players = f"{S3_PREFIX}raw/{fname_players}"
        upload_to_s3(local_players, s3_key_players)

    # --------------- 2) Yesterday games (built from player logs) ---------------- #
    df_yesterday_games = build_yesterday_games_from_players(df_players)

    fname_games = f"games_yesterday_{y_str_file.replace('-', '')}.csv"
    local_games = LOCAL_DATA_DIR / fname_games
    print(f"[games] Saving yesterday games to {local_games}")
    df_yesterday_games.to_csv(local_games, index=False)

    s3_key_games = f"{S3_PREFIX}raw/{fname_games}"
    upload_to_s3(local_games, s3_key_games)

    # --------------- 3) Today + next 4 days schedule ---------------- #
    for i in range(0, 5):  # today_la + 0..4
        d = today_la + timedelta(days=i)
        d_str_file = d.strftime("%Y%m%d")

        df_sched = fetch_schedule_for_date(d)

        fname_sched = f"schedule_{d_str_file}.csv"
        local_sched = LOCAL_DATA_DIR / fname_sched
        print(f"[schedule] Saving schedule for {d} (LA date) to {local_sched}")
        # Even if there are no games, we still create an empty CSV with header
        df_sched.to_csv(local_sched, index=False)

        s3_key_sched = f"{S3_PREFIX}raw/{fname_sched}"
        upload_to_s3(local_sched, s3_key_sched)

    print("Done daily crawl: player logs + yesterday games + next 5 days schedule (LA-based).")


if __name__ == "__main__":
    main()
