#!/usr/bin/env python3
# daily_crawl_and_upload.py

"""
每天增量爬虫：
  1) 抓取昨天所有球员比赛日志（player level）
  2) 用球员日志计算昨天每场比赛的主客队+比分（game level）
  3) 抓取今天 + 未来 4 天的赛程（没有比分）
  4) 所有结果保存到 LOCAL_DATA_DIR 并上传到 S3 raw/ 下面
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

# ----------------- NBA team id -> 缩写，用于赛程 ----------------- #
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


# ----------------- 时间：统一按美东算 ----------------- #

def get_today_yesterday_et() -> Tuple[date, date]:
    """返回 (today_et, yesterday_et)，按美东时间计算。"""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today_et = now_et.date()
    yesterday_et = today_et - timedelta(days=1)
    return today_et, yesterday_et


# ----------------- 自动判断赛季 ----------------- #

def get_current_season(d: date) -> str:
    """
    根据日期判断 NBA 赛季字符串，如 '2024-25'、'2025-26'。

    规则：NBA 新赛季从每年 10 月开始。
      - 10,11,12 月：赛季起始年 = 当前年
      - 1~9 月：赛季起始年 = 当前年 - 1
    """
    if d.month >= 10:
        start_year = d.year
    else:
        start_year = d.year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


# ----------------- 工具：Scoreboard 结果转 dict[name] -> DataFrame ----------------- #

def scoreboard_frames(sb: scoreboardv2.ScoreboardV2):
    data = sb.get_dict()
    frames = {}
    for rs in data.get("resultSets", []):
        name = rs.get("name")
        headers = rs.get("headers", [])
        rows = rs.get("rowSet", [])
        frames[name] = pd.DataFrame(rows, columns=headers)
    return frames


# ----------------- 1) 昨天球员日志 ----------------- #

def fetch_yesterday_player_logs(yesterday_et: date):
    """
    抓取昨天的球员 logs，返回 DataFrame 和日期字符串：
      - date_str_file: 'YYYY-MM-DD'（用于文件名）
    """
    # 文件名用 YYYY-MM-DD，方便你现有 pipeline
    date_str_file = yesterday_et.strftime("%Y-%m-%d")
    # 调 API 用 MM/DD/YYYY（nba_api 通常这样）
    date_str_api = yesterday_et.strftime("%m/%d/%Y")

    season_str = get_current_season(yesterday_et)

    print(f"[player logs] season={season_str}, date={date_str_api} (ET) ...")

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
    用昨天的球员 logs，构造每场比赛的主客队 + 比分：

    输出列：
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
        raise ValueError(f"player logs 缺少列：{missing}")

    # 每队每场的总得分 + 对应 MATCHUP
    team_stats = (
        df_players
        .groupby(["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP"], as_index=False)["PTS"]
        .sum()
    )

    # MATCHUP 中： 'XXX vs YYY' = XXX 主场； 'XXX @ YYY' = XXX 客场
    # 更稳的写法：只要含有 '@' 就是客场，否则视为主场
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

    # 确保日期格式统一
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    # 一场比赛一行
    games = games[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_SCORE", "AWAY_SCORE"]].drop_duplicates()

    print(f"[games-from-players] built {len(games)} games")
    return games


# ----------------- 2) 赛程（今天 + 未来 4 天） ----------------- #

def fetch_schedule_for_date(game_date_et: date) -> pd.DataFrame:
    """
    用 ScoreboardV2 抓某一天的赛程。

    输出列：
      - GAME_DATE
      - HOME_TEAM
      - AWAY_TEAM
    """
    # 调 API 用 MM/DD/YYYY
    date_str_api = game_date_et.strftime("%m/%d/%Y")
    print(f"[schedule] Fetching schedule for {date_str_api} (ET) ...")

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
        return pd.DataFrame(columns=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"])

    g = game_header[["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].copy()

    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE_EST"]).dt.strftime("%Y-%m-%d")
    g["HOME_TEAM"] = g["HOME_TEAM_ID"].astype(int).map(TEAM_ID_TO_ABBR)
    g["AWAY_TEAM"] = g["VISITOR_TEAM_ID"].astype(int).map(TEAM_ID_TO_ABBR)

    result = g[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]].copy()
    return result


# ----------------- 上传到 S3 的小工具 ----------------- #

def upload_to_s3(local_path: Path, s3_key: str) -> None:
    """上传本地文件到 S3."""
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

    today_et, yesterday_et = get_today_yesterday_et()
    print(f"[time] ET today={today_et}, yesterday={yesterday_et}")

    # --------------- 1) 昨天的球员日志 ---------------- #
    df_players, y_str_file = fetch_yesterday_player_logs(yesterday_et)

    if df_players.empty:
        print("[player logs] No games found for yesterday -> 不产生增量 player_logs 文件。")
    else:
        # 保存 & 上传昨天的球员增量
        fname_players = f"player_logs_daily_{y_str_file.replace('-', '')}.csv"
        local_players = LOCAL_DATA_DIR / fname_players
        print(f"[player logs] Saving to {local_players}")
        df_players.to_csv(local_players, index=False)

        s3_key_players = f"{S3_PREFIX}raw/{fname_players}"
        upload_to_s3(local_players, s3_key_players)

    # --------------- 2) 昨天的赛程 + 结果（用球员 logs 算） ---------------- #
    df_yesterday_games = build_yesterday_games_from_players(df_players)

    fname_games = f"games_yesterday_{y_str_file.replace('-', '')}.csv"
    local_games = LOCAL_DATA_DIR / fname_games
    print(f"[games] Saving yesterday games to {local_games}")
    df_yesterday_games.to_csv(local_games, index=False)

    s3_key_games = f"{S3_PREFIX}raw/{fname_games}"
    upload_to_s3(local_games, s3_key_games)

    # --------------- 3) 今天 + 未来 4 天的赛程 ---------------- #
    for i in range(0, 5):  # today_et + 0..4
        d = today_et + timedelta(days=i)
        d_str_file = d.strftime("%Y%m%d")

        df_sched = fetch_schedule_for_date(d)

        fname_sched = f"schedule_{d_str_file}.csv"
        local_sched = LOCAL_DATA_DIR / fname_sched
        print(f"[schedule] Saving schedule for {d} to {local_sched}")
        # 没有比赛也会生成只有表头的 CSV
        df_sched.to_csv(local_sched, index=False)

        s3_key_sched = f"{S3_PREFIX}raw/{fname_sched}"
        upload_to_s3(local_sched, s3_key_sched)

    print("Done daily crawl: player logs + yesterday games + next 5 days schedule (ET-based).")


if __name__ == "__main__":
    main()
