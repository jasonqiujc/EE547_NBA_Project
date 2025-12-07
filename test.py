from datetime import date
from nba_api.stats.endpoints import scoreboardv2, leaguegamelog

# 1) 看赛程
sb = scoreboardv2.ScoreboardV2(game_date="12/09/2025", league_id="00", day_offset=0)
frames = sb.get_data_frames()
print(frames[0].head())  # GameHeader，看看是不是那两场 NBA Cup

# 2) 看球员日志
lg = leaguegamelog.LeagueGameLog(
    player_or_team_abbreviation="P",
    season="2025-26",
    season_type_all_star="Regular Season",
    date_from_nullable="12/09/2025",
    date_to_nullable="12/09/2025",
)
print(len(lg.get_data_frames()[0]))
