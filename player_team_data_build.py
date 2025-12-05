import pandas as pd
import numpy as np



# 修改成真实数据的路径

TEAM_PATH = "/Users/yaojifan/Documents/EE547-finalproject/EE547_NBA_Project/data/team_game_features.csv"
PLAYER_FILES = [
    "/Users/yaojifan/Documents/EE547-finalproject/EE547_NBA_Project/data/player_logs_clean_season_202223.csv",
    "/Users/yaojifan/Documents/EE547-finalproject/EE547_NBA_Project/data/player_logs_clean_season_202324.csv",
    "/Users/yaojifan/Documents/EE547-finalproject/EE547_NBA_Project/data/player_logs_clean_season_202425.csv",
]


# 球员特征

feature_cols = [
    "MIN", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK",
    "TOV", "PF", "PTS",
    "PLUS_MINUS"
]

N_PER_TEAM = 15
PLAYER_INPUT_DIM = N_PER_TEAM * 2 * len(feature_cols)   # 30 × 20 = 600


# team 特征

TEAM_FEATURES = [
    "roll5_PTS_FOR",
    "roll5_PTS_AGAINST",
    "roll5_point_diff",
    "roll10_PTS_FOR",
    "roll10_point_diff",
    "roll10_win_rate",
    "season_win_rate",
]


# 读取数据

df_team = pd.read_csv(TEAM_PATH)

dfs = [pd.read_csv(f) for f in PLAYER_FILES]
df_player = pd.concat(dfs, ignore_index=True)

rows = []


# 主循环：按比赛生成

for game_id, team_group in df_team.groupby("GAME_ID"):

    if len(team_group) != 2:
        continue

    team_A = team_group.iloc[0]
    team_B = team_group.iloc[1]

    # ===== 提取球员特征矩阵 =====
    def get_player_matrix(team_row):
        tid, gid = team_row["TEAM_ID"], team_row["GAME_ID"]
        p = df_player[(df_player["TEAM_ID"] == tid) &
                      (df_player["GAME_ID"] == gid)]

        # 选前 15
        p = p.sort_values("MIN", ascending=False).head(N_PER_TEAM)
        arr = p[feature_cols].to_numpy(float)

        # 不足补 0
        if arr.shape[0] < N_PER_TEAM:
            pad = np.zeros((N_PER_TEAM - arr.shape[0], len(feature_cols)))
            arr = np.vstack([arr, pad])

        return arr

    A_mat = get_player_matrix(team_A)
    B_mat = get_player_matrix(team_B)

    # flatten: 600 维
    player_flat = np.vstack([A_mat, B_mat]).flatten().tolist()

    # ===== team 特征 =====
    A_team = team_A[TEAM_FEATURES].fillna(0).tolist()
    B_team = team_B[TEAM_FEATURES].fillna(0).tolist()

    # ===== 得分 =====
    score_A = team_A["PTS_FOR"]
    score_B = team_A["PTS_AGAINST"]   # 等同于 team_B["PTS_FOR"]

    row = player_flat + A_team + B_team + [score_A, score_B]
    rows.append(row)

# 输出 DataFrame
columns = (
    [f"f{i+1}" for i in range(PLAYER_INPUT_DIM)] +
    [f"A_{t}" for t in TEAM_FEATURES] +
    [f"B_{t}" for t in TEAM_FEATURES] +
    ["score_A", "score_B"]
)
#修改成保存训练数据的路径
df = pd.DataFrame(rows, columns=columns)
df.to_csv("nba_616_features.csv", index=False)
