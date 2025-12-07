#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
player_team_data_build.py (AWS S3 Version)

功能：
  - 从 S3 下载 team_game_features.csv
  - 从 S3 下载多个赛季的 player_logs_clean_season_xxx.csv
  - 生成组合后的训练数据（例如 600 + team + opponent features）
  - 输出为：nba_616_features.csv（保存在本地当前目录）

注意：
  - 所有路径都从 S3 读取
  - EC2 中临时下载到 /tmp 目录
"""

import pandas as pd
import numpy as np
from pathlib import Path
import boto3

from config_aws import S3_BUCKET, S3_PREFIX, AWS_REGION


# ---------------------------
# S3 设置
# ---------------------------

RAW_PREFIX = f"{S3_PREFIX}raw/"   # e.g. datasets/nba_project/raw/

TEAM_FILE_S3 = RAW_PREFIX + "team_game_features.csv"

PLAYER_FILES_S3 = [
    RAW_PREFIX + "player_logs_clean_season_202223.csv",
    RAW_PREFIX + "player_logs_clean_season_202324.csv",
    RAW_PREFIX + "player_logs_clean_season_202425.csv",
]


s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------
# 下载工具函数
# ---------------------------

def download_from_s3(s3_key: str) -> Path:
    """
    下载 S3 文件到 /tmp，并返回本地路径。
    """
    local_path = Path("/tmp") / Path(s3_key).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[S3] Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
    s3.download_file(S3_BUCKET, s3_key, str(local_path))
    return local_path


# ---------------------------
# 读取数据
# ---------------------------

print("\n===== Loading TEAM features from S3 =====")
team_local = download_from_s3(TEAM_FILE_S3)
df_team = pd.read_csv(team_local)

print(f"Loaded team_game_features.csv: {len(df_team)} rows")

dfs = []
print("\n===== Loading PLAYER features from S3 =====")
for s3_key in PLAYER_FILES_S3:
    local_csv = download_from_s3(s3_key)
    df = pd.read_csv(local_csv)
    dfs.append(df)
    print(f"Loaded {s3_key}: {len(df)} rows")

df_player = pd.concat(dfs, ignore_index=True)
print(f"Total player logs: {len(df_player)} rows")


# ---------------------------
# 定义特征
# ---------------------------

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
PLAYER_INPUT_DIM = N_PER_TEAM * 2 * len(feature_cols)  # 30 players × 20 features = 600

TEAM_FEATURES = [
    "roll5_PTS_FOR",
    "roll5_PTS_AGAINST",
    "roll5_point_diff",
    "roll10_PTS_FOR",
    "roll10_point_diff",
    "roll10_win_rate",
    "season_win_rate",
]


# ---------------------------
# 构建训练数据
# ---------------------------

rows = []


def get_player_matrix(team_row):
    """返回一个球队在一场比赛中的 15 名球员 × feature matrix"""
    tid, gid = team_row["TEAM_ID"], team_row["GAME_ID"]
    p = df_player[(df_player["TEAM_ID"] == tid) &
                  (df_player["GAME_ID"] == gid)]

    # 选前 15（按上场时间排序）
    p = p.sort_values("MIN", ascending=False).head(N_PER_TEAM)
    arr = p[feature_cols].to_numpy(float)

    # 不足 15 人用 0 补齐
    if arr.shape[0] < N_PER_TEAM:
        pad = np.zeros((N_PER_TEAM - arr.shape[0], len(feature_cols)))
        arr = np.vstack([arr, pad])

    return arr


print("\n===== Building combined dataset =====")

for game_id, team_group in df_team.groupby("GAME_ID"):
    if len(team_group) != 2:
        continue

    team_A = team_group.iloc[0]
    team_B = team_group.iloc[1]

    # 球员矩阵：shape = (15 players × features)
    A_mat = get_player_matrix(team_A)
    B_mat = get_player_matrix(team_B)

    # 展平 600 维
    player_flat = np.vstack([A_mat, B_mat]).flatten().tolist()

    # team features
    A_team = team_A[TEAM_FEATURES].fillna(0).tolist()
    B_team = team_B[TEAM_FEATURES].fillna(0).tolist()

    # scores
    score_A = team_A["PTS_FOR"]
    score_B = team_A["PTS_AGAINST"]

    row = player_flat + A_team + B_team + [score_A, score_B]
    rows.append(row)


# ---------------------------
# 输出 CSV
# ---------------------------

columns = (
    [f"f{i+1}" for i in range(PLAYER_INPUT_DIM)] +
    [f"A_{t}" for t in TEAM_FEATURES] +
    [f"B_{t}" for t in TEAM_FEATURES] +
    ["score_A", "score_B"]
)

df_out = pd.DataFrame(rows, columns=columns)

output_path = Path("nba_616_features.csv")
df_out.to_csv(output_path, index=False)

print(f"\n===== DONE =====")
print(f"Saved training dataset to: {output_path.resolve()}")
print(f"Total samples: {len(df_out)}")
