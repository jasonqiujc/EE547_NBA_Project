#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
player_team_data_build.py (AWS S3 version)

Goal:
  - Download team_game_features.csv from S3
  - Download multiple seasons of player_logs_clean_season_xxx.csv from S3
  - Build combined training data: player (600 dims) + team + opponent features
  - Save as nba_616_features.csv in the current directory
"""

import pandas as pd
import numpy as np
from pathlib import Path
import boto3

from config_aws import S3_BUCKET, S3_PREFIX, AWS_REGION


# ---------------------------
# S3 paths
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
# S3 download helper
# ---------------------------

def download_from_s3(s3_key: str) -> Path:
    """
    Download S3 object to /tmp and return local path.
    """
    local_path = Path("/tmp") / Path(s3_key).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[S3] Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
    s3.download_file(S3_BUCKET, s3_key, str(local_path))
    return local_path


# ---------------------------
# Load data
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
# Feature definitions
# ---------------------------

feature_cols = [
    "MIN", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK",
    "TOV", "PF", "PTS",
    "PLUS_MINUS",
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
# Build training rows
# ---------------------------

rows = []


def get_player_matrix(team_row):
    """
    Return a (N_PER_TEAM x num_features) matrix for one team in one game.
    """
    tid, gid = team_row["TEAM_ID"], team_row["GAME_ID"]
    p = df_player[(df_player["TEAM_ID"] == tid) & (df_player["GAME_ID"] == gid)]

    # Top N_PER_TEAM players by minutes
    p = p.sort_values("MIN", ascending=False).head(N_PER_TEAM)
    arr = p[feature_cols].to_numpy(float)

    # Pad to N_PER_TEAM if fewer players
    if arr.shape[0] < N_PER_TEAM:
        pad = np.zeros((N_PER_TEAM - arr.shape[0], len(feature_cols)))
        arr = np.vstack([arr, pad])

    return arr


print("\n===== Building combined dataset =====")

for game_id, team_group in df_team.groupby("GAME_ID"):
    # Expect exactly home + away
    if len(team_group) != 2:
        continue

    team_A = team_group.iloc[0]
    team_B = team_group.iloc[1]

    # Player matrices
    A_mat = get_player_matrix(team_A)
    B_mat = get_player_matrix(team_B)

    # Flatten player features (600 dims)
    player_flat = np.vstack([A_mat, B_mat]).flatten().tolist()

    # Team-level features
    A_team = team_A[TEAM_FEATURES].fillna(0).tolist()
    B_team = team_B[TEAM_FEATURES].fillna(0).tolist()

    # Targets (scores)
    score_A = team_A["PTS_FOR"]
    score_B = team_A["PTS_AGAINST"]

    row = player_flat + A_team + B_team + [score_A, score_B]
    rows.append(row)


# ---------------------------
# Save CSV
# ---------------------------

columns = (
    [f"f{i + 1}" for i in range(PLAYER_INPUT_DIM)] +
    [f"A_{t}" for t in TEAM_FEATURES] +
    [f"B_{t}" for t in TEAM_FEATURES] +
    ["score_A", "score_B"]
)

df_out = pd.DataFrame(rows, columns=columns)

output_path = Path("nba_616_features.csv")
df_out.to_csv(output_path, index=False)

print("\n===== DONE =====")
print(f"Saved training dataset to: {output_path.resolve()}")
print(f"Total samples: {len(df_out)}")
