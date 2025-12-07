#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/ec2-user/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
TRAIN_LOG="$LOG_DIR/training_$(date +"%Y%m%d").log"
LAUNCHER_LOG="$LOG_DIR/cron_launcher.log"

# Create log dir
mkdir -p "$LOG_DIR"

# Record cron trigger time
echo "[CRON] run_daily_training.sh triggered at $(date)" >> "$LAUNCHER_LOG"

# Go to project folder
cd "$PROJECT_DIR"

# Activate venv
source venv/bin/activate

# Write start log
echo "=== Training START at $(date) ===" >> "$TRAIN_LOG"

# Run training
python run_daily_training.py >> "$TRAIN_LOG" 2>&1

# Write finish log
echo "=== Training END at $(date) ===" >> "$TRAIN_LOG"
echo "[CRON] run_daily_training.sh finished at $(date)" >> "$LAUNCHER_LOG"

