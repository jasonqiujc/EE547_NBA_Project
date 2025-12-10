#!/bin/bash
# Cron-safe wrapper for running run_daily_training.py

set -euo pipefail

PROJECT_DIR="/home/ec2-user/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/cron_training.log"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOCKFILE="$LOG_DIR/run_training_cron.lock"

mkdir -p "$LOG_DIR"

# Redirect all script output to logfile
exec >> "$LOG_FILE" 2>&1

echo "=== [$(date)] [cron] start run_training_cron.sh (pid=$$) ==="

# Prevent concurrent executions
if [ -e "$LOCKFILE" ]; then
  echo "[WARN] $(date) Lock file exists ($LOCKFILE); another run may still be active. Exiting."
  exit 0
fi

echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

cd "$PROJECT_DIR" || {
  echo "[ERROR] $(date) Failed to cd into $PROJECT_DIR"
  exit 1
}

echo "[INFO] $(date) Running run_daily_training.py using $PYTHON ..."

if "$PYTHON" run_daily_training.py; then
  echo "[INFO] $(date) run_daily_training.py completed successfully."
else
  rc=$?
  echo "[ERROR] $(date) run_daily_training.py failed with exit code $rc"
  exit $rc
fi

echo "=== [$(date)] [cron] run_training_cron.sh finished (pid=$$) ==="

