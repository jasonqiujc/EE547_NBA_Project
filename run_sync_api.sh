#!/bin/bash
# Safe wrapper for running sync_api_files_from_s3.py from cron

set -euo pipefail

PROJECT_DIR="/home/ec2-user/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/cron_sync_api.log"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOCKFILE="$LOG_DIR/run_sync_api.lock"

mkdir -p "$LOG_DIR"

# Redirect ALL output (stdout + stderr) to log file
exec >> "$LOG_FILE" 2>&1

echo "=== [$(date)] [cron] start run_sync_api.sh (pid=$$) ==="

# Prevent overlapping sync jobs
if [ -e "$LOCKFILE" ]; then
  echo "[WARN] $(date) sync job already running (lock exists), exit."
  exit 0
fi

echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

cd "$PROJECT_DIR" || {
  echo "[ERROR] $(date) Failed to cd to $PROJECT_DIR"
  exit 1
}

echo "[INFO] $(date) Running sync_api_files_from_s3.py ..."

if "$PYTHON" sync_api_files_from_s3.py; then
  echo "[INFO] $(date) sync_api_files_from_s3.py finished successfully."
else
  rc=$?
  echo "[ERROR] $(date) sync_api_files_from_s3.py failed with exit code $rc"
  exit $rc
fi

echo "=== [$(date)] [cron] end run_sync_api.sh (pid=$$) ==="
