#!/bin/bash
# Safe wrapper for running run_daily_training.py from cron

set -euo pipefail

PROJECT_DIR="/home/ec2-user/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/cron_training.log"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOCKFILE="$LOG_DIR/run_training_cron.lock"

mkdir -p "$LOG_DIR"

# 把整个脚本的输出都重定向到日志文件
exec >> "$LOG_FILE" 2>&1

echo "=== [$(date)] [cron] start run_training_cron.sh (pid=$$) ==="

# 防止重复运行：如果上一次还没跑完，直接退出
if [ -e "$LOCKFILE" ]; then
  echo "[WARN] $(date) lock file exists ($LOCKFILE), another run may still be running. Exit."
  exit 0
fi

echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

cd "$PROJECT_DIR" || { 
  echo "[ERROR] $(date) Failed to cd to $PROJECT_DIR"; 
  exit 1; 
}

echo "[INFO] $(date) Calling run_daily_training.py with $PYTHON ..."

if "$PYTHON" run_daily_training.py; then
  echo "[INFO] $(date) run_daily_training.py finished successfully."
else
  rc=$?
  echo "[ERROR] $(date) run_daily_training.py failed with exit code $rc"
  exit $rc
fi

echo "=== [$(date)] [cron] run_training_cron.sh finished (pid=$$) ==="
