#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/jasonqiu/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
CRAWLER_LOG="$LOG_DIR/crawler_$(date +"%Y%m%d").log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Record cron trigger time
echo "[run_daily_crawler.sh] started at $(date)" >> "$LOG_DIR/cron_launcher.log"

cd "$PROJECT_DIR"

# Basic debug info
echo "=== Running crawler at $(date) ===" >> "$CRAWLER_LOG"
echo "Python: $PYTHON_BIN" >> "$CRAWLER_LOG"
echo "PWD: $(pwd)" >> "$CRAWLER_LOG"

# Execute crawler script
"$PYTHON_BIN" "$PROJECT_DIR/daily_crawl_and_upload.py" >> "$CRAWLER_LOG" 2>&1

# Completion time
echo "[run_daily_crawler.sh] finished at $(date)" >> "$LOG_DIR/cron_launcher.log"
