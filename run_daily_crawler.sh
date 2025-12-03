#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/jasonqiu/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
CRAWLER_LOG="$LOG_DIR/crawler_$(date +"%Y%m%d").log"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 记录启动时间（写入 cron log）
echo "[run_daily_crawler.sh] started at $(date)" >> "$LOG_DIR/cron_launcher.log"

# 切换到项目目录
cd "$PROJECT_DIR"

# 写入一些 debug 信息（方便定位问题）
echo "=== Running crawler at $(date) ===" >> "$CRAWLER_LOG"
echo "Using Python: $PYTHON_BIN" >> "$CRAWLER_LOG"
echo "PWD: $(pwd)" >> "$CRAWLER_LOG"

# 运行 Python 程序
"$PYTHON_BIN" "$PROJECT_DIR/daily_crawl_and_upload.py" >> "$CRAWLER_LOG" 2>&1

# 完成时间
echo "[run_daily_crawler.sh] finished at $(date)" >> "$LOG_DIR/cron_launcher.log"
