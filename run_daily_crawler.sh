#!/bin/bash

# 1) 切到项目目录
cd /Users/jasonqiu/Documents/vscode/EE547/EE547_NBA_Project

# 2) 激活虚拟环境
source venv/bin/activate

# 3) 运行爬虫 + 上传脚本
python daily_crawl_and_upload.py >> \
  /Users/jasonqiu/Documents/vscode/EE547/EE547_NBA_Project/logs/crawler_$(date +"%Y%m%d").log 2>&1

