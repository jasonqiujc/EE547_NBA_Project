#!/bin/bash

# 切到项目目录（EC2 上的路径）
cd /home/ec2-user/EE547_NBA_Project

# 激活虚拟环境
source venv/bin/activate

# 运行训练脚本，带日志
python run_daily_training.py >> logs/training_$(date +"%Y%m%d").log 2>&1
