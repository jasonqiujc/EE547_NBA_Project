# NBA Daily Pipeline Deployment Guide

This document explains how to deploy and operate the full end-to-end daily pipeline used in this project.  
The system consists of:

- Local daily crawler  
- AWS S3 data storage  
- EC2 daily training pipeline  
- EC2 daily API sync  
- FastAPI backend serving predictions  

The goal is to allow any new user to reproduce the entire system from scratch.

---

## 1. System Architecture

```
Local Machine (Daily Crawler)
        ↓ uploads raw data
AWS S3 Bucket (datasets/nba_project/raw)
        ↓ EC2 downloads raw data
EC2 Daily Training Job
        ↓ uploads trained model
AWS S3 Bucket (datasets/nba_project/models)
        ↓ EC2 sync job downloads data
FastAPI Server (uses latest model + data files)
```

---

## 2. Requirements

### Local Machine
- macOS or Linux  
- Python 3.9+  
- Git  
- AWS CLI with S3 read/write credentials  

### AWS Resources
- S3 bucket: ee547-nba-project  
- S3 prefix: datasets/nba_project/  
- EC2 instance: Amazon Linux 2 or 2023  
- Instance type: t3.micro or larger  
- IAM role with S3 read/write permissions  

---

## 3. Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/EE547_NBA_Project.git
cd EE547_NBA_Project
```

---

## 4. Python Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. AWS Configuration

Edit the file:

```
config_aws.py
```

Example configuration:

```python
AWS_REGION = "us-west-1"
S3_BUCKET = "ee547-nba-project"
S3_PREFIX = "datasets/nba_project/"

# Local machine example:
# LOCAL_DATA_DIR = Path("/Users/<yourname>/EE547_NBA_Project/data")

# EC2 machine:
LOCAL_DATA_DIR = Path("/home/ec2-user/EE547_NBA_Project/data")
```

---

## 6. Local Daily Crawler

The crawler script:

```
daily_crawl_and_upload.py
```

This script:

- Fetches yesterday's player logs  
- Builds yesterday's games  
- Fetches schedule for today + next 4 days  
- Uploads all raw CSVs to S3  

### Run manually:

```bash
python daily_crawl_and_upload.py
```

### Set up cron on macOS:

```
crontab -e
```

Add:

```cron
55 16 * * * cd /Users/<yourname>/EE547_NBA_Project && \
/Users/<yourname>/EE547_NBA_Project/venv/bin/python daily_crawl_and_upload.py \
>> /Users/<yourname>/EE547_NBA_Project/logs/cron_crawl.log 2>&1
```

---

## 7. EC2 Setup

### Clone and install:

```bash
cd /home/ec2-user
git clone https://github.com/<YOUR_USERNAME>/EE547_NBA_Project.git
cd EE547_NBA_Project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 8. Training Wrapper (Cron Safe)

File: `run_training_cron.sh`

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/ec2-user/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/cron_training.log"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOCKFILE="$LOG_DIR/run_training_cron.lock"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1

echo "[START] Training (cron)"

if [ -e "$LOCKFILE" ]; then
  echo "[WARN] Training already running. Exit."
  exit 0
fi

echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

cd "$PROJECT_DIR"
$PYTHON run_daily_training.py

echo "[END] Training completed"
```

Make executable:

```bash
chmod +x run_training_cron.sh
```

---

## 9. Sync Wrapper

File: `run_sync_api.sh`

```bash
#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/ec2-user/EE547_NBA_Project"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/cron_sync_api.log"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOCKFILE="$LOG_DIR/run_sync_api.lock"

mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1

echo "[START] Sync (cron)"

if [ -e "$LOCKFILE" ]; then
  echo "[WARN] Sync already running. Exit."
  exit 0
fi

echo "$$" > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

cd "$PROJECT_DIR"
$PYTHON sync_api_files_from_s3.py

echo "[END] Sync completed"
```

Make executable:

```bash
chmod +x run_sync_api.sh
```

---

## 10. EC2 Cron Configuration

Enable cron:

```bash
sudo systemctl start crond
sudo systemctl enable crond
```

Edit crontab:

```bash
crontab -e
```

Recommended schedule:

```cron
24 18 * * * /home/ec2-user/EE547_NBA_Project/run_training_cron.sh
30 18 * * * /home/ec2-user/EE547_NBA_Project/run_sync_api.sh
```

---

## 11. FastAPI Server

Start manually:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API reads:

- model files from `data/models/model_latest.pth`  
- game and schedule CSVs from `data/`  

---

## 12. Monitoring

### Training logs:

```
logs/cron_training.log
logs/training_YYYYMMDD.log
```

### Sync logs:

```
logs/cron_sync_api.log
```

### Cron service logs:

```bash
sudo journalctl -u crond -n 50
```

### Check running processes:

```bash
ps aux | grep training
```

---

## 13. Project Structure

## Project Structure

```
EE547_NBA_PROJECT/
│
├── __pycache__/                # Python cache files
├── data/                       # Local data (ignored by git)
├── logs/                       # Training/crawler/sync logs (ignored by git)
├── venv/                       # Python virtual environment (ignored by git)
│
├── .gitignore                  # Git ignore rules
│
├── api_server.py               # FastAPI backend server
├── build_datasets.py           # Build datasets from raw logs
├── build_team_features.py      # Build team-level features for training
├── config_aws.py               # AWS/S3 configuration
├── daily_crawl_and_upload.py   # Local crawler (runs daily)
├── player_team_data_build.py   # Player→team feature aggregation
├── requirements.txt            # Python dependencies
│
├── run_daily_crawler.sh        # Cron wrapper for crawler (local)
├── run_daily_training.py       # Training logic (called by wrapper)
├── run_daily_training.sh       # Manual training shell script
├── run_sync_api.sh             # Cron-safe sync-to-EC2 wrapper
├── run_training_cron.sh        # Cron-safe training wrapper
│
├── sync_api_files_from_s3.py   # Downloads raw/schedule files for API
├── train_model.py              # Model training utilities
├── upload_raw_to_s3.py         # Utility for uploading raw data
```


---

## 14. Summary

Once deployed:

- Local machine uploads raw NBA data daily  
- EC2 trains a new model daily  
- EC2 syncs updated files daily  
- FastAPI always serves predictions using the latest model  

This guide allows any user to deploy the entire system end-to-end.

