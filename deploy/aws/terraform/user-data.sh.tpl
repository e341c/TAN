#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl git

# Docker Engine + Compose plugin
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

usermod -aG docker ubuntu

APP_DIR="${app_dir}"
mkdir -p "$APP_DIR"
chown ubuntu:ubuntu "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
  sudo -u ubuntu git clone --branch ${repo_branch} ${repo_url} "$APP_DIR"
fi

cd "$APP_DIR"
sudo -u ubuntu tee deploy/.env >/dev/null <<ENV
POSTGRES_USER=tan
POSTGRES_PASSWORD=${postgres_password}
POSTGRES_DB=tan_predictions
DATABASE_URL=postgresql://tan:${postgres_password}@postgres:5432/tan_predictions
QUANTILE_MODEL_RUN=2026-05-08T01-04-55
TREND_MODEL_RUN=2026-04-18T07-04-16
TREND_REALTIME_CONFIG=technical_analysis_trainer/models/trend/test_config.yaml
PREDICTION_SYMBOL=BTCUSDT
PREDICTION_INTERVAL=1h
PREDICTION_BINANCE_LIMIT=45
ALLOWED_ORIGINS=http://localhost,http://127.0.0.1
WEB_PORT=80
ENV

# Placeholder dirs for bind mounts (predict CSV uploaded separately)
sudo -u ubuntu mkdir -p technical_analysis_trainer/data/predict

# Initial bring-up (build on instance; CI can switch to GHCR + --no-build later)
sudo -u ubuntu docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build
