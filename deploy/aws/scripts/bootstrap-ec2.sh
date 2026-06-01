#!/usr/bin/env bash
# Bootstrap Ubuntu 22.04 for tan-model (run as root or with sudo).
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/tan-model}"
REPO_URL="${REPO_URL:-https://github.com/YOUR_USER/tan-model.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

if id ubuntu &>/dev/null; then
  usermod -aG docker ubuntu
  RUN_USER=ubuntu
else
  RUN_USER="${SUDO_USER:-root}"
fi

mkdir -p "$APP_DIR"
chown "$RUN_USER:$RUN_USER" "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
  sudo -u "$RUN_USER" git clone --branch "$REPO_BRANCH" "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
if [ ! -f deploy/.env ]; then
  sudo -u "$RUN_USER" cp deploy/.env.example deploy/.env
  echo "Edit $APP_DIR/deploy/.env (POSTGRES_PASSWORD, GHCR images) before starting."
fi

sudo -u "$RUN_USER" mkdir -p technical_analysis_trainer/data/predict

echo "Bootstrap done. Next:"
echo "  1. Upload predict CSV to $APP_DIR/technical_analysis_trainer/data/predict/"
echo "  2. cd $APP_DIR && docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build"
