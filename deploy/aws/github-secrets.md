# GitHub Actions secrets

Configure in **Settings → Secrets and variables → Actions** for the repository.

| Secret | Required | Description |
|--------|----------|-------------|
| `EC2_HOST` | deploy | Public IP or DNS of the EC2 instance (Terraform output `public_ip`) |
| `EC2_USER` | deploy | SSH user, usually `ubuntu` |
| `EC2_SSH_KEY` | deploy | Private key PEM (full contents, including `BEGIN`/`END` lines) |
| `GHCR_TOKEN` | deploy | GitHub PAT with `read:packages` for `docker login ghcr.io` on EC2 |

`GITHUB_TOKEN` is provided automatically for pushing images to GHCR from Actions (`packages: write` permission in workflow).

## EC2 one-time setup

1. Provision host (Terraform or `scripts/bootstrap-ec2.sh`).
2. Clone repo to `/opt/tan-model` (bootstrap does this).
3. Create `deploy/.env` with production values and GHCR image names:

```env
TAN_API_IMAGE=ghcr.io/YOUR_GITHUB_USER/tan-api:latest
TAN_WEB_IMAGE=ghcr.io/YOUR_GITHUB_USER/tan-web:latest
POSTGRES_PASSWORD=...
DATABASE_URL=postgresql://tan:...@postgres:5432/tan_predictions
```

4. Upload `technical_analysis_trainer/data/predict/btcusdt_1h_predict.csv` (not in git).
5. First deploy from Actions or manually:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
cd /opt/tan-model
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml --env-file deploy/.env pull
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml --env-file deploy/.env up -d --no-build
```
