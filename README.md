# TAN
```
quant-regressor/
├── deploy/
│   ├── docker-compose.yml
│   ├── k8s/
│   │   ├── go-infer.yaml
│   │   ├── trainer.yaml
│   │   └── registry.yaml
├── infer/                        # инференс + сигналы + исполнение
│   ├── cmd/main.go
│   ├── internal/
│   │   ├── datafeed/             # WS биржи, ClickHouse writer
│   │   ├── features/             # RSI/MACD/ATR/z-score/σ, кэш окон
│   │   ├── model/                # onnxruntime wrapper, hot-reload
│   │   ├── signal/               # thresholds, vol targeting
│   │   ├── risk/                 # SL/TP, лимиты, cooldown
│   │   ├── exec/                 # ордера, ретраи, отчёты
│   │   ├── api/                  # REST/gRPC (signals, health, metrics)
│   │   └── store/                # ClickHouse, Postgres
│   └── config/
│       └── config.yaml
├── technical-analysis-trainer/   # обучение
│   ├── train.py                  # основной pipeline
│   ├── features.py               # расчёт фич оффлайн
│   ├── dataset.py                # выборки, split, leakage guard
│   ├── model.py                  # LGBMRegressor + early stopping
│   ├── export_onnx.py            # сохранение в ONNX + метаданные
│   ├── eval.py                   # метрики, репорт, PnL sim
│   └── trainer_api.py            # (опц) REST: /train, /status, /latest
├── model-registry/               # S3/MinIO или простой REST-сторож
│   ├── app.py                    # выдаёт версии/артефакты
│   └── storage/                  # {version}/model.onnx, meta.json
└── shared/
    ├── schemas/                  # Avro/JSON схемы сообщений/метаданных
    └── docs/                     # ТЗ, ADR, SLA
```

## Prediction dashboard (local)

Backend serves JSON (`candles`, quantile forecast for the next hour, trend probabilities). The trend pipeline still needs the predict CSV path from [`technical_analysis_trainer/trend_model/test_config.yaml`](technical_analysis_trainer/trend_model/test_config.yaml) (`data_path`) and ONNX/scaler under `trend_model/models/`. The quantile branch expects scaler/ONNX under `technical_analysis_trainer/quantile_model/models/{QUANTILE_MODEL_RUN}/` (override with env `QUANTILE_MODEL_RUN` if your run id differs).

From the **repository root** (`tan-model/`):

1. API (port 8001):

```bash
PYTHONPATH=. uvicorn technical_analysis_trainer.prediction_dashboard_api:app --reload --port 8001
```

2. Frontend (Vite proxies `/api` to 8001):

```bash
cd web && npm install && npm run dev
```

Open `http://localhost:5173`. Use **Refresh** to reload `/api/snapshot`.

## Docker full stack (postgres + api + web + scheduler)

Runs the dashboard behind nginx on port 80, FastAPI backend, PostgreSQL, and an hourly prediction scheduler (supercronic).

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env (POSTGRES_PASSWORD, model run ids)

# Ensure predict CSV exists (not in git):
# technical_analysis_trainer/data/predict/btcusdt_1h_predict.csv

docker compose -f deploy/docker-compose.yml --env-file deploy/.env up --build -d
```

- Dashboard: `http://localhost/` (or `WEB_PORT` from `.env`)
- Health: `curl http://localhost/health`
- API (internal): `http://api:8000/api/snapshot`

Stop: `docker compose -f deploy/docker-compose.yml --env-file deploy/.env down`

## Deploy to AWS EC2

See [deploy/aws/README.md](deploy/aws/README.md) for Terraform (`deploy/aws/terraform/`) or manual bootstrap (`deploy/aws/scripts/bootstrap-ec2.sh`).

GitHub Actions:

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — build web + Docker images on PR/push
- [`.github/workflows/deploy-ec2.yml`](.github/workflows/deploy-ec2.yml) — push to GHCR and deploy to EC2 on `main`

Secrets: [deploy/aws/github-secrets.md](deploy/aws/github-secrets.md). On EC2 use `docker-compose.prod.yml` with `TAN_API_IMAGE` / `TAN_WEB_IMAGE` pointing at GHCR.

## Scheduled predictions (cron + PostgreSQL)

Hourly quantile and trend predictions are persisted to PostgreSQL. The dashboard above stays **live** (on-demand); only the cron job writes to the DB.

### Setup

1. Install deps (includes `psycopg2-binary`):

```bash
pip install -r technical_analysis_trainer/requirements.txt
```

2. Start Postgres only (optional; full stack includes postgres):

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d postgres
```

Or use the full Docker stack above (scheduler writes predictions hourly).

3. Configure env (copy [`.env.predict.example`](.env.predict.example) → `.env.predict`, or use `deploy/.env`):

```bash
cp .env.predict.example .env.predict
export DATABASE_URL=postgresql://tan:tan@localhost:5432/tan_predictions
```

4. Create tables (either Peewee or SQL):

```bash
# Peewee
PYTHONPATH=. python -c "from technical_analysis_trainer.prediction_store import init_db; init_db()"

# Or SQL
psql "$DATABASE_URL" -f technical_analysis_trainer/sql/migrations/001_predictions.sql
```

5. Run once manually:

```bash
set -a && source .env.predict && set +a
PYTHONPATH=. python -m technical_analysis_trainer.run_scheduled_prediction
```

Exit code `0` prints `scheduled_prediction OK {...}`; non-zero on failure (suitable for cron alerts).

**Requirements:** same ONNX/scaler/CSV paths as the live dashboard (`QUANTILE_MODEL_RUN`, trend `test_config.yaml` `data_path`, `trend_model/models/`).

### Cron (1h candles, UTC)

Run **5 minutes after each hour** so Binance has closed the previous bar:

```cron
5 * * * * cd /path/to/tan-model && set -a && . .env.predict && set +a && PYTHONPATH=. /path/to/venv/bin/python -m technical_analysis_trainer.run_scheduled_prediction >> logs/scheduled_predict.log 2>&1
```

On macOS you can use `launchd` with the same `ProgramArguments` instead of crontab.

### Verify

```bash
psql "$DATABASE_URL" -c "SELECT target_ts, q50, created_at FROM quantile_predictions ORDER BY target_ts DESC LIMIT 5;"
psql "$DATABASE_URL" -c "SELECT target_ts, predicted_trend, prob_up, created_at FROM trend_predictions ORDER BY target_ts DESC LIMIT 5;"
```

Re-running the job for the same `target_ts` updates the row (UPSERT), it does not duplicate.