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

## Обучение и протокол оценки

Оба тренера используют walk-forward фолды, scaler на train, ансамбль по seed'ам
и обязательные baseline'ы. Метрика без baseline ничего не значит.

```bash
# Quantile: 3 фолда x 5 seeds, conformal-калибровка, HAR-RV benchmark
python -m technical_analysis_trainer.models.quantile.train

# Trend: triple-barrier таргет, ансамбль + GBM baseline
python -m technical_analysis_trainer.models.trend.train_v2

# Проверка протокола (splits, embargo, conformal, HAR)
python -m technical_analysis_trainer.core.test_validation
```

### Правила, зашитые в код

**Выбор конфигурации — только по val.** `core.validation.select_by_val`
выбрасывает исключение при попытке выбрать по test-метрике. Причина: в старом
`train_runs.csv` 184 прогона дали ROC-AUC 0.5305 ± 0.0052 при нулевой
корреляции с любым гиперпараметром (`hidden_size`: +0.000), а лучший прогон
(0.5457) в точности совпадает с ожидаемым максимумом 184 шумовых выборок
(0.5439). Выбор лучшего по test завышает метрику примерно на +2.5σ — в
несколько раз больше реального эджа.

**Embargo между сплитами.** Метки смотрят вперёд на `horizon` (или
`max_holding` для triple barrier) баров; без разрыва последние train-метки
залезают в val.

**Ансамбль вместо лотереи seed'ов.** Разброс между seed'ами доминирует, поэтому
модели усредняются (`QuantileEnsemble`, `TrendEnsemble`) — обе экспортируются в
один ONNX-граф.

### Что показывают метрики

| Модель | Есть эдж | Нет эджа |
|---|---|---|
| Quantile | хвосты: pinball skill **+4.4% против HAR-RV**, покрытие 90.3% при номинале 90% | медиана: skill **+0.0003**, MAE не бьёт persistence |
| Trend | triple barrier 2.5σ: val AUC 0.5237 | next_bar: AUC 0.5102; barrier 1.0σ: 0.4937 (хуже монетки) |

Ширина барьера выбрана по **val** AUC: 1.75σ→0.5141, **2.5σ→0.5237**,
3.5σ→0.5122. По test выиграл бы 3.5σ (0.5390) — это ровно та ошибка, от которой
защищает правило выше.

## Бэктест: прогон trend-модели по истории

Проигрывает ту же модель, что отдаёт прод, по историческим свечам из локальных
CSV — без сети и без БД. В консоль печатает отчёт по метрикам, в
`artifacts/backtest.html` кладёт интерактивный график, где каждое предсказание —
линия от своей свечи до барьера, который модель назвала, длиной ровно в горизонт
(24 бара). Соседние линии красятся разными цветами (палитра из 24 цветов на
24-барный горизонт), так что пересекающиеся линии всегда различимы.

```bash
# последние 500 свечей (по умолчанию, см. apps/backtest_config.yaml)
python -m technical_analysis_trainer.apps.backtest
# произвольный диапазон
python -m technical_analysis_trainer.apps.backtest --start 2026-01-01 --end 2026-06-01
# вся история, только отчёт в консоль
python -m technical_analysis_trainer.apps.backtest --last-bars 0 --no-chart
```

Диапазон, шаг, путь к HTML и предел числа линий задаются в
[`technical_analysis_trainer/apps/backtest_config.yaml`](technical_analysis_trainer/apps/backtest_config.yaml);
любой ключ перебивается флагом CLI (`--help`). `trend_run: null` означает «тот же
прогон, что в проде» — берётся `TREND_MODEL_RUN`, иначе `defaults.trend_run` из
`assembly/config.yaml`.

Модель триплбарьерная, поэтому и оценивается как триплбарьерная: эталон считает
`features.triple_barrier_labels` — та же функция, что делала метки при обучении.
Якоря, у которых за `max_holding` не задет ни один барьер, в точность не входят
(как и при обучении) и показываются отдельной строкой `unresolved`.
Проверка: прогон по тестовому фолду 2 даёт ROC-AUC 0.5860 / accuracy 55.61%
против 0.5855 / 55.40%, которые напечатал `train_v2.py` для того же прогона.

## Backfill: заполнение пропусков в предсказаниях

Дашборд накладывает сохранённые прогнозы на свечи; свеча без строки в БД —
дырка на графике (планировщик лежал, контейнер перезапускался и т.п.).

```bash
# посчитать, но не писать
python -m technical_analysis_trainer.apps.backfill_predictions --bars 200 --dry-run
# заполнить пропуски
python -m technical_analysis_trainer.apps.backfill_predictions --bars 200
# пересчитать всё заново
python -m technical_analysis_trainer.apps.backfill_predictions --bars 200 --force
```

Через API: `POST /api/backfill?bars=200`, статус — `GET /api/backfill-status`.
Автозапуск при старте: `BACKFILL_ON_START=1` (по умолчанию выключен; выполняется
фоном и никогда не блокирует старт API).

### Почему здесь нет утечки

Наивный backfill скармливает модели всю историю, и прогноз «подглядывает» в
собственное будущее — точность в панели статистики становится завышенной.
Здесь гарантия структурная, а не на доверии:

1. каждый входной фрейм (BTC + кросс-рынок) обрезается до `ts <= T` **до**
   расчёта любой фичи — будущих строк просто не существует внутри вызова;
2. обрезка проверяется `assert`, а не предполагается;
3. окно якорится так, что его содержимое совпадает с live-путём в момент T
   (`drop_last` для quantile, `last_row_is_closed` для trend).

Проверка: `python -m technical_analysis_trainer.apps.test_backfill_predictions`
— в частности, прогноз для T обязан быть **побитово одинаковым** с фреймом,
содержащим будущие свечи, и без них.

## Prediction dashboard (local)

Backend serves JSON (`candles`, live quantile forecast for the next hour, trend probabilities, and optional `quantile_history` from PostgreSQL). When `DATABASE_URL` is set and the scheduler has written rows, `/api/snapshot` includes past quantile forecasts for the visible candle window; the chart draws them as faded dashed segments. The trend pipeline needs the predict CSV from [`technical_analysis_trainer/models/trend/test_config.yaml`](technical_analysis_trainer/models/trend/test_config.yaml) (`data_path`) and ONNX/scaler under `models/trend/models/`. Quantile artifacts live under `technical_analysis_trainer/models/quantile/models/{QUANTILE_MODEL_RUN}/` (env `QUANTILE_MODEL_RUN`).

From the **repository root** (`tan-model/`):

1. API (port 8001):

```bash
PYTHONPATH=. uvicorn technical_analysis_trainer.apps.api:app --reload --port 8001
```

2. Frontend (Vite proxies `/api` to 8001):

```bash
cd web && npm install && npm run dev
```

Open `http://localhost:5173`. Use **Refresh** to reload `/api/snapshot`.

## Docker full stack (postgres + api + web + scheduler)

Runs the dashboard behind nginx on port 80, FastAPI backend, PostgreSQL, an hourly prediction scheduler (supercronic), and the Go trading bot (paper mode by default).

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

## Trading bot (Go)

The `trading_bot/` service polls predictions from the API (or PostgreSQL), evaluates a YAML strategy, and executes trades in **paper** or **live** mode.

Config: [`trading_bot/config/strategy.yaml`](trading_bot/config/strategy.yaml) (see [`strategy.example.yaml`](trading_bot/config/strategy.example.yaml)).

| Env var | Purpose |
|---------|---------|
| `STRATEGY_CONFIG` | Path to strategy YAML (default `config/strategy.yaml`) |
| `STATE_PATH` | JSON state file for position + dedupe across restarts (default `/var/lib/trading_bot/state.json`) |
| `TRADE_LOG_PATH` | Append structured trade logs (JSON lines) |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Required only for `bot.mode: live` |

**SL/TP behavior:**
- **live** — after market entry, places Binance protective orders (`STOP_LOSS` + `TAKE_PROFIT` on spot; `STOP_MARKET` + `TAKE_PROFIT_MARKET` on futures). Cancels them on manual close. `SyncPosition` detects exchange-side fills.
- **paper** — checks SL/TP against the last candle close on each cron tick.

Local run (paper, against API on port 8001):

```bash
cd trading_bot
STRATEGY_CONFIG=config/strategy.yaml STATE_PATH=/tmp/trading_bot_state.json \
  go run ./cmd/bot
```

Tests: `cd trading_bot && go test ./...`

Bot health: `curl http://localhost:8081/health` (exposed in Docker on the internal network).

## Deploy to AWS EC2

See [deploy/aws/README.md](deploy/aws/README.md) for Terraform (`deploy/aws/terraform/`) or manual bootstrap (`deploy/aws/scripts/bootstrap-ec2.sh`).

GitHub Actions:

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — build web + Docker images on PR/push
- [`.github/workflows/deploy-ec2.yml`](.github/workflows/deploy-ec2.yml) — push to GHCR and deploy to EC2 on `main`

Secrets: [deploy/aws/github-secrets.md](deploy/aws/github-secrets.md). On EC2 use `docker-compose.prod.yml` with `TAN_API_IMAGE` / `TAN_WEB_IMAGE` pointing at GHCR.

## Scheduled predictions (cron + PostgreSQL)

Hourly quantile and trend predictions are persisted to PostgreSQL by the scheduler. The dashboard computes the **current** forecast on demand and overlays **historical** quantile segments from the DB when `DATABASE_URL` is available to the API (Docker stack by default).

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
PYTHONPATH=. python -c "from technical_analysis_trainer.storage.prediction_store import init_db; init_db()"

# Or SQL
psql "$DATABASE_URL" -f technical_analysis_trainer/sql/migrations/001_predictions.sql
```

5. Run once manually:

```bash
set -a && source .env.predict && set +a
PYTHONPATH=. python -m technical_analysis_trainer.apps.scheduler
```

Exit code `0` prints `scheduled_prediction OK {...}`; non-zero on failure (suitable for cron alerts).

**Requirements:** same ONNX/scaler/CSV paths as the live dashboard (`QUANTILE_MODEL_RUN`, `models/trend/test_config.yaml` `data_path`, `models/trend/models/`).

### Cron (1h candles, UTC)

Run **5 minutes after each hour** so Binance has closed the previous bar:

```cron
5 * * * * cd /path/to/tan-model && set -a && . .env.predict && set +a && PYTHONPATH=. /path/to/venv/bin/python -m technical_analysis_trainer.apps.scheduler >> logs/scheduled_predict.log 2>&1
```

On macOS you can use `launchd` with the same `ProgramArguments` instead of crontab.

### Verify

```bash
psql "$DATABASE_URL" -c "SELECT target_ts, q50, created_at FROM quantile_predictions ORDER BY target_ts DESC LIMIT 5;"
psql "$DATABASE_URL" -c "SELECT target_ts, predicted_trend, prob_up, created_at FROM trend_predictions ORDER BY target_ts DESC LIMIT 5;"
```

Re-running the job for the same `target_ts` updates the row (UPSERT), it does not duplicate.