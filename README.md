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