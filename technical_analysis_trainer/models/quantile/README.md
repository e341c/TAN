# Quantile LSTM — прогноз распределения доходности

Модуль учит **одну LSTM** выдавать сразу несколько **условных квантилей** будущей **лог-доходности** по окну последних свечей.

## Запуск обучения

```bash
python -m technical_analysis_trainer.models.quantile.train
```

Артефакты сохраняются в `models/quantile/models/{timestamp}/`:
- `model_{timestamp}.onnx`
- `scaler_{timestamp}.pkl`

Инференс в production — через `assembly` (`QuantilePlugin`), не напрямую из этого модуля.

## Архитектура

- `QuantileLSTM` в [`model.py`](model.py): LSTM → линейный слой → квантили **0.1, 0.5, 0.9**
- Loss: pinball (`QuantileLoss`)
- Признаки: 7 каналов (OHLC, log_return, volume_log, rsi_scaled) — см. [`features.py`](features.py)

Подробное описание метрик, baseline и ограничений — см. историческую документацию в git history `quantile_model/README.md`.
