"""
Cron entrypoint: fetch candles, run quantile + trend models, persist to PostgreSQL.

From repo root:
  PYTHONPATH=. python -m technical_analysis_trainer.run_scheduled_prediction
"""
from __future__ import annotations

import asyncio
import os
import sys

from technical_analysis_trainer.prediction_store import (
    init_db,
    save_prediction_run,
    summary_to_json,
)
from technical_analysis_trainer.real_time_predictions import (
    compute_predictions_from_df,
    fetch_last_candles,
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


async def run() -> dict:
    symbol = _env("PREDICTION_SYMBOL", "BTCUSDT")
    interval = _env("PREDICTION_INTERVAL", "1h")
    limit = int(_env("PREDICTION_BINANCE_LIMIT", "45"))
    quantile_run_id = _env("QUANTILE_MODEL_RUN", "2026-05-08T01-04-55")

    init_db()

    data = await fetch_last_candles(symbol, interval, limit)
    if data is None or len(data) == 0:
        raise RuntimeError("No candle data from Binance")

    quantile, trend, trend_meta = await compute_predictions_from_df(data)
    quantile["last_close"] = float(data["close"].iloc[-1])
    candle_ts = data["ts"].iloc[-1]

    summary = save_prediction_run(
        quantile,
        trend,
        trend_meta,
        symbol=symbol,
        timeframe=interval,
        candle_ts=candle_ts,
        quantile_run_id=quantile_run_id,
    )
    return summary


def main() -> int:
    try:
        summary = asyncio.run(run())
    except Exception as e:
        print(f"scheduled_prediction ERROR: {e}", file=sys.stderr)
        return 1

    print(f"scheduled_prediction OK {summary_to_json(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
