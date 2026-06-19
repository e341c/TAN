"""
Cron entrypoint: fetch candles, run quantile + trend models, persist to PostgreSQL.

From repo root:
  PYTHONPATH=. python -m technical_analysis_trainer.apps.scheduler
"""
from __future__ import annotations

import asyncio
import os
import sys

from technical_analysis_trainer.assembly.orchestrator import build_snapshot, load_assembly_config
from technical_analysis_trainer.storage.prediction_store import init_db, save_snapshot, summary_to_json


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


async def run() -> dict:
    cfg = load_assembly_config()
    symbol = _env("PREDICTION_SYMBOL", str(cfg.get("symbol", "BTCUSDT")))
    interval = _env("PREDICTION_INTERVAL", str(cfg.get("interval", "1h")))
    limit = int(_env("PREDICTION_BINANCE_LIMIT", str(cfg.get("binance_limit", 45))))
    defaults = cfg.get("defaults") or {}
    quantile_run_id = _env("QUANTILE_MODEL_RUN", str(defaults.get("quantile_run", "")))

    init_db()
    snapshot = await build_snapshot(symbol, interval, limit)
    if snapshot.candles is None or len(snapshot.candles) == 0:
        raise RuntimeError("No candle data from Binance")

    quantile = snapshot.quantile_dict() or {}
    if quantile and "last_close" not in quantile:
        quantile = {**quantile, "last_close": float(snapshot.candles["close"].iloc[-1])}

    summary = save_snapshot(snapshot, quantile_run_id=quantile_run_id or None)
    return summary


def main() -> int:
    try:
        summary = asyncio.run(run())
    except Exception as e:
        print(f"scheduled_prediction ERROR: {e}", file=sys.stderr)
        return 1
    print(summary_to_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
