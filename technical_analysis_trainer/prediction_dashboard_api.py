"""
FastAPI snapshot for the prediction dashboard (candles + quantile + trend).
Run from repo root: PYTHONPATH=. uvicorn technical_analysis_trainer.prediction_dashboard_api:app --reload --port 8001
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from technical_analysis_trainer.real_time_predictions import (
    compute_predictions_from_df,
    fetch_last_candles,
)


def _cors_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost",
        "http://127.0.0.1",
        "http://web",
    ]


app = FastAPI(title="Prediction dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _ts_to_unix_seconds(ts: Any) -> int:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        raise ValueError("Invalid timestamp")
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def _iso_utc(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()


def _serialize_candles(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append(
            {
                "time": _ts_to_unix_seconds(row["ts"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        )
    return out


def _serialize_quantile(preds: dict[str, Any]) -> dict[str, Any]:
    lr = preds.get("log_returns")
    if lr is not None:
        lr = [float(x) for x in np.asarray(lr).ravel()]
    return {
        "next_time": _iso_utc(preds["ts"]),
        "q10": float(preds["q10"]),
        "q50": float(preds["q50"]),
        "q90": float(preds["q90"]),
        "log_returns": lr,
    }


def _serialize_trend(trend: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_trend": trend.get("predicted_trend"),
        "probabilities": trend.get("probabilities"),
        "logits": [float(x) for x in (trend.get("logits") or [])],
    }


@app.get("/api/snapshot")
async def api_snapshot(
    bars: int = Query(200, ge=30, le=1000, description="Number of recent candles to return"),
) -> dict[str, Any]:
    limit = max(bars, 45)
    try:
        data = await fetch_last_candles("BTCUSDT", "1h", limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Binance fetch failed: {e}") from e

    if data is None or len(data) == 0:
        raise HTTPException(status_code=503, detail="No candle data from Binance")

    chart = data.tail(bars).reset_index(drop=True)

    try:
        quant_preds, trend_preds, trend_meta = await compute_predictions_from_df(data)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prediction failed: {e}") from e

    meta: dict[str, Any] = {
        "server_time": datetime.now(timezone.utc).isoformat(),
        "symbol": "BTCUSDT",
        "interval": "1h",
        "bars": int(bars),
    }
    if trend_meta:
        lt = trend_meta.get("last_ts")
        meta["trend_run_id"] = trend_meta.get("trend_run_id")
        meta["trend_last_ts"] = _iso_utc(lt) if lt is not None else None

    return {
        "candles": _serialize_candles(chart),
        "quantile": _serialize_quantile(quant_preds),
        "trend": _serialize_trend(trend_preds),
        "meta": meta,
    }
