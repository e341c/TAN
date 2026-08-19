"""
FastAPI snapshot for the prediction dashboard (candles + quantile + trend).
Run from repo root:
  PYTHONPATH=. uvicorn technical_analysis_trainer.apps.api:app --reload --port 8001
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from technical_analysis_trainer.assembly.orchestrator import build_snapshot
from technical_analysis_trainer.models.quantile.features import required_history
from technical_analysis_trainer.runtime.binance import fetch_last_candles
from technical_analysis_trainer.storage.prediction_metrics import compute_model_stats
from technical_analysis_trainer.storage.prediction_store import (
    list_quantile_predictions,
    list_trend_predictions,
    save_snapshot,
)

logger = logging.getLogger(__name__)


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


_backfill_state: dict[str, Any] = {"status": "idle", "result": None, "error": None}


async def _run_startup_backfill() -> None:
    from technical_analysis_trainer.apps.backfill_predictions import backfill

    bars = int(os.environ.get("BACKFILL_BARS", "200"))
    _backfill_state["status"] = "running"
    try:
        res = await backfill(bars=bars)
        _backfill_state.update(status="done", result=res.as_dict(), error=None)
        logger.info("startup backfill complete: %s", res.as_dict())
    except Exception as e:  # noqa: BLE001
        _backfill_state.update(status="failed", error=f"{type(e).__name__}: {e}")
        logger.warning("startup backfill failed: %s", e)


@app.on_event("startup")
async def _maybe_backfill() -> None:
    """
    Optionally fill prediction gaps for the displayed range at boot.

    Off by default and always detached: filling 200 candles takes minutes, and
    the dashboard must come up immediately regardless. A failure here is logged
    and surfaced on /api/backfill-status, never raised — a gap in the chart is a
    far smaller problem than an API that will not start.
    """
    if os.environ.get("BACKFILL_ON_START", "").strip().lower() not in {"1", "true", "yes"}:
        return
    if not os.environ.get("DATABASE_URL", "").strip():
        logger.warning("BACKFILL_ON_START set but DATABASE_URL is not; skipping")
        return
    import asyncio

    asyncio.create_task(_run_startup_backfill())


@app.get("/api/backfill-status")
def api_backfill_status() -> dict[str, Any]:
    return dict(_backfill_state)


@app.post("/api/backfill")
async def api_backfill(
    bars: int = Query(200, ge=30, le=1000),
    dry_run: bool = Query(False),
    max_fill: int | None = Query(None, ge=1, le=1000),
) -> dict[str, Any]:
    """Trigger a gap fill on demand."""
    if not os.environ.get("DATABASE_URL", "").strip():
        raise HTTPException(status_code=503, detail="DATABASE_URL is not set")
    from technical_analysis_trainer.apps.backfill_predictions import backfill

    try:
        res = await backfill(bars=bars, dry_run=dry_run, max_fill=max_fill)
        return res.as_dict()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Backfill failed: {e}") from e


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
        # Full predicted distribution. The model emits five levels; q10/q90 above
        # are interpolated from them for backward compatibility.
        "quantiles": preds.get("quantiles"),
        "quantile_levels": preds.get("quantile_levels"),
        "sigma": preds.get("sigma"),
    }


def _serialize_trend(trend: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_trend": trend.get("predicted_trend"),
        "probabilities": trend.get("probabilities"),
        "logits": [float(x) for x in (trend.get("logits") or [])],
        # Ship the target definition with the label. A bare "UP"/"DOWN" is
        # ambiguous between a 1h close and a 24h barrier touch, and both the web
        # UI and the trading bot need to know which one they are looking at.
        "target": trend.get("target"),
        "horizon_hours": trend.get("horizon_hours"),
    }


@app.get("/api/snapshot")
async def api_snapshot(
    bars: int = Query(200, ge=30, le=1000, description="Number of recent candles to return"),
) -> dict[str, Any]:
    # Fetch enough history for the slowest feature to warm up, independent of how
    # many bars the chart shows. Sizing this off `bars` alone means a small chart
    # request silently starves the model of history.
    limit = max(bars, required_history() + 60)
    try:
        data = await fetch_last_candles("BTCUSDT", "1h", limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Binance fetch failed: {e}") from e

    if data is None or len(data) == 0:
        raise HTTPException(status_code=503, detail="No candle data from Binance")

    chart = data.tail(bars).reset_index(drop=True)

    try:
        snapshot = await build_snapshot("BTCUSDT", "1h", limit, candles_df=data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prediction failed: {e}") from e

    quant_preds = snapshot.quantile_dict() or {}
    trend_preds = snapshot.trend_dict() or {}
    trend_meta = snapshot.trend_meta()

    # Persist the forecast we just computed. Without this the dashboard produces
    # a fresh prediction on every page load and throws it away, so the only rows
    # that ever reach the database are the scheduler's — and any hour the
    # scheduler misses is a permanent hole. Never fatal: the chart must render
    # even when the database is down.
    if os.environ.get("DATABASE_URL", "").strip() and quant_preds and trend_preds:
        try:
            await asyncio.to_thread(
                save_snapshot, snapshot, quantile=quant_preds,
                trend=trend_preds, trend_meta=trend_meta,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("snapshot save failed: %s", e)

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

    quantile_history: list[dict[str, Any]] = []
    trend_history: list[dict[str, Any]] = []
    if len(chart) > 0:
        candle_min = pd.Timestamp(chart["ts"].iloc[0])
        candle_max = pd.Timestamp(chart["ts"].iloc[-1])
        if candle_min.tzinfo is None:
            candle_min = candle_min.tz_localize("UTC")
        else:
            candle_min = candle_min.tz_convert("UTC")
        if candle_max.tzinfo is None:
            candle_max = candle_max.tz_localize("UTC")
        else:
            candle_max = candle_max.tz_convert("UTC")
        candle_ts_min = candle_min.to_pydatetime()
        candle_ts_max = candle_max.to_pydatetime()
        try:
            quantile_history = list_quantile_predictions(
                "BTCUSDT",
                "1h",
                candle_ts_min=candle_ts_min,
                candle_ts_max=candle_ts_max,
                limit=bars + 10,
            )
        except Exception as e:
            logger.warning("quantile_history fetch failed: %s", e)
            quantile_history = []
        try:
            trend_history = list_trend_predictions(
                "BTCUSDT",
                "1h",
                candle_ts_min=candle_ts_min,
                candle_ts_max=candle_ts_max,
                limit=bars + 10,
            )
        except Exception as e:
            logger.warning("trend_history fetch failed: %s", e)
            trend_history = []

    return {
        "candles": _serialize_candles(chart),
        "quantile": _serialize_quantile(quant_preds),
        "quantile_history": quantile_history,
        "trend": _serialize_trend(trend_preds),
        "trend_history": trend_history,
        "meta": meta,
    }


def _load_trend_target_cfg() -> dict[str, Any]:
    """Read the trend target definition from its config, with safe defaults."""
    import yaml

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "trend", "config.yaml",
    )
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        cfg = {}
    return {
        "target_mode": str(cfg.get("target_mode", "next_bar")),
        "triple_barrier": cfg.get("triple_barrier") or {},
    }


@app.get("/api/model-stats")
async def api_model_stats(
    limit: int = Query(200, ge=10, le=1000, description="Max mature predictions to evaluate"),
) -> dict[str, Any]:
    if not os.environ.get("DATABASE_URL", "").strip():
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not set; model stats require PostgreSQL",
        )
    try:
        # The trend model's target must match how its predictions are scored.
        # A triple-barrier model graded against the next bar's sign measures a
        # different question entirely and reports a meaningless accuracy.
        trend_cfg = _load_trend_target_cfg()
        return await compute_model_stats(
            "BTCUSDT",
            "1h",
            limit=limit,
            trend_target_mode=trend_cfg["target_mode"],
            trend_tb_cfg=trend_cfg["triple_barrier"],
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model stats failed: {e}") from e
