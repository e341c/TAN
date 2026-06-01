"""
PostgreSQL persistence for quantile and trend predictions (Peewee).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from peewee import (
    CharField,
    DateTimeField,
    DoubleField,
    Model,
    Proxy,
    TextField,
)
from playhouse.db_url import connect

db_proxy = Proxy()
_database = None


def get_database():
    global _database
    if _database is not None:
        return _database
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Example: postgresql://user:pass@localhost:5432/tan_predictions"
        )
    _database = connect(url)
    db_proxy.initialize(_database)
    return _database


class QuantilePrediction(Model):
    created_at = DateTimeField(default=lambda: datetime.now(timezone.utc))
    symbol = CharField(max_length=32)
    timeframe = CharField(max_length=16)
    candle_ts = DateTimeField()
    target_ts = DateTimeField()
    model_run_id = CharField(max_length=64, null=True)
    last_close = DoubleField(null=True)
    q10 = DoubleField()
    q50 = DoubleField()
    q90 = DoubleField()
    log_return_q10 = DoubleField(null=True)
    log_return_q50 = DoubleField(null=True)
    log_return_q90 = DoubleField(null=True)

    class Meta:
        database = db_proxy
        table_name = "quantile_predictions"
        indexes = ((("symbol", "timeframe", "target_ts"), True),)


class TrendPrediction(Model):
    created_at = DateTimeField(default=lambda: datetime.now(timezone.utc))
    symbol = CharField(max_length=32)
    timeframe = CharField(max_length=16)
    candle_ts = DateTimeField()
    target_ts = DateTimeField()
    model_run_id = CharField(max_length=64, null=True)
    trend_last_ts = DateTimeField(null=True)
    predicted_trend = CharField(max_length=16)
    prob_down = DoubleField(null=True)
    prob_up = DoubleField(null=True)
    logits = TextField(null=True)

    class Meta:
        database = db_proxy
        table_name = "trend_predictions"
        indexes = ((("symbol", "timeframe", "target_ts"), True),)


def init_db() -> None:
    db = get_database()
    db.create_tables([QuantilePrediction, TrendPrediction], safe=True)


def _to_utc_datetime(ts: Any) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def _trend_probs(trend: dict[str, Any]) -> tuple[float | None, float | None]:
    probs = trend.get("probabilities") or {}
    down = probs.get("DOWN")
    if down is None:
        down = probs.get("down")
    up = probs.get("UP")
    if up is None:
        up = probs.get("up")
    return (
        float(down) if down is not None else None,
        float(up) if up is not None else None,
    )


def save_prediction_run(
    quantile: dict[str, Any],
    trend: dict[str, Any],
    trend_meta: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    candle_ts: Any,
    quantile_run_id: str,
) -> dict[str, Any]:
    """
    Upsert quantile + trend rows for the same target hour in one transaction.
    """
    now = datetime.now(timezone.utc)
    target_ts = _to_utc_datetime(quantile["ts"])
    candle_dt = _to_utc_datetime(candle_ts)

    lr = quantile.get("log_returns")
    if lr is not None:
        lr_arr = [float(x) for x in lr]
    else:
        lr_arr = [None, None, None]

    trend_last = trend_meta.get("last_ts")
    trend_last_dt = _to_utc_datetime(trend_last) if trend_last is not None else None

    logits = trend.get("logits")
    if logits is not None:
        logits_val = json.dumps([float(x) for x in logits])
    else:
        logits_val = None

    prob_down, prob_up = _trend_probs(trend)
    trend_run_id = trend_meta.get("trend_run_id")
    if trend_run_id is not None:
        trend_run_id = str(trend_run_id)

    last_close = None
    if "last_close" in quantile:
        last_close = float(quantile["last_close"])

    q_row = {
        "created_at": now,
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_ts": candle_dt,
        "target_ts": target_ts,
        "model_run_id": quantile_run_id or None,
        "last_close": last_close,
        "q10": float(quantile["q10"]),
        "q50": float(quantile["q50"]),
        "q90": float(quantile["q90"]),
        "log_return_q10": lr_arr[0] if len(lr_arr) > 0 else None,
        "log_return_q50": lr_arr[1] if len(lr_arr) > 1 else None,
        "log_return_q90": lr_arr[2] if len(lr_arr) > 2 else None,
    }

    t_row = {
        "created_at": now,
        "symbol": symbol,
        "timeframe": timeframe,
        "candle_ts": candle_dt,
        "target_ts": target_ts,
        "model_run_id": trend_run_id,
        "trend_last_ts": trend_last_dt,
        "predicted_trend": str(trend.get("predicted_trend", "")),
        "prob_down": prob_down,
        "prob_up": prob_up,
        "logits": logits_val,
    }

    q_update = {
        QuantilePrediction.created_at: now,
        QuantilePrediction.candle_ts: candle_dt,
        QuantilePrediction.model_run_id: q_row["model_run_id"],
        QuantilePrediction.last_close: q_row["last_close"],
        QuantilePrediction.q10: q_row["q10"],
        QuantilePrediction.q50: q_row["q50"],
        QuantilePrediction.q90: q_row["q90"],
        QuantilePrediction.log_return_q10: q_row["log_return_q10"],
        QuantilePrediction.log_return_q50: q_row["log_return_q50"],
        QuantilePrediction.log_return_q90: q_row["log_return_q90"],
    }

    t_update = {
        TrendPrediction.created_at: now,
        TrendPrediction.candle_ts: candle_dt,
        TrendPrediction.model_run_id: t_row["model_run_id"],
        TrendPrediction.trend_last_ts: trend_last_dt,
        TrendPrediction.predicted_trend: t_row["predicted_trend"],
        TrendPrediction.prob_down: prob_down,
        TrendPrediction.prob_up: prob_up,
        TrendPrediction.logits: logits_val,
    }

    db = get_database()
    with db.atomic():
        QuantilePrediction.insert(q_row).on_conflict(
            conflict_target=[
                QuantilePrediction.symbol,
                QuantilePrediction.timeframe,
                QuantilePrediction.target_ts,
            ],
            update=q_update,
        ).execute()
        TrendPrediction.insert(t_row).on_conflict(
            conflict_target=[
                TrendPrediction.symbol,
                TrendPrediction.timeframe,
                TrendPrediction.target_ts,
            ],
            update=t_update,
        ).execute()

    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "target_ts": target_ts.isoformat(),
        "candle_ts": candle_dt.isoformat(),
        "q50": q_row["q50"],
        "predicted_trend": t_row["predicted_trend"],
        "quantile_run_id": quantile_run_id,
        "trend_run_id": trend_run_id,
    }
    return summary


def summary_to_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False)
