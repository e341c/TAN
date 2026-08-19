from __future__ import annotations

import asyncio
import re
from datetime import datetime

import pandas as pd
from binance.client import Client

_KLINE_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _klines_to_df(klines: list) -> pd.DataFrame:
    if not klines:
        raise RuntimeError("No data returned from Binance")
    df = pd.DataFrame(klines, columns=_KLINE_COLUMNS)
    # `trades` must be kept: the trend model trains on avg_trade_size_{24,72,168},
    # which are volume/trades. Dropping it here left those three features to be
    # mean-filled at inference, pushing the input vector into a region the model
    # never saw in training — the served model then predicted a single class
    # regardless of market conditions.
    df = df[["ts", "open", "high", "low", "close", "volume", "trades"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "trades": float,
        }
    ).drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)


def _to_utc_timestamp(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def fetch_last_candles_sync(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    return _klines_to_df(klines)


def fetch_klines_range_sync(
    symbol: str,
    interval: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    client = Client()
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must be >= start")
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        end_ts.strftime("%Y-%m-%d %H:%M:%S"),
    )
    return _klines_to_df(klines)


async def fetch_last_candles(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_last_candles_sync, symbol, interval, limit)


async def fetch_klines_range(
    symbol: str,
    interval: str,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_klines_range_sync, symbol, interval, start, end)


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    m = re.match(r"^(\d+)\s*([mhdw])$", interval.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"Unsupported interval: {interval!r}")
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return pd.Timedelta(minutes=n)
    if unit == "h":
        return pd.Timedelta(hours=n)
    if unit == "d":
        return pd.Timedelta(days=n)
    if unit == "w":
        return pd.Timedelta(weeks=n)
    raise ValueError(f"Unsupported interval unit: {unit!r}")
