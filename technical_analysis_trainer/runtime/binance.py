from __future__ import annotations

import asyncio

import pandas as pd
from binance.client import Client


def fetch_last_candles_sync(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not klines:
        raise RuntimeError("No data returned from Binance")
    df = pd.DataFrame(
        klines,
        columns=[
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
        ],
    )
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        }
    )


async def fetch_last_candles(symbol: str, interval: str, limit: int = 30) -> pd.DataFrame:
    return await asyncio.to_thread(fetch_last_candles_sync, symbol, interval, limit)
