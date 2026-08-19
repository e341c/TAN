"""Fetch Binance klines (+ optional funding) to CSV."""
from binance.client import Client
from datetime import datetime, timedelta
import argparse
import os
import time

import pandas as pd
import requests


def fetch_klines(symbol: str, interval: str, months: int) -> pd.DataFrame:
    client = Client()
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30 * months)
    print(f"📅 Fetching {months} months of {interval} klines for {symbol}")
    klines_all = []
    chunk_start = start_time
    chunk_size_days = 5
    while chunk_start < end_time:
        chunk_end = min(chunk_start + timedelta(days=chunk_size_days), end_time)
        try:
            klines = client.get_historical_klines(
                symbol,
                interval,
                chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception as e:
            print(f"⚠️ Error fetching {chunk_start:%Y-%m-%d}: {e}")
            time.sleep(1)
            continue
        if not klines:
            break
        klines_all.extend(klines)
        chunk_start = chunk_end
        time.sleep(0.25)
    if not klines_all:
        raise RuntimeError("No kline data fetched.")
    df = pd.DataFrame(
        klines_all,
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
    df = df[["ts", "open", "high", "low", "close", "volume", "trades"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float, "trades": int}
    ).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def fetch_funding_rate(symbol: str, months: int) -> pd.DataFrame:
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=30 * months)
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_rows = []
    chunk_start = start_time
    while chunk_start < end_time:
        params = {
            "symbol": symbol,
            "startTime": int(chunk_start.timestamp() * 1000),
            "endTime": int(min(chunk_start + timedelta(days=30), end_time).timestamp() * 1000),
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        chunk_start = pd.to_datetime(data[-1]["fundingTime"], unit="ms") + timedelta(hours=8)
        time.sleep(0.25)
    if not all_rows:
        raise RuntimeError("No funding rate data fetched.")
    df = pd.DataFrame(all_rows)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def merge_funding_rate(klines_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    df = klines_df.sort_values("ts").reset_index(drop=True)
    funding_df = funding_df.sort_values("ts")
    df = pd.merge_asof(df, funding_df, on="ts", direction="backward")
    df["funding_rate"] = df["funding_rate"].fillna(0.0)
    return df


def fetch_data(symbol: str, interval: str, months: int, output_dir: str, with_funding: bool = True):
    klines_df = fetch_klines(symbol, interval, months)
    if with_funding:
        try:
            funding_df = fetch_funding_rate(symbol, months)
            df = merge_funding_rate(klines_df, funding_df)
        except Exception as e:
            print(f"⚠️ Could not fetch funding rate: {e}")
            df = klines_df
    else:
        df = klines_df
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol.lower()}_{interval}.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {output_path} ({len(df)} rows)")


def check_freshness(
    output_dir: str,
    symbols: list[str],
    interval: str = "1h",
    max_age_days: int = 2,
    min_rows: int = 20000,
) -> int:
    """
    Report stale or short training files. Returns the number of problems.

    Two separate failures worth catching, both of which have bitten this repo:

      * **stale** — training data ending months before serving time means the
        model is fitted to a different volatility regime than the one it runs in;
      * **short** — a benchmark with far less history than the main symbol makes
        its cross-market features forward-filled constants during training and
        real values at inference, which is silent train/serve skew.
    """
    import pandas as pd

    problems = 0
    lengths: dict[str, int] = {}
    for sym in symbols:
        path = os.path.join(output_dir, f"{sym.lower()}_{interval}.csv")
        if not os.path.isfile(path):
            print(f"MISSING  {path}")
            problems += 1
            continue
        df = pd.read_csv(path)
        last = pd.to_datetime(df["ts"], utc=True).max()
        age = (pd.Timestamp.now(tz="UTC") - last).days
        lengths[sym] = len(df)
        flags = []
        if age > max_age_days:
            flags.append(f"STALE({age}d)")
        if len(df) < min_rows:
            flags.append(f"SHORT({len(df)} rows)")
        status = " ".join(flags) if flags else "ok"
        print(f"{status:22} {os.path.basename(path):22} rows={len(df):6} last={last.date()}")
        if flags:
            problems += 1

    # A benchmark much shorter than the main symbol is the skew case above.
    if len(lengths) > 1:
        longest = max(lengths.values())
        for sym, n in lengths.items():
            if n < longest * 0.5:
                print(
                    f"⚠ {sym} has {n} rows vs {longest} for the longest series — its "
                    "cross-market features will be forward-filled for most of training."
                )
                problems += 1
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical data from Binance.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--also", nargs="*", default=[], metavar="SYMBOL")
    parser.add_argument("--check", action="store_true", help="report staleness, do not fetch")
    parser.add_argument("--max-age-days", type=int, default=2)
    args = parser.parse_args()

    if args.check:
        n = check_freshness(
            args.output, [args.symbol, *args.also], args.interval, args.max_age_days
        )
        print(f"\n{n} problem(s)")
        raise SystemExit(1 if n else 0)

    fetch_data(args.symbol, args.interval, args.months, args.output, with_funding=not args.no_funding)
    for extra in args.also:
        fetch_data(extra, args.interval, args.months, args.output, with_funding=False)
